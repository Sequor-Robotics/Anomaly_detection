import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import print_n_txt
from MDN.loss import mdn_loss,mdn_eval,mdn_uncertainties
from MDN.network import MixtureDensityNetwork
from VAE.network import VAE
from VAE.loss import VAE_loss,VAE_eval
from tools.dataloader import mixquality_dataset
from pathlib import Path
from tqdm.auto import tqdm
import threading
import itertools
import sys
import time



class _Spinner:
    def __init__(self, text="Loading", interval=0.12):
        self.text = text
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return

        def run():
            for ch in itertools.cycle("|/-\\"):
                if self._stop.is_set():
                    break
                sys.stdout.write(f"\r{self.text} {ch}")
                sys.stdout.flush()
                time.sleep(self.interval)

            # clear line
            sys.stdout.write("\r" + " " * (len(self.text) + 4) + "\r")
            sys.stdout.flush()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self, final_text=None):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if final_text:
            print(final_text)


class solver():

    def __init__(self, args, device, SEED):
        self.EPOCH = args.epoch
        self.device = device
        self.lr = args.lr
        self.wd = args.wd
        self.lr_rate = args.lr_rate
        self.lr_step = args.lr_step
        self.CLIP = 1
        self.SEED=SEED
        self.neg_case = args.neg_case

        self.load_iter(args)
        self.load_model(args)


    def load_model(self,args):

        if args.mode== 'vae':
            self.model      = VAE( x_dim=self.data_dim[0], 
                                   h_dim=args.h_dim, z_dim=args.z_dim ).to(self.device)
            self.train_func = self.train_VAE
            self.eval_func  = self.eval_ood_VAE

        elif args.mode == 'mdn':
            self.model      = MixtureDensityNetwork( name='mdn',x_dim=self.data_dim[0], y_dim=self.data_dim[1],k=args.k,h_dims=[128,128], actv=nn.ReLU(),
                                                     sig_max=args.sig_max, mu_min=-3, mu_max=+3, dropout=args.dropout ).to(self.device)
            self.train_func = self.train_mdn
            self.eval_func  = self.eval_ood_mdn


    def init_param(self):
        self.model.init_param()


    def load_iter(self, args):
        root = str(Path(args.root).resolve() / "Data")

        sp = _Spinner("Loading datasets / dataloaders")
        sp.start()

        try:
            # train data
            self.train_dataset = mixquality_dataset(
                root=root, train=True, norm=args.norm, frame=args.frame,
                exp_case=args.exp_case, neg_case=args.neg_case
            )
            self.train_iter = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=args.batch, shuffle=False
            )
            torch.manual_seed(self.SEED)

            # test expert
            self.test_e_dataset = mixquality_dataset(
                root=root, train=False, neg=False, norm=args.norm, frame=args.frame,
                exp_case=args.exp_case
            )
            self.test_e_iter = torch.utils.data.DataLoader(
                self.test_e_dataset, batch_size=args.batch, shuffle=False
            )
            torch.manual_seed(self.SEED)

            # test negative
            self.test_n_dataset = mixquality_dataset(
                root=root, train=False, neg=True, norm=args.norm, frame=args.frame,
                neg_case=self.neg_case
            )
            self.test_n_iter = torch.utils.data.DataLoader(
                self.test_n_dataset, batch_size=args.batch, shuffle=False
            )

            self.data_dim = [self.train_dataset.x.size(-1), self.train_dataset.y.size(-1)]

        except Exception as e:
            sp.stop(final_text=f"[ERROR] Loading failed: {e}")
            raise

        sp.stop(final_text="\nDone!")
        print(f"[DATA] #samples | train={len(self.train_dataset)} | test_e(ID)={len(self.test_e_dataset)} | test_n(OOD)={len(self.test_n_dataset)}")



    ### VAE
    def train_VAE(self, f):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.lr_rate, step_size=self.lr_step)

        train_l2, test_l2 = [], []

        epoch_bar = tqdm(range(self.EPOCH), desc="Train(VAE)", unit="epoch", dynamic_ncols=True)

        for epoch in epoch_bar:
            loss_sum = 0.0

            batch_bar = tqdm(self.train_iter, desc=f"Epoch {epoch+1}/{self.EPOCH}", unit="batch",
                            leave=False, dynamic_ncols=True)

            for batch_in, batch_out in batch_bar:
                x_reconst, mu, logvar = self.model.forward(batch_in.to(self.device))
                loss_out = VAE_loss(batch_in.to(self.device), x_reconst, mu, logvar)
                loss = torch.mean(loss_out['loss'])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP)
                optimizer.step()

                loss_sum += float(loss.item())
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            loss_avg = loss_sum / max(1, len(self.train_iter))

            # epoch-end eval
            train_out    = self.test_eval_VAE(self.train_iter, 'cuda')
            test_in_out  = self.test_eval_VAE(self.test_e_iter, 'cuda')
            test_ood_out = self.test_eval_VAE(self.test_n_iter, 'cuda')

            # summary
            epoch_bar.set_postfix(
                loss=f"{loss_avg:.3f}",
                train=f"{train_out['total']:.4f}",
                test=f"{test_in_out['total']:.4f}"
            )

            # log
            if (epoch % 10) == 0:
                print_n_txt(f, f"\nepoch: [{epoch}/{self.EPOCH}] loss: [{loss_avg:.3f}] "
                        f"train_loss:[{train_out['total']:.4f}] test_loss: [{test_in_out['total']:.4f}]")
                print_n_txt(f, f"[ID]  recon avg: [{test_in_out['recon']:.3f}] kl_div avg: [{test_in_out['kl_div']:.3f}]")
                print_n_txt(f, f"[OOD] recon avg: [{test_ood_out['recon']:.3f}] kl_div avg: [{test_ood_out['kl_div']:.3f}]")

            train_l2.append(train_out['total'])
            test_l2.append(test_in_out['total'])

        return train_l2, test_l2

    def eval_ood_VAE(self,data_iter,device):
        with torch.no_grad():
            n_total= 0
            recon_ , kl_  = list(),list()
            self.model.eval() # evaluate (affects DropOut and BN)
            for batch_in,batch_out in data_iter:
                #batch_in = torch.cat((batch_in,batch_out),dim=1)
                x_recon, mu, logvar = self.model.forward(batch_in.to(device))
                loss_out = VAE_eval(batch_in.to(self.device), x_recon, mu, logvar)
                recon   = loss_out['reconst_loss'] # [N x D]
                kl  = loss_out['kl_div'] # [N x D]
                recon_ += recon.cpu().numpy().tolist()
                kl_ += kl.cpu().numpy().tolist()
                n_total += batch_in.size(0)
            self.model.train() # back to train mode
            out_eval = {'recon_' : recon_,'kl_' : kl_}
        return out_eval

    def test_eval_VAE (self, data_iter, device):
        with torch.no_grad():
            n_total,recon,kl_div,total_loss = 0,0,0,0
            self.model.eval() 
            for batch_in,batch_out in data_iter:
                #batch_in = torch.cat((batch_in,batch_out),dim=1)
                x_reconst, mu, logvar = self.model.forward(batch_in.to(device))
                loss_out = VAE_loss(batch_in.to(self.device), x_reconst, mu, logvar)
                recon += torch.sum(loss_out['reconst_loss'])
                kl_div += torch.sum(loss_out['kl_div'])
                total_loss += torch.sum(loss_out['loss'])
                n_total += batch_in.size(0)
            recon_avg=(recon/n_total).detach().cpu().item()
            kl_avg = (kl_div/n_total).detach().cpu().item()
            total_avg = (total_loss/n_total).detach().cpu().item()
            self.model.train() # back to train mode
            out_eval = {'recon':recon_avg,'kl_div':kl_avg, 'total':total_avg}
        return out_eval


    ### MDN
    def train_mdn(self, f):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.lr_rate, step_size=self.lr_step)

        train_l2, test_l2 = [], []

        epoch_bar = tqdm(range(self.EPOCH), desc="Train(MDN)", unit="epoch", dynamic_ncols=True)

        for epoch in epoch_bar:
            loss_sum = 0.0

            batch_bar = tqdm(self.train_iter, desc=f"Epoch {epoch+1}/{self.EPOCH}", unit="batch",
                            leave=False, dynamic_ncols=True)

            for batch_in, batch_out in batch_bar:
                out = self.model.forward(batch_in.to(self.device))
                pi, mu, sigma = out['pi'], out['mu'], out['sigma']

                loss_out = mdn_loss(pi, mu, sigma, batch_out.to(self.device))
                loss = torch.mean(loss_out['nll'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.item())
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            loss_avg = loss_sum / max(1, len(self.train_iter))

            train_out    = self.test_eval_mdn(self.train_iter, 'cuda')
            test_in_out  = self.test_eval_mdn(self.test_e_iter, 'cuda')
            test_ood_out = self.test_eval_mdn(self.test_n_iter, 'cuda')

            epoch_bar.set_postfix(
                loss=f"{loss_avg:.3f}",
                train=f"{train_out['l2_norm']:.4f}",
                test=f"{test_in_out['l2_norm']:.4f}"
            )

            print_n_txt(f, f"\nepoch: [{epoch}/{self.EPOCH}] loss: [{loss_avg:.3f}] "
                        f"train_l2:[{train_out['l2_norm']:.4f}] test_l2: [{test_in_out['l2_norm']:.4f}]")

            print_n_txt(f, f"[ID]  epis avg: [{test_in_out['epis']:.3f}] "
                        f"alea avg: [{test_in_out['alea']:.3f}] pi_entropy avg: [{test_in_out['pi_entropy']:.3f}]")

            print_n_txt(f, f"[OOD] epis avg: [{test_ood_out['epis']:.3f}] "
                        f"alea avg: [{test_ood_out['alea']:.3f}] pi_entropy avg: [{test_ood_out['pi_entropy']:.3f}]")

            train_l2.append(train_out['l2_norm'])
            test_l2.append(test_in_out['l2_norm'])

        return train_l2, test_l2

    def eval_ood_mdn(self,data_iter,device):
        with torch.no_grad():
            n_total= 0
            pi_entropy_ , epis_ ,alea_  = list(),list(),list()
            self.model.eval() # evaluate (affects DropOut and BN)
            for batch_in,_ in data_iter:
                # Foraward path
                mdn_out     = self.model.forward(batch_in.to(device))
                pi,mu,sigma = mdn_out['pi'],mdn_out['mu'],mdn_out['sigma']

                unct_out    = mdn_uncertainties(pi,mu,sigma)
                epis_unct   = unct_out['epis'] # [N x D]
                alea_unct   = unct_out['alea'] # [N x D]
                pi_entropy  = unct_out['pi_entropy'] # [N]


                ### Need to deterimine a single value for ood score for each sample
                ### should collapse D-directional dim.

                # ## a) mean
                # epis_unct = torch.mean(epis_unct,dim=-1)
                # alea_unct = torch.mean(alea_unct,dim=-1)
                
                ## b) max
                epis_unct,_ = torch.max(epis_unct,dim=-1)
                alea_unct,_ = torch.max(alea_unct,dim=-1)        
                
                epis_ += epis_unct.cpu().numpy().tolist()
                alea_ += alea_unct.cpu().numpy().tolist()
                pi_entropy_ += pi_entropy.cpu().numpy().tolist()

                n_total += batch_in.size(0)
            self.model.train() # back to train mode 
            out_eval = {'epis_' : epis_,'alea_' : alea_,'pi_entropy_':pi_entropy_}
        return out_eval
    
    def test_eval_mdn(self, data_iter, device):
        with torch.no_grad():
            n_total,l2_sum,epis_unct_sum,alea_unct_sum,entropy_pi_sum = 0,0,0,0,0
            self.model.eval() # evaluate (affects DropOut and BN)
            for batch_in,batch_out in data_iter:
                # Foraward path
                mdn_out     = self.model.forward(batch_in.to(device))
                pi,mu,sigma = mdn_out['pi'],mdn_out['mu'],mdn_out['sigma']

                l2        = mdn_eval(pi,mu,sigma,batch_out.to(device))['l2_mean'] # [N]
                unct_out    = mdn_uncertainties(pi,mu,sigma)
                epis_unct   = unct_out['epis'] # [N x D]
                alea_unct   = unct_out['alea'] # [N x D]
                entropy_pi  = unct_out['pi_entropy'] # [N]
                entropy_pi_sum  += torch.sum(entropy_pi)
                epis_unct,_ = torch.max(epis_unct,dim=-1)
                alea_unct,_ = torch.max(alea_unct,dim=-1)
                # epis_unct_sum += torch.sum(torch.mean(epis_unct,dim=-1))
                # alea_unct_sum += torch.sum(torch.mean(alea_unct,dim=-1))
                epis_unct_sum += torch.sum(epis_unct,dim=-1)
                alea_unct_sum += torch.sum(alea_unct,dim=-1)

                l2_sum += torch.sum(l2) # [N]
                n_total += batch_in.size(0)
            entropy_pi_avg=(entropy_pi_sum/n_total).detach().cpu().item()
            epis      = (epis_unct_sum/n_total).detach().cpu().item()
            alea      = (alea_unct_sum/n_total).detach().cpu().item()
            l2      = (l2_sum/n_total).detach().cpu().item()
            self.model.train() # back to train mode 
            out_eval = {'l2_norm':l2,'epis':epis,'alea':alea,
                        'pi_entropy':entropy_pi_avg}
        return out_eval