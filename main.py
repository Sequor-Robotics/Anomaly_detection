from solver import solver

from tools.utils import print_n_txt,Logger
from tools.measure_ood import measure
import torch
import random
import numpy as np
from datetime import datetime
import argparse
import os

from plot import plot_run


parser = argparse.ArgumentParser()
parser.add_argument('--root'    , type=str, default='.'  , help='root directory of the dataset')
parser.add_argument('--id'      , type=int, default=1    , help='id')
parser.add_argument('--mode'    , type=str, default='mdn', help='mdn vae')
parser.add_argument('--gpu'     , type=int, default=0    , help='gpu id')
parser.add_argument('--frame'   , type=int, default=1    , help='frame')
parser.add_argument('--exp_case', type=int, nargs='+'    , default=[1,2,3],help='expert case')

parser.add_argument('--epoch'     , type=int  , default=100 , help='epoch')
parser.add_argument('--lr'        , type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int  , default=128 , help='batch size')
parser.add_argument('--wd'        , type=float, default=1e-4, help='weight decay')
parser.add_argument('--dropout'   , type=float, default=0.25, help='dropout rate')
parser.add_argument('--lr_rate'   , type=float, default=0.9 , help='learning rate schedular rate')
parser.add_argument('--lr_step'   , type=int  , default=50  , help='learning rate schedular rate')

# Parser for MDN
parser.add_argument('--k', type=int,default=10,help='number of mixtures')
parser.add_argument('--norm', type=int,default=1,help='normalize dataset elementwise')
parser.add_argument('--sig_max', type=float,default=1,help='sig max')

# Parser for VAE
parser.add_argument('--h_dim', type=int, nargs='+', default=[20],help='h dim for vae')
parser.add_argument('--z_dim', type=int, default=10, help='z dim for vae')

# Parser for Variants
parser.add_argument('--lambda_mmd', type=float,default=10.0,help='lambda for mmd')     # Maximum Mean Discrepancy
parser.add_argument('--lambda_z', type=float,default=0.1,help='lambda for z')
parser.add_argument('--sigma', type=float,default=1.0,help='sigma for mmd')
parser.add_argument('--num_embeddings', type=int,default=512,help='number of embeddings for vq')
parser.add_argument('--commitment_cost', type=float,default=0.25,help='commitment cost for vq')

args = parser.parse_args()

SEED = 0
EPOCH = args.epoch

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device='cuda'

Solver = solver(args,device=device,SEED=SEED)
Solver.init_param()
# [init, total-init]

if args.mode == 'mdn':
    method = ['epis_','alea_','pi_entropy_']
    
elif args.mode == 'vae':
    method = ['recon_','kl_']

elif args.mode == 'vqvae':
    method = ['recon_','vq_']

elif args.mode == 'wae':
    method = ['recon_','mmd_']

elif args.mode == 'rae':
    method = ['recon_','zreg_']
    
else:
    raise NotImplementedError

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")   # YYYYMMDD


DIR = './res/' + run_name + '_{}_{}/'.format(args.mode, args.id)
DIR2     = os.path.join(DIR, 'ckpt/')

os.makedirs(DIR2, exist_ok=True)

log     = Logger(os.path.join(DIR, 'log.json'),
                 exp_case=Solver.test_e_dataset.case,
                 neg_case=Solver.test_n_dataset.case)

txtName = os.path.join(DIR, 'log.txt')
f       = open(txtName, 'w')             # Open txt file

print_n_txt(_f=f,_chars='Text name: '+txtName)
print_n_txt(_f=f,_chars=str(args))

train_l2, test_l2 = Solver.train_func(f)
log.train_res(train_l2,test_l2)

id_eval  = Solver.eval_func(Solver.test_e_iter,'cuda')
ood_eval = Solver.eval_func(Solver.test_n_iter,'cuda')

auroc, aupr = {},{}
for m in method:
    temp1, temp2 = measure(id_eval[m],ood_eval[m])
    strTemp = ("\n%s AUROC: [%.3f] AUPR: [%.3f]"%(m[:-1],temp1,temp2))
    print_n_txt(_f=f,_chars= strTemp)
    auroc[m] = temp1
    aupr[m]  = temp2

log.ood(id_eval,ood_eval,auroc,aupr)
torch.save(Solver.model.state_dict(),DIR2+'model.pt')
log.save()

# Plot
try:
    plot_run(run_name, args.mode, args.id)
except Exception as e:
    print_n_txt(_f=f, _chars=f"[WARN] plotting failed: {e}")