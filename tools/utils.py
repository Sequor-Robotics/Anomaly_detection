import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm


def print_n_txt(_f,_chars,_addNewLine=True,_DO_PRINT=True):
    if _addNewLine: _f.write(_chars+'\n')
    else: _f.write(_chars)
    _f.flush();os.fsync(_f.fileno()) # Write to txt
    if _DO_PRINT:
        try:
            tqdm.write(_chars)
        except Exception:
            print(_chars)

def get_methods(mode: str):
    if mode == 'mdn':
        return ['epis_', 'alea_', 'pi_entropy_']
    if mode == 'vae':
        return ['recon_', 'kl_']
    raise NotImplementedError

def set_seed_and_device(args, seed=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = "cuda"
    return seed, device

class Logger():
    def __init__(self, path):
        self.path=path
    
    def train_res(self,train_l2,test_l2):
        self.train_l2 = train_l2
        self.test_l2 = test_l2

    def ood(self,id_eval,ood_eval,auroc,aupr):
        self.id_eval = id_eval
        self.ood_eval = ood_eval
        self.auroc = auroc
        self.aupr = aupr

    def save(self):
        try:
            os.remove(self.path)
        except:
            pass
        data = {}
        with open(self.path,'w') as json_file:
            data['train_l2']=self.train_l2
            data['test_l2']=self.test_l2
            data['id_eval'] = self.id_eval
            data['ood_eval'] = self.ood_eval
            data['auroc'] = self.auroc
            data['aupr'] = self.aupr
            json.dump(data,json_file, indent=4)