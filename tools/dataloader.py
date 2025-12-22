from tools.mixquality import MixQuality
from torchvision import transforms
import torch.utils.data as data
import torch

class mixquality_dataset(data.Dataset):
    def __init__(self, root='./dataset/mixquality/', train=True, neg=False,
             norm=True, exp_case=[1,2,3], neg_case=None, frame=1):

        mix = MixQuality(root=root,train=train,neg=neg,norm=norm,exp_case=exp_case,neg_case=neg_case,frame=frame)
        self.x = mix.x
        self.y = mix.y
        self.e_label = mix.e_label
        self.case = mix.case
        # self.path = mix.path

        self.exp_dirs = getattr(mix, "exp_list", None)   # Ex: ['expert_xxx_1', 'expert_yyy_2', ...]
        self.neg_dirs = getattr(mix, "neg_list", None)   # Ex: ['neg_xxx_1', 'neg_xxx_2', ...]

        self.neg_seq      = getattr(mix, "neg_seq_sel", None)
        self.neg_trial    = getattr(mix, "neg_trial_sel", None)
        self.neg_scenario = getattr(mix, "neg_scenario_sel", None)

        self.exp_train_scenarios = getattr(mix, "exp_train_scenarios", None)
        self.exp_test_scenarios  = getattr(mix, "exp_test_scenarios", None)
        self.exp_split_stats     = getattr(mix, "exp_split_stats", None)


    def __getitem__(self, index):
        data, target = self.x[index], self.y[index]
        return data,target
    
    def __len__(self):
        return len(self.x)