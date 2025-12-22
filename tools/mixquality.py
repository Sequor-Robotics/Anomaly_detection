import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
import re
from collections import defaultdict


MAX_N_OBJECTS = 5

def get_name_lists(root_dir: str, neg_case="all", train=True, neg=False) -> Tuple[List[str], List[str]]:

    root_path = Path(root_dir)

    exp_data_name_list: List[str] = []
    neg_data_name_list: List[str] = []

    for item in root_path.iterdir():
        if not item.is_dir():
            continue

        name = item.name

        if name.startswith("expert_"):
            exp_data_name_list.append(name)
        elif name.startswith("neg_"):
            neg_data_name_list.append(name)

    if neg_case == "all" and train is True and neg is False:
        
        print("\nExpert data scenario names")
        for exp_trial in sorted(exp_data_name_list):
            print(exp_trial)
        
        print("\nNegative data scenario names")
        for neg_trial in sorted(neg_data_name_list):
            print(neg_trial)

    return exp_data_name_list, neg_data_name_list


def build_state(frame_data: dict):
    """
    Build 1D state vector at a frame from processed.json
    """
    
    pos     = frame_data.get("position") or [0.0, 0.0]
    lin_vel = frame_data.get("lin_vel")  or [0.0, 0.0]
    lin_acc = frame_data.get("lin_acc")  or [0.0, 0.0]
    ang_vel = frame_data.get("ang_vel")

    if ang_vel is None:
        ang_vel = 0.0

    obj_pos = frame_data.get("obj_pos")
    if obj_pos is None or not isinstance(obj_pos, (list, tuple)) or len(obj_pos) != 2:
        obj_pos = [0.0, 0.0]

    # scan data
    scan = frame_data.get("scan") or []
    if len(scan) > 0:
        ranges = np.array([p[1] for p in scan], dtype=np.float32)
        finite = np.isfinite(ranges)
        if not finite.any():
            ranges = np.zeros_like(ranges)
        ranges_list = ranges.tolist()
    else:
        ranges_list = []

    ### Define state vector
    state_vec = [
        float(lin_vel[0]), float(lin_vel[1]),
        float(lin_acc[0]), float(lin_acc[1]),
        float(ang_vel),
        float(obj_pos[0]), float(obj_pos[1]),
    ]
    state_vec.extend(ranges_list)

    return state_vec



def load_expert_dataset(path, frame, exp_data_name_list, simple_stride=1):
    """
    path : Dataset root dir
    frame: length of sequence

    Return:
      rt   : (N, D_in)  FloatTensor (Input)
      act  : (N, D_out) FloatTensor (Target)
      case : (N, 7)     FloatTensor (dummy)
      file : (N,)       np.ndarray  (sample source info)
    """

    rt   = []
    act  = []
    case = []
    file = []
    exp_scenario = []

    trial_win_counts = {}
    trial_frame_counts = {}
    scen_win_counts = defaultdict(int)

    # 기존 R3 코드에서는 road/hazard 정보를 7차원으로 넣었으므로
    # 구조를 맞추기 위해 일단 7차원 dummy 0벡터 사용
    dummy_case = [0.0] * 7

    for data_path in exp_data_name_list:
        scenario = re.sub(r'_\d+$', '', data_path)
        scenario_root = Path(path) / data_path
        if not scenario_root.exists():
            print(f"[load_expert_dataset] Warning: {scenario_root} not found, skip.")
            continue

        scenario_name = scenario_root.name
        processed_path = scenario_root / f"{scenario_name}_prcd.json"

        if not processed_path.exists():
            print(f"[load_expert_dataset] Warning: {processed_path} not found, skip.")
            continue

        # Read '*_processed.json'
        with open(processed_path, "r") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        n_frames = len(frames)

        if n_frames < frame:
            print(f"[load_expert_dataset] Warning: {processed_path} has only {n_frames} frames (< frame={frame}), skip.")
            continue

        is_simple_expert = data_path.startswith("expert_") and data_path.count("_") == 1
        stride = int(simple_stride) if is_simple_expert else 1
        stride = max(1, stride)

        n_windows = ( n_frames - frame ) / stride + 1
        trial_frame_counts[data_path] = n_frames
        trial_win_counts[data_path] = n_windows
        scen_win_counts[scenario] += n_windows

        # Sliding window
        for seq in range(0, n_frames - frame + 1, stride):
            exp_scenario.append(scenario)
            data_vec = []
            target_vec = []

            for it in range(frame):
                fr = frames[seq + it]
                s = build_state(fr)

                # State (input)
                data_vec.extend(s)

                # Action (output)
                if it == frame - 1:
                    target_vec.append(float(s[2]))  # lin_acc x
                    target_vec.append(float(s[3]))  # lin_acc y
                    target_vec.append(float(s[4]))  # ang_vel

            rt.append(data_vec)
            act.append(target_vec)
            case.append(dummy_case)

            # # sample source
            # file.append(str(processed_path) + f":{seq}")

    # No data case ... dummy
    if len(rt) == 0:
        print("[load_expert_dataset] Warning: no expert data loaded.")
        return (
            torch.empty(0, 0),
            torch.empty(0, 0),
            torch.empty(0, len(dummy_case)),
            np.asarray(file),
        )
    
    # expert data sample distribution
    if len(scen_win_counts) > 0:
        total_w = int(sum(scen_win_counts.values()))
        print("\n[Expert sample distribution] (sliding windows)")
        print(f"  frame={frame}  total_windows={total_w}  n_trials={len(trial_win_counts)}  n_scenarios={len(scen_win_counts)}")

        # print("  - per trial dir")
        # for trial in sorted(trial_win_counts.keys(), key=lambda k: trial_win_counts[k], reverse=True):
        #     w = int(trial_win_counts[trial])
        #     fr = int(trial_frame_counts.get(trial, -1))
        #     pct = (100.0 * w / total_w) if total_w > 0 else 0.0
        #     print(f"    {trial:<40s} frames={fr:6d}  windows={w:6d}  ({pct:5.1f}%)")

        # print("  - per scenario prefix")
        for scen in sorted(scen_win_counts.keys(), key=lambda k: scen_win_counts[k], reverse=True):
            w = int(scen_win_counts[scen])
            pct = (100.0 * w / total_w) if total_w > 0 else 0.0
            print(f"    {scen:<40s} windows={w:6d}  ({pct:5.1f}%)")

    rt_tensor   = torch.FloatTensor(rt)
    act_tensor  = torch.FloatTensor(act)
    case_tensor = torch.FloatTensor(case)
    file_arr    = np.asarray(file)

    return rt_tensor, act_tensor, case_tensor, file_arr, exp_scenario



def load_negative_dataset(path, frame, neg_data_name_list):
    """
    path : Dataset root dir
    frame: length of sequence

    Return:
      rt   : (N, D_in)  FloatTensor (Input)
      act  : (N, D_out) FloatTensor (Target)
      case : (N, 7)     FloatTensor (dummy)
      file : (N,)       np.ndarray  (sample source info)
      neg_scenario : (N,) list[str]  (scenario name per sample)
    """

    rt = []
    act = []
    case = []
    file = []
    neg_scenario = []

    neg_trial = []   # 
    neg_seq   = []   # 


    dummy_case = [0.0] * 7

    for data_path in neg_data_name_list:

        scenario_prefix = re.sub(r'_\d+$', '', data_path)

        scenario_root = Path(path) / data_path
        if not scenario_root.exists():
            continue

        processed_path = scenario_root / f"{scenario_root.name}_prcd.json"
        if not processed_path.exists():
            continue

        with open(processed_path, "r") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        n_frames = len(frames)

        if n_frames < frame:
            continue

        # sliding window (sample-wise)
        for seq in range(0, n_frames - frame + 1):
            data_vec = []
            target_vec = []

            for it in range(frame):
                fr = frames[seq + it]
                s = build_state(fr)
                data_vec.extend(s)

                if it == frame - 1:
                    target_vec.append(float(s[2]))  # lin_acc x
                    target_vec.append(float(s[3]))  # lin_acc y
                    target_vec.append(float(s[4]))  # ang_vel

            rt.append(data_vec)
            act.append(target_vec)
            case.append(dummy_case)

            neg_scenario.append(scenario_prefix)
            neg_trial.append(data_path)  # (neg_xxx_3)
            neg_seq.append(seq)          # window start frame


    if len(rt) == 0:
        print("[load_negative_dataset] Warning: no negative data loaded.")
        return (
            torch.empty(0, 0),
            torch.empty(0, 0),
            torch.empty(0, len(dummy_case)),
            np.asarray(file),
            [],
        )

    rt_tensor   = torch.FloatTensor(rt)
    act_tensor  = torch.FloatTensor(act)
    case_tensor = torch.FloatTensor(case)
    file_arr    = np.asarray(file)

    return rt_tensor, act_tensor, case_tensor, file_arr, neg_scenario, neg_trial, neg_seq



torch.manual_seed(0)


class MixQuality():
    def __init__(self, root, train=True, neg=False, norm=True,
             exp_case=[1,2,3], neg_case=None, frame=1):
        
        self.neg_trial_sel = None

        self.neg_case_filter = neg_case if neg else None

        self.train = train
        self.neg   = neg

        name_neg_case = "all" if neg_case is None else neg_case
        self.exp_list, self.neg_list = get_name_lists(root,
                                                      neg_case=name_neg_case,
                                                      train=self.train,
                                                      neg=self.neg)
        
        self.e_in, self.e_target, self.e_case, self.file_expert, self.exp_scenario   = load_expert_dataset(root,frame,self.exp_list, simple_stride=70)
        self.n_in, self.n_target, self.n_case, self.file_negative, self.neg_scenario, self.neg_trial, self.neg_seq = load_negative_dataset(root,frame,self.neg_list)
        
        """
        e_in     : Expert input data samples
        e_target : Expert target data samples
        n_in     : Negative input data samples
        n_target : Negative target data samples
        """
        
        self.e_size = self.e_in.size(0)
        self.n_size = self.n_in.size(0)

        
        # True = Z-score standardization
        # False = min-max
        self.norm = norm

        self.frame = frame

        in_list = []
        t_list  = []

        if self.e_in.numel() > 0:
            in_list.append(self.e_in)
            t_list.append(self.e_target)

        if self.n_in.numel() > 0:
            in_list.append(self.n_in)
            t_list.append(self.n_target)

        if len(in_list) == 0:
            raise RuntimeError("No expert or negative data loaded. Check your data root and folder names.")

        all_in = torch.cat(in_list, dim=0)   # (N_total, D_in)
        all_t  = torch.cat(t_list,  dim=0)   # (N_total, D_out)

        self.mean_in = all_in.mean(dim=0)
        self.std_in  = all_in.std(dim=0)
        self.mean_t  = all_t.mean(dim=0)
        self.std_t   = all_t.std(dim=0)

        """
        mean_in : (D_in)  Buffer to store mean value of input data (Expert+Negative)
        std_in  : (D_in)  Buffer to store std value of input data (Expert+Negative)
        mean_t  : (D_out) Buffer to store mean value of target data (Expert+Negative)
        std_t   : (D_out) Buffer to store std value of target data (Expert+Negative)
        """

        self.load()
        self.normalize()


    def load(self):
        # =====================================================
        # Expert scenario-wise 8:2 split
        # =====================================================
        rng = torch.Generator()
        rng.manual_seed(0)

        # scenario -> list of expert sample indices
        scen_to_idx = {}
        for i, scen in enumerate(self.exp_scenario):
            scen_to_idx.setdefault(scen, []).append(i)

        train_idx = []
        test_idx = []

        for scen, idx_list in scen_to_idx.items():
            idx = torch.tensor(idx_list, dtype=torch.long)
            perm = idx[torch.randperm(len(idx), generator=rng)]

            n_train = int(len(perm) * 0.8)

            if n_train <= 0:
                n_train = max(1, len(perm) - 1)

            train_idx.append(perm[:n_train])
            test_idx.append(perm[n_train:])

        train_idx = torch.cat(train_idx)
        test_idx = torch.cat(test_idx)

        # =====================================================
        # TRAIN / TEST branching
        # =====================================================
        if self.train:
            # ======================
            # TRAIN : expert only
            # ======================
            self.x = self.e_in[train_idx]
            self.y = self.e_target[train_idx]
            self.case = self.e_case[train_idx]
            self.e_label = train_idx.size(0)

        else:
            # ======================
            # TEST
            # ======================
            if not self.neg:
                # ---------- ID test ----------
                self.x = self.e_in[test_idx]
                self.y = self.e_target[test_idx]
                self.case = self.e_case[test_idx]
                self.e_label = test_idx.size(0)

            else:
                # ---------- OOD test (STRICT filtering; no silent fallback) ----------
                filters = self.neg_case_filter

                def _match(name: str, f: str) -> bool:
                    # support: exact, prefix wildcard
                    #  - "neg_straight"        (exact scenario)
                    #  - "neg_straight_2"      (exact trial dir)
                    #  - "neg_straight*"       (prefix)
                    #  - "neg_straight_*"      (prefix)
                    f = f.strip()
                    if f.endswith("_*"):
                        return name.startswith(f[:-2])
                    if f.endswith("*"):
                        return name.startswith(f[:-1])
                    return name == f

                if filters is not None and len(filters) > 0:
                    if isinstance(filters, str):
                        filters = [filters]

                    mask = torch.zeros(len(self.neg_scenario), dtype=torch.bool)
                    for i, scen in enumerate(self.neg_scenario):
                        trial = self.neg_trial[i]
                        ok = False
                        for f in filters:
                            if _match(scen, f) or _match(trial, f):
                                ok = True
                                break
                        mask[i] = ok

                    n_match = int(mask.sum().item())
                    if n_match == 0:
                        raise ValueError(
                            f"[NEG FILTER ERROR] No negative samples matched neg_case={filters}. "
                            f"This is strict on purpose to prevent mixing ALL negatives by mistake."
                        )

                    self.x = self.n_in[mask]
                    self.y = self.n_target[mask]
                    self.case = self.n_case[mask]

                    # debugging meta (what actually got selected)
                    self.neg_scenario_sel = [s for s, keep in zip(self.neg_scenario, mask.tolist()) if keep]
                    self.neg_trial_sel    = [t for t, keep in zip(self.neg_trial,   mask.tolist()) if keep]
                    self.neg_seq_sel      = [s for s, keep in zip(self.neg_seq,     mask.tolist()) if keep]

                else:
                    # if user did NOT request a filter, then load all negatives (this is intentional)
                    self.x = self.n_in
                    self.y = self.n_target
                    self.case = self.n_case

                    self.neg_scenario_sel = self.neg_scenario
                    self.neg_trial_sel = self.neg_trial
                    self.neg_seq_sel = self.neg_seq


                # ID label size는 유지 (기존 코드 의미)
                self.e_label = test_idx.size(0)

            self.exp_train_scenarios = sorted(set(self.exp_scenario[i] for i in train_idx.tolist()))
            self.exp_test_scenarios  = sorted(set(self.exp_scenario[i] for i in test_idx.tolist()))



    # def normaize(self):
    #     # except LiDAR data
    #     if self.norm:
    #         self.x = (self.x - self.mean_in)/(self.std_in)
    #         self.y = (self.y - self.mean_t)/(self.std_t)
    #         self.x[self.x != self.x] = 0
    #         self.y[self.y != self.y] = 0
    #     else:
    #         self.max_in = torch.max(torch.cat((self.e_in,self.n_in),dim=0),dim=0)[0]
    #         self.min_in = torch.min(torch.cat((self.e_in,self.n_in),dim=0),dim=0)[0]
    #         self.max_t = torch.max(torch.cat((self.e_target,self.n_target),dim=0),dim=0)[0]
    #         self.min_t = torch.min(torch.cat((self.e_target,self.n_target),dim=0),dim=0)[0]
    #         self.x = (self.x - self.min_in)/(self.max_in-self.min_in)
    #         self.y = (self.y - self.min_t)/(self.max_t-self.min_t)

    def normalize(self):
        """
        Input x shape: (N, D_in) where D_in = frame * (7 + lidar_dim)
        - per frame:
            0~4 : z-normalization (or min-max when self.norm==0)
            5~6 : divide by 5
            7~  : LiDAR (keep as-is)
        Target y: (N, 3) -> normalize as usual (z or min-max)
        """
        D = self.x.size(1)
        F = self.frame

        if F <= 0 or (D % F) != 0:
            raise ValueError(f"[normalize] Invalid shapes: D_in={D}, frame={F}")

        per_frame_dim = D // F
        non_lidar_dim = 7
        lidar_dim = per_frame_dim - non_lidar_dim

        if lidar_dim < 0:
            raise ValueError(f"[normalize] per_frame_dim={per_frame_dim} < non_lidar_dim={non_lidar_dim}")

        # ---- build indices across frames ----
        idx_z = []    # per-frame 0~4
        idx_div = []  # per-frame 5~6
        for i in range(F):
            base = i * per_frame_dim
            idx_z.extend(range(base + 0, base + 5))  # 0,1,2,3,4
            idx_div.extend(range(base + 5, base + 7))  # 5,6

        idx_z = torch.tensor(idx_z, dtype=torch.long, device=self.x.device)
        idx_div = torch.tensor(idx_div, dtype=torch.long, device=self.x.device)

        eps = 1e-8

        if self.norm:
            # 0~4: z-score
            self.x[:, idx_z] = (self.x[:, idx_z] - self.mean_in[idx_z]) / (self.std_in[idx_z] + eps)

            # 5~6: divide by 5 (NO z-score)
            self.x[:, idx_div] = 1 - ( self.x[:, idx_div] / 5.0 )

            # target y: z-score (as usual)
            self.y = (self.y - self.mean_t) / (self.std_t + eps)

            # NaN guard
            self.x[self.x != self.x] = 0
            self.y[self.y != self.y] = 0

        else:
            # min-max for 0~4 only, divide-by-5 for 5~6, LiDAR unchanged
            in_list = []
            t_list = []
            if self.e_in.numel() > 0:
                in_list.append(self.e_in)
                t_list.append(self.e_target)
            if self.n_in.numel() > 0 and self.n_in.size(1) == self.e_in.size(1):
                in_list.append(self.n_in)
                t_list.append(self.n_target)

            all_in = torch.cat(in_list, dim=0)
            all_t = torch.cat(t_list, dim=0)

            max_in = all_in.max(dim=0)[0]
            min_in = all_in.min(dim=0)[0]
            max_t = all_t.max(dim=0)[0]
            min_t = all_t.min(dim=0)[0]

            # 0~4: min-max
            denom_in_z = (max_in[idx_z] - min_in[idx_z]) + eps
            self.x[:, idx_z] = (self.x[:, idx_z] - min_in[idx_z]) / denom_in_z

            # 5~6: divide by 5 (NO min-max)
            self.x[:, idx_div] = 1 - ( self.x[:, idx_div] / 5.0 )

            # target y: min-max (as usual)
            denom_t = (max_t - min_t) + eps
            self.y = (self.y - min_t) / denom_t

            # NaN guard
            self.x[self.x != self.x] = 0
            self.y[self.y != self.y] = 0



if __name__ == '__main__':
    m = MixQuality(root='../Data/',train=True,neg=False)

    for i in range(200):
        print(m.x[i][5:7])