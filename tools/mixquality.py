import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
import re
from collections import defaultdict


def get_name_lists(root_dir: str, neg_case="all", train=True, neg=False,
                   exp_case=None) -> Tuple[List[str], List[str]]:
    """
    Folder layout:
      Data/
        E1/001/...
        ...
        N6/123/...

    Returns:
      exp_data_name_list: ["E1/001", "E1/002", ...]
      neg_data_name_list: ["N1/001", "N2/030", ...]
    """

    root_path = Path(root_dir)

    # exp_case=[1,2,3] -> {"E1","E2","E3"}
    exp_scen_allow = None
    if exp_case is not None:
        try:
            exp_scen_allow = {f"E{int(x)}" for x in exp_case}  # selected expert scenarios
        except Exception:
            exp_scen_allow = set()

    exp_trials: List[str] = []
    neg_trials: List[str] = []

    if not root_path.is_dir():
        return exp_trials, neg_trials

    scen_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])

    for scen_dir in scen_dirs:
        scen_id = scen_dir.name
        if not re.match(r'^[EN]\d+$', scen_id):
            continue

        if scen_id.startswith("E") and exp_scen_allow is not None and scen_id not in exp_scen_allow:
            continue

        trial_dirs = sorted([p for p in scen_dir.iterdir() if p.is_dir() and p.name.isdigit()])

        for td in trial_dirs:
            trial_ref = f"{scen_id}/{td.name}"  # ex) "E1/030"
            if scen_id.startswith("E"):
                exp_trials.append(trial_ref)
            else:
                neg_trials.append(trial_ref)

    if neg_case == "all" and train is True and neg is False:
        print("\nExpert trial refs")
        for line in _summarize_trial_refs(exp_trials):
            print("  " + line)

        print("\nNegative trial refs")
        for line in _summarize_trial_refs(neg_trials):
            print("  " + line)

        print(f"\n[INFO] #expert_trials={len(exp_trials)}  #neg_trials={len(neg_trials)}")


    return exp_trials, neg_trials

def _summarize_trial_refs(trial_refs: List[str]) -> List[str]:
    """
    Compress refs
      ["N1/001", "N1/002", ..., "N1/031", "N1/033"]
    into
      ["N1/001 ~ N1/031, N1/033"]
    """
    # scenario -> sorted unique trial ints
    scen_map = {}
    for ref in trial_refs:
        parts = re.split(r"[\\/]", str(ref))
        if len(parts) != 2:
            continue
        scen, trial = parts[0], parts[1]
        if not trial.isdigit():
            continue
        scen_map.setdefault(scen, set()).add(int(trial))

    lines = []
    for scen in sorted(scen_map.keys()):
        nums = sorted(scen_map[scen])
        if not nums:
            continue

        # compress consecutive ranges
        ranges = []
        start = prev = nums[0]
        for n in nums[1:]:
            if n == prev + 1:
                prev = n
                continue
            ranges.append((start, prev))
            start = prev = n
        ranges.append((start, prev))

        # format
        chunks = []
        for a, b in ranges:
            if a == b:
                chunks.append(f"{scen}/{a:03d}")
            else:
                chunks.append(f"{scen}/{a:03d} ~ {scen}/{b:03d}")

        lines.append(", ".join(chunks))

    return lines


def build_state(frame_data: dict):
    """
    Build 1D state vector at a frame from .json data
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

def _resolve_trial_json(trial_dir: Path, scenario_id: str, trial_id: str) -> Path | None:
    """
    {scenario_id}_{trial_id}.json  (ex: E1_030.json)
    Fallback: any *.json inside trial_dir
    """
    cand = trial_dir / f"{scenario_id}_{trial_id}.json"
    if cand.exists():
        return cand

    js = sorted([p for p in trial_dir.glob("*.json") if p.is_file()])
    if not js:
        return None

    # if multiple, prefer ones that start with scenario_id_
    for p in js:
        if p.name.startswith(f"{scenario_id}_"):
            return p
    return js[0]

def load_expert_dataset(path, frame, exp_data_name_list, train, simple_stride=1):
    """
    path : Dataset root dir
    frame: length of sequence
    exp_data_name_list: ["E1/001", "E1/002", ...]

    Return:
      rt   : (N, D_in)  FloatTensor (Input)
      act  : (N, D_out) FloatTensor (Target)
      file : (N,)       np.ndarray  (sample source info)
    """

    rt, act, file = [], [], []
    exp_scenario  = []

    trial_win_counts   = {}
    trial_frame_counts = {}
    scen_win_counts    = defaultdict(int)

    for trial_ref in exp_data_name_list:
        # trial_ref = "E1/030"
        parts = re.split(r"[\\/]", str(trial_ref))
        if len(parts) != 2:
            print(f"[load_expert_dataset] Warning: invalid trial_ref={trial_ref}, skip.")
            continue

        scenario_id, trial_id = parts[0], parts[1]

        trial_dir = Path(path) / scenario_id / trial_id
        if not trial_dir.exists():
            print(f"[load_expert_dataset] Warning: {trial_dir} not found, skip.")
            continue

        json_path = _resolve_trial_json(trial_dir, scenario_id, trial_id)
        if json_path is None or not json_path.exists():
            print(f"[load_expert_dataset] Warning: json not found under {trial_dir}, skip.")
            continue

        # Read .json data file
        with open(json_path, "r") as f:
            data = json.load(f)

        frames   = data.get("frames", [])
        n_frames = len(frames)

        if n_frames < frame:
            print(f"[load_expert_dataset] Warning: {json_path} has only {n_frames} frames (< frame={frame}), skip.")
            continue

        # [NOTE] Apply stride ONLY for E1 case (too dense sampling on E1 exacerbates performance of model)
        if scenario_id == "E1":
            stride = max(1, int(simple_stride))
        else:
            stride = 1

        n_windows                     = (n_frames - frame) // stride + 1
        trial_frame_counts[trial_ref] = n_frames
        trial_win_counts[trial_ref]   = n_windows
        scen_win_counts[scenario_id] += n_windows

        for seq in range(0, n_frames - frame + 1, stride):

            exp_scenario.append(scenario_id)   # E1/E2/E3
            data_vec   = []
            target_vec = []

            for it in range(frame):
                fr = frames[seq + it]
                s = build_state(fr)
                data_vec.extend(s)

                if it == frame - 1:
                    target_vec.append(float(s[2]))  # ax
                    target_vec.append(float(s[3]))  # ay
                    target_vec.append(float(s[4]))  # w

            rt.append(data_vec)
            act.append(target_vec)
            file.append(f"{scenario_id}/{trial_id}:{seq}")


    if len(rt) == 0:
        print("[load_expert_dataset] Warning: no expert data loaded.")
        return (
            torch.empty(0, 0),
            torch.empty(0, 0),
            np.asarray(file),
            [],
        )

    if len(scen_win_counts) > 0:
        total_w = int(sum(scen_win_counts.values()))
        if train:
            print("\n[Expert sample distribution] (sliding windows)")
            print(f"  frame={frame}  total_windows={total_w}  num_trials={len(trial_win_counts)}  num_scenarios={len(scen_win_counts)}")
            for scen in sorted(scen_win_counts.keys(), key=lambda k: scen_win_counts[k], reverse=True):
                w = int(scen_win_counts[scen])
                pct = (100.0 * w / total_w) if total_w > 0 else 0.0
                print(f"    {scen:<8s} windows={w:6d}  ({pct:5.1f}%)")

    return (
        torch.FloatTensor(rt),
        torch.FloatTensor(act),
        np.asarray(file),
        exp_scenario,
    )

def load_negative_dataset(path, frame, neg_data_name_list):
    """
    path : Dataset root dir
    frame: length of sequence
    neg_data_name_list: ["N1/001", "N2/030", ...]

    Return:
      rt   : (N, D_in)  FloatTensor (Input)
      act  : (N, D_out) FloatTensor (Target)
      file : (N,)       np.ndarray  (sample source info)
      neg_scenario : (N,) list[str]  (scenario name per sample)
    """

    rt, act, file = [], [], []
    neg_scenario = []
    neg_trial = []

    for trial_ref in neg_data_name_list:
        # trial_ref = "N1/010"
        parts = re.split(r"[\\/]", str(trial_ref))
        if len(parts) != 2:
            continue

        scenario_id, trial_id = parts[0], parts[1]

        trial_dir = Path(path) / scenario_id / trial_id
        if not trial_dir.exists():
            continue

        json_path = _resolve_trial_json(trial_dir, scenario_id, trial_id)
        if json_path is None or not json_path.exists():
            continue

        # Read .json data file
        with open(json_path, "r") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        n_frames = len(frames)
        if n_frames < frame:
            continue

        for seq in range(0, n_frames - frame + 1):
            data_vec = []
            target_vec = []

            for it in range(frame):
                fr = frames[seq + it]
                s = build_state(fr)
                data_vec.extend(s)

                if it == frame - 1:
                    target_vec.append(float(s[2]))
                    target_vec.append(float(s[3]))
                    target_vec.append(float(s[4]))

            rt.append(data_vec)
            act.append(target_vec)
            file.append(f"{scenario_id}/{trial_id}:{seq}")

            neg_scenario.append(scenario_id)               # "N1"
            neg_trial.append(f"{scenario_id}/{trial_id}")  # "N1/030"

    if len(rt) == 0:
        print("[load_negative_dataset] Warning: no negative data loaded.")
        return (
            torch.empty(0, 0),
            torch.empty(0, 0),
            np.asarray(file),
            [],
            [],
            [],
        )

    return (
        torch.FloatTensor(rt),
        torch.FloatTensor(act),
        np.asarray(file),
        neg_scenario,
        neg_trial,
    )



torch.manual_seed(0)


class MixQuality():
    def __init__(self, root, train=True, neg=False, norm=True,
             exp_case=None, neg_case=None, frame=10):
        
        self.neg_trial_sel = None

        self.neg_case_filter = neg_case if neg else None

        self.train = train
        self.neg   = neg

        name_neg_case = "all" if neg_case is None else neg_case
        self.exp_list, self.neg_list = get_name_lists(root,
                                                      neg_case=name_neg_case,
                                                      train=self.train,
                                                      neg=self.neg,
                                                      exp_case=exp_case )

        self.e_in, self.e_target, self.file_expert, self.exp_scenario   = load_expert_dataset(root,frame,self.exp_list, train, simple_stride=150)
        self.n_in, self.n_target, self.file_negative, self.neg_scenario, self.neg_trial = load_negative_dataset(root,frame,self.neg_list)
        
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

        # Expert scenario-wise 8:2 split

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
            perm = idx[torch.randperm(len(idx), generator=rng)]  # random select

            n_train = int(len(perm) * 0.8)

            if n_train <= 0:
                n_train = max(1, len(perm) - 1)

            train_idx.append(perm[:n_train])
            test_idx.append(perm[n_train:])

        train_idx = torch.cat(train_idx)
        test_idx  = torch.cat(test_idx)


        # TRAIN (80% expert)
        if self.train:
            self.x = self.e_in[train_idx]
            self.y = self.e_target[train_idx]
            self.e_label = train_idx.size(0)

        # TEST (20% expert + negative)
        else:
            # TEST (ID)
            if not self.neg:

                self.x = self.e_in[test_idx]
                self.y = self.e_target[test_idx]
                self.e_label = test_idx.size(0)

            # TEST (OOD)
            else:
                filters = self.neg_case_filter

                def _match(name: str, f: str) -> bool:

                    # support:
                    #  - "N1"        (scenario)
                    #  - "N1/030"    (specific trial)
                    #  - "N1*" or "N1_*" (prefix)
                    #  - "N1_030"    (alias)

                    f = f.strip()
                    if f.endswith("_*"):
                        return name.startswith(f[:-2])
                    if f.endswith("*"):
                        return name.startswith(f[:-1])
                    return name == f

                if filters is not None and len(filters) > 0:
                    if isinstance(filters, str):
                        filters = [filters]

                    mask = torch.zeros(len(self.neg_scenario), dtype=torch.bool)  # Mask out undesired negative scenario (if filters=None, use all of negative data)
                    for i, scen in enumerate(self.neg_scenario):
                        trial = self.neg_trial[i]
                        trial_alias = trial.replace("/", "_")  # "N1/030" -> "N1_030"
                        ok = False
                        for f in filters:
                            if _match(scen, f) or _match(trial, f) or _match(trial_alias, f):
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

                    # save elements where mask=True
                    self.neg_scenario_sel = [s for s, keep in zip(self.neg_scenario, mask.tolist()) if keep]
                    self.neg_trial_sel    = [t for t, keep in zip(self.neg_trial,    mask.tolist()) if keep]

                else:
                    # load all negatives
                    self.x = self.n_in
                    self.y = self.n_target

                    self.neg_scenario_sel = self.neg_scenario
                    self.neg_trial_sel = self.neg_trial

                self.e_label = test_idx.size(0)

            self.exp_train_scenarios = sorted(set(self.exp_scenario[i] for i in train_idx.tolist()))
            self.exp_test_scenarios  = sorted(set(self.exp_scenario[i] for i in test_idx.tolist()))


    def normalize(self):
        """
        Input x shape: (N, D_in) where D_in = frame * (7 + lidar_dim)
        - per frame:
            0~4 : z-normalization or min-max (depending on self.norm)
            5~6 : 1 - ( VALUE / 5 )
            7~  : LiDAR (ALREADY PROCESSED)
        Target y: (N, 3) -> normalize (z or min-max)
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

        # Build indices
        idx_z   = []    # per-frame 0~4
        idx_div = []    # per-frame 5~6
        for i in range(F):
            base = i * per_frame_dim
            idx_z.extend(range(base + 0, base + 5))    # 0,1,2,3,4
            idx_div.extend(range(base + 5, base + 7))  # 5,6

        idx_z = torch.tensor(idx_z, dtype=torch.long, device=self.x.device)
        idx_div = torch.tensor(idx_div, dtype=torch.long, device=self.x.device)

        eps = 1e-8

        if self.norm:
            # 0~4
            self.x[:, idx_z] = (self.x[:, idx_z] - self.mean_in[idx_z]) / (self.std_in[idx_z] + eps)

            # 5~6
            self.x[:, idx_div] = 1 - ( self.x[:, idx_div] / 5.0 )

            # target y: as usual
            self.y = (self.y - self.mean_t) / (self.std_t + eps)

            # NaN guard
            self.x[self.x != self.x] = 0
            self.y[self.y != self.y] = 0

        else:
            # min-max for 0~4 only
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

            # 0~4
            denom_in_z = (max_in[idx_z] - min_in[idx_z]) + eps
            self.x[:, idx_z] = (self.x[:, idx_z] - min_in[idx_z]) / denom_in_z

            # 5~6
            self.x[:, idx_div] = 1 - ( self.x[:, idx_div] / 5.0 )

            # target y: as usual
            denom_t = (max_t - min_t) + eps
            self.y = (self.y - min_t) / denom_t

            # NaN guard
            self.x[self.x != self.x] = 0
            self.y[self.y != self.y] = 0




if __name__ == '__main__':
    m = MixQuality(root='../Data/',train=True,neg=False)

    # for i in range(200):
    #     print(m.x[i][5:7])