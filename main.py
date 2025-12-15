from solver import solver
from tools.utils import print_n_txt, Logger
from tools.measure_ood import measure
from plot import plot_run

import torch
import random
import numpy as np
from datetime import datetime
import argparse
import os
import re
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================= Argument Parser =================
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='.', help='project root (parent of Data)')
parser.add_argument('--id', type=int, default=1)
parser.add_argument('--mode', type=str, default='mdn')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--frame', type=int, default=1)
parser.add_argument('--exp_case', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--neg_case', type=str, nargs='+', default=None, help='negative scenario or trial names')



parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--lr_rate', type=float, default=0.75)    # 0.9
parser.add_argument('--lr_step', type=int, default=25)        # 50

# MDN
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--norm', type=int, default=1)
parser.add_argument('--sig_max', type=float, default=1)

# VAE
parser.add_argument('--h_dim', type=int, nargs='+', default=[20])
parser.add_argument('--z_dim', type=int, default=10)

# Variants
parser.add_argument('--lambda_mmd', type=float, default=10.0)
parser.add_argument('--lambda_z', type=float, default=0.1)
parser.add_argument('--sigma', type=float, default=1.0)
parser.add_argument('--num_embeddings', type=int, default=512)
parser.add_argument('--commitment_cost', type=float, default=0.25)

args = parser.parse_args()

# ================= Seed / Device =================
SEED = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = "cuda"


# ================= Negative Scenario Discovery =================
def discover_neg_scenarios(project_root: str):
    """
    Scan <project_root>/Data and automatically extract:
      - scenario name
      - neg_case (from trailing _N)

    Returns:
      {
        scenario_name: {
            "dirs":  [list of directories],
            "cases": [list of int]
        }
      }
    """
    neg_map = {}
    data_root = Path(project_root).resolve() / "Data"

    if not data_root.is_dir():
        raise RuntimeError(f"[ERROR] Data directory not found: {data_root}")

    for d in data_root.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith("neg"):
            continue

        m = re.search(r'_(\d+)$', d.name)
        if m is None:
            raise RuntimeError(f"[ERROR] Cannot parse case id from {d.name}")

        case_id = int(m.group(1))
        scenario_name = re.sub(r'_\d+$', '', d.name)

        neg_map.setdefault(scenario_name, {"dirs": [], "cases": []})
        neg_map[scenario_name]["dirs"].append(str(d))
        neg_map[scenario_name]["cases"].append(case_id)

    return neg_map


# ================= Solver (TRAIN + ALL NEG) =================
Solver = solver(args, device=device, SEED=SEED)
Solver.init_param()

if args.mode == 'mdn':
    method = ['epis_', 'alea_', 'pi_entropy_']
elif args.mode == 'vae':
    method = ['recon_', 'kl_']
else:
    raise NotImplementedError


# ================= Run Dir =================
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
DIR = f'./res/{run_name}_{args.mode}_{args.id}/'
DIR2 = os.path.join(DIR, 'ckpt/')
os.makedirs(DIR2, exist_ok=True)

log = Logger(
    os.path.join(DIR, 'log.json'),
    exp_case=Solver.test_e_dataset.case,
    neg_case=Solver.test_n_dataset.case
)

txt_path = os.path.join(DIR, 'log.txt')
f = open(txt_path, 'w')
print_n_txt(f, 'Text name: ' + txt_path)
print_n_txt(f, str(args))


# ================= Train =================
train_l2, test_l2 = Solver.train_func(f)
log.train_res(train_l2, test_l2)


# ================= Eval (ALL NEG) =================
id_eval = Solver.eval_func(Solver.test_e_iter, device)
ood_eval = Solver.eval_func(Solver.test_n_iter, device)

auroc, aupr = {}, {}
for m in method:
    r1, r2 = measure(id_eval[m], ood_eval[m])
    print_n_txt(f, f"\n{m[:-1]} AUROC: [{r1:.3f}] AUPR: [{r2:.3f}]\n")
    auroc[m] = r1
    aupr[m] = r2

log.ood(id_eval, ood_eval, auroc, aupr)


# ================= Scenario-wise NEG (ISOLATED SOLVER) =================
neg_scenarios = discover_neg_scenarios(args.root)
trained_state = Solver.model.state_dict()

scenario_eval = {}
scenario_auroc = {}
scenario_aupr = {}

trial_score_plots = {}   # summary to log.json
ood_plot_dir = os.path.join(DIR, "ood_score_plots")
os.makedirs(ood_plot_dir, exist_ok=True)

for scen_name, info in neg_scenarios.items():
    print_n_txt(f, f"\n[INFO] Scenario-wise eval: {scen_name}")

    args_s = argparse.Namespace(**vars(args))
    args_s.root = args.root
    args_s.neg_case = [scen_name]

    Solver_s = solver(args_s, device=device, SEED=SEED)
    Solver_s.init_param()
    Solver_s.model.load_state_dict(trained_state)

    print(
    f"[CHECK] {scen_name} | neg_case={args_s.neg_case} | "
    f"N_neg_samples={len(Solver_s.test_n_dataset)}")

    x = Solver_s.test_n_dataset.x
    print(
        f"[CHECK DATA] {scen_name} | "
        f"x_mean={x.mean().item():.6f}, x_std={x.std().item():.6f}"
    )

    id_eval_s = Solver_s.eval_func(Solver_s.test_e_iter, device)
    ood_eval_s = Solver_s.eval_func(Solver_s.test_n_iter, device)

    a1, a2 = {}, {}
    for m in method:
        r1, r2 = measure(id_eval_s[m], ood_eval_s[m])
        a1[m] = r1
        a2[m] = r2

    scenario_eval[scen_name] = {
        "id": id_eval_s,
        "ood": ood_eval_s
    }
    scenario_auroc[scen_name] = a1
    scenario_aupr[scen_name] = a2

    # # ================= Trial-wise OOD score plot =================
    # # trial dirs: 예) .../Data/neg_turnleft_1, .../Data/neg_turnleft_2 ...
    # trial_score_plots.setdefault(scen_name, {})

    # # mdn이면 epis_, vae면 recon_만 사용
    # if args.mode == "mdn":
    #     score_key = "epis_"
    # elif args.mode == "vae":
    #     score_key = "recon_"
    # else:
    #     # 필요하면 다른 모드도 여기서 지정 가능
    #     score_key = "recon_"

    # # info["dirs"]는 discover_neg_scenarios가 모아둔 trial 디렉토리 경로들
    # for trial_dir in sorted(info["dirs"]):
    #     trial_name = Path(trial_dir).name  # neg_(scenario)_n

    #     args_t = argparse.Namespace(**vars(args))
    #     args_t.root = args.root
    #     args_t.neg_case = [trial_name]     # ★ trial 하나만 선택

    #     Solver_t = solver(args_t, device=device, SEED=SEED)
    #     Solver_t.init_param()
    #     Solver_t.model.load_state_dict(trained_state)

    #     # OOD score 계산 (AUROC/AUPR 필요 없음)
    #     ood_eval_t = Solver_t.eval_func(Solver_t.test_n_iter, device)
    #     scores = ood_eval_t.get(score_key, None)

    #     if scores is None or len(scores) == 0:
    #         print_n_txt(f, f"[WARN] {trial_name}: no scores for key={score_key}")
    #         continue

    #     # x축: window 시작 frame index (seq)
    #     seq = getattr(Solver_t.test_n_dataset, "neg_seq", None)
    #     if seq is None or len(seq) != len(scores):
    #         x = list(range(len(scores)))
    #     else:
    #         x = seq

    #     # (선택) CSV로도 저장하면 후처리 편함
    #     csv_path = os.path.join(ood_plot_dir, f"{trial_name}_{score_key}.csv")
    #     scores = np.asarray(scores, dtype=float)
    #     x = np.asarray(x, dtype=int)
    #     np.savetxt(csv_path, np.column_stack([x, scores]), delimiter=",",
    #                header="seq,score", comments="")

    #     # PNG plot 저장
    #     png_path = os.path.join(ood_plot_dir, f"{trial_name}_{score_key}.png")
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(x, scores)
    #     plt.xlabel("window start frame index (seq)")
    #     plt.ylabel(f"OOD score ({score_key})")
    #     plt.title(f"{trial_name} | window_len(frame)={args.frame}")
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig(png_path, dpi=200)
    #     plt.close()

    #     trial_score_plots[scen_name][trial_name] = {
    #         "score_key": score_key,
    #         "png": png_path,
    #         "csv": csv_path,
    #         "n_samples": len(scores),
    #     }

    #     print_n_txt(f, f"[SAVE] {trial_name} -> {png_path}")




# ================= Save =================
torch.save(Solver.model.state_dict(), os.path.join(DIR2, 'model.pt'))
log.save()

log_path = os.path.join(DIR, 'log.json')
with open(log_path, 'r') as jf:
    log_json = json.load(jf)

log_json["scenario_ood_eval"] = scenario_eval
log_json["scenario_auroc"] = scenario_auroc
log_json["scenario_aupr"] = scenario_aupr
# log_json["trial_ood_score_plots"] = trial_score_plots


with open(log_path, 'w') as jf:
    json.dump(log_json, jf, indent=2)

# ================= Plot (MUST BE LAST) =================
try:
    plot_run(run_name, args.mode, args.id)
except Exception as e:
    print_n_txt(f, f"[WARN] plotting failed: {e}")
