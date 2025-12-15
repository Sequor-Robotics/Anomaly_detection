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


# ================= Argument Parser =================
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='.', help='project root (parent of Data)')
parser.add_argument('--id', type=int, default=1)
parser.add_argument('--mode', type=str, default='mdn')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--frame', type=int, default=1)
parser.add_argument('--exp_case', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--neg_case', type=int, nargs='+', default=None, help='negative scenario case ids (parsed automatically if None)')


parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--lr_rate', type=float, default=0.9)
parser.add_argument('--lr_step', type=int, default=50)

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
elif args.mode == 'vqvae':
    method = ['recon_', 'vq_']
elif args.mode == 'wae':
    method = ['recon_', 'mmd_']
elif args.mode == 'rae':
    method = ['recon_', 'zreg_']
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
    print_n_txt(f, f"\n{m[:-1]} AUROC: [{r1:.3f}] AUPR: [{r2:.3f}]")
    auroc[m] = r1
    aupr[m] = r2

log.ood(id_eval, ood_eval, auroc, aupr)


# ================= Scenario-wise NEG (ISOLATED SOLVER) =================
neg_scenarios = discover_neg_scenarios(args.root)
trained_state = Solver.model.state_dict()

scenario_eval = {}
scenario_auroc = {}
scenario_aupr = {}

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


# ================= Save =================
torch.save(Solver.model.state_dict(), os.path.join(DIR2, 'model.pt'))
log.save()

log_path = os.path.join(DIR, 'log.json')
with open(log_path, 'r') as jf:
    log_json = json.load(jf)

log_json["scenario_ood_eval"] = scenario_eval
log_json["scenario_auroc"] = scenario_auroc
log_json["scenario_aupr"] = scenario_aupr

with open(log_path, 'w') as jf:
    json.dump(log_json, jf, indent=2)

# ================= Plot (MUST BE LAST) =================
try:
    plot_run(run_name, args.mode, args.id)
except Exception as e:
    print_n_txt(f, f"[WARN] plotting failed: {e}")
