from solver import solver
from tools.utils import *
from tools.measure_ood import measure
from plot import plot_run
import torch
from datetime import datetime
import argparse
import os
import json
import re
from pathlib import Path
from tools.data_summary import collect_distances_to_csv

def _trial_sort_key(trial_ref: str):
    """
    Sort by trial numbers:
      - "N1/030" -> ("N1", 30)
      - "N1_030" -> ("N1", 30)
      - "030"    -> ("", 30)
    """
    s = str(trial_ref)
    m = re.search(r'([/_])(\d+)$', s)
    if m:
        base = s[:m.start(1)]
        num = int(m.group(2))
        return (base, num)
    if s.isdigit():
        return ("", int(s))
    
    return (s, float("inf"))


def run_experiment(args, device: str, seed: int):

    # Solver
    Solver = solver(args, device=device, SEED=seed)
    Solver.init_param()
    method = get_methods(args.mode)

    # Set Dirs
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    DIR = f'./res/{run_name}_{args.mode}_{args.frame}/'
    DIR2 = os.path.join(DIR, 'ckpt/')
    os.makedirs(DIR2, exist_ok=True)

    log = Logger(os.path.join(DIR, 'log.json'))
    txt_path = os.path.join(DIR, 'log.txt')

    with open(txt_path, 'w') as f:
        print_n_txt(f, '\nText name: ' + txt_path)
        print_n_txt(f, str(args))

        # Record used data
        exp_trials_used = sorted(set(getattr(Solver.test_e_dataset, "exp_dirs", []) or []))
        neg_trials_used = sorted(set(getattr(Solver.test_n_dataset, "neg_dirs", []) or []))

        print_n_txt(f, "\n[DATA USED]", _DO_PRINT = False)
        print_n_txt(f, f"Expert trials loaded: {exp_trials_used}", _DO_PRINT = False)
        print_n_txt(f, f"Negative trials loaded: {neg_trials_used}\n", _DO_PRINT = False)

        print_n_txt(f, f"[EXPERT TRAIN SCENARIOS] {getattr(Solver.train_dataset, 'exp_train_scenarios', None)}", _DO_PRINT = False)
        print_n_txt(f, f"[EXPERT TEST  SCENARIOS] {getattr(Solver.test_e_dataset, 'exp_test_scenarios', None)}", _DO_PRINT = False)

        # Train
        train_l2, test_l2 = Solver.train_func(f)
        log.train_res(train_l2, test_l2)

        # Evaluate All negatives
        id_eval  = Solver.eval_func(Solver.test_e_iter, device)
        ood_eval = Solver.eval_func(Solver.test_n_iter, device)

        auroc, aupr = {}, {}
        for m in method:
            r1, r2 = measure(id_eval[m], ood_eval[m])
            print_n_txt(f, f"\n{m[:-1]} AUROC: [{r1:.4f}] AUPR: [{r2:.4f}]\n")
            auroc[m] = r1
            aupr[m]  = r2
        log.ood(id_eval, ood_eval, auroc, aupr)

        # Scenario-wise Eval
        scenario_eval      = {}
        scenario_auroc     = {}
        scenario_aupr      = {}
        scenario_dirs_used = {}

        neg_scen_list = getattr(Solver.test_n_dataset, "neg_scenario", None) \
                        or getattr(Solver.test_n_dataset, "neg_scenario_sel", None)
        neg_trial_list = getattr(Solver.test_n_dataset, "neg_trial", None) \
                         or getattr(Solver.test_n_dataset, "neg_trial_sel", None)

        if neg_scen_list is None or len(neg_scen_list) == 0:
            print_n_txt(f, "[WARN] Scenario-wise eval skipped: test_n_dataset has no neg_scenario metadata.")
        else:
            # sanity check: lengths must match (to catch drop/filter bugs)
            for m in method:
                if len(ood_eval[m]) != len(neg_scen_list):
                    raise RuntimeError(
                        f"[Scenario-wise eval ERROR] Length mismatch: len(ood_eval[{m}])={len(ood_eval[m])} != "
                        f"len(neg_scen_list)={len(neg_scen_list)}. "
                        f"Make sure DataLoader shuffle=False and neg_scenario is aligned with dataset order."
                    )

            scen_to_indices = {}
            for i, scen in enumerate(neg_scen_list):
                scen_to_indices.setdefault(scen, []).append(i)

            # scenario -> unique trial refs
            if neg_trial_list is not None and len(neg_trial_list) == len(neg_scen_list):
                for scen, idxs in scen_to_indices.items():
                    trials = sorted({neg_trial_list[i] for i in idxs}, key=_trial_sort_key)
                    scenario_dirs_used[scen] = trials
            else:
                for scen in scen_to_indices.keys():
                    scenario_dirs_used[scen] = []

            for scen_name, idxs in sorted(scen_to_indices.items(), key=lambda kv: kv[0]):
                print_n_txt(f, f"\n[INFO] Scenario-wise eval: {scen_name} | N={len(idxs)}")

                ood_eval_s = {}
                a1, a2 = {}, {}

                for m in method:
                    ood_scores_s = [ood_eval[m][i] for i in idxs]
                    ood_eval_s[m] = ood_scores_s

                    r1, r2 = measure(id_eval[m], ood_scores_s)
                    a1[m] = r1
                    a2[m] = r2

                    print_n_txt(f, f"  - {m[:-1]} AUROC: [{r1:.4f}]  AUPR: [{r2:.4f}]")

                scenario_eval[scen_name]  = {"id": id_eval, "ood": ood_eval_s}
                scenario_auroc[scen_name] = a1
                scenario_aupr[scen_name]  = a2


    # Save to local files
    torch.save(Solver.model.state_dict(), os.path.join(DIR2, 'model.pt'))
    log.save()

    log_path = os.path.join(DIR, 'log.json')
    with open(log_path, 'r') as jf:
        log_json = json.load(jf)

    log_json["scenario_ood_eval"] = scenario_eval
    log_json["scenario_auroc"] = scenario_auroc
    log_json["scenario_aupr"] = scenario_aupr

    log_json["data_used"] = {
        "expert_trials_loaded": exp_trials_used,
        "neg_trials_loaded": neg_trials_used,
        "neg_scenario_trials": scenario_dirs_used,
    }

    log_json["expert_split"] = {
        "train_scenarios": getattr(Solver.train_dataset, "exp_train_scenarios", None),
        "test_scenarios":  getattr(Solver.test_e_dataset, "exp_test_scenarios", None),
        "stats": getattr(Solver.train_dataset, "exp_split_stats", None),
    }

    with open(log_path, 'w') as jf:
        json.dump(log_json, jf, indent=2)

    return run_name, DIR


def main():

    ### Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',     type=str,   default='.', help='project root dir'                            )
    parser.add_argument('--mode',     type=str,   default='vae'                                                   )
    parser.add_argument('--gpu',      type=int,   default=0                                                       )
    parser.add_argument('--frame',    type=int,   default=10                                                      )
    parser.add_argument('--exp_case', type=int,   default=None, nargs='+', help='exp scenario numbers to include' )
    parser.add_argument('--neg_case', type=str,   default=None, nargs='+', help='neg scenario names to include'   )
    parser.add_argument('--epoch',    type=int,   default=240                                                     )
    parser.add_argument('--lr',       type=float, default=1e-3                                                    )
    parser.add_argument('--batch',    type=int,   default=128                                                     )
    parser.add_argument('--wd',       type=float, default=1e-4                                                    )
    parser.add_argument('--dropout',  type=float, default=0.25                                                    )
    parser.add_argument('--lr_rate',  type=float, default=0.8                                                     )
    parser.add_argument('--lr_step',  type=int,   default=20                                                      )
    # MDN
    parser.add_argument('--k',       type=int,   default=10 )
    parser.add_argument('--norm',    type=int,   default=1  )
    parser.add_argument('--sig_max', type=float, default=1  )
    # VAE
    parser.add_argument('--h_dim', type=list, nargs='+', default=[256] )
    parser.add_argument('--z_dim', type=int , default=64               )
    args = parser.parse_args()


    ### Set seed / device
    seed, device = set_seed_and_device(args, seed=0)
    run_name, _ = run_experiment(args, device=device, seed=seed)


    ### Dataset summary
    DATA_ROOT = Path("./Anomaly_detection_dataset").resolve()
    collect_distances_to_csv(DATA_ROOT, include_missing=False)


    ### Plot
    try:
        plot_run(run_name, args.mode, args.frame)
    except Exception as e:
        print(f"[WARN] plotting failed: {e}")


if __name__ == "__main__":
    main()
