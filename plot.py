import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tools.measure_ood import measure
import re
from sklearn.metrics import precision_recall_curve



def _extract_frame_from_logtxt(base_dir):
    log_txt = os.path.join(base_dir, 'log.txt')
    if not os.path.isfile(log_txt):
        return "unknown"

    with open(log_txt) as f:
        for line in f:
            if 'Namespace(' in line and 'frame=' in line:
                m = re.search(r'frame=(\d+)', line)
                if m:
                    return m.group(1)
    return "unknown"


def _load_log(run_name, mode, exp_id):
    base_dir = os.path.join('./res', f'{run_name}_{mode}_{exp_id}')
    with open(os.path.join(base_dir, 'log.json')) as f:
        data = json.load(f)
    return data, base_dir


def plot_run(run_name, mode, exp_id):

    data, base_dir = _load_log(run_name, mode, exp_id)

    train_l2 = data['train_l2']
    test_l2  = data['test_l2']
    auroc    = data['auroc']
    aupr     = data['aupr']
    id_eval  = data['id_eval']
    ood_eval = data['ood_eval']

    scenario_eval  = data.get("scenario_ood_eval", {})
    scenario_auroc = data.get("scenario_auroc", {})
    scenario_aupr  = data.get("scenario_aupr", {})

    if mode == 'mdn':
        method = ['epis_']
    elif mode == 'vae':
        method = ['recon_']
    else:
        raise NotImplementedError

    frame = _extract_frame_from_logtxt(base_dir)

    # ===== Base Plot (ALL NEG) =====
    def make_plot(tag, id_eval_cur, ood_eval_cur, auroc_cur, aupr_cur):
        fig, axes = plt.subplots(len(method) + 1, 3,
                                 figsize=(18, 4 * (len(method) + 1)),
                                 squeeze=False)

        ep = np.arange(1, len(train_l2) + 1)
        axes[0, 0].plot(ep, train_l2, '--', label='train')
        axes[0, 0].plot(ep, test_l2, label='test')
        axes[0, 0].legend()
        axes[0, 0].set_title("L2 over epochs")

        axes[0, 1].axis('off')
        lines = [
            f"mode: {mode}, id: {exp_id}",
            f"run: {run_name}",
            f"frame: {frame}",
            f"scenario: {tag}",
            "",
            "AUROC / AUPR"
        ]
        for m in method:
            lines.append(f"{m[:-1]}: {auroc_cur[m]:.3f} / {aupr_cur[m]:.3f}")
        axes[0, 1].text(0.01, 0.99, "\n".join(lines),
                        va='top', ha='left')
        axes[0, 2].axis('off')

        for i, m in enumerate(method):
            id_s  = np.asarray(id_eval_cur[m])
            ood_s = np.asarray(ood_eval_cur[m])


            axes[i+1, 0].hist(id_s,  100, alpha=0.5, density=True, label='ID')
            axes[i+1, 0].hist(ood_s, 100, alpha=0.5, density=True, label='OOD')
            axes[i+1, 0].legend()

            # ROC
            _, _, fpr, tpr = measure(id_s.tolist(), ood_s.tolist(), plot=True)
            axes[i+1, 1].plot(fpr, tpr)
            axes[i+1, 1].plot([0, 1], [0, 1], 'k--')
            axes[i+1, 1].set_title(f"{m[:-1]} ROC")
            axes[i+1, 1].set_xlabel("FPR")
            axes[i+1, 1].set_ylabel("TPR")

            # PR
            gt = np.concatenate([np.zeros(len(id_s), dtype=int), np.ones(len(ood_s), dtype=int)])
            scores = np.concatenate([id_s, ood_s])
            precision, recall, _ = precision_recall_curve(gt, scores)

            axes[i+1, 2].plot(recall, precision)
            axes[i+1, 2].set_title(f"{m[:-1]} PR")
            axes[i+1, 2].set_xlabel("Recall")
            axes[i+1, 2].set_ylabel("Precision")
            axes[i+1, 2].set_xlim([0.0, 1.0])
            axes[i+1, 2].set_ylim([0.0, 1.0])

            # (optional) PR baseline = positive rate
            pos_rate = gt.mean()
            axes[i+1, 2].plot([0, 1], [pos_rate, pos_rate], 'k--')


        plt.tight_layout()
        out = os.path.join(base_dir, f"plot_{mode}_{exp_id}_{tag}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[INFO] saved {out}")

    make_plot("all", id_eval, ood_eval, auroc, aupr)

    for scen in scenario_eval:
        make_plot(
            scen,
            scenario_eval[scen]["id"],    # 🔥 시나리오별 ID score
            scenario_eval[scen]["ood"],   # 🔥 시나리오별 OOD score
            scenario_auroc[scen],
            scenario_aupr[scen]
        )


# ===== CLI =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mdn')
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--run', type=str, default='latest')
    args = parser.parse_args()

    if args.run == 'latest':
        dirs = [d for d in os.listdir('./res') if d.endswith(f"_{args.mode}_{args.id}")]
        dirs.sort()
        run_name = dirs[-1].replace(f"_{args.mode}_{args.id}", "")
    else:
        run_name = args.run

    plot_run(run_name, args.mode, args.id)


if __name__ == "__main__":
    main()
