import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tools.measure_ood import measure
import re


def _extract_frame_from_logtxt(base_dir: str):

    log_txt_path = os.path.join(base_dir, 'log.txt')
    if not os.path.isfile(log_txt_path):
        return None

    frame_val = None
    with open(log_txt_path, 'r') as f:
        for line in f:
            if 'Namespace(' in line and 'frame=' in line:
                m = re.search(r'frame=(\d+)', line)
                if m:
                    frame_val = int(m.group(1))
                    break

    return frame_val



def _load_log(run_name: str, mode: str, exp_id: int):

    # ./res/{run_name}_{mode}_{id}/log.json
    base_dir = os.path.join('./res', f'{run_name}_{mode}_{exp_id}')
    log_path = os.path.join(base_dir, 'log.json')

    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"log.json not found: {log_path}")

    with open(log_path, 'r') as f:
        data = json.load(f)

    return data, base_dir



def plot_run(run_name: str, mode: str, exp_id: int):
    """
    main.py caller function
    """
    data, base_dir = _load_log(run_name, mode, exp_id)

    train_l2 = data['train_l2']      # list[epoch] of float
    test_l2  = data['test_l2']       # list[epoch] of float
    auroc    = data['auroc']         # dict: metric -> float
    aupr     = data['aupr']          # dict: metric -> float
    id_eval  = data['id_eval']       # dict: metric -> list[score]
    ood_eval = data['ood_eval']      # dict: metric -> list[score]

    # select (mode - metric key) pairs
    if mode == 'mdn':
        method = ['epis_', 'alea_', 'pi_entropy_']
    elif mode == 'vae':
        method = ['recon_', 'kl_']
    elif mode == 'vqvae':
        method = ['recon_', 'vq_']
    elif mode == 'wae':
        method = ['recon_', 'mmd_']
    elif mode == 'rae':
        method = ['recon_', 'zreg_']
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")


    n_methods = len(method)

    # ---------- Figure / Subplots ----------
    fig, axes = plt.subplots(
        n_methods + 1, 2,
        figsize=(12, 4 * (n_methods + 1)),
        squeeze=False
    )

    # ---- (row 0, col 0) L2 train / test ----
    ax_l2 = axes[0, 0]
    epochs = np.arange(1, len(train_l2) + 1)

    ax_l2.plot(epochs, train_l2, '--', label='train L2')
    ax_l2.plot(epochs, test_l2, label='test L2')
    ax_l2.set_xlabel("Epoch")
    ax_l2.set_ylabel("L2 error")
    ax_l2.set_title("Train / Test L2 over epochs")
    ax_l2.legend()
    ax_l2.grid(True, linestyle=':')

    # ---- (row 0, col 1) AUROC/AUPR summary ----
    ax_text = axes[0, 1]
    ax_text.axis('off')

    frame_val = _extract_frame_from_logtxt(base_dir)
    frame_str = str(frame_val) if frame_val is not None else "unknown"

    text_lines = [
        f"mode: {mode}, id: {exp_id}",
        f"run:  {run_name}",
        f"frame: {frame_str}",
        "",
        "OOD metrics (AUROC / AUPR):"
    ]
    for m in method:
        m_name = m[:-1]
        if m in auroc and m in aupr:
            text_lines.append(
                f"  {m_name:12s}: AUROC={auroc[m]:.3f}, AUPR={aupr[m]:.3f}"
            )

    ax_text.text(
        0.01, 0.99,
        "\n".join(text_lines),
        va='top', ha='left',
        fontsize=10
    )

    # ---------- Histogtram per method + ROC ----------
    for idx, m in enumerate(method):
        row = idx + 1
        m_name = m[:-1]

        id_scores  = np.asarray(id_eval[m])
        ood_scores = np.asarray(ood_eval[m])

        # (row, col=0) score distribution (ID vs OOD)
        ax_hist = axes[row, 0]
        bins = 30

        ax_hist.hist(id_scores,  bins=bins, alpha=0.6,
                     label='ID (expert)', density=True)
        ax_hist.hist(ood_scores, bins=bins, alpha=0.6,
                     label='OOD (negative)', density=True)
        ax_hist.set_title(f"{m_name} score distribution")
        ax_hist.set_xlabel("score")
        ax_hist.set_ylabel("density")
        ax_hist.legend()
        ax_hist.grid(True, linestyle=':')

        # (row, col=1) ROC curve
        ax_roc = axes[row, 1]

        auroc_m, aupr_m, fpr, tpr = measure(
            id_scores.tolist(),
            ood_scores.tolist(),
            plot=True
        )

        ax_roc.plot(fpr, tpr,
                    label=f"ROC (AUROC={auroc_m:.3f}, AUPR={aupr_m:.3f})")
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"{m_name} ROC curve")
        ax_roc.legend()
        ax_roc.grid(True, linestyle=':')

        ax_roc.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax_roc.yaxis.set_major_locator(mticker.MaxNLocator(5))

    plt.tight_layout()

    out_path = os.path.join(base_dir, f"plot_{mode}_{exp_id}.png")
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved figure to: {out_path}")


def _find_latest_run(mode: str, exp_id: int):

    if not os.path.isdir('./res'):
        raise FileNotFoundError("./res directory not found")

    suffix = f"_{mode}_{exp_id}"

    subdirs = [
        d for d in os.listdir('./res')
        if os.path.isdir(os.path.join('./res', d)) and d.endswith(suffix)
    ]
    if not subdirs:
        raise FileNotFoundError(
            f"No run folders matching '*{suffix}' under ./res"
        )

    subdirs.sort()
    latest_dir = subdirs[-1]  # Ex: '20251209_010203_mdn_1'

    run_name = latest_dir[:-len(suffix)]
    return run_name



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mdn',
                        help='mdn / vae / wae / rae / vqvae')
    parser.add_argument('--id', type=int, default=1,
                        help='experiment id (same as main.py --id)')
    parser.add_argument('--run', type=str, default='latest',
                        help="run folder name under ./res/ ")
    args = parser.parse_args()

    if args.run == 'latest':
        run_name = _find_latest_run(args.mode, args.id)  # latest
    else:
        run_name = args.run
        # Esistence Check - './res/{run_name}_{mode}_{id}'
        run_dir = os.path.join('./res', f'{run_name}_{args.mode}_{args.id}')
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(
                f"Run folder not found: {run_dir}"
            )

    print(f"[INFO] Using run_name = {run_name}")
    plot_run(run_name, args.mode, args.id)


if __name__ == "__main__":
    main()