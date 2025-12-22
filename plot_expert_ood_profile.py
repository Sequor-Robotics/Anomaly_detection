# plot_expert_ood_profile.py
import os
import re
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch

# --- hotfix: dtype/device에 NoneType(클래스)가 들어오면 None으로 강제 ---
_torch_empty_orig = torch.empty

def _fix_factory_kw(v, kind):
    # kind: "dtype" or "device"
    if v is None:
        return None
    # type(None) 또는 NoneType 클래스면 None으로
    if v is type(None):
        return None
    if isinstance(v, type) and getattr(v, "__name__", "") == "NoneType":
        return None

    # dtype/device로 올 수 없는 타입이면 None으로 떨어뜨리기 (안전망)
    if kind == "dtype":
        if not isinstance(v, torch.dtype):
            return None
    if kind == "device":
        if not isinstance(v, torch.device):
            return None
    return v

def _torch_empty_fix(*args, **kwargs):
    if "dtype" in kwargs:
        kwargs["dtype"] = _fix_factory_kw(kwargs["dtype"], "dtype")
    if "device" in kwargs:
        kwargs["device"] = _fix_factory_kw(kwargs["device"], "device")
    return _torch_empty_orig(*args, **kwargs)

torch.empty = _torch_empty_fix
# -------------------------------------------------------------------

import matplotlib.pyplot as plt

# repo modules (패치 이후에 import!)
from MDN.network import MixtureDensityNetwork
from MDN.loss import mdn_uncertainties
from VAE.network import VAE
from VAE.loss import VAE_eval


def normalize_like_mixquality(X, frame_len, scan_len, mean_x, std_x, mode="z", div_flip=False):
    """
    X: (N, D) raw windows
    mode: "z" only (너 pipeline의 self.norm==1에 해당)
    div_flip: 네가 1 - (x/5)로 쓰고 있으면 True, 그냥 /5면 False
    """
    eps = 1e-8
    D = X.shape[1]
    per_frame_dim = (7 + scan_len)

    # per-frame indices
    idx_z = []    # 0~4
    idx_div = []  # 5~6
    for i in range(frame_len):
        base = i * per_frame_dim
        idx_z.extend(range(base + 0, base + 5))
        idx_div.extend(range(base + 5, base + 7))

    idx_z = np.asarray(idx_z, dtype=np.int64)
    idx_div = np.asarray(idx_div, dtype=np.int64)

    Xn = X.copy().astype(np.float32)

    # 0~4: z-score만 적용
    Xn[:, idx_z] = (Xn[:, idx_z] - mean_x[idx_z]) / (std_x[idx_z] + eps)

    # 5~6: /5 (또는 1 - /5)
    if div_flip:
        Xn[:, idx_div] = 1.0 - (Xn[:, idx_div] / 5.0)
    else:
        Xn[:, idx_div] = (Xn[:, idx_div] / 5.0)

    # LiDAR(7~)는 그대로 둠

    Xn = np.where(np.isfinite(Xn), Xn, 0.0)
    return Xn


# -----------------------------
# Helpers: parse training args from log.txt
# -----------------------------
def parse_args_from_logtxt(res_dir: Path) -> dict:
    log_txt = res_dir / "log.txt"
    out = {}
    if not log_txt.is_file():
        return out

    txt = log_txt.read_text(errors="ignore")

    def pick_int(key):
        m = re.search(rf"{key}=([0-9]+)", txt)
        return int(m.group(1)) if m else None

    def pick_float(key):
        m = re.search(rf"{key}=([0-9eE\.\+\-]+)", txt)
        return float(m.group(1)) if m else None

    def pick_bool(key):
        m = re.search(rf"{key}=(True|False)", txt)
        return (m.group(1) == "True") if m else None

    out["frame"] = pick_int("frame")
    out["norm"] = pick_bool("norm")
    out["k"] = pick_int("k")
    out["sig_max"] = pick_float("sig_max")
    out["dropout"] = pick_float("dropout")
    out["h_dim"] = pick_int("h_dim")
    out["z_dim"] = pick_int("z_dim")
    return out


# -----------------------------
# Data reading (expert_* / *_prcd.json)
# -----------------------------
def find_scenarios(data_root: Path, prefix: str):
    return sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith(prefix)])


def load_frames_from_prcd(scenario_dir: Path):
    scen = scenario_dir.name
    prcd = scenario_dir / f"{scen}_prcd.json"
    if not prcd.is_file():
        raise FileNotFoundError(f"missing {prcd}")
    with open(prcd, "r") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    return frames


def infer_scan_len(frames):
    # find first non-empty scan length
    for fr in frames:
        scan = fr.get("scan") or []
        if len(scan) > 0:
            return len(scan)
    return 0


def build_state_fixed(fr, scan_len: int):
    pos     = fr.get("position") or [0.0, 0.0]
    lin_vel = fr.get("lin_vel")  or [0.0, 0.0]
    lin_acc = fr.get("lin_acc")  or [0.0, 0.0]
    obj_pos = fr.get("obj_pos")  or [5.0, 5.0]
    ang_vel = fr.get("ang_vel")
    if ang_vel is None:
        ang_vel = 0.0

    # scan -> ranges
    scan = fr.get("scan") or []
    if len(scan) > 0:
        ranges = np.array([p[1] for p in scan], dtype=np.float32)
        finite = np.isfinite(ranges)
        if finite.any():
            ranges = ranges
        else:
            ranges = np.zeros_like(ranges)
        ranges = ranges.tolist()
    else:
        ranges = []

    # pad/trim to fixed length
    if scan_len > 0:
        if len(ranges) < scan_len:
            ranges = ranges + [0.0] * (scan_len - len(ranges))
        elif len(ranges) > scan_len:
            ranges = ranges[:scan_len]

    state_vec = [
        float(lin_vel[0]), float(lin_vel[1]),
        float(lin_acc[0]), float(lin_acc[1]),
        float(ang_vel),
        float(obj_pos[0]), float(obj_pos[1]),
    ]
    state_vec.extend(ranges)
    return state_vec


def extract_time(fr, fallback_idx: int):
    # 다양한 키 대응 (없으면 index)
    for k in ["timestamp", "time", "t", "stamp"]:
        if k in fr:
            try:
                return float(fr[k])
            except Exception:
                pass
    return float(fallback_idx)


def make_windows(frames, frame_len: int, scan_len: int):
    """
    returns:
      X: (N, Din)
      Y: (N, 3)  (for MDN)
      T: (N,)    time of last frame in window
    """
    n = len(frames)
    if n < frame_len:
        return None, None, None

    din_per = 7 + scan_len
    X = np.zeros((n - frame_len + 1, frame_len * din_per), dtype=np.float32)
    Y = np.zeros((n - frame_len + 1, 3), dtype=np.float32)
    T = np.zeros((n - frame_len + 1,), dtype=np.float32)

    for seq in range(0, n - frame_len + 1):
        buf = []
        for it in range(frame_len):
            s = build_state_fixed(frames[seq + it], scan_len)
            buf.extend(s)

            if it == frame_len - 1:
                # target = last frame action(=acc/ang_vel)
                Y[seq, 0] = float(s[2])  # lin_acc x
                Y[seq, 1] = float(s[3])  # lin_acc y
                Y[seq, 2] = float(s[4])  # ang_vel
                T[seq] = extract_time(frames[seq + it], fallback_idx=(seq + it))

        X[seq, :] = np.asarray(buf, dtype=np.float32)

    return X, Y, T


# -----------------------------
# Global normalization (mean/std) over expert+neg (optional)
# -----------------------------
def welford_update(mean, m2, count, x):
    # x: (B, D)
    if x.size == 0:
        return mean, m2, count
    x = x.astype(np.float64)
    if mean is None:
        mean = np.zeros((x.shape[1],), dtype=np.float64)
        m2 = np.zeros((x.shape[1],), dtype=np.float64)
        count = 0

    for i in range(x.shape[0]):
        count += 1
        delta = x[i] - mean
        mean += delta / count
        delta2 = x[i] - mean
        m2 += delta * delta2
    return mean, m2, count


def compute_global_mean_std(data_root: Path, frame_len: int, include_neg: bool):
    scen_paths = find_scenarios(data_root, "expert_")
    if include_neg:
        scen_paths += find_scenarios(data_root, "neg_")

    mean_x = None
    m2_x = None
    cnt_x = 0

    mean_y = np.zeros((3,), dtype=np.float64)
    m2_y = np.zeros((3,), dtype=np.float64)
    cnt_y = 0

    # scan_len은 시나리오마다 다르면 곤란하므로,
    # 가장 많이 쓰일 가능성이 높은 "첫 expert 시나리오" 기준으로 고정
    if len(scen_paths) == 0:
        raise RuntimeError("No scenarios found under data_root")
    first_frames = load_frames_from_prcd(scen_paths[0])
    scan_len = infer_scan_len(first_frames)

    for scen in scen_paths:
        frames = load_frames_from_prcd(scen)
        X, Y, _ = make_windows(frames, frame_len, scan_len)
        if X is None:
            continue
        mean_x, m2_x, cnt_x = welford_update(mean_x, m2_x, cnt_x, X)

        # Y (3D) update
        Y64 = Y.astype(np.float64)
        for i in range(Y64.shape[0]):
            cnt_y += 1
            d = Y64[i] - mean_y
            mean_y += d / cnt_y
            d2 = Y64[i] - mean_y
            m2_y += d * d2

    if cnt_x < 2:
        raise RuntimeError("Not enough samples to compute std")

    var_x = m2_x / (cnt_x - 1)
    std_x = np.sqrt(np.maximum(var_x, 1e-12))

    var_y = m2_y / max(cnt_y - 1, 1)
    std_y = np.sqrt(np.maximum(var_y, 1e-12))

    return scan_len, mean_x.astype(np.float32), std_x.astype(np.float32), mean_y.astype(np.float32), std_y.astype(np.float32)


def normalize_z(x, mean, std):
    x = (x - mean) / std
    # NaN -> 0 (학습 코드와 동일한 효과)
    x = np.where(np.isfinite(x), x, 0.0)
    return x


# -----------------------------
# OOD scoring
# -----------------------------
@torch.no_grad()
def score_mdn(model, X, batch_size=512, device="cuda"):
    model.eval()
    scores = {"epis_": [], "alea_": [], "pi_entropy_": []}

    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
        out = model.forward(xb)
        pi, mu, sigma = out["pi"], out["mu"], out["sigma"]
        u = mdn_uncertainties(pi, mu, sigma)

        epis = u["epis"]          # [N, D]
        alea = u["alea"]          # [N, D]
        ent  = u["pi_entropy"]    # [N]

        # 학습 eval과 동일: D축 max
        epis, _ = torch.max(epis, dim=-1)
        alea, _ = torch.max(alea, dim=-1)

        scores["epis_"].extend(epis.detach().cpu().numpy().tolist())
        scores["alea_"].extend(alea.detach().cpu().numpy().tolist())
        scores["pi_entropy_"].extend(ent.detach().cpu().numpy().tolist())

    model.train()
    return scores


@torch.no_grad()
def score_vae(model, X, batch_size=512, device="cuda"):
    model.eval()
    scores = {"recon_": [], "kl_": []}

    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
        x_recon, mu, logvar = model.forward(xb)

        loss_out = VAE_eval(xb, x_recon, mu, logvar)
        recon = loss_out["reconst_loss"]
        kl    = loss_out["kl_div"]

        # ---- recon을 (B,)로 만들기 ----
        # 가능한 경우:
        # - recon: scalar
        # - recon: (B,)
        # - recon: (B, D) / (B, D, ...)
        if recon.ndim == 0:
            recon_1d = recon.expand(xb.shape[0])
        elif recon.ndim == 1:
            recon_1d = recon
        else:
            recon_1d = recon.reshape(recon.shape[0], -1).mean(dim=1)

        # ---- kl도 (B,)로 만들기 ----
        if kl.ndim == 0:
            kl_1d = kl.expand(xb.shape[0])
        elif kl.ndim == 1:
            kl_1d = kl
        else:
            kl_1d = kl.reshape(kl.shape[0], -1).mean(dim=1)

        scores["recon_"].extend(recon_1d.detach().cpu().tolist())
        scores["kl_"].extend(kl_1d.detach().cpu().tolist())

    model.train()
    return scores

def parse_thresholds(thresholds_str: str):
    """
    'recon_=0.12,epis_=0.5' -> {'recon_':0.12, 'epis_':0.5}
    """
    if not thresholds_str:
        return {}
    out = {}
    for part in thresholds_str.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        out[k.strip()] = float(v.strip())
    return out


def find_runs(mask: np.ndarray, min_len: int = 1, merge_gap: int = 0):
    """
    mask: 1D bool array
    returns list of (start, end) with end exclusive.
    """
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    runs = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        # [i, j) is a run
        if (j - i) >= max(1, min_len):
            runs.append([i, j])
        i = j

    # merge close runs
    if merge_gap > 0 and len(runs) > 1:
        merged = [runs[0]]
        for s, e in runs[1:]:
            ps, pe = merged[-1]
            if s - pe <= merge_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        runs = merged

    return [(int(s), int(e)) for s, e in runs]


def plot_profile(t, s, title, out_png,
                 threshold=None, min_bad_len=1, merge_gap=0):
    t = np.asarray(t)
    s = np.asarray(s)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, s, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel("OOD score")
    ax.grid(True, alpha=0.3)

    if threshold is not None:
        # threshold line
        ax.axhline(float(threshold), linestyle="--", linewidth=1, label=f"thr={threshold:g}")

        bad = s > float(threshold)
        runs = find_runs(bad, min_len=min_bad_len, merge_gap=merge_gap)

        # shade bad segments
        for (st, ed) in runs:
            # t가 sample index이면 그냥 st~ed-1, time이면 t[st]~t[ed-1]
            x0 = t[st]
            x1 = t[ed - 1] if ed - 1 < len(t) else t[-1]
            ax.axvspan(x0, x1, alpha=0.2)

        # highlight bad points
        if bad.any():
            ax.plot(t[bad], s[bad], linestyle="None", marker="o", markersize=2, label="bad")

        # annotate summary
        if len(runs) > 0:
            ax.text(
                0.01, 0.01,
                f"bad_segments={len(runs)}  (min_len={min_bad_len}, merge_gap={merge_gap})",
                transform=ax.transAxes,
                ha="left", va="bottom"
            )

        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)



def top_spikes(t, s, k=20):
    idx = np.argsort(-s)[:k]
    rows = []
    for i in idx:
        rows.append((int(i), float(t[i]), float(s[i])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./Data", help="folder that contains expert_* / neg_*")
    ap.add_argument("--res_dir", type=str, required=True, help="e.g. ./res/<run>_<mode>_<id>")
    ap.add_argument("--mode", type=str, default=None, choices=["mdn", "vae"], help="override mode (optional)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--include_neg_for_norm", action="store_true", help="use expert+neg for mean/std (recommended)")
    ap.add_argument("--frame", type=int, default=None, help="override frame length (optional)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--scenario_regex", type=str, default=None, help="only expert scenarios matching regex")
    ap.add_argument("--spike_topk", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.000025,
                help="manual threshold; highlight segments where score > threshold")
    ap.add_argument("--min_bad_len", type=int, default=1,
                    help="minimum consecutive samples to treat as a bad segment")
    ap.add_argument("--merge_gap", type=int, default=0,
                    help="merge bad segments if gap between them <= merge_gap samples")
    ap.add_argument("--thresholds", type=str, default=None,
                    help="per-method thresholds, e.g. 'recon_=0.12,epis_=0.5' (overrides --threshold)")

    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    res_dir = Path(args.res_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (res_dir / "expert_profiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = res_dir / "ckpt" / "model.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")

    parsed = parse_args_from_logtxt(res_dir)
    frame_len = args.frame if args.frame is not None else (parsed.get("frame") or 1)

    mode = args.mode if args.mode else None
    if mode is None:
        # res_dir name ends with _<mode>_<id> in your pipeline
        m = re.search(r"_(mdn|vae)_\d+$", res_dir.name)
        mode = m.group(1) if m else "mdn"

    # 1) compute global mean/std for normalization
    scan_len, mean_x, std_x, mean_y, std_y = compute_global_mean_std(
        data_root=data_root,
        frame_len=frame_len,
        include_neg=args.include_neg_for_norm
    )

    din = frame_len * (7 + scan_len)
    ydim = 3

    # 2) load model
    device = args.device
    if mode == "mdn":
        k = parsed.get("k") or 5
        sig_max = parsed.get("sig_max") or 1.0
        dropout = parsed.get("dropout") or 0.0

        model = MixtureDensityNetwork(
            name="mdn",
            x_dim=din,
            y_dim=ydim,
            k=k,
            h_dims=[128, 128],
            actv=torch.nn.ReLU(),
            sig_max=sig_max,
            mu_min=-3, mu_max=+3,
            dropout=dropout,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        h_dim = parsed.get("h_dim") or [256]
        z_dim = parsed.get("z_dim") or 64
        if isinstance(h_dim, int):
            h_dim = [h_dim]
        model = VAE(x_dim=din, h_dim=h_dim, z_dim=z_dim).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

    # 3) iterate expert scenarios
    scenarios = find_scenarios(data_root, "expert_")
    if args.scenario_regex:
        rgx = re.compile(args.scenario_regex)
        scenarios = [s for s in scenarios if rgx.search(s.name)]

    if len(scenarios) == 0:
        raise RuntimeError("No expert_* scenarios matched")

    # summary CSV
    csv_path = out_dir / f"expert_ood_spikes_{mode}.csv"
    with open(csv_path, "w") as fcsv:
        fcsv.write("scenario,method,rank,sample_idx,time,score\n")

        for scen_dir in scenarios:
            scen_name = scen_dir.name
            frames = load_frames_from_prcd(scen_dir)
            X, Y, T = make_windows(frames, frame_len, scan_len)
            if X is None:
                print(f"[SKIP] {scen_name}: too short (< frame={frame_len})")
                continue

            Xn = normalize_like_mixquality(X, frame_len, scan_len, mean_x, std_x, div_flip=True)

            if mode == "mdn":
                out = score_mdn(model, Xn, batch_size=args.batch, device=device)
                methods = ["epis_", "alea_", "pi_entropy_"]
            else:
                out = score_vae(model, Xn, batch_size=args.batch, device=device)
                methods = ["recon_"]

            # per-method plot + spikes
            for m in methods:
                s = np.asarray(out[m], dtype=np.float32)
                # t = T  # already aligned to last frame in window
                t = np.arange(len(s), dtype=np.int32)

                thr_map = parse_thresholds(args.thresholds)
                thr = thr_map.get(m, args.threshold)  # method별 있으면 그걸 쓰고, 없으면 공통 threshold

                out_png = out_dir / f"{scen_name}__{m}.png"
                plot_profile(
                    t, s,
                    title=f"{scen_name} | {mode} | {m} | frame={frame_len}",
                    out_png=str(out_png),
                    threshold=thr,
                    min_bad_len=args.min_bad_len,
                    merge_gap=args.merge_gap
                )


                spikes = top_spikes(t, s, k=args.spike_topk)
                for r, (idx, tt, sc) in enumerate(spikes, start=1):
                    fcsv.write(f"{scen_name},{m},{r},{idx},{tt},{sc}\n")

            print(f"[OK] saved profiles: {scen_name}")

    print(f"[DONE] plots -> {out_dir}")
    print(f"[DONE] spikes csv -> {csv_path}")


if __name__ == "__main__":
    main()
