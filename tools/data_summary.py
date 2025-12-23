import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _is_scenario_dir(p: Path) -> bool:
    # Data/ E1, E2, ..., N1, N2, ...
    return p.is_dir() and re.match(r"^[EN]\d+$", p.name) is not None


def _is_trial_dir(p: Path) -> bool:
    # each scenario floder/ 001, 002, ...
    return p.is_dir() and p.name.isdigit()


def _read_distance_txt(path: Path) -> Optional[float]:
    try:
        txt = path.read_text(encoding="utf-8").strip()
        if txt == "":
            return None
        return float(txt.split()[0])
    except Exception:
        return None


def collect_distances_to_csv(
    data_root: Path,
    out_csv: Path | None = None,
    include_missing: bool = False,
) -> Path:
    """
    Scan:
      data_root/
        E1/001/E1_001_distance.txt
        ...
        N6/123/N6_123_distance.txt

    Write CSV with:
      - trial rows
      - scenario subtotal rows
      - grand total row
    """
    data_root = Path(data_root).resolve()
    if out_csv is None:
        out_csv = data_root / "distance_summary.csv"
    else:
        out_csv = Path(out_csv).resolve()

    rows: List[Dict] = []
    scenario_sum: Dict[str, float] = {}
    scenario_count: Dict[str, int] = {}
    scenario_missing: Dict[str, int] = {}

    grand_total = 0.0
    grand_count = 0
    grand_missing = 0

    scenario_dirs = sorted([p for p in data_root.iterdir() if _is_scenario_dir(p)], key=lambda p: p.name)

    for scen_dir in scenario_dirs:
        scen_id = scen_dir.name
        trial_dirs = sorted([p for p in scen_dir.iterdir() if _is_trial_dir(p)], key=lambda p: int(p.name))

        s_sum = 0.0
        s_count = 0
        s_miss = 0

        for td in trial_dirs:
            trial_id = td.name  # already digits
            dist_path = td / f"{scen_id}_{trial_id}_distance.txt"

            dist_val = _read_distance_txt(dist_path) if dist_path.is_file() else None

            if dist_val is None:
                s_miss += 1
                grand_missing += 1
                if include_missing:
                    rows.append({
                        "row_type": "trial",
                        "scenario": scen_id,
                        "trial": trial_id,
                        "distance_m": "",
                        "distance_txt": str(dist_path) if dist_path.exists() else "",
                        "status": "missing_or_invalid",
                    })
                continue

            s_sum += dist_val
            s_count += 1
            grand_total += dist_val
            grand_count += 1

            rows.append({
                "row_type": "trial",
                "scenario": scen_id,
                "trial": trial_id,
                "distance_m": f"{dist_val:.4f}",
                "distance_txt": str(dist_path),
                "status": "ok",
            })

        scenario_sum[scen_id] = s_sum
        scenario_count[scen_id] = s_count
        scenario_missing[scen_id] = s_miss

        # scenario subtotal row
        rows.append({
            "row_type": "scenario_total",
            "scenario": scen_id,
            "trial": "",
            "distance_m": f"{s_sum:.4f}",
            "distance_txt": "",
            "status": f"trials_ok={s_count}, missing={s_miss}",
        })

    # grand total row
    rows.append({
        "row_type": "grand_total",
        "scenario": "ALL",
        "trial": "",
        "distance_m": f"{grand_total:.4f}",
        "distance_txt": "",
        "status": f"trials_ok={grand_count}, missing={grand_missing}",
    })

    # write csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_type", "scenario", "trial", "distance_m", "distance_txt", "status"]
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[DONE] saved: {out_csv}")
    return out_csv



if __name__ == "__main__":

    DATA_ROOT = Path("./Data").resolve()
    collect_distances_to_csv(DATA_ROOT, include_missing=False)
