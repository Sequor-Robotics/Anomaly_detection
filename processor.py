from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

from tools.data_parser import (
    infer_scenario_trial,
    get_single_mcap,
    find_trial_dirs_from_src,
    parse_one_mcap_to_parsed_dir,
    export_frames_to_json_named,
)
from tools.data_processor import (
    Read_parsed_data,
    resampling,
    load_pos_from_json,
    compute_total_dist,
    save_dist,
    save_trj_svg,
)


"""
processor.py:
  Data parsing & processing from raw ros bag (.mcap) data

Usage:
  Indicate dirs of raw data

  # 1) .mcap directly
  python processor.py --src /path/to/raw_Data.mcap --interp linear --extras

  # 2) folder including single .mcap file
  python processor.py --src /path/to/trial/Data --interp linear --extras

  # 3) parent folder (handle multiple folders)
  python processor.py --src /path/to/parent/folder --interp linear --extras

Outputs:
  _parsed/...
  {SCEN}_{TRIAL}.json
  {SCEN}_{TRIAL}_trajectory.svg   (if --extras)
  {SCEN}_{TRIAL}_distance.txt     (if --extras)
"""


def run_one_trial(trial_dir: Path, interp_method: str, extras: bool,
                  overwrite: bool, overwrite_parsed: bool, extras_only: bool):
    trial_dir = trial_dir.resolve()
    scenario_id, trial_id = infer_scenario_trial(trial_dir)

    json_path = (trial_dir / f"{scenario_id}_{trial_id}.json").resolve()

    # extras-only (w/o parsing & processing)
    if extras_only:
        if not json_path.is_file():
            raise FileNotFoundError(
                f"[extras-only] JSON not found: {json_path}\n"
                f"Generate it first (run without --extras-only)."
            )

        positions  = load_pos_from_json(json_path)
        total_dist = compute_total_dist(positions)

        dist_path = trial_dir / f"{scenario_id}_{trial_id}_distance.txt"
        traj_path = trial_dir / f"{scenario_id}_{trial_id}_trajectory.svg"

        if dist_path.exists() and not overwrite:
            print(f"[SKIP] distance exists: {dist_path}")
        else:
            save_dist(total_dist, trial_dir, scenario_id, trial_id)

        if traj_path.exists() and not overwrite:
            print(f"[SKIP] trajectory exists: {traj_path}")
        else:
            save_trj_svg(trial_dir, positions, total_dist, scenario_id, trial_id)

        print(f"[DONE][extras-only] {scenario_id}/{trial_id}  (dir={trial_dir})")
        return

    # raw data parsing & processing
    mcap_path = get_single_mcap(trial_dir)
    parsed_dir = trial_dir / "_parsed"

    parse_one_mcap_to_parsed_dir(mcap_path, parsed_dir, overwrite=overwrite_parsed)

    data_raw = Read_parsed_data(parsed_dir)
    frames   = resampling(data_raw, method=interp_method)

    export_frames_to_json_named(frames, trial_dir, scenario_id, trial_id, overwrite=overwrite)

    if not json_path.is_file():
        raise FileNotFoundError(f"[ERROR] Expected JSON not found: {json_path}")
    
    if extras:
        positions  = load_pos_from_json(json_path)
        total_dist = compute_total_dist(positions)

        save_dist(total_dist, trial_dir, scenario_id, trial_id)
        save_trj_svg(trial_dir, positions, total_dist, scenario_id, trial_id)

    print(f"[DONE] {scenario_id}/{trial_id}  (dir={trial_dir})")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",              type=str,  required=True,       help="Path to .mcap or directory containing it" )
    ap.add_argument("--interp",           type=str,  default="linear",    choices=["linear", "poly", "spline"]            )
    ap.add_argument("--extras",           action="store_true", help="Also export trajectory svg + distance txt"           )
    ap.add_argument("--extras-only",      action="store_true",
                help="Skip parsing/processing. Only (re)generate distance/trajectory from existing {SCEN}_{TRIAL}.json"   )
    ap.add_argument("--overwrite",        action="store_true", help="Overwrite final JSON/outputs if exist"               )
    ap.add_argument("--overwrite_parsed", action="store_true", help="Re-parse even if _parsed exists"                     )
    args = ap.parse_args()

    src = Path(args.src)
    trial_dirs = find_trial_dirs_from_src(src)
    if not trial_dirs:
        raise RuntimeError(f"No trial directories with single .mcap found under: {src.resolve()}")

    print(f"[INFO] Found {len(trial_dirs)} trial dir(s)")
    for td in trial_dirs:
        run_one_trial(
            td,
            interp_method=args.interp,
            extras=args.extras,
            overwrite=args.overwrite,
            overwrite_parsed=args.overwrite_parsed,
            extras_only=args.extras_only
        )


if __name__ == "__main__":
    main()
