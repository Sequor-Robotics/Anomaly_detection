import json
from pathlib import Path

from PySide6.QtWidgets import QAbstractItemView

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSpinBox,
    QDoubleSpinBox, QListView, QTreeView,
    QTabWidget, QComboBox, QLineEdit, QTextEdit, QCheckBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import sys
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from augment_lrflip import augment_folder



MAX_DIST_DEFAULT = 5.0  # data_processor.py의 MAX_DIST와 맞추기


def scan_to_xy(scan, max_dist=MAX_DIST_DEFAULT):
    """
    scan: list of [angle, range]
    range가 0~1이면 normalize로 간주하고 meters로 복원
    아니면 meters로 간주
    """
    scan = np.asarray(scan, dtype=float)
    if scan.size == 0:
        return np.array([]), np.array([])

    angles = scan[:, 0]
    ranges = scan[:, 1]

    # heuristic: normalized면 대부분 0~1
    try:
        if np.nanmax(ranges) <= 1.00001:
            dist = (1.0 - ranges) * max_dist
        else:
            dist = ranges
    except Exception:
        dist = ranges

    x = dist * np.cos(angles)
    y = dist * np.sin(angles)
    return x, y


# -----------------------
# Time-series compare utilities
# -----------------------
TS_KEYS = [
    "timestamp", "time_stamp", "ts", "t", "time",
    "scan_timestamp", "laser_ts", "scan_ts",
]

SERIES_KEYS = {
    # NOTE: 필요하면 여기 후보 키를 더 추가하세요.
    "lin_acc":  ["lin_acc", "imu_lin_acc", "acc", "accel", "linear_acceleration"],
    "ang_vel":  ["ang_vel", "imu_ang_vel", "gyro", "gyr", "angular_velocity"],
    "pos":      ["pos", "position", "odom_pos"],
    "lin_vel":  ["lin_vel", "vel", "velocity", "odom_linvel"],
}


def _get_frames_from_json_obj(obj, p: Path | None = None):
    """JSON root가 dict(frames) 또는 list 인 경우 모두 처리."""
    if isinstance(obj, dict):
        # 가장 흔한 케이스
        if "frames" in obj and isinstance(obj["frames"], list):
            return obj["frames"]
        # 다른 키 이름도 방어적으로 시도
        for k in ["data", "sequence", "items"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        # dict 안에 list-of-dict가 들어있는 경우
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        raise ValueError(f"Can't find frames list in dict JSON: {p}")

    if isinstance(obj, list):
        return obj

    raise ValueError(f"Unsupported JSON root type: {type(obj)} in {p}")


def _to_vec(v):
    """스칼라/리스트/딕셔너리 형태를 1D float 벡터로 변환."""
    if v is None:
        return None
    if isinstance(v, (int, float, np.number)):
        return np.array([float(v)], dtype=float)
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, dtype=float).reshape(-1)
    if isinstance(v, dict):
        if all(k in v for k in ("x", "y", "z")):
            return np.array([float(v["x"]), float(v["y"]), float(v["z"])], dtype=float)
        if all(k in v for k in ("w", "x", "y", "z")):
            return np.array([float(v["w"]), float(v["x"]), float(v["y"]), float(v["z"])], dtype=float)
        for kk in ["data", "value", "vec", "xyz"]:
            if kk in v:
                return _to_vec(v[kk])
    return None


def extract_timeseries(frames: list[dict]):
    """frames(list[dict])에서 timestamp + 주요 시계열(벡터)을 뽑아온다.

    반환:
      {
        "ts": (N,),
        "lin_acc": (N,D) or None, "lin_acc_path": str|None,
        ...
      }
    """
    # timestamp
    ts = []
    for fr in frames:
        found = None
        for k in TS_KEYS:
            if k in fr:
                found = fr[k]
                break
        ts.append(float(found) if found is not None else np.nan)
    ts = np.asarray(ts, dtype=float)

    out: dict[str, object] = {"ts": ts}

    # series (root에서 바로 찾고, 없으면 1-level nested dict에서도 찾아봄)
    for name, keys in SERIES_KEYS.items():
        vecs = []
        chosen = None

        for fr in frames:
            v = None

            # root
            for k in keys:
                if k in fr:
                    v = fr[k]
                    chosen = chosen or k
                    break

            # 1-level nested dict
            if v is None:
                for kk, vv in fr.items():
                    if isinstance(vv, dict):
                        for k in keys:
                            if k in vv:
                                v = vv[k]
                                chosen = chosen or f"{kk}.{k}"
                                break
                    if v is not None:
                        break

            vecs.append(_to_vec(v))

        maxdim = max((len(x) for x in vecs if x is not None), default=0)
        if maxdim == 0:
            out[name] = None
            out[name + "_path"] = None
            continue

        arr = np.full((len(vecs), maxdim), np.nan, dtype=float)
        for i, x in enumerate(vecs):
            if x is None:
                continue
            arr[i, : len(x)] = x

        out[name] = arr
        out[name + "_path"] = chosen

    return out


def _segments_from_mask(mask: np.ndarray):
    """boolean mask에서 True 구간 (start,end) list 반환 (end inclusive)."""
    if mask.size == 0:
        return []
    m = mask.astype(bool)
    # leading/trailing
    diff = np.diff(m.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0])
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        ends = ends + [len(m) - 1]
    return list(zip(starts, ends))


class CompareTab(QWidget):
    """두 개의 JSON(.json)에서 시계열을 추출해 시각적으로 비교하는 탭.

    - 파일 2개 선택
    - (lin_acc / ang_vel / pos / lin_vel) 중 선택
    - dim 선택
    - 두 시계열 + 차이(Δ) 플롯
    - |Δ|가 threshold를 넘는 구간들의 인덱스 지표 출력
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.path_a: Path | None = None
        self.path_b: Path | None = None
        self.data_a: dict | None = None
        self.data_b: dict | None = None

        root = QVBoxLayout(self)

        # --- file pickers ---
        file_box = QVBoxLayout()
        root.addLayout(file_box)

        self.line_a = QLineEdit()
        self.line_a.setReadOnly(True)
        self.line_b = QLineEdit()
        self.line_b.setReadOnly(True)

        row_a = QHBoxLayout()
        row_a.addWidget(QLabel("JSON A"))
        row_a.addWidget(self.line_a, stretch=1)
        btn_a = QPushButton("Browse...")
        row_a.addWidget(btn_a)
        file_box.addLayout(row_a)

        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("JSON B"))
        row_b.addWidget(self.line_b, stretch=1)
        btn_b = QPushButton("Browse...")
        row_b.addWidget(btn_b)
        file_box.addLayout(row_b)

        # --- options ---
        opt = QHBoxLayout()
        root.addLayout(opt)

        opt.addWidget(QLabel("Series"))
        self.series_cb = QComboBox()
        self.series_cb.addItems(list(SERIES_KEYS.keys()))
        opt.addWidget(self.series_cb)

        opt.addWidget(QLabel("dim"))
        self.dim_sb = QSpinBox()
        self.dim_sb.setMinimum(0)
        self.dim_sb.setMaximum(0)
        self.dim_sb.setKeyboardTracking(False)
        opt.addWidget(self.dim_sb)

        opt.addWidget(QLabel("|Δ| threshold"))
        self.thr_sb = QDoubleSpinBox()
        self.thr_sb.setDecimals(6)
        self.thr_sb.setRange(0.0, 1e9)
        self.thr_sb.setSingleStep(0.1)
        self.thr_sb.setValue(0.5)
        opt.addWidget(self.thr_sb)

        self.use_ts = QCheckBox("x=timestamp(if any)")
        self.use_ts.setChecked(False)  # trimming은 index 기반이어서 기본은 index
        opt.addWidget(self.use_ts)

        btn_plot = QPushButton("Plot + Analyze")
        opt.addWidget(btn_plot)

        opt.addStretch(1)

        # --- plots ---
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax1.grid(True)
        self.ax2.grid(True)
        root.addWidget(self.canvas, stretch=1)

        # --- text output ---
        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setPlaceholderText("Load two JSON files to start.")
        root.addWidget(self.out, stretch=0)

        # --- signals ---
        btn_a.clicked.connect(lambda: self._pick_and_load(which="a"))
        btn_b.clicked.connect(lambda: self._pick_and_load(which="b"))
        btn_plot.clicked.connect(self.plot_and_analyze)
        self.series_cb.currentIndexChanged.connect(self._refresh_dim_limit)

    # -----------------------
    # Loading
    # -----------------------
    def _pick_and_load(self, which: str):
        path, _ = QFileDialog.getOpenFileName(self, "Open JSON", "", "JSON (*.json)")
        if not path:
            return

        p = Path(path)
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            frames = _get_frames_from_json_obj(obj, p=p)
            data = extract_timeseries(frames)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {p.name}\n\n{e}")
            return

        if which == "a":
            self.path_a = p
            self.data_a = data
            self.line_a.setText(str(p))
        else:
            self.path_b = p
            self.data_b = data
            self.line_b.setText(str(p))

        self._render_detected_paths()
        self._refresh_dim_limit()

    def _render_detected_paths(self):
        lines = []
        if self.path_a:
            lines.append(f"[A] {self.path_a.name}")
            if self.data_a:
                for k in SERIES_KEYS.keys():
                    lines.append(f"  - {k}: {self.data_a.get(k + '_path')}")
        if self.path_b:
            lines.append(f"\n[B] {self.path_b.name}")
            if self.data_b:
                for k in SERIES_KEYS.keys():
                    lines.append(f"  - {k}: {self.data_b.get(k + '_path')}")
        self.out.setPlainText("\n".join(lines))

    def _refresh_dim_limit(self):
        key = self.series_cb.currentText()
        # data_a/b 중 하나라도 있으면 dim max를 잡는다.
        maxdim = 0
        for data in (self.data_a, self.data_b):
            if not data:
                continue
            arr = data.get(key)
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                maxdim = max(maxdim, arr.shape[1])
        maxdim = max(1, maxdim)
        self.dim_sb.setMaximum(maxdim - 1)

    # -----------------------
    # Plot + Analyze
    # -----------------------
    def plot_and_analyze(self):
        if not self.data_a or not self.data_b:
            QMessageBox.information(self, "Need files", "Please load JSON A and JSON B first.")
            return

        key = self.series_cb.currentText()
        dim = int(self.dim_sb.value())
        thr = float(self.thr_sb.value())

        A = self.data_a.get(key)
        B = self.data_b.get(key)

        if A is None or B is None:
            QMessageBox.information(self, "Missing series", f"Series '{key}' not found in one of the files.")
            return
        if not (isinstance(A, np.ndarray) and isinstance(B, np.ndarray)) or A.ndim != 2 or B.ndim != 2:
            QMessageBox.information(self, "Invalid series", f"Series '{key}' is not a 2D numeric array.")
            return
        if A.shape[1] <= dim or B.shape[1] <= dim:
            QMessageBox.information(
                self, "Dim out of range",
                f"dim={dim} is out of range.\nA shape={A.shape}, B shape={B.shape}"
            )
            return

        n = min(len(A), len(B))
        yA = A[:n, dim]
        yB = B[:n, dim]
        d = yB - yA

        # x-axis
        x = np.arange(n, dtype=float)
        if self.use_ts.isChecked():
            tsA = self.data_a.get("ts")
            if isinstance(tsA, np.ndarray) and np.isfinite(tsA[:n]).any():
                x = tsA[:n].astype(float)

        # clean plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.grid(True)
        self.ax2.grid(True)

        nameA = self.path_a.name if self.path_a else "A"
        nameB = self.path_b.name if self.path_b else "B"

        self.ax1.plot(x, yA, label=f"{key}[{dim}] {nameA}")
        self.ax1.plot(x, yB, label=f"{key}[{dim}] {nameB}")
        self.ax1.set_title(f"{key}[{dim}] : A vs B")
        self.ax1.set_xlabel("timestamp" if self.use_ts.isChecked() else "frame index")
        self.ax1.legend()

        self.ax2.plot(x, d, label=f"Δ = (B - A)  {key}[{dim}]")
        self.ax2.set_title(f"Difference: {key}[{dim}]")
        self.ax2.set_xlabel("timestamp" if self.use_ts.isChecked() else "frame index")
        self.ax2.legend()

        # segments where abs(d) > thr
        mask = np.isfinite(d) & (np.abs(d) > thr)
        segs = _segments_from_mask(mask)

        # shade segments
        if segs:
            # for shading, use x-index positions, not timestamps, so make segments on index
            # (x itself can be ts, but segments computed on index)
            for s, e in segs:
                # x might be timestamp; shading by x[s], x[e]
                self.ax1.axvspan(x[s], x[e], alpha=0.15)
                self.ax2.axvspan(x[s], x[e], alpha=0.15)

        self.canvas.draw_idle()

        # text report
        lines = []
        lines.append("=== Detected paths ===")
        lines.append(f"A: {nameA}")
        lines.append(f"  - {key}: {self.data_a.get(key + '_path')}")
        lines.append(f"B: {nameB}")
        lines.append(f"  - {key}: {self.data_b.get(key + '_path')}")
        lines.append("")
        lines.append(f"=== Analyze: {key}[{dim}]  (n={n}) ===")
        lines.append(f"threshold |Δ| > {thr}")
        if np.isfinite(d).any():
            lines.append(f"diff mean={np.nanmean(d):.6g}, std={np.nanstd(d):.6g}, maxabs={np.nanmax(np.abs(d)):.6g}")
        else:
            lines.append("diff stats: all NaN")

        if not segs:
            lines.append("\nNo segments exceed threshold.")
        else:
            lines.append(f"\nSegments exceed threshold: {len(segs)}")
            first_bad = segs[0][0]
            last_bad = segs[-1][1]
            lines.append(f"first_bad_index={first_bad}, last_bad_index={last_bad}")

            for i, (s, e) in enumerate(segs, start=1):
                seg_max = float(np.nanmax(np.abs(d[s:e+1]))) if np.isfinite(d[s:e+1]).any() else float("nan")
                lines.append(f"  {i:02d}) [{s}..{e}] len={e-s+1}, max|Δ|={seg_max:.6g}")

            # optional: suggest stable longest range (complement of bad)
            stable_mask = ~mask
            stable_segs = _segments_from_mask(stable_mask)
            if stable_segs:
                # pick longest stable segment
                stable_segs_sorted = sorted(stable_segs, key=lambda t: (t[1] - t[0]), reverse=True)
                ss, ee = stable_segs_sorted[0]
                lines.append(f"\nLongest stable segment (|Δ|<=thr): [{ss}..{ee}] len={ee-ss+1}")

        self.out.setPlainText("\n".join(lines))


class LabelerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiDAR Label Tool")

        self.json_path = None
        self.data = None
        self.frames = None
        self.idx = 0

        self.max_dist = MAX_DIST_DEFAULT

        # -----------------------
        # Tabs
        # -----------------------
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        labeler_tab = QWidget()
        tabs.addTab(labeler_tab, "LiDAR label (obj_pos)")
        self._init_labeler_tab(labeler_tab)

        compare_tab = CompareTab(self)
        tabs.addTab(compare_tab, "Time-series compare")

    # -----------------------
    # UI: labeler tab
    # -----------------------
    def _init_labeler_tab(self, parent: QWidget):
        main = QVBoxLayout(parent)

        # Top row: open/save + frame navigation
        top = QHBoxLayout()
        main.addLayout(top)

        btn_open = QPushButton("Open prcd.json")
        btn_save = QPushButton("Save (overwrite)")
        btn_saveas = QPushButton("Save As...")
        top.addWidget(btn_open)
        top.addWidget(btn_save)
        top.addWidget(btn_saveas)

        top.addWidget(QLabel("Frame"))
        self.spin = QSpinBox()
        self.spin.setMinimum(0)
        self.spin.setMaximum(0)
        self.spin.setKeyboardTracking(False)
        top.addWidget(self.spin)

        btn_prev = QPushButton("Prev")
        btn_next = QPushButton("Next")
        top.addWidget(btn_prev)
        top.addWidget(btn_next)

        top.addWidget(QLabel("MAX_DIST"))
        self.maxdist_spin = QDoubleSpinBox()
        self.maxdist_spin.setRange(0.1, 1000.0)
        self.maxdist_spin.setSingleStep(0.1)
        self.maxdist_spin.setDecimals(2)
        self.maxdist_spin.setValue(self.max_dist)
        self.maxdist_spin.setKeyboardTracking(False)
        top.addWidget(self.maxdist_spin)

        # Middle row: default obj_pos + batch button
        row2 = QHBoxLayout()
        main.addLayout(row2)

        row2.addWidget(QLabel("Default obj_pos (x,y)"))
        self.def_x = QDoubleSpinBox()
        self.def_y = QDoubleSpinBox()
        self.def_x.setRange(-1e6, 1e6)
        self.def_y.setRange(-1e6, 1e6)
        self.def_x.setDecimals(3)
        self.def_y.setDecimals(3)
        self.def_x.setValue(5.0)
        self.def_y.setValue(5.0)
        row2.addWidget(self.def_x)
        row2.addWidget(self.def_y)

        btn_batch = QPushButton("Batch: set default obj_pos to ALL frames in selected scenarios")
        row2.addWidget(btn_batch)

        # Row: delete frames by keeping range
        cut_row = QHBoxLayout()
        main.addLayout(cut_row)

        cut_row.addWidget(QLabel("Keep frames [a..b]"))
        self.cut_a = QSpinBox()
        self.cut_b = QSpinBox()
        self.cut_a.setMinimum(0)
        self.cut_b.setMinimum(0)
        self.cut_a.setMaximum(0)
        self.cut_b.setMaximum(0)
        self.cut_a.setKeyboardTracking(False)
        self.cut_b.setKeyboardTracking(False)
        cut_row.addWidget(self.cut_a)
        cut_row.addWidget(self.cut_b)
        btn_cut = QPushButton("Apply keep-range (delete outside)")
        cut_row.addWidget(btn_cut)

        # Row: drop first N frames
        drop_row = QHBoxLayout()
        main.addLayout(drop_row)

        drop_row.addWidget(QLabel("Drop first N frames"))
        self.drop_n = QSpinBox()
        self.drop_n.setMinimum(0)
        self.drop_n.setMaximum(0)  # refresh_limits()에서 갱신
        self.drop_n.setKeyboardTracking(False)
        drop_row.addWidget(self.drop_n)

        btn_drop = QPushButton("Apply drop-first")
        drop_row.addWidget(btn_drop)

        btn_drop_batch = QPushButton("Batch: drop-first N frames (selected scenarios)")
        drop_row.addWidget(btn_drop_batch)



        # Augment button
        aug_row = QHBoxLayout()
        main.addLayout(aug_row)
        btn_aug = QPushButton("Augment: left-right flip (folder)")
        aug_row.addWidget(btn_aug)
        aug_row.addStretch(1)

        # Info + plot
        self.info = QLabel("No file loaded.")
        main.addWidget(self.info)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)
        main.addWidget(self.canvas, stretch=1)

        self.pos_label = QLabel("obj_pos: (none)")
        main.addWidget(self.pos_label)

        # Signals
        btn_open.clicked.connect(self.open_json)
        btn_save.clicked.connect(self.save_overwrite)
        btn_saveas.clicked.connect(self.save_as)
        btn_prev.clicked.connect(lambda: self.goto(self.idx - 1))
        btn_next.clicked.connect(lambda: self.goto(self.idx + 1))
        self.spin.valueChanged.connect(self.goto)
        btn_drop.clicked.connect(self.drop_first_frames)
        btn_drop_batch.clicked.connect(self.batch_drop_first_frames)



        self.maxdist_spin.valueChanged.connect(self.on_maxdist_changed)

        btn_cut.clicked.connect(self.delete_range)
        btn_batch.clicked.connect(self.batch_set_default_obj_pos)
        btn_aug.clicked.connect(self.augment_folder_dialog)

        self.canvas.mpl_connect("button_press_event", self.on_click)

    # -----------------------
    # Augment
    # -----------------------
    def augment_folder_dialog(self):

        dirs = self.pick_directories("Select folders to augment (multi-select)")
        if not dirs:
            return

        reply = QMessageBox.question(
            self,
            "Augment (batch)",
            f"Selected {len(dirs)} folders.\nRun left-right flip augmentation for all of them?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        ok, fail = 0, 0
        failed = []

        for d in dirs:
            try:
                augment_folder(d)  # 기존 단일 폴더 증폭 함수 재사용
                ok += 1
            except Exception as e:
                fail += 1
                failed.append(f"{d} :: {e}")

        msg = f"Done.\nSuccess: {ok}\nFailed: {fail}"
        if fail > 0:
            msg += "\n\nFailed folders (first 5):\n" + "\n".join(failed[:5])

        QMessageBox.information(self, "Augment result", msg)


    # -----------------------
    # Helpers
    # -----------------------
    def on_maxdist_changed(self, v):
        self.max_dist = float(v)
        self.render()

    def current_default_xy(self):
        return float(self.def_x.value()), float(self.def_y.value())

    def set_obj_pos_all_frames(self, frames, default_xy):
        x, y = default_xy
        for fr in frames:
            fr["obj_pos"] = [x, y]

    def fill_missing_obj_pos(self, frames, default_xy):
        """obj_pos 키가 없는 프레임에만 default로 채운다."""
        x, y = default_xy
        for fr in frames:
            if "obj_pos" not in fr or fr["obj_pos"] is None:
                fr["obj_pos"] = [x, y]

    def make_backup_path(self, p: Path) -> Path:
        """foo.json -> foo.json.bak, 이미 있으면 foo.json.bak1, bak2..."""
        base = p.with_suffix(p.suffix + ".bak")
        if not base.exists():
            return base
        k = 1
        while True:
            cand = Path(str(base) + str(k))
            if not cand.exists():
                return cand
            k += 1

    def pick_directories(self, title="Select scenario folders (multi-select)"):
        dlg = QFileDialog(self, title)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)

        for view in dlg.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for view in dlg.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if dlg.exec():
            return [Path(p) for p in dlg.selectedFiles()]
        return []

    def get_frames_from_loaded_data(self, data):
        """data가 dict(frames) 또는 list 인지 처리"""
        if isinstance(data, dict) and "frames" in data:
            return data["frames"]
        if isinstance(data, list):
            return data
        return None

    def refresh_limits(self):
        """현재 frames 길이에 맞춰 spin / cut spin 최대값 갱신"""
        if self.frames is None:
            self.spin.setMaximum(0)
            self.cut_a.setMaximum(0)
            self.cut_b.setMaximum(0)
            if hasattr(self, "drop_n"):
                self.drop_n.setMaximum(0)
            return

        m = max(0, len(self.frames) - 1)
        self.spin.setMaximum(m)
        self.cut_a.setMaximum(m)
        self.cut_b.setMaximum(m)

        # drop first N: 0..len(frames) 허용 (전부 drop도 가능)
        if hasattr(self, "drop_n"):
            self.drop_n.setMaximum(len(self.frames))


    # -----------------------
    # Actions
    # -----------------------
    def open_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open prcd.json", "", "JSON (*.json)")
        if not path:
            return

        self.json_path = Path(path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        frames = self.get_frames_from_loaded_data(self.data)
        if frames is None:
            QMessageBox.critical(self, "Error", "Cannot find 'frames' in JSON.")
            self.data = None
            self.frames = None
            return

        self.frames = frames
        self.idx = 0

        self.refresh_limits()
        self.spin.blockSignals(True)
        self.spin.setValue(0)
        self.spin.blockSignals(False)

        self.info.setText(f"Loaded: {self.json_path.name}\nFrames: {len(self.frames)}")
        self.render()

    def save_overwrite(self):
        if not self.json_path or self.data is None:
            return

        # backup (원본 보존)
        backup = self.make_backup_path(self.json_path)
        try:
            backup.write_text(self.json_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

        # 저장 시: obj_pos 없는 프레임은 default로 채우기
        default_xy = self.current_default_xy()
        self.fill_missing_obj_pos(self.frames, default_xy)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

        QMessageBox.information(self, "Saved", f"Saved to {self.json_path.name}\nBackup: {backup.name}")

    def save_as(self):
        if self.data is None:
            return

        out, _ = QFileDialog.getSaveFileName(self, "Save As", "", "JSON (*.json)")
        if not out:
            return
        outp = Path(out)

        default_xy = self.current_default_xy()
        self.fill_missing_obj_pos(self.frames, default_xy)

        with open(outp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

        QMessageBox.information(self, "Saved", f"Saved to {outp.name}")

    def goto(self, new_idx):
        if self.frames is None:
            return
        new_idx = int(np.clip(int(new_idx), 0, len(self.frames) - 1))
        if new_idx == self.idx:
            self.render()
            return
        self.idx = new_idx
        self.render()

    def delete_range(self):
        if self.frames is None or len(self.frames) == 0:
            return

        a = int(self.cut_a.value())
        b = int(self.cut_b.value())
        if a > b:
            a, b = b, a

        n = len(self.frames)
        a = max(0, min(a, n - 1))
        b = max(0, min(b, n - 1))

        reply = QMessageBox.question(
            self,
            "Confirm keep range",
            f"Keep ONLY frames [{a}..{b}] (inclusive)?\n"
            f"Will keep {b - a + 1} frames.\n"
            f"Total: {n} -> {b - a + 1}",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        kept = self.frames[a:b + 1]
        self.frames[:] = kept
        self.idx = 0

        if len(self.frames) == 0:
            self.idx = 0
        else:
            self.idx = min(self.idx, len(self.frames) - 1)

        self.refresh_limits()
        self.spin.blockSignals(True)
        self.spin.setValue(self.idx)
        self.spin.blockSignals(False)

        self.cut_a.setValue(0)
        self.cut_b.setValue(max(0, len(self.frames) - 1))

        if self.json_path:
            self.info.setText(f"Loaded: {self.json_path.name}\nFrames: {len(self.frames)}")

        self.render()

    def drop_first_frames(self):
        if self.frames is None or len(self.frames) == 0:
            return

        n_drop = int(self.drop_n.value())
        if n_drop <= 0:
            QMessageBox.information(self, "Drop first", "n=0 입니다. 잘라낼 프레임 수를 올려주세요.")
            return

        total = len(self.frames)
        n_drop = min(n_drop, total)

        reply = QMessageBox.question(
            self,
            "Confirm drop-first",
            f"Drop first {n_drop} frames?\n"
            f"Total: {total} -> {total - n_drop}",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # 핵심: in-place로 앞부분 제거 (self.data와의 레퍼런스 유지)
        self.frames[:] = self.frames[n_drop:]

        # 인덱스/스핀/컷 범위 초기화
        self.idx = 0
        self.refresh_limits()

        self.spin.blockSignals(True)
        self.spin.setValue(0)
        self.spin.blockSignals(False)

        self.cut_a.setValue(0)
        self.cut_b.setValue(max(0, len(self.frames) - 1))
        self.drop_n.setValue(0)

        if self.json_path:
            self.info.setText(f"Loaded: {self.json_path.name}\nFrames: {len(self.frames)}")

        self.render()

    def batch_set_default_obj_pos(self):
        dirs = self.pick_directories("Select scenario folders (multi-select)")
        if not dirs:
            return

        default_xy = self.current_default_xy()

        json_files = []
        for d in dirs:
            if d.is_dir():
                json_files.extend(sorted(d.rglob("*_prcd.json")))

        if not json_files:
            QMessageBox.information(self, "Batch", "No *_prcd.json files found under selected folders.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm batch overwrite",
            f"Found {len(json_files)} files.\n"
            f"Set ALL frames obj_pos = [{default_xy[0]:.3f}, {default_xy[1]:.3f}] and overwrite?\n"
            f"(A backup .bak will be created per file.)",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        ok, fail = 0, 0
        failed = []

        for jp in json_files:
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                frames = self.get_frames_from_loaded_data(data)
                if frames is None:
                    raise ValueError("Cannot find frames")

                # 핵심: “모든 프레임” 강제 세팅
                self.set_obj_pos_all_frames(frames, default_xy)

                # backup + overwrite
                backup = self.make_backup_path(jp)
                try:
                    backup.write_text(jp.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception:
                    pass

                with open(jp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                ok += 1
            except Exception as e:
                fail += 1
                failed.append(f"{jp} :: {e}")

        msg = f"Done.\nSuccess: {ok}\nFailed: {fail}"
        if fail > 0:
            msg += "\n\nFailed files (first 5):\n" + "\n".join(failed[:5])

        QMessageBox.information(self, "Batch result", msg)

    def batch_drop_first_frames(self):
        dirs = self.pick_directories("Select scenario folders to drop-first (multi-select)")
        if not dirs:
            return

        n_drop = int(self.drop_n.value()) if hasattr(self, "drop_n") else 0
        if n_drop <= 0:
            QMessageBox.information(self, "Batch drop-first", "n=0 입니다. Drop first N 값을 올려주세요.")
            return

        # 선택한 폴더들 아래의 *_prcd.json 모두 수집
        json_files = []
        for d in dirs:
            if d.is_dir():
                json_files.extend(sorted(d.rglob("*_prcd.json")))

        if not json_files:
            QMessageBox.information(self, "Batch drop-first", "No *_prcd.json files found under selected folders.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm batch drop-first",
            f"Found {len(json_files)} files.\n"
            f"Drop first {n_drop} frames from EACH file and overwrite?\n"
            f"(A backup .bak will be created per file.)",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        ok, fail, skipped = 0, 0, 0
        failed = []

        for jp in json_files:
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                frames = self.get_frames_from_loaded_data(data)
                if frames is None:
                    raise ValueError("Cannot find frames")

                # n_drop이 파일 길이 이상이면 '빈 프레임'이 되므로,
                # 안전하게 스킵하거나(권장) 그냥 비우거나 선택해야 함.
                # 여기서는 스킵(권장)으로 처리.
                if len(frames) <= n_drop:
                    skipped += 1
                    continue

                # 핵심: in-place로 앞부분 제거 (data["frames"] 레퍼런스 유지)
                frames[:] = frames[n_drop:]

                # backup + overwrite
                backup = self.make_backup_path(jp)
                try:
                    backup.write_text(jp.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception:
                    pass

                with open(jp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                ok += 1

            except Exception as e:
                fail += 1
                failed.append(f"{jp} :: {e}")

        msg = f"Done.\nSuccess: {ok}\nSkipped (len<=N): {skipped}\nFailed: {fail}"
        if fail > 0:
            msg += "\n\nFailed files (first 5):\n" + "\n".join(failed[:5])

        QMessageBox.information(self, "Batch drop-first result", msg)


    # -----------------------
    # Plot + click labeling
    # -----------------------
    def on_click(self, event):
        if self.frames is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        frame = self.frames[self.idx]

        # right click = clear (remove key)
        if event.button == 3:
            frame.pop("obj_pos", None)
            self.render()
            return

        # left click = set
        if event.button == 1:
            # display: (x_disp, y_disp) = (y_real, x_real)
            x_real = float(event.ydata)
            y_real = float(event.xdata)
            frame["obj_pos"] = [x_real, y_real]
            self.render()

    def render(self):
        if self.frames is None or len(self.frames) == 0:
            self.ax.clear()
            self.canvas.draw_idle()
            return

        frame = self.frames[self.idx]
        scan = frame.get("scan", None)

        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)
        self.ax.set_title(f"Frame {self.idx}/{len(self.frames) - 1}")

        # draw scan points (real -> display rotated)
        if scan is not None and len(scan) > 0:
            x, y = scan_to_xy(scan, max_dist=self.max_dist)  # real (x,y)
            self.ax.scatter(y, x, s=6)  # display (x_disp, y_disp) = (y, x)

        # origin
        self.ax.scatter([0.0], [0.0], s=30, marker="x")

        # draw obj_pos if exists (real -> display rotated)
        if "obj_pos" in frame and frame["obj_pos"] is not None:
            try:
                ox, oy = frame["obj_pos"]  # real (x,y)
                self.ax.scatter([oy], [ox], s=60, marker="o")  # display (y,x)
                self.pos_label.setText(f"obj_pos: (x={ox:.3f}, y={oy:.3f})")
            except Exception:
                self.pos_label.setText("obj_pos: (invalid)")
        else:
            self.pos_label.setText("obj_pos: (none)")

        # axis labels for rotated display
        self.ax.set_xlabel("y (m)")  # horizontal axis now represents y
        self.ax.set_ylabel("x (m)")  # vertical axis now represents x

        lim = self.max_dist * 1.05
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)

        # make +y appear to the left
        self.ax.invert_xaxis()

        self.canvas.draw_idle()


def main():
    app = QApplication([])
    w = LabelerWindow()
    w.resize(1100, 700)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
