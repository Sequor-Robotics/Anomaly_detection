import json
from pathlib import Path

from PySide6.QtWidgets import QAbstractItemView

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSpinBox,
    QDoubleSpinBox, QListView, QTreeView
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


class LabelerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PRCD JSON Object Position Labeler")

        self.json_path: Path | None = None
        self.data = None
        self.frames = None
        self.idx = 0
        self.max_dist = MAX_DIST_DEFAULT

        # --- UI ---
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        # Plot area (matplotlib)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)

        main.addWidget(self.canvas, stretch=4)

        # Right panel
        side = QVBoxLayout()
        main.addLayout(side, stretch=1)

        self.info = QLabel("No file loaded")
        self.info.setWordWrap(True)
        side.addWidget(self.info)

        btn_open = QPushButton("Open *_prcd.json")
        btn_save = QPushButton("Save (overwrite)")
        btn_save_as = QPushButton("Save As...")
        side.addWidget(btn_open)
        side.addWidget(btn_save)
        side.addWidget(btn_save_as)

        # Frame controls
        ctrl = QHBoxLayout()
        self.spin = QSpinBox()
        self.spin.setMinimum(0)
        self.spin.setMaximum(0)
        self.spin.setKeyboardTracking(False)
        btn_prev = QPushButton("Prev")
        btn_next = QPushButton("Next")
        ctrl.addWidget(btn_prev)
        ctrl.addWidget(self.spin)
        ctrl.addWidget(btn_next)
        side.addLayout(ctrl)

        self.pos_label = QLabel("obj_pos: (none)")
        side.addWidget(self.pos_label)

        hint = QLabel(
            "Left click: set obj_pos\n"
            "Right click: clear obj_pos (delete key)\n"
            "Prev/Next: move frames\n"
            "Note: display axes are rotated: +x up, +y left"
        )
        hint.setStyleSheet("color: gray;")
        side.addWidget(hint)

        # --- Range cut controls ---
        cut_row = QHBoxLayout()
        self.cut_a = QSpinBox()
        self.cut_b = QSpinBox()
        for w in (self.cut_a, self.cut_b):
            w.setMinimum(0)
            w.setMaximum(0)
            w.setKeyboardTracking(False)

        cut_row.addWidget(QLabel("Cut a:"))
        cut_row.addWidget(self.cut_a)
        cut_row.addWidget(QLabel("b:"))
        cut_row.addWidget(self.cut_b)
        side.addLayout(cut_row)

        btn_cut = QPushButton("Keep only frames [a..b]")
        side.addWidget(btn_cut)

        # --- Default obj_pos controls (for save + batch) ---
        default_row = QHBoxLayout()
        self.def_x = QDoubleSpinBox()
        self.def_y = QDoubleSpinBox()
        for w in (self.def_x, self.def_y):
            w.setRange(-1000.0, 1000.0)
            w.setDecimals(3)
            w.setSingleStep(0.1)
        self.def_x.setValue(5.0)
        self.def_y.setValue(5.0)

        default_row.addWidget(QLabel("Default x:"))
        default_row.addWidget(self.def_x)
        default_row.addWidget(QLabel("y:"))
        default_row.addWidget(self.def_y)
        side.addLayout(default_row)

        btn_batch = QPushButton("Set obj_pos default in folders...")
        side.addWidget(btn_batch)

        btn_lrflip = QPushButton("Create LR-flip augmented files in folders...")
        side.addWidget(btn_lrflip)
        btn_lrflip.clicked.connect(self.batch_create_lrflip_augmented)

        side.addStretch(1)

        # --- Signals ---
        btn_open.clicked.connect(self.open_json)
        btn_save.clicked.connect(self.save_overwrite)
        btn_save_as.clicked.connect(self.save_as)

        btn_prev.clicked.connect(lambda: self.goto(self.idx - 1))
        btn_next.clicked.connect(lambda: self.goto(self.idx + 1))
        self.spin.valueChanged.connect(self.goto)

        self.canvas.mpl_connect("button_press_event", self.on_click)

        btn_cut.clicked.connect(self.delete_range)
        btn_batch.clicked.connect(self.batch_set_default_obj_pos)

    # -----------------------
    # Utilities
    # -----------------------
    def batch_create_lrflip_augmented(self):
        dirs = self.pick_directories("Select scenario folders (multi-select)")
        if not dirs:
            return

        suffix = "_aug_lrflip"

        # 각 폴더 단위로 augment_folder 호출
        ok_dirs, fail_dirs = 0, 0
        failed = []

        for d in dirs:
            try:
                augment_folder(
                    Path(d),
                    out_dir=None,          # 같은 폴더에 생성
                    suffix=suffix,
                    overwrite=False,       # 덮어쓰기 X
                    skip_existing=True,    # ✅ 이미 있으면 스킵
                )
                ok_dirs += 1
            except Exception as e:
                fail_dirs += 1
                failed.append(f"{d} :: {e}")

        msg = f"Done.\nSuccess folders: {ok_dirs}\nFailed folders: {fail_dirs}"
        if fail_dirs > 0:
            msg += "\n\nFailed (first 5):\n" + "\n".join(failed[:5])

        QMessageBox.information(self, "LR Flip result", msg)

    def current_default_xy(self):
        return (float(self.def_x.value()), float(self.def_y.value()))

    def fill_missing_obj_pos(self, frames, default_xy):
        """obj_pos가 없거나 None이면 default로 채움"""
        dx, dy = float(default_xy[0]), float(default_xy[1])
        for fr in frames:
            if ("obj_pos" not in fr) or (fr["obj_pos"] is None):
                fr["obj_pos"] = [dx, dy]
            else:
                # 이상한 형태 방어
                try:
                    if len(fr["obj_pos"]) != 2:
                        fr["obj_pos"] = [dx, dy]
                except Exception:
                    fr["obj_pos"] = [dx, dy]

    def set_obj_pos_all_frames(self, frames, default_xy):
        """모든 프레임에 obj_pos를 강제로 default로 세팅"""
        dx, dy = float(default_xy[0]), float(default_xy[1])
        for fr in frames:
            fr["obj_pos"] = [dx, dy]

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
            return
        m = max(0, len(self.frames) - 1)
        self.spin.setMaximum(m)
        self.cut_a.setMaximum(m)
        self.cut_b.setMaximum(m)

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
