"""
App — tkinter GUI for the Split Step Counter.
Lets the user pick a video file or start a live webcam feed, displays the
annotated video, and shows the running split-step count.

Sound cues (feature "audio_cues"): a short high tone for each split step and a
lower one for each swing, so a player drilling alone in front of the webcam
gets feedback without looking at the screen.

Rep target (feature "rep_target"): in a live drill, set how many swings make a
set; the counter shows "12 / 20" and a longer tone marks the set complete.

Threading: analysis runs on a daemon worker thread (VideoAnalysis /
LiveViewAnalysis) that pushes ("frame" | "done" | "error", ...) tuples onto a
small queue; _poll() drains it on the Tk main loop every ~16 ms, since Tk
widgets may only be touched from the main thread.
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .detector import (CANVAS_W, CANVAS_H, GROUNDED, AIRBORNE, LEFT_LEG_UP,
                       RIGHT_LEG_UP, NOT_TRACKED)
from . import features
from .live_view_analysis import LiveViewAnalysis
from .video_analysis import VideoAnalysis

# kind -> (frequency Hz, duration ms) for winsound; elsewhere the Tk bell rings
CUE_TONES = {"split": (1000, 80), "swing": (600, 120), "set": (1300, 450)}


def new_events(prev: tuple[int, int | None], count: int, swings: int | None) -> list[str]:
    """Cue kinds for what happened since the previous frame message. Counts are
    cumulative, so frames dropped from the queue don't lose events (several
    events in one step still get one cue each kind: rapid beeps are noise)."""
    out = []
    if count > prev[0]:
        out.append("split")
    if swings is not None and prev[1] is not None and swings > prev[1]:
        out.append("swing")
    return out


def swing_display(swings: int, target: int) -> tuple[str, bool]:
    """Swing counter text and whether the set is complete (target 0 = no target)."""
    if target <= 0:
        return str(swings), False
    return f"{swings} / {target}", swings >= target


class App:
    BG     = "#1a1a2e"
    BG2    = "#16213e"
    ACCENT = "#0f3460"
    PURPLE = "#533483"
    CYAN   = "#00d4ff"
    GREEN  = "#00ff88"
    ORANGE = "#ff6b35"
    RED    = "#c62828"
    FG     = "#e0e0e0"
    FG_DIM = "#888899"

    # detector state -> (status text, colour)
    _STATUS = {
        AIRBORNE:     ("In hop",       ORANGE),
        LEFT_LEG_UP:  ("Left leg up",  CYAN),
        RIGHT_LEG_UP: ("Right leg up", PURPLE),
        GROUNDED:     ("Grounded",     GREEN),
        NOT_TRACKED:  ("Feet out of view", FG_DIM),
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Split Step Counter")
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG)

        self._queue: queue.Queue              = queue.Queue(maxsize=4)
        self._stop:  threading.Event          = threading.Event()
        self._worker: threading.Thread | None = None
        self._photo:  ImageTk.PhotoImage | None = None
        self._video_path: str | None          = None
        self._flags = features.all_features()
        # With only one model enabled, use it and hide the toggle
        self._use_lstm: tk.BooleanVar         = tk.BooleanVar(value=self._flags["lstm_model"])
        # Live drill choices: label -> drill name (None = split steps only)
        self._drills = {"Split steps only": None}
        for d in ("forehand", "backhand", "serve"):
            if self._flags[f"{d}_training"]:
                self._drills[f"{d.capitalize()} drill"] = d
        self._drill_var = tk.StringVar(value="Split steps only")
        self._hand_var  = tk.StringVar(value="Right-handed")
        self._sound_var = tk.BooleanVar(value=True)
        self._reps_var  = tk.StringVar(value="0")
        self._target, self._set_done = 0, False
        self._last_counts: tuple[int, int | None] = (0, None)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        ctrl = tk.Frame(self.root, bg=self.BG, padx=20, pady=20, width=260)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="Split Step\nCounter",
                 font=("Helvetica", 20, "bold"),
                 bg=self.BG, fg=self.CYAN, justify=tk.CENTER).pack(pady=(0, 24))

        # File section
        self._file_label = tk.Label(ctrl, text="No file selected",
                                    font=("Helvetica", 9), bg=self.BG, fg=self.FG_DIM,
                                    wraplength=220, justify=tk.LEFT)
        self._file_label.pack(anchor=tk.W, pady=(0, 6))

        self._button(ctrl, "Browse video…", self._browse, self.BG2, self.ACCENT,
                     fg=self.FG, pady=6, font=None).pack(fill=tk.X, pady=(0, 6))

        self._btn_process = self._button(ctrl, "Process Video", self._start_video,
                                         self.ACCENT, self.PURPLE, state=tk.DISABLED)
        self._btn_process.pack(fill=tk.X, pady=(0, 4))

        if self._flags["lstm_model"] and self._flags["rules_model"]:
            tk.Checkbutton(ctrl, text="Use LSTM model",
                           variable=self._use_lstm,
                           bg=self.BG, fg=self.FG_DIM,
                           selectcolor=self.BG2,
                           activebackground=self.BG,
                           activeforeground=self.FG,
                           font=("Helvetica", 9)).pack(anchor=tk.W, pady=(0, 4))

        if len(self._drills) > 1:
            row = tk.Frame(ctrl, bg=self.BG)
            row.pack(fill=tk.X, pady=(6, 0))
            ttk.Combobox(row, textvariable=self._drill_var, values=list(self._drills),
                         state="readonly", width=15).pack(side=tk.LEFT)
            ttk.Combobox(row, textvariable=self._hand_var, values=["Right-handed", "Left-handed"],
                         state="readonly", width=11).pack(side=tk.RIGHT)
            if self._flags["rep_target"]:
                reps = tk.Frame(ctrl, bg=self.BG)
                reps.pack(fill=tk.X, pady=(6, 0))
                tk.Label(reps, text="Swings per set (0 = no target)", font=("Helvetica", 9),
                         bg=self.BG, fg=self.FG_DIM).pack(side=tk.LEFT)
                tk.Spinbox(reps, from_=0, to=500, width=5, textvariable=self._reps_var).pack(side=tk.RIGHT)

        if self._flags["audio_cues"]:
            tk.Checkbutton(ctrl, text="Sound cues (split step / swing)",
                           variable=self._sound_var,
                           bg=self.BG, fg=self.FG_DIM,
                           selectcolor=self.BG2,
                           activebackground=self.BG,
                           activeforeground=self.FG,
                           font=("Helvetica", 9)).pack(anchor=tk.W, pady=(6, 0))

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=16)

        # Created even when switched off (other code toggles its state), just never shown
        self._btn_live = self._button(ctrl, "Start Live View", self._start_live,
                                      self.PURPLE, self.ACCENT)
        if self._flags["live_view"]:
            self._btn_live.pack(fill=tk.X, pady=(0, 4))

        self._btn_stop = self._button(ctrl, "Stop", self._request_stop,
                                      self.RED, "#ef5350", state=tk.DISABLED)
        self._btn_stop.pack(fill=tk.X)

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=20)

        tk.Label(ctrl, text="SPLIT STEPS", font=("Helvetica", 10),
                 bg=self.BG, fg=self.FG_DIM).pack()

        self._count_var = tk.StringVar(value="0")
        tk.Label(ctrl, textvariable=self._count_var,
                 font=("Helvetica", 60, "bold"),
                 bg=self.BG, fg=self.CYAN).pack()

        self._status_var = tk.StringVar(value="—")
        self._status_lbl = tk.Label(ctrl, textvariable=self._status_var,
                                    font=("Helvetica", 11),
                                    bg=self.BG, fg=self.FG_DIM)
        self._status_lbl.pack(pady=(2, 0))

        # Live drill swing counter (shown only while a drill runs)
        self._swing_title = tk.StringVar(value="")
        self._swing_var = tk.StringVar(value="")
        self._swing_box = tk.Frame(ctrl, bg=self.BG)
        tk.Label(self._swing_box, textvariable=self._swing_title, font=("Helvetica", 10),
                 bg=self.BG, fg=self.FG_DIM).pack(pady=(14, 0))
        tk.Label(self._swing_box, textvariable=self._swing_var, font=("Helvetica", 36, "bold"),
                 bg=self.BG, fg=self.ORANGE).pack()

        self._canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H,
                                 bg="#0d0d1a", highlightthickness=0)
        self._canvas.pack(side=tk.RIGHT)
        self._draw_placeholder()

    @staticmethod
    def _button(parent, text, command, bg, active_bg, fg="white", pady=8,
                font=("Helvetica", 11, "bold"), state=tk.NORMAL) -> tk.Button:
        """Flat, consistently styled button (caller packs it)."""
        kwargs = {"font": font} if font else {}
        return tk.Button(parent, text=text, command=command, state=state,
                         bg=bg, fg=fg, relief=tk.FLAT, padx=8, pady=pady,
                         cursor="hand2", activebackground=active_bg,
                         activeforeground="white", **kwargs)

    def _draw_placeholder(self):
        self._canvas.delete("all")
        self._canvas.create_text(
            CANVAS_W // 2, CANVAS_H // 2,
            text="No video source\n\nSelect a file or start live view.",
            fill="#333355", font=("Helvetica", 15), justify=tk.CENTER,
        )

    # ── Button actions ────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                ("All files",   "*.*"),
            ],
        )
        if path:
            self._video_path = path
            self._file_label.config(text=os.path.basename(path), fg=self.FG)
            self._btn_process.config(state=tk.NORMAL)

    def _drill_kwargs(self) -> dict:
        drill = self._drills.get(self._drill_var.get())
        hand = "left" if self._hand_var.get().startswith("Left") else "right"
        return {"drill": drill, "hand": hand}

    def _start_video(self):
        if self._video_path:
            self._launch(VideoAnalysis(self._video_path, self._queue, self._stop,
                                       use_lstm=self._use_lstm.get(), **self._drill_kwargs()))

    def _start_live(self):
        self._launch(LiveViewAnalysis(self._queue, self._stop,
                                      use_lstm=self._use_lstm.get(), **self._drill_kwargs()))

    def _request_stop(self):
        self._stop.set()
        self._btn_stop.config(state=tk.DISABLED)

    # ── Worker management ─────────────────────────────────────────────────────

    def _launch(self, worker):
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        self._count_var.set("0")
        drill = worker._drill
        self._last_counts = (0, 0 if drill else None)
        try:
            self._target = max(0, int(self._reps_var.get())) if self._flags["rep_target"] else 0
        except ValueError:
            self._target = 0
        self._set_done = False
        if drill:
            self._swing_title.set(f"{drill.upper()}S")
            self._swing_var.set(swing_display(0, self._target)[0])
            self._swing_box.pack()
        else:
            self._swing_box.pack_forget()
        self._set_status("warming up…", self.FG_DIM)
        self._btn_process.config(state=tk.DISABLED)
        self._btn_live.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)

        self._worker = threading.Thread(target=worker.run, daemon=True)
        self._worker.start()

    # ── UI update loop ────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]

                if kind == "frame":
                    _, rgb, count, state, swings = item
                    img = Image.fromarray(rgb)
                    self._photo = ImageTk.PhotoImage(img)
                    self._canvas.delete("all")
                    self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
                    self._count_var.set(str(count))
                    if self._flags["audio_cues"] and self._sound_var.get():
                        for kind in new_events(self._last_counts, count, swings):
                            self._cue(kind)
                    self._last_counts = (count, swings)
                    if swings is not None:
                        text, done = swing_display(swings, self._target)
                        self._swing_var.set(text)
                        if done and not self._set_done:
                            self._set_done = True
                            self._swing_title.set(self._swing_title.get() + " · SET COMPLETE ✓")
                            if self._flags["audio_cues"] and self._sound_var.get():
                                self._cue("set")
                    self._set_status(*self._STATUS.get(state, self._STATUS[GROUNDED]))

                elif kind == "done":
                    self._count_var.set(str(item[1]))
                    self._set_status("Complete", self.FG_DIM)
                    self._reset_buttons()

                elif kind == "error":
                    messagebox.showerror("Error", item[1])
                    self._draw_placeholder()
                    self._reset_buttons()

        except queue.Empty:
            pass

        self.root.after(16, self._poll)  # ~60 fps

    def _cue(self, kind: str):
        try:
            import winsound
        except ImportError:          # not Windows
            self.root.bell()
            return
        # Beep blocks for its duration; keep it off the UI thread
        threading.Thread(target=winsound.Beep, args=CUE_TONES[kind], daemon=True).start()

    def _set_status(self, text: str, color: str):
        self._status_var.set(text)
        self._status_lbl.config(fg=color)

    def _reset_buttons(self):
        self._btn_live.config(state=tk.NORMAL)
        if self._video_path:
            self._btn_process.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED)

    def _on_close(self):
        self._stop.set()
        self.root.destroy()
