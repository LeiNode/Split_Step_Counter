"""
App — tkinter GUI for the Split Step Counter.
Lets the user pick a video file or start a live webcam feed, displays the
annotated video, and shows the running split-step count.

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
from .live_view_analysis import LiveViewAnalysis
from .video_analysis import VideoAnalysis


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
        self._use_lstm: tk.BooleanVar         = tk.BooleanVar(value=True)

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

        tk.Checkbutton(ctrl, text="Use LSTM model",
                       variable=self._use_lstm,
                       bg=self.BG, fg=self.FG_DIM,
                       selectcolor=self.BG2,
                       activebackground=self.BG,
                       activeforeground=self.FG,
                       font=("Helvetica", 9)).pack(anchor=tk.W, pady=(0, 4))

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=16)

        self._btn_live = self._button(ctrl, "Start Live View", self._start_live,
                                      self.PURPLE, self.ACCENT)
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

    def _start_video(self):
        if self._video_path:
            self._launch(VideoAnalysis(self._video_path, self._queue, self._stop,
                                       use_lstm=self._use_lstm.get()))

    def _start_live(self):
        self._launch(LiveViewAnalysis(self._queue, self._stop,
                                      use_lstm=self._use_lstm.get()))

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
                    _, rgb, count, state = item
                    img = Image.fromarray(rgb)
                    self._photo = ImageTk.PhotoImage(img)
                    self._canvas.delete("all")
                    self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
                    self._count_var.set(str(count))
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
