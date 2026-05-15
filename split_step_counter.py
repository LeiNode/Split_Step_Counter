"""
Split Step Counter — entry point.
Run this file to launch the application.

Requires: pip install opencv-python mediapipe numpy Pillow
The pose model (~6 MB) is downloaded automatically on first run.
"""

import tkinter as tk

from detector import download_model
from app import App


def main():
    download_model()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
