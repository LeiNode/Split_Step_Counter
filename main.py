"""
Split Step Counter — entry point.
Run this file to launch the application: python main.py

Requires: pip install -r requirements.txt
The pose model (~9 MB) is downloaded automatically on first run into models/.
"""

import tkinter as tk

from split_step.detector import download_model
from split_step.app import App


def main():
    download_model()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
