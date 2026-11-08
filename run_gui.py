# -*- coding: utf-8 -*-
"""
Premier League Match Commentator - Spyder / Terminal GUI Launcher
Run this script to start the Streamlit web interface directly.
"""

import sys
import subprocess
import os

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(base_dir, "app.py")

    print("=" * 60)
    print(" Launching Premier League Match Commentator GUI...")
    print(" Opening browser at: http://localhost:8501")
    print(" To stop the app, press the red Stop button in Spyder.")
    print("=" * 60)

    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
