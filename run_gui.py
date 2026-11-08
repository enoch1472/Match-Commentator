# -*- coding: utf-8 -*-
"""
Run this file directly in Spyder (press F5 or the green Play button)
to automatically launch the Streamlit GUI in your web browser.
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
    
    # Launches Streamlit using the current Python environment
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
