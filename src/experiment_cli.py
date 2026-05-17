#!/usr/bin/env python
"""
Experiment Tracking Manager for Chest X-ray Project
Provides easy access to the ml_flow_like tracker UI and run management.
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

TRACKER_PATH = Path("C:\\Users\\Public\\ml_flow_like")


def start_tracker():
    """Start the tracker backend and open the UI in browser."""
    if not TRACKER_PATH.exists():
        print(f"Error: ml_flow_like not found at {TRACKER_PATH}")
        return

    print("[*] Starting ml_flow_like tracker backend...")
    try:
        # Start uvicorn server
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.app:app", "--reload"],
            cwd=str(TRACKER_PATH),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("[+] Backend started")
        
        # Wait for server to start
        time.sleep(3)
        
        # Open browser to UI
        print("[*] Opening tracker UI in browser...")
        webbrowser.open("http://127.0.0.1:8000")
        print("[+] UI opened at http://127.0.0.1:8000")
        print("[*] Press Ctrl+C to stop the tracker")
        
        # Keep process alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[*] Stopping tracker...")
            
    except Exception as e:
        print(f"Error: {e}")


def list_runs():
    """List all recorded experiment runs."""
    import os
    sys.path.insert(0, str(TRACKER_PATH))
    
    try:
        from backend.repository import list_runs as repo_list_runs
        
        runs = repo_list_runs()
        if not runs:
            print("No runs recorded yet.")
            return
        
        print(f"\n{'Run ID':<15} {'Experiment':<35} {'Activation':<14} {'Slope':<8} {'Timestamp':<20} {'Tags':<30}")
        print("-" * 125)
        for run in runs:
            tags = ", ".join(run.get("tags", []))
            params = run.get("parameters", {})
            print(
                f"{run['run_id']:<15} "
                f"{run['experiment_name']:<35} "
                f"{params.get('activation', ''):<14} "
                f"{str(params.get('leaky_relu_slope', '')):<8} "
                f"{run['timestamp']:<20} "
                f"{tags:<30}"
            )
        print(f"\nTotal runs: {len(runs)}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Tracking CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("ui", help="Start tracker UI and open in browser")
    subparsers.add_parser("list", help="List all recorded runs")
    
    args = parser.parse_args()
    
    if args.command == "ui":
        start_tracker()
    elif args.command == "list":
        list_runs()
    else:
        parser.print_help()
