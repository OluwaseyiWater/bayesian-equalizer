import subprocess
import os
import sys
import gpytorch


def run_script(script_path):
    print(f"\n=== Running {script_path} ===")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)



if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    sys.path.insert(0, os.path.abspath("."))
    scripts = [
    "src/training/train_smc.py",
    "experiments/complexity_sweep.py",
    "experiments/pilot_vs_ber.py"
]

    for script in scripts:
        run_script(script)

    print("\n✅ All experiments completed.")

