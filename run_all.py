"""
Orchestrates batch execution of training and sweep experiments using default config.
Useful for automated runs and regeneration of results.
"""

import subprocess
import os


def run_script(script_name):
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run(["python", f"src/training/{script_name}"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error in {script_name}:", result.stderr)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)

    scripts = [
        "train_smc.py",
        "complexity_sweep.py",
        "pilot_vs_ber.py"
    ]

    for script in scripts:
        run_script(script)

    print("\n✅ All experiments completed.")
