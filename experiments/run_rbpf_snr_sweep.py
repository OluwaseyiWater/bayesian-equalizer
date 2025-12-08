import os
import sys
import argparse
import csv
import time
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from sweep_seeds_gp import run_once

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run an SNR sweep for the RBPF-GP equalizer.")
    parser.add_argument("--config", required=True, help="YAML config file to use as a template.")
    parser.add_argument("--out", default="results/rbpf_tv1_gp_snr_sweep.csv", help="Output CSV file for the sweep.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds to run per SNR point.")
    parser.add_argument("--procs", type=int, default=os.cpu_count(), help="Number of parallel processes.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # --- Central Configuration ---
    SNR_SWEEP_DB = list(range(4, 13))
    SEEDS_PER_SNR = args.seeds
    
   
    base_cfg = load_yaml(args.config)
    
    
    code_seed = base_cfg.get('code_seed', 0)

    # --- Build the list of jobs ---
    jobs = []
    for snr in SNR_SWEEP_DB:
        for seed in range(SEEDS_PER_SNR):
            job_cfg = base_cfg.copy()
            job_cfg['snr_db'] = snr
            jobs.append((job_cfg, seed, code_seed))

    print(f"Starting SNR sweep for RBPF-GP across {len(SNR_SWEEP_DB)} SNR points.")
    print(f"Total jobs to run: {len(jobs)}")

    # --- Execute jobs in parallel ---
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.procs) as executor:
        future_to_job = {executor.submit(run_once, cfg, s, cs): (cfg, s) for cfg, s, cs in jobs}
        
        for i, future in enumerate(as_completed(future_to_job), 1):
            cfg, seed = future_to_job[future]
            try:
                result = future.result()
                rows.append(result)
                print(f"[{i}/{len(jobs)}] Completed: SNR={cfg['snr_db']}dB, Seed={seed}, BER={result['ber']:.5f}")
            except Exception as exc:
                print(f"Job for SNR={cfg['snr_db']}dB, Seed={seed} generated an exception: {exc}")

    # --- Save results to a single CSV ---
    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSuccessfully saved RBPF-GP SNR sweep results to {args.out}")
    
    print(f"Total time elapsed: {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    main()