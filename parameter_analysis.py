import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

def analyze_dfe_params(df, output_dir="figures"):
    """Analyzes and plots the impact of DFE hyperparameters."""
    if df.empty:
        print("DFE sweep data is empty. Skipping analysis.")
        return

    # Analyze Lw (Feed-forward filter length)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Lw', y='ber', marker='o', errorbar='sd')
    plt.yscale('log')
    plt.title('DFE Performance vs. Feed-Forward Filter Length (Lw)')
    plt.xlabel('Lw (taps)')
    plt.ylabel('Mean BER')
    plt.grid(True, which='both')
    plt.savefig(os.path.join(output_dir, 'dfe_analysis_Lw.png'), dpi=300)
    plt.close()
    print(f"Saved DFE Lw analysis to {os.path.join(output_dir, 'dfe_analysis_Lw.png')}")

def analyze_rbpf_params(df, output_dir="figures"):
    """Analyzes and plots the impact of RBPF-GP hyperparameters."""
    if df.empty:
        print("RBPF sweep data is empty. Skipping analysis.")
        return
        
    # It's difficult to plot against a dict 'mkws'. We need to parse it.
    try:
        df['ell'] = df['mkws'].apply(lambda x: eval(x).get('ell', np.nan))
    except:
        print("Could not parse 'ell' from 'mkws' column. Skipping length-scale analysis.")
        return

    # Analyze Np (Number of particles)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Np', y='ber', marker='o', errorbar='sd')
    plt.yscale('log')
    plt.title('RBPF-GP Performance vs. Number of Particles (Np)')
    plt.xlabel('Number of Particles (Np)')
    plt.ylabel('Mean BER')
    plt.grid(True, which='both')
    plt.savefig(os.path.join(output_dir, 'rbpf_analysis_Np.png'), dpi=300)
    plt.close()
    print(f"Saved RBPF Np analysis to {os.path.join(output_dir, 'rbpf_analysis_Np.png')}")

    # Analyze ell (GP Length-scale) - This is key for the tracking/noise trade-off
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='ell', y='ber', marker='o', errorbar='sd')
    plt.yscale('log')
    plt.title('RBPF-GP Performance vs. GP Length-Scale (ell)')
    plt.xlabel('Length-Scale ell (A larger value implies slower channel)')
    plt.ylabel('Mean BER')
    plt.grid(True, which='both')
    plt.savefig(os.path.join(output_dir, 'rbpf_analysis_ell.png'), dpi=300)
    plt.close()
    print(f"Saved RBPF length-scale analysis to {os.path.join(output_dir, 'rbpf_analysis_ell.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results.")
    parser.add_argument('--dfe_results', type=str, default='results/dfe_sweep.csv', help='CSV from sweep_dfe.py.')
    parser.add_argument('--rbpf_results', type=str, default='results/rbpf_gp_sweep.csv', help='CSV from sweep_rbpf_gp.py.')
    parser.add_argument('--output_dir', type=str, default='figures', help='Directory to save analysis plots.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load DFE data if available
    try:
        df_dfe = pd.read_csv(args.dfe_results)
        analyze_dfe_params(df_dfe, args.output_dir)
    except FileNotFoundError:
        print(f"Warning: DFE results file not found at {args.dfe_results}")
    
    # Load RBPF-GP data if available
    try:
        df_rbpf = pd.read_csv(args.rbpf_results)
        analyze_rbpf_params(df_rbpf, args.output_dir)
    except FileNotFoundError:
        print(f"Warning: RBPF-GP results file not found at {args.rbpf_results}")