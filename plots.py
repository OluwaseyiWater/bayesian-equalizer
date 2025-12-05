import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Helper function to safely calculate mean, ignoring non-numeric types
def robust_mean(series):
    return pd.to_numeric(series, errors='coerce').mean()

def plot_snr_curves(df, channel, output_dir="figures"):
    """Generates and saves BER vs. SNR plots for a given channel."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Map method names for the legend
    method_map = {
        'trivial': 'Trivial (No EQ)',
        'mmse_le': 'MMSE-LE',
        'mlsd': 'MLSD (Viterbi)',
        'mmse_dfe': 'MMSE-DFE',
        'mmse_dfe_ca_ldpc': 'MMSE-DFE-CA (LDPC)',
        'rbpf_ca': 'RBPF-GP-CA (Proposed)'
    }

    # Identify the correct BER column for each method
    ber_cols = {
        'trivial': 'ber',
        'mmse_le': 'ber',
        'mlsd': 'ber',
        'mmse_dfe': 'ber_dd', # Decision-directed BER
        'mmse_dfe_ca_ldpc': 'msg_ber', # Message BER
        'rbpf_ca': 'ber' # Message BER from RBPF sweeps
    }

    df_channel = df[df['channel'].str.upper() == channel.upper()]
    if df_channel.empty:
        print(f"No data found for channel '{channel}'. Skipping SNR curve plot.")
        return

    for method, display_name in method_map.items():
        if method not in df_channel['method'].unique():
            continue

        ber_col = ber_cols.get(method, 'ber') 
        method_df = df_channel[df_channel['method'] == method]
        
        # Group by SNR and calculate mean BER
        grouped = method_df.groupby('snr_db')[ber_col].apply(robust_mean).reset_index()
        grouped = grouped.dropna()

        if not grouped.empty:
            ax.semilogy(grouped['snr_db'], grouped[ber_col], 'o-', label=display_name)

    ax.set_xlabel("SNR (Es/N0) [dB]")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(f"Equalizer Performance on {channel.upper()} Channel")
    ax.legend()
    ax.set_ylim(bottom=1e-6)
    ax.grid(True, which='both', linestyle='--')
    
    filename = f"ber_vs_snr_{channel.lower()}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    print(f"Saved SNR curve plot to {os.path.join(output_dir, filename)}")
    plt.close()

def plot_method_comparison(df, channel, snr, output_dir="figures"):
    """Generates and saves a bar chart comparing methods at a specific SNR and channel."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    method_map = {
        'trivial': 'Trivial',
        'mmse_le': 'MMSE-LE',
        'mlsd': 'MLSD',
        'mmse_dfe': 'MMSE-DFE',
        'mmse_dfe_ca_ldpc': 'MMSE-DFE-CA',
        'rbpf_gp': 'RBPF-GP-CA'
    }
    ber_cols = {
        'trivial': 'ber', 'mmse_le': 'ber', 'mlsd': 'ber', 'mmse_dfe': 'ber_dd',
        'mmse_dfe_ca_ldpc': 'msg_ber', 'rbpf_ca': 'ber'
    }

    df_filtered = df[(df['channel'].str.upper() == channel.upper()) & (df['snr_db'] == snr)]
    if df_filtered.empty:
        print(f"No data for channel '{channel}' at SNR={snr}dB. Skipping bar chart.")
        return

    methods_present = sorted([m for m in method_map.keys() if m in df_filtered['method'].unique()])
    display_names = [method_map[m] for m in methods_present]
    mean_bers = []
    
    for method in methods_present:
        ber_col = ber_cols.get(method, 'ber') 
        ber_val = robust_mean(df_filtered[df_filtered['method'] == method][ber_col])
        mean_bers.append(ber_val)

    bars = ax.bar(display_names, mean_bers, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(display_names))))

    ax.set_yscale('log')
    ax.set_ylabel("Mean Bit Error Rate (BER)")
    ax.set_title(f"Method Comparison on {channel.upper()} Channel at {snr} dB")
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_ylim(bottom=1e-6)

    # Add BER values on top of bars
    for bar in bars:
        yval = bar.get_height()
        if yval > 0 and not np.isnan(yval):
            ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.2, f'{yval:.2e}', ha='center', va='bottom')

    plt.tight_layout()
    filename = f"method_comparison_{channel.lower()}_{snr}db.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    print(f"Saved method comparison plot to {os.path.join(output_dir, filename)}")
    plt.close()


def load_all_data(results_dir="results"):
    """Loads and concatenates all relevant CSV files from the results directory."""
    all_dfs = []

    # Load data from the general SNR sweep
    snr_sweep_file = os.path.join(results_dir, "snr_sweep_ldpc.csv")
    if os.path.exists(snr_sweep_file):
        df_snr = pd.read_csv(snr_sweep_file)
        if 'channel' not in df_snr.columns:
            df_snr['channel'] = 'TV_AR1' # Assume default if missing
        all_dfs.append(df_snr)
        print(f"Loaded {snr_sweep_file}")

    # Load data from the RBPF-GP sweeps
    for filename in os.listdir(results_dir):
        if "rbpf" in filename and filename.endswith("rbpf_gp_seed_sweep.csv"):
            filepath = os.path.join(results_dir, filename)
            try:
                df_rbpf = pd.read_csv(filepath, comment='#')
                if 'ch' in df_rbpf.columns:
                    df_rbpf = df_rbpf.rename(columns={'ch': 'channel'})
                df_rbpf['method'] = 'rbpf_ca'
                all_dfs.append(df_rbpf)
                print(f"Loaded {filepath}")
            except Exception as e:
                print(f"Could not read {filepath}: {e}")

    if not all_dfs:
        raise FileNotFoundError("No CSV result files found in the 'results' directory.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    essential_cols = ['method', 'snr_db', 'channel']
    full_df.dropna(subset=essential_cols, inplace=True)

    if 'channel' in full_df.columns:
        full_df['channel'] = full_df['channel'].str.upper()
    
    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from simulation sweeps.")
    parser.add_argument('--results_dir', type=str, default='results', help='Directory containing CSV result files.')
    parser.add_argument('--output_dir', type=str, default='figures', help='Directory to save plot images.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    df = load_all_data(args.results_dir)

    # ---- CHANGED: Filter lists to remove any remaining nan/non-string values ----
    channels = [ch for ch in df['channel'].unique() if isinstance(ch, str)]
    snrs = sorted([s for s in df['snr_db'].unique() if not np.isnan(s)])
    # --------------------------------------------------------------------------
    
    print("\n--- Generating Plots ---")
    print(f"Found Channels: {channels}")
    print(f"Found SNRs: {snrs}")

    for channel in channels:
        plot_snr_curves(df, channel, args.output_dir)

    for channel in channels:
        for snr in snrs:
            if snr == int(snr):
                plot_method_comparison(df, channel, snr, args.output_dir)