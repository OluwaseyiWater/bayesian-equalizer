# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse

# def robust_mean(series):
#     """Helper function to safely calculate mean, ignoring non-numeric types."""
#     return pd.to_numeric(series, errors='coerce').mean()

# def plot_snr_curves(df, channel_name, output_dir="figures"):
#     """Generates and saves BER vs. SNR plots for a specific channel."""
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=(10, 7))

#     method_map = {
#         'trivial': 'Trivial (No EQ)',
#         'mmse_le': 'MMSE-LE',
#         'mlsd': 'MLSD (Viterbi)',
#         'mmse_dfe': 'MMSE-DFE',
#         'mmse_dfe_ca_ldpc': 'MMSE-DFE-CA (LDPC)',
#         'rbpf_ca': 'RBPF-GP-CA (Proposed)'
#     }

#     ber_cols = {
#         'trivial': 'ber', 'mmse_le': 'ber', 'mlsd': 'ber', 'mmse_dfe': 'ber_dd',
#         'mmse_dfe_ca_ldpc': 'msg_ber', 'rbpf_ca': 'ber'
#     }

#     df_channel = df[df['channel'] == channel_name]
#     if df_channel.empty:
#         print(f"No data found for channel '{channel_name}'. Skipping plot.")
#         return

#     # --- NEW: Define a small value to replace zeros for log plotting ---
#     ZERO_BER_FLOOR = 1e-7

#     for method_key, display_name in method_map.items():
#         if method_key not in df_channel['method'].unique():
#             continue

#         ber_col = ber_cols.get(method_key, 'ber')
#         method_df = df_channel[df_channel['method'] == method_key]
        
#         grouped = method_df.groupby('snr_db')[ber_col].apply(robust_mean).reset_index()
#         grouped = grouped.dropna(subset=[ber_col])

#         # --- MODIFIED: Condition and zero-handling logic ---
#         if not grouped.empty:
#             # Replace any 0.0 BER values with our small floor value for plotting
#             grouped[ber_col] = grouped[ber_col].replace(0.0, ZERO_BER_FLOOR)
#             ax.semilogy(grouped['snr_db'], grouped[ber_col], 'o-', label=display_name)

#     ax.set_xlabel("SNR (Es/N0) [dB]")
#     ax.set_ylabel("Bit Error Rate (BER)")
#     ax.set_title(f"Equalizer Performance on {channel_name.upper()} Channel")
#     ax.legend(fontsize=12)
#     # Adjust y-axis to accommodate the new floor
#     ax.set_ylim(bottom=ZERO_BER_FLOOR / 2, top=1.0)
#     ax.grid(True, which='both', linestyle='--')
    
#     filename = f"ber_vs_snr_{channel_name.lower()}.png"
#     plt.savefig(os.path.join(output_dir, filename), dpi=300)
#     print(f"Saved SNR curve plot to {os.path.join(output_dir, filename)}")
#     plt.close()


# def load_and_prepare_data(results_dir="results"):
#     """
#     Loads the two specific CSV files, assigns the correct channel name to each,
#     and combines them into a single, clean dataframe.
#     """
#     file_channel_map = {
#         'snr_sweep.csv': 'PR1D',
#         'snr_sweep_ldpc.csv': 'TV_AR1'
#     }
    
#     all_dfs = []
    
#     for filename, channel_name in file_channel_map.items():
#         filepath = os.path.join(results_dir, filename)
#         if os.path.exists(filepath):
#             try:
#                 df = pd.read_csv(filepath)
#                 # ** This is the key step: Assign the channel name **
#                 df['channel'] = channel_name
#                 all_dfs.append(df)
#                 print(f"Successfully loaded '{filepath}' and assigned to channel '{channel_name}'")
#             except Exception as e:
#                 print(f"Error loading {filepath}: {e}")
#         else:
#             print(f"Warning: Data file not found at '{filepath}'. Skipping.")

#     if not all_dfs:
#         raise FileNotFoundError("No valid data files were found. Please check the 'results' directory.")

#     # Combine all loaded data into one big dataframe
#     full_df = pd.concat(all_dfs, ignore_index=True)

#     # --- Standardize method names for consistent plotting ---
#     # This ensures that different names from different files are treated as the same method.
#     method_name_map = {
#         'rbpf_gp': 'rbpf_ca',           # Map 'rbpf_gp' to the key used in our dictionaries
#         'mmse_dfe_ca': 'mmse_dfe_ca_ldpc' # Unify coded DFE names
#     }
#     full_df['method'] = full_df['method'].replace(method_name_map)
    
#     # Clean up any rows with missing essential data
#     full_df.dropna(subset=['method', 'snr_db', 'channel'], inplace=True)
    
#     return full_df


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Plot results from simulation sweeps.")
#     parser.add_argument('--results_dir', type=str, default='results', help='Directory containing CSV result files.')
#     parser.add_argument('--output_dir', type=str, default='figures', help='Directory to save plot images.')
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Load and process the data from the specific CSV files
#     df = load_and_prepare_data(args.results_dir)
    
#     # Get the unique channel names found in the data
#     channels = df['channel'].unique()
    
#     print("\n--- Generating Plots ---")
#     print(f"Found data for channels: {list(channels)}")

#     # Generate a separate plot for each channel
#     for channel in channels:
#         plot_snr_curves(df, channel, args.output_dir)


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def robust_mean(series):
    """Helper function to safely calculate mean, ignoring non-numeric types."""
    # Convert to numeric, coercing errors will turn non-numbers into NaN
    numeric_series = pd.to_numeric(series, errors='coerce')
    # The mean function will automatically skip NaN values
    return numeric_series.mean()

def plot_method_comparison(df, channel, snr, output_dir="figures"):
    """
    Generates and saves a bar chart comparing methods at a specific SNR and channel,
    matching the user-provided example image.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Method names for the plot legend ---
    # The keys here are the standardized names we create in load_and_prepare_data
    method_map = {
        'trivial': 'Trivial',
        'mmse_le': 'MMSE-LE',
        'mlsd': 'MLSD',
        'mmse_dfe': 'MMSE-DFE',
        'mmse_dfe_ca_ldpc': 'MMSE-DFE-CA',
        'rbpf_ca': 'RBPF-GP (Proposed)' # Standardized name
    }
    
    # --- Which column contains the relevant BER for each method ---
    ber_cols = {
        'trivial': 'ber',
        'mmse_le': 'ber',
        'mlsd': 'ber',
        'mmse_dfe': 'ber_dd',
        'mmse_dfe_ca_ldpc': 'msg_ber',
        'rbpf_ca': 'ber'
    }

    # Filter data for the specific channel and SNR
    df_filtered = df[(df['channel'] == channel) & (df['snr_db'] == snr)]
    if df_filtered.empty:
        print(f"No data for channel '{channel}' at SNR={snr}dB. Skipping bar chart.")
        return

    # Get the methods that are actually present in the data for this slice
    methods_present = sorted([m for m in method_map.keys() if m in df_filtered['method'].unique()])
    display_names = [method_map[m] for m in methods_present]
    mean_bers = []
    
    # Calculate the mean BER for each method
    for method in methods_present:
        ber_col = ber_cols.get(method, 'ber')
        ber_val = robust_mean(df_filtered[df_filtered['method'] == method][ber_col])
        mean_bers.append(ber_val)

    # Plot the bars
    bars = ax.bar(display_names, mean_bers, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(display_names))))

    # --- Formatting to match the example ---
    ax.set_yscale('log')
    ax.set_ylabel("Mean Bit Error Rate (BER)", fontsize=14)
    ax.set_title(f"Method Comparison on {channel.upper()} Channel at {snr:.1f} dB", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_ylim(bottom=1e-6) # Set a floor for the y-axis

    # Add BER values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        if yval > 0 and not np.isnan(yval):
            ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.1, f'{yval:.2e}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    
    # Save the figure with a descriptive name
    filename = f"method_comparison_{channel.lower()}_{int(snr)}db.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    print(f"Saved bar chart to {os.path.join(output_dir, filename)}")
    plt.close()


def load_and_prepare_data(results_dir="results"):
    """
    Loads the two specific original CSV files, assigns the correct channel name to each,
    and combines them into a single, clean dataframe.
    """
    file_channel_map = {
        'snr_sweep.csv': 'PR1D',
        'snr_sweep_ldpc.csv': 'TV_AR1'
    }
    
    all_dfs = []
    
    for filename, channel_name in file_channel_map.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['channel'] = channel_name
                all_dfs.append(df)
                print(f"Successfully loaded '{filepath}' and assigned to channel '{channel_name}'")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        else:
            print(f"Warning: Data file not found at '{filepath}'. Skipping.")

    if not all_dfs:
        raise FileNotFoundError("No valid data files were found. Please check the 'results' directory.")

    full_df = pd.concat(all_dfs, ignore_index=True)

    # --- Standardize method names for consistent plotting ---
    method_name_map = {
        'rbpf_gp': 'rbpf_ca',           # Standardize to 'rbpf_ca'
        'mmse_dfe_ca': 'mmse_dfe_ca_ldpc' # Standardize to the longer name
    }
    full_df['method'] = full_df['method'].replace(method_name_map)
    
    full_df.dropna(subset=['method', 'snr_db', 'channel'], inplace=True)
    
    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from simulation sweeps.")
    parser.add_argument('--results_dir', type=str, default='results', help='Directory containing CSV result files.')
    parser.add_argument('--output_dir', type=str, default='figures', help='Directory to save plot images.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    df = load_and_prepare_data(args.results_dir)

    channels = sorted(df['channel'].unique())
    snrs = sorted(df['snr_db'].unique())
    
    print("\n--- Generating Bar Chart Comparisons ---")
    print(f"Found data for channels: {channels}")
    print(f"Found data for SNRs: {snrs}")

    # --- Main Loop to Generate All 18 Plots ---
    for channel in channels:
        for snr in snrs:
            # We removed the 'if snr == int(snr)' to plot for all values
            plot_method_comparison(df, channel, snr, args.output_dir)