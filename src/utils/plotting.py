"""
Utilities for visualizing performance metrics such as BER and channel MSE
against pilot fraction, particle count, or delay-Doppler complexity.
Also includes heatmaps for delay-Doppler gain patterns.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def plot_metric_vs_parameter(x_vals: np.ndarray, y_vals: np.ndarray, metric_name: str, param_name: str, log_x: bool = False, save_path: str = None):
    plt.figure(figsize=(7, 4))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', linewidth=2)
    plt.xlabel(param_name)
    plt.ylabel(metric_name)
    plt.grid(True)
    if log_x:
        plt.xscale('log')
    plt.title(f"{metric_name} vs {param_name}")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_multiple_metrics(x_vals: np.ndarray, metric_dict: dict, param_name: str, log_x: bool = False, save_path: str = None):
    plt.figure(figsize=(7, 4))
    for label, y_vals in metric_dict.items():
        plt.plot(x_vals, y_vals, marker='o', label=label)

    plt.xlabel(param_name)
    plt.ylabel("Metric Value")
    plt.title(f"Performance Metrics vs {param_name}")
    plt.grid(True)
    plt.legend()
    if log_x:
        plt.xscale('log')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_delay_doppler_surface(taps_over_time: List[List[Tuple[float, float]]], time_points: np.ndarray, delay_bins: int = 20, max_delay: float = 10.0, title: str = "Delay-Doppler Channel Magnitude", save_path: str = None):
    """
    Plot a delay-time heatmap of tap magnitudes.

    Args:
        taps_over_time: list of [(delay, amplitude)] per time
        time_points: array of time indices
        delay_bins: number of bins along delay axis
        max_delay: max delay to consider
        title: plot title
        save_path: optional filename
    """
    heatmap = np.zeros((len(time_points), delay_bins))
    bin_edges = np.linspace(0, max_delay, delay_bins + 1)

    for t_idx, taps in enumerate(taps_over_time):
        for delay, amp in taps:
            bin_idx = np.digitize(delay, bin_edges) - 1
            if 0 <= bin_idx < delay_bins:
                heatmap[t_idx, bin_idx] += np.abs(amp)

    plt.figure(figsize=(8, 5))
    extent = [0, max_delay, time_points[-1], time_points[0]]
    plt.imshow(heatmap, aspect='auto', extent=extent, cmap='viridis')
    plt.colorbar(label='|Amplitude|')
    plt.xlabel('Delay (symbols)')
    plt.ylabel('Time Index')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
