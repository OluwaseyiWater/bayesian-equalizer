"""
Resampling methods for Sequential Monte Carlo (SMC) particle filters.
Includes multinomial, systematic, and stratified resampling.
"""

import numpy as np
from typing import Callable


def multinomial_resample(weights: np.ndarray) -> np.ndarray:
    """Multinomial resampling based on weights."""
    n = len(weights)
    indices = np.random.choice(n, size=n, p=weights)
    return indices


def systematic_resample(weights: np.ndarray) -> np.ndarray:
    """
    Systematic resampling with evenly spaced samples.
    More deterministic than multinomial.
    """
    n = len(weights)
    positions = (np.arange(n) + np.random.uniform()) / n

    cumulative_sum = np.cumsum(weights)
    indices = np.zeros(n, dtype=int)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


def stratified_resample(weights: np.ndarray) -> np.ndarray:
    """
    Stratified resampling with random offsets inside each interval.
    """
    n = len(weights)
    positions = (np.random.rand(n) + np.arange(n)) / n

    cumulative_sum = np.cumsum(weights)
    indices = np.zeros(n, dtype=int)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


# Mapping for flexible selection
RESAMPLERS = {
    "multinomial": multinomial_resample,
    "systematic": systematic_resample,
    "stratified": stratified_resample
}


def get_resampler(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name not in RESAMPLERS:
        raise ValueError(f"Unknown resampling method '{name}'")
    return RESAMPLERS[name]
