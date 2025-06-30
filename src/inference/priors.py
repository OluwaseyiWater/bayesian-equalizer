"""
Utility functions for defining prior distributions for Bayesian channel models.
Includes helpers for setting up base measures for delays, Dopplers, and amplitudes.
"""

import numpy as np
from typing import Callable


def uniform_delay_prior(min_delay: float, max_delay: float) -> Callable[[], float]:
    """
    Returns a sampler function for uniform delay prior.

    Args:
        min_delay: minimum delay value
        max_delay: maximum delay value

    Returns:
        A function that samples a delay from Uniform(min_delay, max_delay)
    """
    def sampler():
        return np.random.uniform(min_delay, max_delay)
    return sampler


def uniform_doppler_prior(min_dopp: float, max_dopp: float) -> Callable[[], float]:
    """
    Returns a sampler for uniform Doppler prior.

    Args:
        min_dopp: minimum Doppler shift
        max_dopp: maximum Doppler shift

    Returns:
        A function that samples a Doppler shift
    """
    def sampler():
        return np.random.uniform(min_dopp, max_dopp)
    return sampler


def gaussian_amplitude_prior(mean: float = 0.0, std: float = 1.0) -> Callable[[], float]:
    """
    Returns a sampler for Gaussian prior over complex amplitudes.

    Args:
        mean: mean amplitude
        std: standard deviation

    Returns:
        A function that samples a complex gain from 𝒞𝒩(mean, std²)
    """
    def sampler():
        real = np.random.normal(loc=mean, scale=std / np.sqrt(2))
        imag = np.random.normal(loc=mean, scale=std / np.sqrt(2))
        return real + 1j * imag
    return sampler
