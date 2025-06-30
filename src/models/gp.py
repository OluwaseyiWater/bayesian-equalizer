"""
Gaussian Process prior wrapper using GPyTorch for modeling time-varying or delay-Doppler fading path amplitudes.
This module provides a flexible interface to define GPs with customizable kernels.
"""

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from typing import Optional


class PathGainGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood, kernel_type: str = 'RBF'):
        """
        Initialize a GP model for path gain modeling.

        Args:
            train_x: training inputs (e.g., time steps)
            train_y: training outputs (e.g., complex gain magnitudes)
            likelihood: GPyTorch likelihood object
            kernel_type: one of ['RBF', 'Matern']
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_type == 'RBF':
            base_kernel = RBFKernel()
        elif kernel_type == 'Matern':
            base_kernel = MaternKernel(nu=2.5)
        else:
            raise ValueError("Unsupported kernel type")

        self.covar_module = ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def create_gp_model(train_x: torch.Tensor, train_y: torch.Tensor, kernel_type: str = 'RBF') -> PathGainGP:
    """
    Convenience function to instantiate and return a GP model with likelihood.
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = PathGainGP(train_x, train_y, likelihood, kernel_type=kernel_type)
    return model, likelihood
