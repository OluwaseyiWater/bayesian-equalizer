"""
Dirichlet Process prior implementation (stick-breaking and CRP variants).
This module supports sampling path delays from a DP prior, as used in the
Bayesian nonparametric equalizer.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional

class StickBreakingDP:
    def __init__(self, alpha: float, base_measure: Callable[[], float], max_atoms: int = 50):
        """
        Initialize a Stick-Breaking construction for a Dirichlet Process.

        Args:
            alpha: concentration parameter (higher => more components)
            base_measure: function that samples from the base distribution G0
            max_atoms: truncation level for practical stick-breaking
        """
        self.alpha = alpha
        self.base_measure = base_measure
        self.max_atoms = max_atoms

        self.weights = None  # Stick-breaking weights π_k
        self.atoms = None    # Atom locations θ_k (e.g., delays)

    def sample(self):
        """Sample a discrete measure from the DP (stick-breaking approximation)."""
        betas = np.random.beta(1, self.alpha, size=self.max_atoms)
        remaining_stick = np.cumprod(1 - betas[:-1])
        weights = np.concatenate([[betas[0]], betas[1:] * remaining_stick])
        atoms = [self.base_measure() for _ in range(self.max_atoms)]

        self.weights = weights
        self.atoms = atoms
        return weights, atoms

    def draw(self, n: int) -> List[float]:
        """Draw n samples from the discrete DP measure."""
        if self.weights is None or self.atoms is None:
            self.sample()

        indices = np.random.choice(len(self.atoms), size=n, p=self.weights)
        return [self.atoms[i] for i in indices]


class ChineseRestaurantProcess:
    def __init__(self, alpha: float):
        """
        CRP representation for sampling cluster assignments.

        Args:
            alpha: concentration parameter
        """
        self.alpha = alpha
        self.tables = []  # Each table holds a list of customer indices
        self.assignments = []

    def sample_next(self) -> int:
        """
        Assign the next customer to a table based on CRP probabilities.
        Returns the table index.
        """
        n = len(self.assignments)
        probs = [len(t) / (n + self.alpha) for t in self.tables] + [self.alpha / (n + self.alpha)]
        choice = np.random.choice(len(probs), p=probs)

        if choice == len(self.tables):
            self.tables.append([n])  # New table
        else:
            self.tables[choice].append(n)

        self.assignments.append(choice)
        return choice

    def get_clusters(self) -> List[List[int]]:
        return self.tables
