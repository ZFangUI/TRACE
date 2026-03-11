"""
Conformal prediction framework: calibrate threshold, predict, evaluate.

All score functions share the same conformal wrapper:
    1. Compute scores on calibration set
    2. Set threshold τ = quantile(scores, ⌈(1-α)(n+1)⌉/n)
    3. Evaluate coverage on test set
"""

import numpy as np


def conformal_quantile(scores, alpha):
    """Conformal quantile: exact order statistic.

    τ = the k-th smallest score where k = ⌈(1-α)(n+1)⌉.

    NaN scores are replaced with +inf (treated as maximally nonconforming)
    so they don't silently corrupt the threshold.
    """
    scores = np.asarray(scores, dtype=np.float64)
    # Replace NaN with +inf: NaN scores are "worst case"
    nan_mask = np.isnan(scores)
    if nan_mask.any():
        scores = scores.copy()
        scores[nan_mask] = np.inf
    n = scores.shape[0]
    k = int(np.ceil((1 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


class ConformalPredictor:
    """Unified conformal prediction wrapper for any score function.

    Usage:
        score_fn = NFBallScore(model, device)  # or DiffusionDenoiseScore, etc.
        cp = ConformalPredictor(score_fn, alpha=0.1)
        cp.calibrate(cal_x, cal_y)
        results = cp.evaluate(test_x, test_y)
    """

    def __init__(self, score_fn, alpha=0.1):
        """
        Args:
            score_fn: object with .compute(x, y) → scores [n]
                      and .name attribute
            alpha: error rate (coverage target = 1-α)
        """
        self.score_fn = score_fn
        self.alpha = alpha
        self.tau = None
        self.cal_scores = None

    @property
    def name(self):
        return self.score_fn.name

    def calibrate(self, cal_x, cal_y):
        """Compute calibration threshold.

        Args:
            cal_x: [n_cal, x_dim] calibration inputs
            cal_y: [n_cal, y_dim] calibration outputs
        Returns:
            cal_scores: numpy array of calibration scores
        """
        self.cal_scores = self.score_fn.compute(cal_x, cal_y)
        self.tau = conformal_quantile(self.cal_scores, self.alpha)
        return self.cal_scores

    def evaluate(self, test_x, test_y):
        """Evaluate on test set.

        Returns dict with:
            coverage: fraction of test points covered
            tau: calibrated threshold
            score_mean, score_std: test score statistics
            scores: raw test scores (for analysis)
        """
        assert self.tau is not None, "Must call calibrate() first"
        test_scores = self.score_fn.compute(test_x, test_y)
        covered = test_scores <= self.tau
        return {
            "coverage": float(covered.mean()),
            "tau": float(self.tau),
            "score_mean": float(test_scores.mean()),
            "score_std": float(test_scores.std()),
            "scores": test_scores,
        }

    def predict_grid(self, x_point, y_grid, **kwargs):
        """For visualization: which y's are in the prediction set?

        Args:
            x_point: [x_dim] single conditioning point
            y_grid: [M, y_dim] candidate y values
            **kwargs: forwarded to score_fn.compute_on_grid
                (e.g. n_avg=3 for Diff/FM to smooth boundaries)
        Returns:
            inside: [M] boolean mask
            scores: [M] scores for each y
        """
        assert self.tau is not None, "Must call calibrate() first"
        scores = self.score_fn.compute_on_grid(x_point, y_grid, **kwargs)
        inside = scores <= self.tau
        return inside, scores