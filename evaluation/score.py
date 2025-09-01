import numpy as np


class ScoreCalculators:
    """Static utility class for all uplift metrics."""

    @staticmethod
    def calc_area_under_curve(x, y):
        """Compute the area under a curve using the trapezoidal rule."""
        x, y = x.to_numpy(), y.to_numpy()
        return ((x[1:] - x[:-1]) * (y[:-1] + y[1:])).sum(0) / 2.0

    @staticmethod
    def calc_aucc_score(x_algo, y_algo, x_bsl, y_bsl):
        """AUCC score: relative improvement vs. the straight-line baseline."""
        area_algo = ScoreCalculators.calc_area_under_curve(x_algo, y_algo)
        area_bsl = ScoreCalculators.calc_area_under_curve(x_bsl, y_bsl)
        return np.round((area_algo - area_bsl) / area_bsl, 5)

    @staticmethod
    def calc_qini_score(algo, bsl):
        """QINI score: average cumulative uplift vs. straight-line baseline."""
        return np.round((algo.sum() - bsl.sum()) / len(algo), 5)

    @staticmethod
    def calc_dini_score(algo, bsl):
        """DINI score: weighted QINI with diminishing returns."""
        g = 1 - algo.index / algo.index[-1]
        dini = (g * (algo - bsl)).sum() / len(algo)
        dini = np.sign(dini) * np.abs(dini) ** (2 / 3)
        return np.round(dini, 5)

    @staticmethod
    def calc_ate_error(ite, y_t, y_0):
        """Absolute error between predicted and true ATE."""
        ate = y_t.mean() - y_0.mean()
        ate_hat = ite.mean()
        return np.round(np.abs(ate_hat - ate), 5)