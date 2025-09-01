import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .score import ScoreCalculators
from .curve import CurveGenerator


class ReportGenerator:
    """
    Main entry point for generating uplift evaluation reports.

    Parameters
    ----------
    trt_list : list
        Treatment levels (discounts/deductions).
    trt_type : {'discount', 'deduction'}
    pred_type : {'bcr', 'roi'}
    """

    def __init__(self, trt_list=(100, 95, 90, 85, 80), trt_type="discount", pred_type="bcr"):
        self.trt_list = list(trt_list)
        self.trt_type = trt_type
        self.pred_type = pred_type
        self.cg = CurveGenerator(self.trt_list, trt_type, pred_type)

    # ---------- Helpers ----------
    def _load_df(self, data):
        """Accept either path or DataFrame."""
        return pd.read_csv(data) if isinstance(data, str) else data.copy()

    def _init_pdf(self, save_path, filename):
        """Initialize PdfPages if save_path is provided."""
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            return PdfPages(os.path.join(save_path, filename))
        return None

    # ---------- Multi-treatment RAS-AUCC ----------
    def generate_report_multi_treatment(
        self,
        data,
        n_points=100,
        n_random=20,
        save_path=None,
        pdf_page=None,
        balance_dis_group=True,
    ):
        """Generate RAS-AUCC curve and score."""
        df = self._load_df(data)
        pp = pdf_page or self._init_pdf(save_path, "report_multi_treatment.pdf")

        # Create treatment flag and balance weights
        df["treatment"] = (df.treatment_dis != self.trt_list[0]).astype(int)
        df["first_dis"] = self.trt_list[1]
        if balance_dis_group:
            if "n" not in df.columns:
                df["n"] = 1
            if "blc_w" not in df.columns:
                df["blc_w"] = 1
            ctr = df[df.treatment_dis == self.trt_list[0]]["n"].sum()
            for dis in self.trt_list:
                mask = df.treatment_dis == dis
                df.loc[mask, "blc_w"] = ctr / df.loc[mask, "n"].sum()

        # Curves
        cost, gain = self.cg.generate_series_ras_aucc(df, n_points)
        cost_rand, gain_rand = self.cg.generate_series_ras_aucc(
            df, n_points, n_random, True
        )
        cost_line, gain_line = cost.iloc[[0, -1]], gain.iloc[[0, -1]]

        # Scores
        score = ScoreCalculators.calc_aucc_score(cost, gain, cost_line, gain_line)
        score_rand = ScoreCalculators.calc_aucc_score(
            cost_rand, gain_rand, cost_line, gain_line
        )

        # Plot
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.plot(cost, gain, label="algorithm")
        ax.plot(cost_rand, gain_rand, label="random")
        ax.plot(cost_line, gain_line, label="straight line")
        ax.set_xlabel("Cost")
        ax.set_ylabel("GMV(Gain)" if self.trt_type == "discount" else "Finish Count(Gain)")
        ax.set_title("RAS-AUCC")
        ax.legend()
        ax.text(
            cost.iloc[-1] * 0.5,
            gain.iloc[-1] * 0.3,
            f"RAS-AUCC:{score}\nRandom:{score_rand}",
        )
        if save_path:
            fig.savefig(os.path.join(save_path, "RAS_AUCC.png"))
        if pp and not pdf_page:
            pp.savefig(fig, bbox_inches="tight")
            pp.close()
        return score, score_rand

    # ---------- Single treatment ----------
    def generate_report_single_treatment(
        self,
        data,
        n_aucc=100,
        n_qini=100,
        n_random=20,
        save_path=None,
        pdf_page=None,
    ):
        """Generate AUCC and QINI curves for each treatment vs. control."""
        df = self._load_df(data)
        pp = pdf_page or self._init_pdf(save_path, "report_single_treatment.pdf")

        # Default missing columns
        for col in ("n", "blc_w"):
            if col not in df.columns:
                df[col] = 1
        df["treatment"] = (df.treatment_dis != self.trt_list[0]).astype(int)

        # Subplot grid
        n_sub = len(self.trt_list) - 1
        fig1, axes1 = plt.subplots(
            math.ceil(n_sub / 2),
            2,
            figsize=(16, math.ceil(n_sub / 2) * 6),
        )
        fig2, axes2 = plt.subplots(
            math.ceil(n_sub / 2),
            2,
            figsize=(16, math.ceil(n_sub / 2) * 6),
        )
        fig1.suptitle("AUCC")
        fig2.suptitle("QINI")

        scores = []
        for i, dis in enumerate(self.trt_list[1:], 1):
            d = df[(df.treatment_dis == dis) | (df.treatment_dis == self.trt_list[0])]
            d = d.copy()
            d["ite"] = d[str(dis)] - d[str(self.trt_list[0])]
            d["ite/bcr"] = d["ite"] / d[str(dis)]

            # AUCC
            cost, gain = self.cg.generate_series_aucc(d, n_aucc)
            cost_rand, gain_rand = self.cg.generate_series_aucc(
                d, n_aucc, n_random, True
            )
            cost_line, gain_line = cost.iloc[[0, -1]], gain.iloc[[0, -1]]
            auc_score = ScoreCalculators.calc_aucc_score(
                cost, gain, cost_line, gain_line
            )

            ax = axes1 if n_sub <= 2 else axes1.flat
            idx = i - 1 if n_sub <= 2 else (i - 1) // 2, (i - 1) % 2
            ax = ax[idx] if n_sub > 2 else ax
            ax.plot(cost, gain, label="algorithm")
            ax.plot(cost_line, gain_line, label="straight line")
            ax.plot(cost_rand, gain_rand, label="random")
            ax.set_title(f"AUCC(Discount:{dis})")
            ax.legend()

            # QINI
            gain_q, ratio = self.cg.generate_series_qini(d, n_qini)
            gain_q_rand = self.cg.generate_series_qini(d, n_qini, n_random, True)
            ymax = gain_q.iloc[-1]
            gain_q /= abs(ymax)
            gain_q_rand /= abs(ymax)
            gain_q_line = pd.Series(np.sign(ymax) * gain_q.index / gain_q.index[-1])
            qini = ScoreCalculators.calc_qini_score(gain_q, gain_q_line)
            dini = ScoreCalculators.calc_dini_score(gain_q, gain_q_line)
            ate = ScoreCalculators.calc_ate_error(
                d["ite"], d[d.treatment == 1]["y"], d[d.treatment == 0]["y"]
            )

            ax2 = axes2 if n_sub <= 2 else axes2.flat
            ax2 = ax2[idx] if n_sub > 2 else ax2
            ax2.plot(gain_q, label="algorithm")
            ax2.plot(gain_q_line, label="straight line")
            ax2.plot(gain_q_rand, label="random")
            ax2.plot(ratio, label="trt_ratio")
            ax2.set_title(f"QINI(Discount:{dis})")
            ax2.legend()
            ax2.text(
                gain_q.index[-1] * 0.5,
                gain_q.iloc[-1] * 0.3,
                f"q:{qini}\nd:{dini}\na:{ate}",
            )

            scores.append([auc_score, qini, dini, ate])

        if pp and not pdf_page:
            pp.savefig(fig1, bbox_inches="tight")
            pp.savefig(fig2, bbox_inches="tight")
            pp.close()

        # Build summary dataframe
        idx = list(map(str, self.trt_list[1:])) + ["avg"]
        arr = np.array(scores)
        arr = np.vstack([arr, arr.mean(axis=0)])
        return pd.DataFrame(
            arr, columns=["aucc", "qini", "dini", "ate_error"], index=idx
        )

    # ---------- Unified entry ----------
    def generate_report(
        self,
        data,
        save_path=None,
        n_ras_aucc=100,
        n_aucc=100,
        n_qini=100,
        n_random=20,
        val_auc=None,
    ):
        """Generate both RAS-AUCC and single-treatment reports."""
        pp = self._init_pdf(save_path, "report.pdf")
        ras_algo, ras_rand = self.generate_report_multi_treatment(
            data, n_ras_aucc, n_random, save_path, pp
        )
        single = self.generate_report_single_treatment(
            data, n_aucc, n_qini, n_random, save_path, pp
        )
        summary = single.iloc[[len(self.trt_list) - 2]].copy()
        summary["algo_ras_aucc"] = ras_algo
        summary["random_ras_aucc"] = ras_rand
        if val_auc:
            summary["auc"] = val_auc
        pp.close()
        return summary.round(4)