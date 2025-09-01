import pandas as pd
import numpy as np
from tqdm import trange


class CurveGenerator:
    """
    Generate AUCC, QINI and RAS-AUCC curves (with optional random baselines)
    for uplift evaluation.
    """

    def __init__(self, trt_list, trt_type, pred_type):
        trt_list = sorted(trt_list)
        # Discount: descending; Deduction: ascending
        self.trt_list = trt_list[::-1] if trt_type == "discount" else trt_list
        self.trt_type = trt_type
        self.pred_type = pred_type

    # ---------- ROI marginal calculations ----------
    def generate_roi(self, df, remove_non_convex=False, random=False):
        """
        Add marginal ROI for each treatment level.
        Columns prefixed with 'roi_{discount}' will be created/overwritten.
        """
        def _roi_single(x):
            # Random baseline (shuffle ROI)
            if random:
                x[len(self.trt_list) : 2 * len(self.trt_list)] = np.sort(
                    np.random.rand(len(self.trt_list))
                )[::-1]
                return x
            # If predictions are already ROI, copy them
            if self.pred_type == "roi":
                x[len(self.trt_list) : 2 * len(self.trt_list)] = x[: len(self.trt_list)]
                return x

            roi_list, idx_list, roi_map = [], [0], {}
            for i in range(1, len(self.trt_list)):
                # Remove non-convex points
                while True:
                    prev = idx_list[-1]
                    if self.trt_type == "discount":
                        roi = (x[i] - x[prev]) * 100 / (
                            x[i] * (100 - self.trt_list[i])
                            - x[prev] * (100 - self.trt_list[prev])
                        )
                    else:  # deduction
                        roi = (x[i] - x[prev]) / (
                            self.trt_list[i] - self.trt_list[prev]
                        )
                    if len(roi_list) == 0 or roi >= roi_list[-1]:
                        break
                    roi_list.pop()
                    idx_list.pop()
                    roi_map.pop(idx_list[-1], None)

                roi_list.append(roi)
                idx_list.append(i)
                roi_map[i] = roi

            # Fill ROI columns
            for k, v in roi_map.items():
                x[len(self.trt_list) + k] = v

            # Auxiliary columns for alpha calculation
            x[-3] = self.trt_list[idx_list[1]]  # first valid discount for control
            try:
                next_idx = idx_list[idx_list.index(self.trt_list.index(x[-2])) + 1]
                x[-1] = self.trt_list[next_idx]
            except (ValueError, IndexError):
                x[-1] = -100
            return x

        df = df.copy()
        cols = [str(d) for d in self.trt_list] + [f"roi_{d}" for d in self.trt_list]
        cols += ["first_dis", "treatment_dis", "treatment_next_dis"]
        arr = df[cols].to_numpy()
        for i in range(len(arr)):
            arr[i] = _roi_single(arr[i])
        df[cols] = arr
        return df

    def expand_samples(self, df):
        """
        Replicate each sample once per treatment level (except control) to
        enable RAS-AUCC sorting.
        """
        def alpha_filter(x, dis):
            return 1 if x == dis else 0

        dfs = []
        for i in range(1, len(self.trt_list)):
            dis = self.trt_list[i]
            tmp = df.copy()
            tmp["dis_of_pred"] = dis
            tmp["pred"] = tmp[str(dis)]
            tmp["edge_roi_of_pred"] = tmp[f"roi_{dis}"]

            # Split into three groups
            df1 = tmp[
                (tmp.treatment_dis != self.trt_list[0])
                & (tmp.treatment_next_dis == dis)
            ]
            df2 = tmp[
                (tmp.treatment_dis != self.trt_list[0])
                & (tmp.treatment_next_dis != dis)
            ]
            df3 = tmp[tmp.treatment_dis == self.trt_list[0]]

            # Alpha coefficients
            df1["alpha"] = -1
            df2["alpha"] = df2.treatment_dis.apply(alpha_filter, args=(dis,))
            df3["alpha"] = df3.first_dis.apply(alpha_filter, args=(dis,))

            dfs += [df1, df2, df3]

        df = pd.concat(dfs, ignore_index=True)
        return df[df.edge_roi_of_pred != -1]

    # ---------- Core cumulative calculations ----------
    def _cumsum_aucc(self, df):
        """Compute cumulative cost and GMV gain for AUCC/MT-AUCC."""
        gmv1 = df["alpha"] * df["blc_w"] * df["price"] * df["y"] * df["treatment"]
        gmv0 = df["alpha"] * df["blc_w"] * df["price"] * df["y"] * (1 - df["treatment"])

        if self.trt_type == "discount":
            cost1 = (
                df["alpha"]
                * df["blc_w"]
                * df["y"]
                * df["treatment"]
                * df["price"]
                * (1 - df["treatment_dis"] / 100)
            )
            cost0 = (
                df["alpha"]
                * df["blc_w"]
                * df["y"]
                * (1 - df["treatment"])
                * df["price"]
                * (1 - df["treatment_dis"] / 100)
            )
        else:  # deduction
            cost1 = (
                df["alpha"]
                * df["blc_w"]
                * df["y"]
                * df["treatment"]
                * df["treatment_dis"]
            )
            cost0 = (
                df["alpha"]
                * df["blc_w"]
                * df["y"]
                * (1 - df["treatment"])
                * df["treatment_dis"]
            )

        g1, g0 = gmv1.cumsum(), gmv0.cumsum()
        c1, c0 = cost1.cumsum(), cost0.cumsum()
        p1 = (df["alpha"] * df["blc_w"] * df["treatment"] * df["n"]).cumsum()
        p0 = (df["alpha"] * df["blc_w"] * (1 - df["treatment"]) * df["n"]).cumsum()

        gain = g1 * p0 / p1 - g0
        cost = c1 * p0 / p1 - c0
        gain.iloc[0] = cost.iloc[0] = 0
        return cost, gain

    def _sample_curve(self, cost, gain, n_points):
        """Down-sample cumulative curves to n_points equidistant in cost."""
        step = cost.iloc[-1] / (n_points - 1)
        idx = [(cost - step * i).abs().argmin() for i in range(n_points)]
        return cost.iloc[idx], gain.iloc[idx]

    # ---------- Public generators ----------
    def generate_series_aucc(self, df, n_points, n_random=None, random=False):
        if random:
            costs, gains = [], []
            for _ in range(n_random):
                tmp = df.sample(frac=1).reset_index(drop=True)
                c, g = self._cumsum_aucc(tmp)
                costs.append(c)
                gains.append(g)
            cost = pd.concat(costs, axis=1).mean(axis=1)
            gain = pd.concat(gains, axis=1).mean(axis=1)
        else:
            df = df.sort_values("ite/bcr", ascending=False).reset_index(drop=True)
            cost, gain = self._cumsum_aucc(df)
        return self._sample_curve(cost, gain, n_points)

    def generate_series_qini(self, df, n_points, n_random=None, random=False):
        """Return normalized QINI curve points."""
        def _cumsum_qini(df):
            f1 = (df["y"] * df["treatment"]).cumsum()
            f0 = (df["y"] * (1 - df["treatment"])).cumsum()
            p1 = (df["n"] * df["treatment"]).cumsum()
            p0 = (df["n"] * (1 - df["treatment"])).cumsum()
            gain = f1 - f0 * p1 / p0
            gain.iloc[0] = 0
            ratio = p1 / p0
            ratio.iloc[0] = 0
            return gain, ratio

        if random:
            gains = []
            for _ in range(n_random):
                tmp = df.sample(frac=1).reset_index(drop=True)
                g, _ = _cumsum_qini(tmp)
                gains.append(g)
            gain = pd.concat(gains, axis=1).mean(axis=1)
            return gain[:: max(1, len(df) // (n_points - 1))]
        else:
            df = df.sort_values("ite", ascending=False).reset_index(drop=True)
            gain, ratio = _cumsum_qini(df)
            idx = list(range(0, len(df), max(1, len(df) // (n_points - 1))))
            return gain.iloc[idx], ratio.iloc[idx]

    def generate_series_ras_aucc(self, df, n_points, n_random=20, random=False):
        """Generate RAS-AUCC curves, with optional random baseline."""
        if random:
            costs, gains = [], []
            df_raw = df
            for _ in trange(n_random):
                tmp = self.generate_roi(df_raw.copy(), False, True)
                tmp = self.expand_samples(tmp).sort_values(
                    "edge_roi_of_pred", ascending=False
                )
                c, g = self._cumsum_aucc(tmp)
                costs.append(c)
                gains.append(g)
            cost = pd.concat(costs, axis=1).mean(axis=1)
            gain = pd.concat(gains, axis=1).mean(axis=1)
        else:
            tmp = self.generate_roi(df.copy(), True, False)
            tmp = self.expand_samples(tmp).sort_values(
                "edge_roi_of_pred", ascending=False
            )
            cost, gain = self._cumsum_aucc(tmp)
        return self._sample_curve(cost, gain, n_points)