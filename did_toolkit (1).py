"""DiD toolkit for ECON 5200 PS3, extending Lab 11."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

NEVER = 10**9


def two_by_two(df, outcome, treated, post):
    """Calculate a canonical two-by-two DiD estimate."""
    g = df.groupby([treated, post])[outcome].mean()
    return float(
        (g[1, 1] - g[1, 0])
        - (g[0, 1] - g[0, 0])
    )


def att_by_cohort_period(df, time, outcome, cohort):
    """Estimate ATT(g,t) with not-yet-treated controls."""
    rows = []

    cohorts = sorted(
        g for g in df[cohort].unique()
        if np.isfinite(g) and g < NEVER
    )

    for g in cohorts:
        treated = df[df[cohort] == g]
        baseline = g - 1

        for t in sorted(df[time].unique()):
            if t < g:
                continue

            controls = df[df[cohort] > t]

            if controls.empty:
                continue

            tr_pre = treated.loc[
                treated[time] == baseline, outcome
            ].mean()
            tr_post = treated.loc[
                treated[time] == t, outcome
            ].mean()

            co_pre = controls.loc[
                controls[time] == baseline, outcome
            ].mean()
            co_post = controls.loc[
                controls[time] == t, outcome
            ].mean()

            att = (tr_post - tr_pre) - (co_post - co_pre)

            rows.append({
                "g": g,
                "t": t,
                "e": t - g,
                "att": att,
                "n": int((treated[time] == t).sum())
            })

    return pd.DataFrame(rows)


def overall_att(cells):
    """Aggregate cohort-period ATTs."""
    return float(
        np.average(cells["att"], weights=cells["n"])
    )


def event_study(cells):
    """Aggregate effects by event time."""
    rows = []

    for e, group in cells.groupby("e"):
        rows.append({
            "e": e,
            "att": np.average(
                group["att"], weights=group["n"]
            )
        })

    return pd.DataFrame(rows)


def clustered_inference(df, formula, cluster, param):
    """Run OLS with cluster-robust inference."""
    fit = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster]}
    )

    return {
        "estimate": float(fit.params[param]),
        "se": float(fit.bse[param]),
        "t": float(fit.tvalues[param]),
        "p_value": float(fit.pvalues[param]),
        "fit": fit
    }


def cluster_se_ratio(fit_iid, fit_clustered, param):
    """Ratio of clustered SE to default SE."""
    return float(
        fit_clustered.bse[param] / fit_iid.bse[param]
    )


def wild_cluster_bootstrap(df, formula, restricted_formula,
                           cluster="board", param="treated"):
    """Exact null-imposed wild cluster bootstrap."""
    from itertools import product

    full = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster]}
    )
    null = smf.ols(restricted_formula, data=df).fit()

    X = full.model.exog
    names = full.model.exog_names
    idx = names.index(param)
    X0 = np.delete(X, idx, axis=1)

    boards = df[cluster].to_numpy()
    unique_boards = np.unique(boards)
    G = len(unique_boards)
    N = len(df)
    K = np.linalg.matrix_rank(X)

    Q, _ = np.linalg.qr(X0, mode="reduced")

    def remove_fe(v):
        return v - Q @ (Q.T @ v)

    x = remove_fe(X[:, idx])
    denominator = x @ x

    fitted = null.fittedvalues.to_numpy()
    residuals = null.resid.to_numpy()

    vectors = np.array(
        list(product([-1.0, 1.0], repeat=G))
    )

    board_index = np.searchsorted(
        unique_boards, boards
    )
    weights = vectors[:, board_index].T

    y_star = (
        fitted[:, None]
        + residuals[:, None] * weights
    )

    y_resid = remove_fe(y_star)
    beta_star = (x @ y_resid) / denominator
    errors = y_resid - x[:, None] * beta_star

    scores = np.vstack([
        (
            x[boards == b, None]
            * errors[boards == b, :]
        ).sum(axis=0)
        for b in unique_boards
    ])

    correction = (
        G / (G - 1)
        * (N - 1) / (N - K)
    )

    se_star = np.sqrt(
        correction * (scores ** 2).sum(axis=0)
        / denominator ** 2
    )

    t_star = beta_star / se_star
    actual_t = float(full.tvalues[param])

    p_value = float(
        np.mean(np.abs(t_star) >= abs(actual_t) - 1e-12)
    )

    return {
        "estimate": float(full.params[param]),
        "naive_p": float(full.pvalues[param]),
        "bootstrap_p": p_value,
        "n_vectors": len(vectors),
        "actual_t": actual_t
    }


if __name__ == "__main__":
    test = pd.DataFrame({
        "treated": [0, 0, 1, 1],
        "post": [0, 1, 0, 1],
        "y": [2.0, 3.0, 4.0, 7.0]
    })

    assert np.isclose(
        two_by_two(test, "y", "treated", "post"),
        2.0
    )

    print("Module basic assertions PASSED")
