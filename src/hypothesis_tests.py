from scipy.stats import (
    ttest_ind,
    chi2_contingency,
    f_oneway
)

import pandas as pd
import numpy as np


# =========================================================
# ANOVA TEST
# =========================================================
def run_anova_test(
    df,
    group_col,
    target_col
):
    """
    Run one-way ANOVA test
    """

    groups = [
        group[target_col].dropna().values
        for _, group in df.groupby(group_col)
    ]

    f_stat, p_value = f_oneway(*groups)

    decision = (
        "Reject H0"
        if p_value < 0.05
        else "Fail to Reject H0"
    )

    return {
        "Test": "ANOVA",
        "F-Statistic": round(float(f_stat), 4),
        "P-Value": float(p_value),
        "Decision": decision
    }

# =========================================================
# T-TEST
# =========================================================
def run_ttest(
    df,
    group_col,
    target_col,
    group_a,
    group_b
):

    a = df[
        df[group_col] == group_a
    ][target_col].dropna()

    b = df[
        df[group_col] == group_b
    ][target_col].dropna()

    t_stat, p_value = ttest_ind(
        a,
        b,
        equal_var=False
    )

    decision = (
        "Reject H0"
        if p_value < 0.05
        else "Fail to Reject H0"
    )

    return {
        "Test": "T-Test",
        "T-Statistic": round(float(t_stat), 4),
        "P-Value": float(p_value),
        "Decision": decision
    }

# =========================================================
# CHI-SQUARE TEST
# =========================================================
def run_chi_square_test(
    df,
    group_col,
    target_col
):

    contingency_table = pd.crosstab(
        df[group_col],
        df[target_col]
    )

    chi2, p_value, dof, expected = chi2_contingency(
        contingency_table
    )

    decision = (
        "Reject H0"
        if p_value < 0.05
        else "Fail to Reject H0"
    )

    return {
        "Test": "Chi-Square",
        "Chi2": round(float(chi2), 4),
        "P-Value": float(p_value),
        "Decision": decision
    }