"""The P(benefit)/P(harm) bounds table and the summary table."""
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

from .formatting import fmt
from .styles import BENEFIT_CSS, CATE_CSS, HARM_CSS, NO_HARM_CSS

HARM_COLUMNS = [("p(harm)", "lower"), ("p(harm)", "upper")]
BENEFIT_COLUMNS = [("p(benefit)", "lower"), ("p(benefit)", "upper")]
CATE_COLUMNS = [("CATE", "")]


# --- Bounds table -----------------------------------------------------------

def add_bound_columns(agg_df: pd.DataFrame, p_benefit: dict, p_harm: dict) -> pd.DataFrame:
    """Return the RCT aggregate with P(harm) and P(benefit) bound columns.

    p_benefit, p_harm: sex (0 = male, 1 = female) -> [lower, upper] bounds.
    """
    # Every row comes from the RCT, so is_rct is constant: drop it from the
    # index and record its meaning in the column name instead.
    if "is_rct" in agg_df.index.names:
        agg_df = agg_df.droplevel("is_rct")
    agg_df = agg_df.rename(columns={"outcome": "rct_outcome"}, level=0)

    # "sex" is an index level of agg_df, not a column, so it is selected
    # through the index; both quantities are intervals, so each needs a column
    # per bound.
    sex = agg_df.index.get_level_values("sex")
    agg_df[("p(harm)", "lower")] = [p_harm[s][0] for s in sex]
    agg_df[("p(harm)", "upper")] = [p_harm[s][1] for s in sex]
    agg_df[("p(benefit)", "lower")] = [p_benefit[s][0] for s in sex]
    agg_df[("p(benefit)", "upper")] = [p_benefit[s][1] for s in sex]
    return agg_df


def highlight_quantities(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Colour P(harm) by sex (0 = male, 1 = female), CATE blue, P(benefit) gold.
    """
    styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
    is_female = frame.index.get_level_values("sex") == 1
    styles.loc[is_female, HARM_COLUMNS] = NO_HARM_CSS
    styles.loc[~is_female, HARM_COLUMNS] = HARM_CSS
    styles.loc[:, CATE_COLUMNS] = CATE_CSS
    styles.loc[:, BENEFIT_COLUMNS] = BENEFIT_CSS
    return styles


def style_bounds_table(agg_df: pd.DataFrame) -> Styler:
    """Green where the drug cannot harm anyone (P(harm) = 0), red where it can."""
    return agg_df.style.apply(highlight_quantities, axis=None).format("{:.4f}")


# --- Summary table ----------------------------------------------------------

SUBGROUPS = {
    "Female": "sex == 1",
    "Male": "sex == 0",
    "Both sexes": "sex in [0, 1]",
}

ROW_YT = "P(y_t)  — recovery under treatment (RCT)"
ROW_YC = "P(y_c)  — recovery under control (RCT)"
ROW_Y = "P(y)  — recovery observed (survey)"
ROW_T = "P(t)  — chose the drug (survey)"
ROW_CATE = "CATE = P(y_t) − P(y_c)"
ROW_BENEFIT = "P(benefit)"
ROW_HARM = "P(harm) = P(benefit) − CATE"
ROW_MONOTONICITY = "Monotonicity: P(harm) = 0 ?"


def estimate_quantities(rct: pd.DataFrame, obs: pd.DataFrame) -> dict:
    """Tian-Pearl bounds on P(benefit)/P(harm) for one sample of patients."""
    p_yt = rct.query("treatment == 1")["outcome"].mean()
    p_yc = rct.query("treatment == 0")["outcome"].mean()
    p_y = obs["outcome"].mean()

    p_t = obs["treatment"].mean()
    p_y_given_t = obs.query("treatment == 1")["outcome"].mean()
    p_y_given_c = obs.query("treatment == 0")["outcome"].mean()

    p_t_and_y = p_t * p_y_given_t
    p_t_and_not_y = p_t * (1 - p_y_given_t)
    p_c_and_y = (1 - p_t) * p_y_given_c
    p_c_and_not_y = (1 - p_t) * (1 - p_y_given_c)

    lower = np.max([
        0,
        p_yt - p_yc,
        p_y - p_yc,
        p_yt - p_y,
    ])
    upper = np.min([
        p_yt,
        1 - p_yc,
        p_t_and_y + p_c_and_not_y,
        p_yt - p_yc + p_t_and_not_y + p_c_and_y,
    ])

    cate = p_yt - p_yc
    p_benefit = np.array([lower, upper])
    return {
        "p_yt": p_yt,
        "p_yc": p_yc,
        "p_y": p_y,
        "p_t": p_t,
        "cate": cate,
        "p_benefit": p_benefit,
        "p_harm": p_benefit - cate,  # equation (10)
    }


def estimate_subgroups(
    df: pd.DataFrame,
    df_obs: pd.DataFrame,
    subgroups: dict = SUBGROUPS,
) -> dict:
    """estimate_quantities for every subgroup: name -> query condition."""
    return {
        name: estimate_quantities(
            df.query(f"is_rct == 1 and {condition}"),
            df_obs.query(f"is_rct == 0 and {condition}"),
        )
        for name, condition in subgroups.items()
    }


def monotonicity(results: dict, tol: float) -> dict:
    """Monotonicity holds when the whole P(harm) interval sits at 0 (up to tol)."""
    return {
        name: bool(np.all(np.abs(result["p_harm"]) < tol))
        for name, result in results.items()
    }


def build_summary(results: dict, monotonic: dict, tol: float) -> pd.DataFrame:
    """One column per subgroup, one row per quantity, formatted as text.

    Identified quantities are point estimates; P(benefit) and P(harm) are
    intervals unless their bounds collapse (width < tol).
    """
    summary = pd.DataFrame(
        {
            name: [
                fmt(result["p_yt"], tol),
                fmt(result["p_yc"], tol),
                fmt(result["p_y"], tol),
                fmt(result["p_t"], tol),
                fmt(result["cate"], tol),
                fmt(result["p_benefit"], tol),
                fmt(result["p_harm"], tol) + (" (≈ 0)" if monotonic[name] else ""),
                "yes — P(benefit) = CATE"
                if monotonic[name]
                else "no — P(benefit) > CATE",
            ]
            for name, result in results.items()
        },
        index=[
            ROW_YT,
            ROW_YC,
            ROW_Y,
            ROW_T,
            ROW_CATE,
            ROW_BENEFIT,
            ROW_HARM,
            ROW_MONOTONICITY,
        ],
    )
    summary.index.name = "Quantity"
    return summary


def style_summary(summary: pd.DataFrame, monotonic: dict) -> Styler:
    """Green where the drug never harms (P(harm) = 0, so P(benefit) = CATE)."""

    def highlight_monotonicity(frame: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
        highlighted = [ROW_CATE, ROW_BENEFIT, ROW_HARM, ROW_MONOTONICITY]
        for name in frame.columns:
            styles.loc[highlighted, name] = (
                NO_HARM_CSS if monotonic[name] else HARM_CSS
            )
        return styles

    return summary.style\
        .apply(highlight_monotonicity, axis=None)\
        .set_caption(
            "Green: P(harm) = 0, so the drug never harms and the experimental CATE "
            "already equals P(benefit). "
            "Red: P(harm) > 0, so P(benefit) ≠ CATE and the RCT alone overstates "
            "who is actually helped."
        )\
        .set_table_styles([
            {"selector": "caption",
             "props": "caption-side: bottom; padding-top: 12px; font-size: 90%;"
                      " text-align: left; color: #444;"},
            {"selector": "th", "props": "text-align: left; padding: 4px 10px;"},
            {"selector": "td", "props": "text-align: right; padding: 4px 10px;"},
        ])
