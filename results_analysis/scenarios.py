"""Response types of the sampled males and the three policy diagrams.

All shares are estimates computed from the simulated samples, so every
function takes the upstream estimates (P(y_t), P(y_c), the P(benefit) and
P(harm) bounds), the sample size n and the tolerance used to collapse an
interval into a point estimate.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .flow import draw_flow
from .formatting import fmt_count, fmt_interval
from .styles import (
    CHOICE_EDGE,
    CHOICE_FACE,
    DEATH_EDGE,
    DEATH_FACE,
    NEUTRAL_EDGE,
    NEUTRAL_FACE,
    SURVIVAL_EDGE,
    SURVIVAL_FACE,
)


@dataclass(frozen=True)
class ResponseTypes:
    """Recovery rates and the four response types of one subgroup."""
    p_yt: float           # P(y_t), recovery under treatment (RCT)
    p_yc: float           # P(y_c), recovery under control (RCT)
    benefit: np.ndarray   # [lower, upper] bounds on P(benefit)
    harm: np.ndarray      # [lower, upper] bounds on P(harm)
    always: np.ndarray    # survive either way, P(y_t, y_c)
    never: np.ndarray     # die either way, P(y'_t, y'_c)


def observational_sample_size(df_obs: pd.DataFrame, sex: int) -> int:
    """Number of patients of the given sex (0 = male) in the survey."""
    return len(df_obs.query(f"is_rct == 0 and sex == {sex}"))


def response_types(p_yt, p_yc, p_benefit, p_harm) -> ResponseTypes:
    """Split a subgroup into the four response types.

    The bounds pin down the split through
        P(y_t) = P(benefit) + P(survives either way)
        P(y_c) = P(harm)    + P(survives either way)
    These are sample estimates, not population values, so the two bounds can
    cross by a hair in a finite sample; the shares are clipped to the [0, 1]
    they have to lie in, and they may not add up to the last decimal.
    """
    return ResponseTypes(
        p_yt=p_yt,
        p_yc=p_yc,
        benefit=p_benefit,
        harm=p_harm,
        always=np.clip(np.sort(p_yc - p_harm), 0, 1),
        never=np.clip(np.sort(1 - p_yt - p_harm), 0, 1),
    )


def print_response_types(types: ResponseTypes, tol: float, label: str = "Male") -> None:
    print(f"{label} response types, estimated from the sample")
    print(f"  benefit (drug saves them):     {fmt_interval(types.benefit, tol)}")
    print(f"  harm (drug kills them):        {fmt_interval(types.harm, tol)}")
    print(f"  survive either way:            {fmt_interval(types.always, tol)}")
    print(f"  die either way:                {fmt_interval(types.never, tol)}")

    crossing = float(types.benefit[0] - types.benefit[1])
    if crossing > 0:
        print(
            f"\nNote: the estimated bounds on P(benefit) cross by {crossing:.4f}; "
            "with this sample\nthe two bounds meet, and the gap is sampling noise."
        )


def plot_not_approved(types: ResponseTypes, n: int, tol: float) -> None:
    """Scenario 1 - the drug is not approved: every sampled male stays
    untreated (c), so the experimental control arm estimates what happens to
    all of them."""
    p_yc = types.p_yc
    p_death_no_drug = 1 - p_yc

    nodes = {
        "root": (
            0.02, 0.50,
            f"Males in the sample\nn = {n:,}",
            NEUTRAL_FACE, NEUTRAL_EDGE,
        ),
        "no_drug": (
            0.26, 0.50,
            "The drug is not approved\nnobody can take it (c)",
            CHOICE_FACE, CHOICE_EDGE,
        ),
        "survived": (
            0.58, 0.72,
            "Survived (y)\n"
            f"P(y_c) = {p_yc:.3f}   n ≈ {round(p_yc * n):,}\n"
            f"   survive either way: {fmt_interval(types.always, tol)}\n"
            f"   would have been harmed: {fmt_interval(types.harm, tol)}",
            SURVIVAL_FACE, SURVIVAL_EDGE,
        ),
        "died": (
            0.58, 0.26,
            "Died (y')\n"
            f"P(y'_c) = {p_death_no_drug:.3f}   "
            f"n ≈ {round(p_death_no_drug * n):,}\n"
            f"   would have benefited: {fmt_interval(types.benefit, tol)}\n"
            f"   die either way: {fmt_interval(types.never, tol)}",
            DEATH_FACE, DEATH_EDGE,
        ),
    }

    edges = [
        ("root", "no_drug", ""),
        ("no_drug", "survived", f"{p_yc:.1%}"),
        ("no_drug", "died", f"{p_death_no_drug:.1%}"),
    ]

    draw_flow(
        nodes,
        edges,
        "Scenario 1 - the drug is not approved: no benefit, no harm",
        figsize=(11.5, 5),
        note=(
            "Nobody is harmed by the drug, but nobody is saved by it either: the "
            f"{fmt_interval(types.benefit, tol)} of these males it would have rescued\n"
            "die with everyone else who cannot survive untreated."
        ),
    )


def plot_prescribed_to_all(types: ResponseTypes, n: int, tol: float) -> None:
    """Scenario 2 - the drug is approved and the physician prescribes it to
    every male who attends, so every sampled male lives out his treated
    outcome (t) and the experimental treatment arm estimates what happens to
    all of them. The deaths split in two: those the drug causes and those it
    cannot prevent."""
    p_yt, p_yc = types.p_yt, types.p_yc
    p_death_all_treated = 1 - p_yt
    p_harm_mid = float(np.mean(types.harm))
    p_never_mid = float(np.mean(types.never))

    nodes = {
        "root": (
            0.01, 0.50,
            f"Males attending the physician\nn = {n:,}",
            NEUTRAL_FACE, NEUTRAL_EDGE,
        ),
        "prescribed": (
            0.29, 0.50,
            "The drug is approved\nand prescribed to all of them (t)",
            CHOICE_FACE, CHOICE_EDGE,
        ),
        "survived": (
            0.62, 0.82,
            "Survived (y)\n"
            f"P(y_t) = {p_yt:.3f}   n ≈ {round(p_yt * n):,}\n"
            f"   saved by the drug: {fmt_interval(types.benefit, tol)}\n"
            f"   survive either way: {fmt_interval(types.always, tol)}",
            SURVIVAL_FACE, SURVIVAL_EDGE,
        ),
        "died_harmed": (
            0.68, 0.45,
            "Died, killed by the drug (y')\n"
            f"P(harm) = {fmt_interval(types.harm, tol)}   {fmt_count(types.harm, n)}\n"
            "   they would have survived untreated",
            DEATH_FACE, DEATH_EDGE,
        ),
        "died_anyway": (
            0.62, 0.12,
            "Died, and would have died anyway (y')\n"
            f"P(y'_t, y'_c) = {fmt_interval(types.never, tol)}   "
            f"{fmt_count(types.never, n)}\n"
            "   the drug changes nothing for them",
            DEATH_FACE, DEATH_EDGE,
        ),
    }

    edges = [
        ("root", "prescribed", ""),
        ("prescribed", "survived", f"{p_yt:.1%}"),
        ("prescribed", "died_harmed", f"{p_harm_mid:.1%}", 0.5),
        ("prescribed", "died_anyway", f"{p_never_mid:.1%}"),
    ]

    draw_flow(
        nodes,
        edges,
        "Scenario 2 - the drug is prescribed to everyone: benefit and harm together",
        note=(
            "Treating everyone collects the whole benefit - "
            f"{fmt_interval(types.benefit, tol)} of these males are saved - but also "
            f"inflicts the whole harm: {fmt_interval(types.harm, tol)} die who would "
            f"have survived untreated.\nSurvival is {p_yt:.3f}, up from "
            f"{p_yc:.3f} in scenario 1, and that single number hides which of "
            f"the {p_death_all_treated:.3f} deaths the drug itself caused."
        ),
        figsize=(11.5, 5.8),
    )


def plot_biomarker(types: ResponseTypes, n: int, tol: float) -> None:
    """Scenario 3 - the biomarker tells the physician which males the drug
    would harm: those are kept off it, every other male is prescribed it.
    Three groups come out of that split. The shares are sample estimates,
    hence the intervals."""
    p_yt, p_yc = types.p_yt, types.p_yc
    p_prescribed = np.clip(np.sort(1 - types.harm), 0, 1)
    p_survival_biomarker = np.clip(np.sort(p_yt + types.harm), 0, 1)

    # Rates inside the prescribed group; the males kept off the drug all survive.
    p_prescribed_mid = float(np.mean(p_prescribed))
    p_survive_on_drug = p_yt / p_prescribed_mid

    nodes = {
        "root": (
            0.01, 0.50,
            f"Males attending the physician\nn = {n:,}",
            NEUTRAL_FACE, NEUTRAL_EDGE,
        ),
        "prescribed": (
            0.27, 0.72,
            "Biomarker: would not be harmed\n→ the drug is prescribed (t)\n"
            f"P = {fmt_interval(p_prescribed, tol)}   {fmt_count(p_prescribed, n)}",
            CHOICE_FACE, CHOICE_EDGE,
        ),
        "withheld": (
            0.27, 0.20,
            "Biomarker: would be harmed\n→ kept off the drug (c)\n"
            f"P(harm) = {fmt_interval(types.harm, tol)}   {fmt_count(types.harm, n)}",
            CHOICE_FACE, CHOICE_EDGE,
        ),
        "survived_on_drug": (
            0.64, 0.87,
            "Survived on the drug (y)\n"
            f"P(y_t) = {p_yt:.3f}   n ≈ {round(p_yt * n):,}\n"
            f"   saved by the drug: {fmt_interval(types.benefit, tol)}\n"
            f"   survive either way: {fmt_interval(types.always, tol)}",
            SURVIVAL_FACE, SURVIVAL_EDGE,
        ),
        "died_on_drug": (
            0.64, 0.55,
            "Died on the drug (y')\n"
            f"P(y'_t, y'_c) = {fmt_interval(types.never, tol)}   "
            f"{fmt_count(types.never, n)}\n"
            "   they die either way, the drug changes nothing",
            DEATH_FACE, DEATH_EDGE,
        ),
        "survived_off_drug": (
            0.64, 0.18,
            "Survived by avoiding it (y)\n"
            f"P(harm) = {fmt_interval(types.harm, tol)}   {fmt_count(types.harm, n)}\n"
            "   every one of them survives, none is harmed",
            SURVIVAL_FACE, SURVIVAL_EDGE,
        ),
    }

    edges = [
        ("root", "prescribed", f"{p_prescribed_mid:.1%}"),
        ("root", "withheld", f"{1 - p_prescribed_mid:.1%}"),
        ("prescribed", "survived_on_drug", f"{p_survive_on_drug:.1%}", 0.35),
        ("prescribed", "died_on_drug", f"{1 - p_survive_on_drug:.1%}", 0.35),
        ("withheld", "survived_off_drug", "100%"),
    ]

    draw_flow(
        nodes,
        edges,
        "Scenario 3 - the physician prescribes by biomarker: benefit without harm",
        note=(
            "Nobody is killed by the drug any more: the "
            f"{fmt_interval(types.harm, tol)} it would have killed survive by avoiding "
            f"it, and the {fmt_interval(types.benefit, tol)} it rescues still take "
            "it.\nSurvival is P(y_t) + P(harm) = "
            f"{fmt_interval(p_survival_biomarker, tol)}, against {p_yt:.3f} when "
            f"every male is treated (scenario 2) and {p_yc:.3f} when none is "
            "(scenario 1)."
        ),
        figsize=(12, 6.5),
    )
