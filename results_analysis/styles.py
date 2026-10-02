"""Shared colours: green for no harm / survival, red for harm / death,
blue for the experimental CATE and for choices, gold for P(benefit)."""

# Table cells (pandas Styler CSS).
NO_HARM_CSS = "background-color: #d6efd6; color: #14532d; font-weight: 600"
HARM_CSS = "background-color: #fadcd9; color: #7f1d1d; font-weight: 600"
CATE_CSS = "background-color: #d8e6fb; color: #1e3a8a; font-weight: 600"
BENEFIT_CSS = "background-color: #f8e7b2; color: #78500f; font-weight: 600"

# Flow-diagram boxes (face colour, edge colour).
CHOICE_FACE, CHOICE_EDGE = "#d8e6fb", "#1e3a8a"
SURVIVAL_FACE, SURVIVAL_EDGE = "#d6efd6", "#14532d"
DEATH_FACE, DEATH_EDGE = "#fadcd9", "#7f1d1d"
NEUTRAL_FACE, NEUTRAL_EDGE = "#f0f0f0", "#444444"
