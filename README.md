# Personalized Decision Making

A **CausalPython demo** on Individual Treatment Effects (ITEs). It is based
entirely on Scott Mueller and Judea Pearl's paper,
[*Personalized Decision Making – A Conceptual Introduction*](https://doi.org/10.1515/jci-2022-0050)
(Journal of Causal Inference, 2023). Its examples, numbers and formulas all
come from that paper. The bounds it uses come from Tian & Pearl (2000).

## The core idea

An ITE is a rung-3 (counterfactual) quantity. It is the difference between one
individual's outcomes under two alternative interventions:

$$\text{ITE}(u) = Y_t(u) - Y_c(u) \qquad \text{vs.} \qquad \text{CATE}(c) = E[Y_t - Y_c \mid C = c]$$

Equal averages can hide harm. A randomized trial identifies only
$\text{ATE} = P(\text{benefit}) - P(\text{harm})$. Combining it with
observational data bounds each term for a given group:

|                                        | Female | Male |
|----------------------------------------|--------|------|
| CATE, from the trial                   | 0.28   | 0.28 |
| Recovery among those who chose the drug| 0.27   | 0.70 |
| P(benefit)                             | 0.28   | 0.49 |
| P(harm)                                | 0.00   | 0.21 |

Female and male patients have the same CATE, but about 21% of male patients
would survive without the drug and die with it. For female patients, the drug
harms no one.

## Notebooks

- **`ITE_Numerical_Example.ipynb`** reproduces the paper's example. It takes
  the published trial and survey tables and computes the Tian–Pearl bounds on
  P(benefit) and P(harm) for female and male patients.
- **`ITE_Data_Generation_Example.ipynb`** simulates a trial and an
  observational study, then estimates benefit and harm as intervals from those
  samples. It ends by comparing three policies for 2,000 simulated male
  patients: the drug is not approved, it is prescribed to everyone, or it is
  withheld from those it would harm.

The `results_analysis/` package holds the tables and diagrams used in the
notebooks' results sections. [`docs/IndividualTreatmentEffects.pdf`](docs/IndividualTreatmentEffects.pdf)
is a short slide overview of the demo.

## Getting started

You need [uv](https://docs.astral.sh/uv/) installed. The project requires
Python ≥ 3.14, and uv downloads it if it isn't already on your machine.

```bash
git clone https://github.com/HornerAlberto/personalized-decision-making.git
cd personalized-decision-making
uv sync
```

`uv sync` creates a `.venv` with all dependencies. To work in VS Code, open a
notebook and pick the `.venv` kernel. To use Jupyter instead, run:

```bash
uv run --with jupyter jupyter lab
```

## Reference

Mueller, S., & Pearl, J. (2023). Personalized decision making – A conceptual
introduction. *Journal of Causal Inference*, 11(1), 20220050.

## License

[MIT](LICENSE)
