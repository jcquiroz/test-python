import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"


def make_fake_dataset(n=100, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0, scale=1, size=n)
    z = rng.normal(loc=5, scale=2, size=n)
    noise = rng.normal(loc=0, scale=1, size=n)
    y = 3 + 2 * x - 0.5 * z + noise
    return pd.DataFrame({"x": x, "y": y, "z": z})


def plot_actual_vs_predicted(actual, predicted, title="Actual vs. Predicted y"):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2, color=BASELINE, zorder=1)
    ax.text(hi, hi, "y = x", color=INK_MUTED, fontsize=9, ha="right", va="bottom")

    ax.scatter(
        actual,
        predicted,
        s=36,
        color=SERIES_1,
        alpha=0.85,
        edgecolors=SURFACE,
        linewidths=0.5,
        zorder=2,
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.set_title(title, color=INK_PRIMARY, fontsize=13, pad=12)
    ax.set_xlabel("Actual y", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Predicted y", color=INK_SECONDARY, fontsize=10)

    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    fig.tight_layout()
    plt.show()


def plot_3d_regression(
    df, model, grid_size=20, response="y", predict_transform=None, title=None
):
    x_seq = np.linspace(df["x"].min(), df["x"].max(), grid_size)
    z_seq = np.linspace(df["z"].min(), df["z"].max(), grid_size)
    x_grid, z_grid = np.meshgrid(x_seq, z_seq)

    raw_pred = model.predict(
        pd.DataFrame({"x": x_grid.ravel(), "z": z_grid.ravel()})
    ).to_numpy()
    if predict_transform is not None:
        raw_pred = predict_transform(raw_pred)
    y_grid = raw_pred.reshape(x_grid.shape)

    fig = plt.figure(figsize=(7, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor(SURFACE)

    ax.plot_surface(
        x_grid,
        z_grid,
        y_grid,
        color=SERIES_1,
        alpha=0.35,
        linewidth=0,
        antialiased=True,
        zorder=1,
    )

    ax.scatter(
        df["x"],
        df["z"],
        df[response],
        s=30,
        color=SERIES_2,
        edgecolors=SURFACE,
        linewidths=0.4,
        depthshade=True,
        zorder=2,
    )

    ax.set_title(
        title or f"Regression Plane: {model.model.formula}",
        color=INK_PRIMARY,
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("x", color=INK_SECONDARY, fontsize=10, labelpad=8)
    ax.set_ylabel("z", color=INK_SECONDARY, fontsize=10, labelpad=8)
    ax.set_zlabel(response, color=INK_SECONDARY, fontsize=10, labelpad=8)
    ax.tick_params(colors=INK_MUTED, labelsize=8)

    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(SURFACE)
        pane.pane.set_edgecolor(GRIDLINE)
        pane.line.set_color(BASELINE)

    fig.tight_layout()
    plt.show()


def compare_models(models):
    rows = []
    for model in models:
        rmse = np.sqrt(np.mean(model.resid**2))
        rows.append(
            {
                "Formula": model.model.formula,
                "R-squared": model.rsquared,
                "Adj. R-squared": model.rsquared_adj,
                "RMSE": rmse,
                "AIC": model.aic,
                "BIC": model.bic,
                "Log-Likelihood": model.llf,
                "F-statistic": model.fvalue,
                "Df Model": model.df_model,
            }
        )
    return pd.DataFrame(rows).set_index("Formula")


def main():
    df = make_fake_dataset()

    model = smf.ols(formula="y ~ x + z", data=df).fit()
    print(model.summary())

    plot_actual_vs_predicted(
        df["y"],
        model.fittedvalues,
        title=f"Actual vs. Predicted y ({model.model.formula})",
    )
    plot_3d_regression(df, model)

    model_interaction = smf.ols(formula="y ~ x*z", data=df).fit()
    print(model_interaction.summary())

    plot_actual_vs_predicted(
        df["y"],
        model_interaction.fittedvalues,
        title=f"Actual vs. Predicted y ({model_interaction.model.formula})",
    )
    plot_3d_regression(df, model_interaction)

    comparison = compare_models([model, model_interaction])
    print("\nModel comparison:")
    print(comparison.round(4).to_string())


if __name__ == "__main__":
    main()
