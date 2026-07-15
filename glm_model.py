import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from linear_model import make_fake_dataset, plot_actual_vs_predicted, plot_3d_regression


def fit_score(actual, predicted):
    resid = actual - predicted
    rmse = np.sqrt(np.mean(resid**2))
    mae = np.mean(np.abs(resid))
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return rmse, mae, r2


def compare_on_gamma_scale(actual_gamma, entries):
    """entries: iterable of (label, model, family_link, predicted) tuples."""
    rows = []
    for label, model, family_link, predicted in entries:
        rmse, mae, r2 = fit_score(actual_gamma, predicted)
        # Prefer the log-likelihood-based BIC (bic_llf) so GLM and OLS report the
        # same formula; GLMResults exposes it, OLSResults' plain .bic already is one.
        bic = getattr(model, "bic_llf", model.bic)
        rows.append(
            {
                "Model": label,
                "Formula": model.model.formula,
                "Family/Link": family_link,
                "RMSE (y_gamma)": rmse,
                "MAE (y_gamma)": mae,
                "R-squared (y_gamma)": r2,
                "Native AIC": model.aic,
                "Native BIC": bic,
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def main():
    df = make_fake_dataset()
    df["y_gamma"] = np.exp(df["y"])

    linear_model = smf.ols(formula="y ~ x + z", data=df).fit()
    print(linear_model.summary())

    glm_model = smf.glm(
        formula="y_gamma ~ x + z",
        data=df,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    print(glm_model.summary())

    glm_pred_gamma = glm_model.fittedvalues
    linear_pred_gamma = np.exp(linear_model.fittedvalues)

    plot_actual_vs_predicted(
        df["y_gamma"],
        glm_pred_gamma,
        title=f"Actual vs. Predicted y_gamma (GLM: {glm_model.model.formula})",
    )
    plot_3d_regression(
        df,
        glm_model,
        response="y_gamma",
        title=f"GLM Gamma-Log Surface: {glm_model.model.formula}",
    )

    plot_actual_vs_predicted(
        df["y_gamma"],
        linear_pred_gamma,
        title=f"Actual vs. Predicted y_gamma (Linear, exp-transformed: {linear_model.model.formula})",
    )
    plot_3d_regression(
        df,
        linear_model,
        response="y_gamma",
        predict_transform=np.exp,
        title=f"Linear Model Surface (exp-transformed): {linear_model.model.formula}",
    )

    comparison = compare_on_gamma_scale(
        df["y_gamma"],
        [
            ("GLM (Gamma, log link)", glm_model, "Gamma/log", glm_pred_gamma),
            (
                "Linear, exp-transformed",
                linear_model,
                "Gaussian/identity",
                linear_pred_gamma,
            ),
        ],
    )
    print("\nModel comparison (predictions compared on the y_gamma = exp(y) scale):")
    print(comparison.round(4).to_string())
    print(
        "\nNote: Native AIC/BIC come from different likelihoods (Gamma on y_gamma vs. "
        "Gaussian on y) and are not directly comparable across models; RMSE/MAE/R-squared "
        "above are computed on the common y_gamma scale and are the fair comparison."
    )


if __name__ == "__main__":
    main()
