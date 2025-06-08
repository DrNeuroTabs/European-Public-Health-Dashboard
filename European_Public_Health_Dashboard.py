import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters
import requests
import gzip
from io import BytesIO
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pycountry
from requests.exceptions import HTTPError

# NEW Bayesian imports
import pymc3 as pm
import arviz as az

# --------------------------------------------------------------------------
# (All your constants: EU_CODES, NEIGHBORS, SEX_NAME_MAP, CAUSE_NAME_MAP, etc.)
# … [unchanged from your original script] …
# --------------------------------------------------------------------------

def compute_bayes_factor(pair_df: pd.DataFrame,
                         maxlag: int,
                         prior_sd: float = 1.0,
                         draws: int = 1000,
                         tune: int = 1000) -> float:
    """
    Savage–Dickey Bayes‐factor approximation for 'causer → caused'.
    We regress causedₜ on causerₜ₋₁...ₜ₋ₗₐg, put Normal(0,prior_sd) priors,
    then BF₁₀ ≈ max_posterior_sd / prior_sd.
    """
    # Prepare series
    caused, causer = pair_df.columns
    df = pair_df.dropna()
    if len(df) < maxlag + 5:
        return np.nan

    # Build X matrix of lags of causer
    Ys = df[caused].values[maxlag:]
    Xs = np.column_stack([
        df[causer].values[maxlag - lag:-lag]
        for lag in range(1, maxlag + 1)
    ])

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=prior_sd, shape=maxlag)
        sigma = pm.HalfCauchy("sigma", beta=1)
        mu = pm.math.dot(Xs, beta)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=Ys)
        trace = pm.sample(
            draws,
            tune=tune,
            cores=1,
            progressbar=False,
            target_accept=0.9
        )

    # posterior SD of each lag‐coefficient
    posterior_sd = trace["beta"].std(axis=0)
    # Bayes‐factor in favor of H₁ (coeff ≠ 0)
    bf10_per_lag = posterior_sd / prior_sd
    return float(bf10_per_lag.max())


def main():
    st.set_page_config(layout="wide", page_title="European Public Health Dashboard")
    st.title("European Public Health Dashboard")
    st.markdown("by Younes Adam Tabi")

    # … [all your data‐loading, sidebar filters, joinpoints, forecasts, regression,
    #     clustering sections remain unchanged] …

    # --- Global Bayesian Causality ------------------------------------------
    st.markdown("---")
    st.header("Global Bayesian Causality")
    st.markdown(
        "We fit, for each pair (A→B), a Bayesian regression of Bₜ on Aₜ₋₁…ₜ₋ₗₐg, "
        "compute Savage–Dickey Bayes-factor (BF₁₀) for A→B, and show log₁₀(BF₁₀) in a heatmap "
        "and directed network (arrowhead for A→B when BF₁₀ ≥ threshold)."
    )

    country_list = sorted(df["CountryFull"].dropna().unique())
    sel_countries = st.multiselect("Select countries (default: all)",
                                   country_list, default=country_list)
    gl_maxlag = st.slider("Max lag (yrs)", 1, 5, 2, key="gl_lag_bayes")
    bf_thresh = st.number_input("BF₁₀ cutoff for an arrow", 1.0, 100.0, 3.0, 0.5)

    if len(sel_countries) >= 2:
        df_g = df[
            (df["Cause"] == cause_code) &
            (df["CountryFull"].isin(sel_countries)) &
            (df["Sex"] == "T") &
            (df["Year"].between(*year_range))
        ]
        pivot_gc = df_g.pivot_table(
            index="Year", columns="CountryFull", values="Rate", aggfunc="mean"
        )
        common = [c for c in sel_countries if c in pivot_gc.columns]

        if len(common) >= 2:
            bf_mat = pd.DataFrame(np.nan, index=common, columns=common)
            with st.spinner("Sampling Bayesian models… this can take a bit"):
                for causer in common:
                    for caused in common:
                        if causer == caused:
                            continue
                        data_pair = pivot_gc[[caused, causer]].dropna()
                        bf = compute_bayes_factor(data_pair, gl_maxlag)
                        bf_mat.loc[causer, caused] = bf

            # Heatmap of log10 BF₁₀
            fig_hm = px.imshow(
                np.log10(bf_mat),
                text_auto=".2f",
                labels={
                    "x": "Predictor →",
                    "y": "Target ↓",
                    "color": "log₁₀(BF₁₀)"
                },
                title="Global Bayesian Causality (log₁₀ BF₁₀)"
            )
            st.plotly_chart(fig_hm)

            # Directed network
            edges = [
                (i, j)
                for i in common for j in common
                if i != j and pd.notna(bf_mat.loc[i, j]) and bf_mat.loc[i, j] >= bf_thresh
            ]
            theta = np.linspace(0, 2 * np.pi, len(common), endpoint=False)
            pos = {n: (np.cos(t), np.sin(t)) for n, t in zip(common, theta)}

            fig_net = go.Figure()
            # 1) nodes
            nx_, ny_ = zip(*(pos[n] for n in common))
            fig_net.add_trace(go.Scatter(
                x=nx_, y=ny_,
                mode="markers+text",
                marker=dict(size=20),
                text=common,
                textposition="bottom center",
                hoverinfo="text"
            ))
            # 2) directed arrows
            for src, dst in edges:
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                fig_net.add_annotation(
                    x=x1, y=y1, ax=x0, ay=y0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="black"
                )
            fig_net.update_layout(
                title=f"Global Bayesian Network (BF₁₀ ≥ {bf_thresh})",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=600
            )
            st.plotly_chart(fig_net)

    # --- Neighbor‐Based Bayesian Causality ----------------------------------
    st.markdown("---")
    st.header("Neighbor‐Based Bayesian Causality")
    st.markdown(
        "Same Bayes‐factor approach, but focussing on one country and its immediate neighbors."
    )
    base_full = st.selectbox("Focal country", country_list,
                             index=country_list.index("Germany"))
    base_code = REV_COUNTRY_NAME_MAP.get(base_full, base_full)
    nbr_codes = NEIGHBORS.get(base_code, [])
    map_df = pd.DataFrame({
        "Country": [base_code] + nbr_codes,
        "Role":   ["Focal"] + ["Neighbor"] * len(nbr_codes)
    })
    map_df["CountryFull"] = map_df["Country"].map(COUNTRY_NAME_MAP)
    map_df["iso_alpha"]   = map_df["Country"].apply(alpha3_from_a2)
    st.plotly_chart(px.choropleth(
        map_df, locations="iso_alpha", color="Role",
        hover_name="CountryFull", locationmode="ISO-3",
        scope="europe", title="Focal & Neighbors"
    ))

    if nbr_codes:
        gb = [base_code] + nbr_codes
        df_n = df[
            (df["Cause"] == cause_code) &
            (df["Country"].isin(gb)) &
            (df["Sex"] == "T") &
            (df["Year"].between(*year_range))
        ]
        pivot_n = df_n.pivot_table(
            index="Year", columns="Country", values="Rate", aggfunc="mean"
        )
        common_codes = [c for c in gb if c in pivot_n.columns]
        if len(common_codes) >= 2:
            nbr_maxlag = st.slider("Neighbor max lag (yrs)", 1, 5, 2, key="nbr_lag_bayes")
            nbr_bf_thresh = st.number_input("Neighbor BF₁₀ cutoff", 1.0, 100.0, 3.0, 0.5, key="nbr_bf")

            bf_n = pd.DataFrame(np.nan, index=common_codes, columns=common_codes)
            with st.spinner("Sampling neighbor Bayesian models…"):
                for causer in common_codes:
                    for caused in common_codes:
                        if causer == caused:
                            continue
                        pair = pivot_n[[caused, causer]].dropna()
                        bf_n.loc[causer, caused] = compute_bayes_factor(pair, nbr_maxlag)

            fig_hm_n = px.imshow(
                np.log10(bf_n.rename(index=map(lambda c: COUNTRY_NAME_MAP[c], bf_n.index),
                                      columns=map(lambda c: COUNTRY_NAME_MAP[c], bf_n.columns))),
                text_auto=".2f",
                labels={
                    "x": "Predictor →",
                    "y": "Target ↓",
                    "color": "log₁₀(BF₁₀)"
                },
                title="Neighbor-Based Bayesian Heatmap"
            )
            st.plotly_chart(fig_hm_n)

            edges_n = [
                (i, j)
                for i in common_codes for j in common_codes
                if i != j and pd.notna(bf_n.loc[i, j]) and bf_n.loc[i, j] >= nbr_bf_thresh
            ]
            common_full = [COUNTRY_NAME_MAP[c] for c in common_codes]
            theta_n = np.linspace(0, 2 * np.pi, len(common_full), endpoint=False)
            pos_n = {n: (np.cos(t), np.sin(t)) for n, t in zip(common_full, theta_n)}

            fig_net_n = go.Figure()
            x_n, y_n = zip(*(pos_n[n] for n in common_full))
            fig_net_n.add_trace(go.Scatter(
                x=x_n, y=y_n,
                mode="markers+text",
                marker=dict(size=20),
                text=common_full,
                textposition="bottom center"
            ))
            for src, dst in edges_n:
                x0, y0 = pos_n[COUNTRY_NAME_MAP[src]]
                x1, y1 = pos_n[COUNTRY_NAME_MAP[dst]]
                fig_net_n.add_annotation(
                    x=x1, y=y1, ax=x0, ay=y0,
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1, arrowwidth=1
                )
            fig_net_n.update_layout(
                title=f"Neighbor Network (BF₁₀ ≥ {nbr_bf_thresh})",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False), height=600
            )
            st.plotly_chart(fig_net_n)

    st.markdown("---")
    st.info(
        "Adjust your Bayes-factor cutoff to tune how strong the evidence for causality "
        "must be before drawing an arrow."
    )


if __name__ == "__main__":
    # make sure all your variables like `cause_code`, `year_range`, `df`, etc.
    # are in scope here (as in your original `main()`), or refactor accordingly.
    main()
