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

# --------------------------------------------------------------------------
# CONSTANTS (EU_CODES, NEIGHBORS, SEX_NAME_MAP, CAUSE_NAME_MAP, COUNTRY_NAME_MAP, etc.)
# … copy exactly from your original script …
# --------------------------------------------------------------------------

EU_CODES = [
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE",
    "IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"
]

NEIGHBORS = {
    "AT":["DE","CZ","SK","HU","SI","IT"],
    "BE":["FR","DE","NL","LU"],
    "BG":["RO","EL"],
    "HR":["SI","HU"],
    "CY":[],
    "CZ":["DE","PL","SK","AT"],
    "DK":["DE"],
    "EE":["LV"],
    "FI":["SE"],
    "FR":["BE","LU","DE","IT","ES"],
    "DE":["DK","PL","CZ","AT","FR","LU","BE","NL"],
    "EL":["BG"],
    "HU":["AT","SK","RO","HR"],
    "IE":[],
    "IT":["FR","AT","SI"],
    "LV":["EE","LT"],
    "LT":["LV","PL"],
    "LU":["BE","DE","FR"],
    "MT":[],
    "NL":["BE","DE"],
    "PL":["DE","CZ","SK","LT"],
    "PT":["ES"],
    "RO":["BG","HU"],
    "SK":["CZ","PL","HU","AT"],
    "SI":["IT","AT","HU","HR"],
    "ES":["FR","PT"],
    "SE":["FI"]
}

SEX_NAME_MAP = {"T": "Total", "M": "Male", "F": "Female"}
REV_SEX_NAME = {v: k for k, v in SEX_NAME_MAP.items()}

CAUSE_NAME_MAP = {
    # … all your cause mappings …
    "TOTAL":"Total",
    "A_B":"Certain infectious and parasitic diseases (A00-B99)",
    # etc.
    "U072":"COVID-19, virus not identified"
}
REV_CAUSE_NAME_MAP = {v: k for k, v in CAUSE_NAME_MAP.items()}

COUNTRY_NAME_MAP = {c.alpha_2: c.name for c in pycountry.countries}
COUNTRY_NAME_MAP.update({
    "FX":"France (Metropolitan)",
    "EU":"European Union",
    "Europe":"Europe"
})
REV_COUNTRY_NAME_MAP = {v: k for k, v in COUNTRY_NAME_MAP.items()}

FACTOR_IDS = {
    # … your factor IDs …
    "Unmet needs": "hlth_ehis_un1e"
}

def alpha3_from_a2(a2: str):
    c = pycountry.countries.get(alpha_2=a2)
    return c.alpha_3 if c else None

# --------------------------------------------------------------------------
# DATA-LOADING UTILITIES (load_eurostat_series, load_data, load_all_factors)
# JOINPOINT/FORECAST/CLUSTER FUNCTIONS
# … copy exactly from your original script …
# --------------------------------------------------------------------------

@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    # … unchanged …

@st.cache_data
def load_data() -> pd.DataFrame:
    # … unchanged …

@st.cache_data
def load_all_factors() -> pd.DataFrame:
    # … unchanged …

def detect_change_points(ts, pen: float = 3) -> list:
    # … unchanged …

def compute_joinpoints_and_apc(df_sub: pd.DataFrame) -> pd.DataFrame:
    # … unchanged …

def plot_joinpoints_comparative(df_sub: pd.DataFrame, title: str):
    # … unchanged …

def plot_segmented_fit_series(df_sub: pd.DataFrame, title: str):
    # … unchanged …

def get_prophet_forecast(df_sub: pd.DataFrame, periods: int) -> pd.DataFrame:
    # … unchanged …

def get_arima_forecast(df_sub: pd.DataFrame, periods: int) -> pd.DataFrame:
    # … unchanged …

def get_ets_forecast(df_sub: pd.DataFrame, periods: int) -> pd.DataFrame:
    # … unchanged …

def forecast_mortality(df_sub: pd.DataFrame, periods: int, method: str, title: str):
    # … unchanged …

# --------------------------------------------------------------------------
# NEW: BIC-based Bayes‐factor approximation
# --------------------------------------------------------------------------
def compute_bayes_factor_bic(pair_df: pd.DataFrame, maxlag: int) -> float:
    """
    Approximate BF10 via BIC comparison of:
      - Null:    target_t ~ 1
      - Alternate: target_t ~ lags 1…maxlag of cause_t
    """
    df = pair_df.dropna()
    if df.shape[0] < maxlag + 3:
        return np.nan

    target, cause = df.columns
    Y = df[target].values[maxlag:]
    # Build design for alternate model
    X_alt = np.column_stack([
        df[cause].values[maxlag - lag:-lag] for lag in range(1, maxlag + 1)
    ])
    X_alt = sm.add_constant(X_alt)
    # Null model: intercept only
    X_null = np.ones((len(Y), 1))

    m0 = sm.OLS(Y, X_null).fit()
    m1 = sm.OLS(Y, X_alt).fit()

    bic0, bic1 = m0.bic, m1.bic
    # BF10 ≈ exp((BIC_null - BIC_alt)/2)
    return float(np.exp((bic0 - bic1) / 2.0))

# --------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="European Public Health Dashboard")
    st.title("European Public Health Dashboard")
    st.markdown("by Younes Adam Tabi")

    # Load & label mortality data
    df = load_data()
    df["CountryFull"] = df["Country"].map(COUNTRY_NAME_MAP)
    df["CauseFull"]   = df["Cause"].map(CAUSE_NAME_MAP)
    df["SexFull"]     = df["Sex"].map(SEX_NAME_MAP)

    # Sidebar controls (unchanged)…
    countries    = sorted(df["CountryFull"].dropna().unique())
    country_full = st.sidebar.selectbox("Country", countries, index=countries.index("European Union"))
    country_code = REV_COUNTRY_NAME_MAP.get(country_full, country_full)
    causes       = sorted(df[df["Country"]==country_code]["CauseFull"].dropna().unique())
    cause_full   = st.sidebar.selectbox("Cause of Death", causes)
    cause_code   = REV_CAUSE_NAME_MAP.get(cause_full, cause_full)
    sex_sel      = st.sidebar.multiselect("Sex", ["Total","Male","Female"], default=["Total"])
    sex_codes    = [REV_SEX_NAME[s] for s in sex_sel]
    yrs          = sorted(df["Year"].unique())
    year_range   = st.sidebar.slider("Historical Years", yrs[0], yrs[-1], (yrs[0], yrs[-1]))
    forecast_yrs = st.sidebar.slider("Forecast Horizon (yrs)", 1, 30, 10)
    method       = st.sidebar.selectbox("Forecast Method", ["Prophet","ARIMA","ETS","Ensemble"])

    # … Mortality trends, forecasts, health‐factor regression, cluster analysis …
    # (copy all of your existing sections up through cluster analysis unchanged)

    # --- Global Bayesian Causality ------------------------------------------
    st.markdown("---")
    st.header("Global Bayesian Causality")
    st.markdown(
        "For each pair (A→B), we compare:\n"
        "- **Null**:  Bₜ ~ 1\n"
        "- **Alt**:   Bₜ ~ Aₜ₋₁…ₜ₋ₗₐg\n"
        "and approximate Bayes‐factor BF₁₀ = exp((BIC_null – BIC_alt)/2). "
        "Heatmap shows log₁₀(BF₁₀), and directed arrows are drawn when BF₁₀ ≥ threshold."
    )

    country_list = sorted(df["CountryFull"].dropna().unique())
    sel_countries = st.multiselect("Select countries (default: all)", country_list, default=country_list)
    gl_maxlag = st.slider("Max lag (yrs)", 1, 5, 2, key="gl_lag_bf")
    bf_thresh = st.number_input("BF₁₀ cutoff for arrow", 1.0, 100.0, 3.0, 0.5)

    if len(sel_countries) >= 2:
        df_g = df[
            (df["Cause"] == cause_code) &
            (df["CountryFull"].isin(sel_countries)) &
            (df["Sex"] == "T") &
            (df["Year"].between(*year_range))
        ]
        pivot_gc = df_g.pivot_table(index="Year", columns="CountryFull", values="Rate", aggfunc="mean")
        common = [c for c in sel_countries if c in pivot_gc.columns]

        if len(common) >= 2:
            bf_mat = pd.DataFrame(np.nan, index=common, columns=common)
            with st.spinner("Computing Bayes‐factors…"):
                for causer in common:
                    for caused in common:
                        if causer == caused:
                            continue
                        pair = pivot_gc[[caused, causer]]
                        bf_mat.loc[causer, caused] = compute_bayes_factor_bic(pair, gl_maxlag)

            # Heatmap of log10(BF₁₀)
            fig_hm = px.imshow(
                np.log10(bf_mat),
                text_auto=".2f",
                labels={"x":"Predictor →","y":"Target ↓","color":"log₁₀(BF₁₀)"},
                title="Global Bayesian Causality (log₁₀ BF₁₀)"
            )
            st.plotly_chart(fig_hm)

            # Directed network
            edges = [
                (i, j)
                for i in common for j in common
                if i != j
                and pd.notna(bf_mat.loc[i, j])
                and bf_mat.loc[i, j] >= bf_thresh
            ]
            theta = np.linspace(0, 2*np.pi, len(common), endpoint=False)
            pos = {n:(np.cos(t), np.sin(t)) for n,t in zip(common, theta)}

            fig_net = go.Figure()
            # nodes
            nx_, ny_ = zip(*(pos[n] for n in common))
            fig_net.add_trace(go.Scatter(
                x=nx_, y=ny_,
                mode="markers+text",
                marker=dict(size=20),
                text=common,
                textposition="bottom center",
                hoverinfo="text"
            ))
            # arrows
            for src, dst in edges:
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                fig_net.add_annotation(
                    x=x1, y=y1, ax=x0, ay=y0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3, arrowsize=1, arrowwidth=1
                )
            fig_net.update_layout(
                title=f"Global Network (BF₁₀ ≥ {bf_thresh})",
                xaxis=dict(visible=False), yaxis=dict(visible=False), height=600
            )
            st.plotly_chart(fig_net)

    # --- Neighbor‐Based Bayesian Causality ----------------------------------
    st.markdown("---")
    st.header("Neighbor‐Based Bayesian Causality")
    st.markdown(
        "Same BF₁₀ approach, but focused on one country and its immediate neighbors."
    )

    base_full  = st.selectbox("Focal country", country_list, index=country_list.index("Germany"))
    base_code  = REV_COUNTRY_NAME_MAP.get(base_full, base_full)
    nbr_codes  = NEIGHBORS.get(base_code, [])
    map_df     = pd.DataFrame({
        "Country":[base_code]+nbr_codes,
        "Role":   ["Focal"]+["Neighbor"]*len(nbr_codes)
    })
    map_df["CountryFull"] = map_df["Country"].map(COUNTRY_NAME_MAP)
    map_df["iso_alpha"]   = map_df["Country"].apply(alpha3_from_a2)
    st.plotly_chart(px.choropleth(
        map_df, locations="iso_alpha", color="Role",
        hover_name="CountryFull", locationmode="ISO-3",
        scope="europe", title="Focal & Neighbors"
    ))

    if nbr_codes:
        gb      = [base_code] + nbr_codes
        df_n    = df[
            (df["Cause"] == cause_code) &
            (df["Country"].isin(gb)) &
            (df["Sex"] == "T") &
            (df["Year"].between(*year_range))
        ]
        pivot_n = df_n.pivot_table(index="Year", columns="Country", values="Rate", aggfunc="mean")
        common_codes = [c for c in gb if c in pivot_n.columns]

        if len(common_codes) >= 2:
            nbr_maxlag  = st.slider("Neighbor max lag (yrs)", 1, 5, 2, key="nbr_lag_bf_nbr")
            nbr_bf_thresh = st.number_input("Neighbor BF₁₀ cutoff", 1.0, 100.0, 3.0, 0.5, key="nbr_bf_thresh")

            bf_n = pd.DataFrame(np.nan, index=common_codes, columns=common_codes)
            with st.spinner("Computing neighbor Bayes‐factors…"):
                for causer in common_codes:
                    for caused in common_codes:
                        if causer == caused:
                            continue
                        pair = pivot_n[[caused, causer]]
                        bf_n.loc[causer, caused] = compute_bayes_factor_bic(pair, nbr_maxlag)

            # Heatmap
            labels = {c: COUNTRY_NAME_MAP[c] for c in common_codes}
            fig_hm_n = px.imshow(
                np.log10(bf_n.rename(index=labels, columns=labels)),
                text_auto=".2f",
                labels={"x":"Predictor →","y":"Target ↓","color":"log₁₀(BF₁₀)"},
                title="Neighbor-Based Bayesian Heatmap"
            )
            st.plotly_chart(fig_hm_n)

            # Directed neighbor network
            edges_n = [
                (i, j)
                for i in common_codes for j in common_codes
                if i != j
                and pd.notna(bf_n.loc[i, j])
                and bf_n.loc[i, j] >= nbr_bf_thresh
            ]
            full_names = [COUNTRY_NAME_MAP[c] for c in common_codes]
            theta_n = np.linspace(0, 2*np.pi, len(full_names), endpoint=False)
            pos_n = {n:(np.cos(t),np.sin(t)) for n,t in zip(full_names, theta_n)}

            fig_net_n = go.Figure()
            x_n, y_n = zip(*(pos_n[n] for n in full_names))
            fig_net_n.add_trace(go.Scatter(
                x=x_n, y=y_n,
                mode="markers+text",
                marker=dict(size=20),
                text=full_names,
                textposition="bottom center"
            ))
            for src, dst in edges_n:
                x0, y0 = pos_n[COUNTRY_NAME_MAP[src]]
                x1, y1 = pos_n[COUNTRY_NAME_MAP[dst]]
                fig_net_n.add_annotation(
                    x=x1, y=y1, ax=x0, ay=y0,
                    showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1
                )
            fig_net_n.update_layout(
                title=f"Neighbor Network (BF₁₀ ≥ {nbr_bf_thresh})",
                xaxis=dict(visible=False), yaxis=dict(visible=False), height=600
            )
            st.plotly_chart(fig_net_n)

    st.markdown("---")
    st.info("Adjust the BF₁₀ cutoffs to tune how strong the evidence must be before drawing an arrow.")

if __name__ == "__main__":
    main()
