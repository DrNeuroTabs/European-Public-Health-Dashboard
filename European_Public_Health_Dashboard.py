import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters
import requests
import gzip
from io import BytesIO
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pycountry
from requests.exceptions import HTTPError
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# --------------------------------------------------------------------------
# CONSTANTS
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
    "TOTAL": "Total",
    "A_B": "Certain infectious and parasitic diseases (A00-B99)",
    "A15-A19_B90": "Tuberculosis",
    "B15-B19_B942": "Viral hepatitis and sequelae of viral hepatitis",
    "B180-B182": "Chronic viral hepatitis B and C",
    "B20-B24": "Human immunodeficiency virus [HIV] disease",
    "A_B_OTH": "Other infectious and parasitic diseases (A00-B99)",
    "C00-D48": "Neoplasms",
    "C": "Malignant neoplasms (C00-C97)",
    "C00-C14": "Malignant neoplasm of lip, oral cavity, pharynx",
    "C15": "Malignant neoplasm of oesophagus",
    "C16": "Malignant neoplasm of stomach",
    "C18-C21": "Malignant neoplasm of colon, rectum, anus",
    "C22": "Malignant neoplasm of liver and intrahepatic bile ducts",
    "C25": "Malignant neoplasm of pancreas",
    "C32": "Malignant neoplasm of larynx",
    "C33_C34": "Malignant neoplasm of trachea, bronchus and lung",
    "C43": "Malignant melanoma of skin",
    "C50": "Malignant neoplasm of breast",
    "C53": "Malignant neoplasm of cervix uteri",
    "C54_C55": "Malignant neoplasm of other parts of uterus",
    "C56": "Malignant neoplasm of ovary",
    "C61": "Malignant neoplasm of prostate",
    "C64": "Malignant neoplasm of kidney, except renal pelvis",
    "C67": "Malignant neoplasm of bladder",
    "C70-C72": "Malignant neoplasm of brain and CNS",
    "C73": "Malignant neoplasm of thyroid gland",
    "C81-C86": "Hodgkin disease and lymphomas",
    "C88_C90_C96": "Other lymphoid & haematopoietic neoplasms",
    "C91-C95": "Leukaemia",
    "C_OTH": "Other malignant neoplasms (C00-C97)",
    "D00-D48": "Non-malignant neoplasms",
    "D50-D89": "Diseases of blood & blood-forming organs",
    "E": "Endocrine, nutritional & metabolic diseases",
    "E10-E14": "Diabetes mellitus",
    "E_OTH": "Other endocrine, nutritional & metabolic diseases",
    "F": "Mental & behavioural disorders",
    "F01_F03": "Dementia",
    "F10": "Alcohol-related mental disorders",
    "TOXICO": "Drug dependence & toxicomania",
    "F_OTH": "Other mental & behavioural disorders",
    "G_H": "Nervous system & sense organs diseases",
    "G20": "Parkinson disease",
    "G30": "Alzheimer disease",
    "G_H_OTH": "Other nervous system & sense organ diseases",
    "I": "Circulatory system diseases",
    "I20-I25": "Ischaemic heart diseases",
    "I21_I22": "Acute myocardial infarction",
    "I20_I23-I25": "Other ischaemic heart diseases",
    "I30-I51": "Other heart diseases",
    "I60-I69": "Cerebrovascular diseases",
    "I_OTH": "Other circulatory diseases",
    "J": "Respiratory system diseases",
    "J09-J11": "Influenza (including swine flu)",
    "J12-J18": "Pneumonia",
    "J40-J47": "Chronic lower respiratory diseases",
    "J45_J46": "Asthma",
    "J40-J44_J47": "Other lower respiratory diseases",
    "J_OTH": "Other respiratory diseases",
    "K": "Digestive system diseases",
    "K25-K28": "Ulcer of stomach & duodenum",
    "K70_K73_K74": "Chronic liver disease",
    "K72-K75": "Other liver diseases",
    "K_OTH": "Other digestive diseases",
    "L": "Skin & subcutaneous tissue diseases",
    "M": "Musculoskeletal system diseases",
    "RHEUM_ARTHRO": "Rheumatoid arthritis & arthrosis",
    "M_OTH": "Other musculoskeletal diseases",
    "N": "Genitourinary system diseases",
    "N00-N29": "Kidney & ureter diseases",
    "N_OTH": "Other genitourinary diseases",
    "O": "Pregnancy, childbirth & puerperium",
    "P": "Perinatal conditions",
    "Q": "Congenital malformations, deformations and chromosomal abnormalities",
    "R": "Symptoms & abnormal clinical and laboratory findings",
    "R95": "Sudden infant death syndrome",
    "R96-R99": "Ill-defined & unknown causes of mortality",
    "R_OTH": "Other signs & lab findings",
    "V01-Y89": "External causes of morbidity and mortality",
    "ACC": "Accidents",
    "V_Y85": "Transport accidents",
    "ACC_OTH": "Other accidents",
    "W00-W19": "Falls",
    "W65-W74": "Accidental drowning and submersion",
    "X60-X84_Y870": "Intentional self-harm",
    "X40-X49": "Accidental poisoning by and exposure to noxious substances",
    "X85-Y09_Y871": "Assault",
    "Y10-Y34_Y872": "Event of undetermined intent",
    "V01-Y89_OTH": "Other external causes of morbidity and mortality",
    "A-R_V-Y": "All causes (A00-R99 & V01-Y89)",
    "U071": "COVID-19, virus identified",
    "U072": "COVID-19, virus not identified"
}
REV_CAUSE_NAME_MAP = {v: k for k, v in CAUSE_NAME_MAP.items()}

COUNTRY_NAME_MAP = {c.alpha_2: c.name for c in pycountry.countries}
COUNTRY_NAME_MAP.update({
    "FX": "France (Metropolitan)",
    "EU": "European Union",
    "Europe": "Europe"
})
REV_COUNTRY_NAME_MAP = {v: k for k, v in COUNTRY_NAME_MAP.items()}

FACTOR_IDS = {
    "BMI by citizenship": "hlth_ehis_bm1c",
    "Phys activity by citizenship": "hlth_ehis_pe9c",
    "Fruit & veg by citizenship": "hlth_ehis_fv3c",
    "Smoking by citizenship": "hlth_ehis_sk1c",
    "Social support by citizenship": "hlth_ehis_ss1c",
    "Health care expenditure by provider": "hlth_sha11_hp",
    "Staff – physicians": "hlth_rs_prs2",
    "Staff – hospital": "hlth_rs_prshp2",
    "Available beds in hospitals": "hlth_rs_bdsrg2",
    "Beds in nursing and other residential long-term care facilities": "hlth_rs_bdltc",
    "Imaging devices": "hlth_rs_medim",
    "Beds hospital": "hlth_rs_bds2",
    "Consultations": "hlth_ehis_am1e",
    "Med use prescribed": "hlth_ehis_md1e",
    "Med use non-prescribed": "hlth_ehis_md2e",
    "Home care": "hlth_ehis_am7e",
    "Unmet needs": "hlth_ehis_un1e"
}

def alpha3_from_a2(a2: str):
    """Convert alpha-2 to alpha-3 country codes."""
    c = pycountry.countries.get(alpha_2=a2)
    return c.alpha_3 if c else None

# --------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------
@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    """Load Eurostat data series with proper error handling."""
    base = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
    endpoints = [
        f"{base}/{dataset_id}?format=TSV&compressed=true",
        f"{base}/{dataset_id}?format=TSV"
    ]
    raw = None
    for url in endpoints:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            if "compressed=true" in url:
                buf = BytesIO(r.content)
                with gzip.GzipFile(fileobj=buf) as gz:
                    raw = pd.read_csv(gz, sep="\t", low_memory=False)
            else:
                raw = pd.read_csv(BytesIO(r.content), sep="\t", low_memory=False)
            break
        except HTTPError:
            raw = None
    
    if raw is None:
        files = glob.glob(f"/mnt/data/estat_{dataset_id}*.tsv")
        if files:
            raw = pd.read_csv(files[0], sep="\t", low_memory=False)
    
    if raw is None:
        raise HTTPError(f"Could not fetch or find local file for {dataset_id}")

    first = raw.columns[0]
    dims = first.split("\\")[0].split(",")
    raw = raw.rename(columns={first: "series_keys"})
    keys = raw["series_keys"].str.split(",", expand=True)
    keys.columns = dims
    df = pd.concat([keys, raw.drop(columns=["series_keys"])], axis=1)

    years = [c for c in df.columns if c not in dims]
    long = df.melt(id_vars=dims, value_vars=years,
                   var_name="Year", value_name="raw_rate")
    long["Year"] = long["Year"].astype(int)
    long["Rate"] = pd.to_numeric(long["raw_rate"].replace(":", np.nan), errors="coerce")

    mask = pd.Series(True, index=long.index)
    if "unit" in dims:
        uv = "RT" if "RT" in long["unit"].unique() else ("NR" if "NR" in long["unit"].unique() else None)
        if uv:
            mask &= (long["unit"] == uv)
    if "freq" in dims:
        mask &= (long["freq"] == "A")
    if "age" in dims:
        mask &= (long["age"] == "TOTAL")
    if "resid" in dims:
        mask &= (long["resid"] == "TOT_IN")

    sub = long[mask].copy()
    rename = {"geo": "Region", "sex": "Sex"}
    others = [d for d in dims if d not in ("geo", "sex", "freq", "unit", "age", "resid")]
    if len(others) == 1:
        rename[others[0]] = "Category"
    out = sub.rename(columns=rename)
    cols = ["Region", "Year", "Category", "Sex", "Rate"]
    return out[[c for c in cols if c in out.columns]]

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load mortality data from Eurostat."""
    def ld(ds):
        x = load_eurostat_series(ds).rename(columns={"Region": "Country", "Category": "Cause"})
        return x.dropna(subset=["Rate"])
    
    hist = ld("hlth_cd_asdr")
    mod  = ld("hlth_cd_asdr2")
    mod  = mod[mod["Country"].str.fullmatch(r"[A-Z]{2}")]
    df   = pd.concat([hist, mod], ignore_index=True).sort_values(["Country", "Cause", "Sex", "Year"])
    
    # Add EU and Europe aggregates
    df_eu  = df[df["Country"].isin(EU_CODES)].groupby(["Year", "Cause", "Sex"], as_index=False)["Rate"].mean()
    df_eu["Country"] = "EU"
    df_eur = df.groupby(["Year", "Cause", "Sex"], as_index=False)["Rate"].mean()
    df_eur["Country"] = "Europe"
    
    return pd.concat([df, df_eu, df_eur], ignore_index=True)

@st.cache_data
def load_all_factors() -> pd.DataFrame:
    """Load health factor data from Eurostat."""
    frames = []
    for name, ds in FACTOR_IDS.items():
        try:
            f = load_eurostat_series(ds).rename(columns={"Region": "Country"})
        except HTTPError:
            continue
        f.loc[f["Country"].str.startswith("EU"), "Country"] = "EU"
        if "Sex" not in f.columns:
            f["Sex"] = "T"
        if "Category" in f.columns:
            f = f[f["Category"] == "TOTAL"]
        f = f[["Country", "Year", "Sex", "Rate"]].copy()
        f["FactorName"] = name
        frames.append(f)
    
    if not frames:
        return pd.DataFrame(columns=["Country", "Year", "Sex", "Rate", "FactorName"])
    return pd.concat(frames, ignore_index=True)

# --------------------------------------------------------------------------
# CHANGEPOINT DETECTION (formerly "joinpoints")
# --------------------------------------------------------------------------
def detect_change_points(ts, pen: float = 3) -> list:
    """Detect changepoints in time series using PELT algorithm."""
    ts = pd.Series(ts).dropna()
    if len(ts) < 2:
        return []
    algo = rpt.Pelt(model="l2").fit(ts.values)
    try:
        return algo.predict(pen=pen)
    except BadSegmentationParameters:
        return []

def compute_changepoints_and_apc(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Compute changepoints and Annual Percent Change (APC) for each segment.
    Note: This uses changepoint detection, not traditional joinpoint regression.
    """
    recs = []
    for sex in df_sub["Sex"].unique():
        part = df_sub[df_sub["Sex"] == sex].sort_values("Year")
        yrs, vals = part["Year"].values, part["Rate"].values
        bkps = detect_change_points(vals)[:-1]
        segs = np.split(np.arange(len(yrs)), bkps) if bkps else [np.arange(len(yrs))]
        
        for seg in segs:
            sy, ey = int(yrs[seg].min()), int(yrs[seg].max())
            sv = vals[seg]
            if len(sv) < 2 or np.all(np.isnan(sv)):
                recs.append({
                    "Sex": SEX_NAME_MAP[sex],
                    "start_year": sy,
                    "end_year": ey,
                    "slope": np.nan,
                    "APC_pct": np.nan
                })
            else:
                slope = sm.OLS(sv, sm.add_constant(yrs[seg])).fit().params[1]
                apc = (slope / np.nanmean(sv)) * 100
                recs.append({
                    "Sex": SEX_NAME_MAP[sex],
                    "start_year": sy,
                    "end_year": ey,
                    "slope": slope,
                    "APC_pct": apc
                })
    
    return pd.DataFrame(recs)

# --------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS
# --------------------------------------------------------------------------
def plot_changepoints_comparative(df_sub: pd.DataFrame, title: str):
    """Plot comparative trends across sexes."""
    df_sub["SexFull"] = df_sub["Sex"].map(SEX_NAME_MAP)
    fig = px.line(df_sub, x="Year", y="Rate", color="SexFull", 
                  title=title, markers=True)
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

def plot_segmented_fit_series(df_sub: pd.DataFrame, title: str):
    """Plot time series with segmented linear fits."""
    sub = df_sub.sort_values("Year")
    yrs, rates = sub["Year"].values, sub["Rate"].values
    bkps = detect_change_points(rates)[:-1]
    segs = np.split(np.arange(len(yrs)), bkps) if bkps else [np.arange(len(yrs))]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yrs, y=rates, mode="markers+lines", 
                            name="Observed", line=dict(color='lightgray')))
    
    palette = px.colors.qualitative.Dark24
    for i, seg in enumerate(segs):
        idx, vals = yrs[seg], rates[seg]
        if len(vals) >= 2 and not np.all(np.isnan(vals)):
            fit = sm.OLS(vals, sm.add_constant(idx)).fit().predict(sm.add_constant(idx))
            fig.add_trace(go.Scatter(
                x=idx, y=fit, mode="lines",
                line=dict(color=palette[i % len(palette)], width=3),
                name=f"Segment {i+1} ({idx[0]}-{idx[-1]})"
            ))
    
    fig.update_layout(title=title, xaxis_title="Year", yaxis_title="Rate",
                     hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap_temporal(df_sub: pd.DataFrame, title: str):
    """Create heatmap showing temporal patterns across countries."""
    pivot = df_sub.pivot_table(index="Country", columns="Year", values="Rate")
    
    fig = px.imshow(pivot, 
                    labels=dict(x="Year", y="Country", color="Rate"),
                    title=title,
                    aspect="auto",
                    color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# FORECASTING
# --------------------------------------------------------------------------
def get_prophet_forecast(df_sub: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Generate Prophet forecast."""
    dfp = df_sub[["Year", "Rate"]].rename(columns={"Year": "ds", "Rate": "y"})
    dfp["ds"] = pd.to_datetime(dfp["ds"].astype(str) + "-01-01", format="%Y-%m-%d")
    m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
    m.fit(dfp)
    fut = m.make_future_dataframe(periods=periods, freq="YS")
    fc = m.predict(fut)
    return pd.DataFrame({"Year": fc["ds"].dt.year, "Prophet": fc["yhat"]})

def get_arima_forecast(df_sub: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Generate ARIMA forecast."""
    ser = df_sub.set_index("Year")["Rate"]
    res = ARIMA(ser, order=(1,1,1)).fit()
    preds = res.forecast(periods)
    yrs = np.arange(ser.index.max()+1, ser.index.max()+1+periods)
    return pd.DataFrame({"Year": yrs, "ARIMA": preds.values})

def get_ets_forecast(df_sub: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Generate ETS forecast."""
    ser = df_sub.set_index("Year")["Rate"]
    m = ExponentialSmoothing(ser, trend="add", seasonal=None).fit(optimized=True)
    preds = m.forecast(periods)
    yrs = np.arange(ser.index.max()+1, ser.index.max()+1+periods)
    return pd.DataFrame({"Year": yrs, "ETS": preds.values})

def forecast_mortality(df_sub: pd.DataFrame, periods: int, method: str) -> pd.DataFrame:
    """Generate mortality forecasts using specified method."""
    prop = get_prophet_forecast(df_sub, periods)
    ari = get_arima_forecast(df_sub, periods)
    ets = get_ets_forecast(df_sub, periods)
    fc = prop.merge(ari, on="Year").merge(ets, on="Year")
    
    if method == "Prophet":
        fc["Forecast"] = fc["Prophet"]
    elif method == "ARIMA":
        fc["Forecast"] = fc["ARIMA"]
    elif method == "ETS":
        fc["Forecast"] = fc["ETS"]
    else:  # Ensemble
        fc["Forecast"] = fc[["Prophet", "ARIMA", "ETS"]].mean(axis=1)
    
    hist = df_sub[["Year", "Rate"]].rename(columns={"Rate": "History"})
    return hist.merge(fc[["Year", "Forecast"]], on="Year", how="outer")

# --------------------------------------------------------------------------
# GRANGER CAUSALITY ANALYSIS (corrected terminology)
# --------------------------------------------------------------------------
def compute_granger_causality_bic(pair_df: pd.DataFrame, maxlag: int) -> dict:
    """
    Compute Granger causality using BIC approximation.
    Returns BIC-based Bayes Factor and p-value.
    """
    df_pair = pair_df.dropna()
    if df_pair.shape[0] < maxlag + 3:
        return {"BF10": np.nan, "p_value": np.nan}
    
    target, cause = df_pair.columns
    Y = df_pair[target].values[maxlag:]
    X_alt = np.column_stack([df_pair[cause].values[maxlag-lag:-lag] 
                             for lag in range(1, maxlag+1)])
    X_alt = sm.add_constant(X_alt)
    X_null = np.ones((len(Y), 1))
    
    m0 = sm.OLS(Y, X_null).fit()
    m1 = sm.OLS(Y, X_alt).fit()
    
    # BIC approximation to Bayes Factor
    bf10 = float(np.exp((m0.bic - m1.bic) / 2.0))
    
    # F-test for significance
    f_stat = ((m0.ssr - m1.ssr) / maxlag) / (m1.ssr / m1.df_resid)
    p_value = 1 - stats.f.cdf(f_stat, maxlag, m1.df_resid)
    
    return {"BF10": bf10, "p_value": p_value}

def draw_directed_network(nodes, edges, title, edge_labels=None):
    """Draw directed network with improved layout."""
    try:
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        angles = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
        fig, ax = plt.subplots(figsize=(8, 8))
        pos = {nodes[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(len(nodes))}
        
        if edges:
            nx.draw_networkx_edges(
                G, pos,
                arrows=True, arrowsize=20, width=2,
                edge_color='gray',
                ax=ax,
                connectionstyle="arc3,rad=0.1"
            )
            
            # Add edge labels if provided
            if edge_labels:
                edge_labels_dict = {(e[0], e[1]): f"BF={edge_labels.get((e[0], e[1]), '')}" 
                                   for e in edges if (e[0], e[1]) in edge_labels}
                if edge_labels_dict:
                    nx.draw_networkx_edge_labels(G, pos, edge_labels_dict, font_size=8, ax=ax)
        
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', ax=ax)
        
        # Better label placement
        radius = 1.4
        for i, node in enumerate(nodes):
            angle = angles[i]
            x, y = np.cos(angle)*radius, np.sin(angle)*radius
            ha = "left" if np.cos(angle) > 0 else "right"
            va = "bottom" if np.sin(angle) > 0 else "top"
            ax.text(x, y, node, ha=ha, va=va,
                    fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='black', 
                             boxstyle='round,pad=0.5', alpha=0.8),
                    zorder=3)
        
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
        ax.set_axis_off()
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error drawing network: {str(e)}")
        plt.close('all')

# --------------------------------------------------------------------------
# NEW ANALYSES
# --------------------------------------------------------------------------
def compute_excess_mortality(df, baseline_years, comparison_year, country_code, cause_code):
    """
    Compute excess mortality compared to baseline period.
    """
    baseline = df[
        (df["Country"] == country_code) &
        (df["Cause"] == cause_code) &
        (df["Sex"] == "T") &
        (df["Year"].between(*baseline_years))
    ]["Rate"].mean()
    
    comparison = df[
        (df["Country"] == country_code) &
        (df["Cause"] == cause_code) &
        (df["Sex"] == "T") &
        (df["Year"] == comparison_year)
    ]["Rate"].values
    
    if len(comparison) == 0:
        return None
    
    excess = ((comparison[0] - baseline) / baseline) * 100
    return {
        "baseline_rate": baseline,
        "comparison_rate": comparison[0],
        "excess_pct": excess
    }

def compute_spatial_autocorrelation(df, year, cause_code):
    """
    Compute Moran's I for spatial autocorrelation.
    """
    # Get rates for all EU countries
    rates_df = df[
        (df["Country"].isin(EU_CODES)) &
        (df["Cause"] == cause_code) &
        (df["Sex"] == "T") &
        (df["Year"] == year)
    ][["Country", "Rate"]].set_index("Country")
    
    if len(rates_df) < 5:
        return None
    
    # Create spatial weights matrix based on neighbors
    countries = rates_df.index.tolist()
    n = len(countries)
    W = np.zeros((n, n))
    
    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if c2 in NEIGHBORS.get(c1, []):
                W[i, j] = 1
    
    # Row-normalize
    row_sums = W.sum(axis=1)
    W = np.divide(W, row_sums[:, np.newaxis], 
                  where=row_sums[:, np.newaxis] != 0)
    
    # Compute Moran's I
    rates = rates_df["Rate"].values
    mean_rate = rates.mean()
    z = rates - mean_rate
    
    numerator = np.sum(W * np.outer(z, z))
    denominator = np.sum(z ** 2)
    
    moran_i = (n / W.sum()) * (numerator / denominator)
    
    return {"morans_i": moran_i, "n_countries": n}

def compare_with_benchmark(df, country_code, cause_code, year_range):
    """
    Compare country performance with EU average.
    """
    country_data = df[
        (df["Country"] == country_code) &
        (df["Cause"] == cause_code) &
        (df["Sex"] == "T") &
        (df["Year"].between(*year_range))
    ][["Year", "Rate"]].rename(columns={"Rate": "Country"})
    
    eu_data = df[
        (df["Country"] == "EU") &
        (df["Cause"] == cause_code) &
        (df["Sex"] == "T") &
        (df["Year"].between(*year_range))
    ][["Year", "Rate"]].rename(columns={"Rate": "EU Average"})
    
    comparison = country_data.merge(eu_data, on="Year")
    comparison["Difference"] = comparison["Country"] - comparison["EU Average"]
    comparison["Pct_Difference"] = (comparison["Difference"] / comparison["EU Average"]) * 100
    
    return comparison

# --------------------------------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------------------------------
def main():
    try:
        st.set_page_config(
            layout="wide",
            page_title="European Public Health Dashboard",
            page_icon="🏥"
        )
        
        st.title("🏥 European Public Health Dashboard")
        st.markdown("### Advanced Mortality Trend Analysis & Forecasting")
        st.markdown("*Developed by Younes Adam Tabi*")
        st.markdown("---")
        
        # Load data
        with st.spinner("Loading data..."):
            df = load_data()
            df["CountryFull"] = df["Country"].map(COUNTRY_NAME_MAP).fillna(df["Country"])
            df["CauseFull"] = df["Cause"].map(CAUSE_NAME_MAP).fillna(df["Cause"])
            df["SexFull"] = df["Sex"].map(SEX_NAME_MAP).fillna(df["Sex"])
    
        # Sidebar filters
        st.sidebar.header("📊 Data Filters")
    
        countries = sorted(df["CountryFull"].dropna().unique())
        country_full = st.sidebar.selectbox(
        "Country", 
        countries,
        index=countries.index("European Union") if "European Union" in countries else 0
    )
    country_code = REV_COUNTRY_NAME_MAP.get(country_full, country_full)
    
    causes = sorted(df[df["Country"] == country_code]["CauseFull"].dropna().unique())
    cause_full = st.sidebar.selectbox("Cause of Death", causes)
    cause_code = REV_CAUSE_NAME_MAP.get(cause_full, cause_full)
    
    sex_sel = st.sidebar.multiselect(
        "Sex", 
        ["Total", "Male", "Female"],
        default=["Total"]
    )
    sex_codes = [REV_SEX_NAME[s] for s in sex_sel]
    
    yrs = sorted(df["Year"].unique())
    year_range = st.sidebar.slider(
        "Analysis Period", 
        yrs[0], yrs[-1],
        (yrs[0], yrs[-1])
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔮 Forecast Settings")
    forecast_yrs = st.sidebar.slider("Forecast Horizon (years)", 1, 30, 10)
    method = st.sidebar.selectbox(
        "Forecast Method",
        ["Ensemble", "Prophet", "ARIMA", "ETS"]
    )
    
    # Initialize result storage
    changepoint_df = pd.DataFrame()
    forecasts = {}
    
    # ======================================================================
    # SECTION 1: TEMPORAL TREND ANALYSIS
    # ======================================================================
    st.header(f"📈 Temporal Trend Analysis: {cause_full}")
    st.markdown(f"**Country:** {country_full} | **Period:** {year_range[0]}–{year_range[1]}")
    
    df_filtered = df[
        (df["Country"] == country_code) &
        (df["Cause"] == cause_code) &
        (df["Sex"].isin(sex_codes)) &
        (df["Year"].between(*year_range))
    ]
    
    if df_filtered.empty:
        st.warning("⚠️ No data available for selected filters.")
    else:
        # Changepoint analysis
        st.subheader("🔍 Changepoint Detection & Trend Segmentation")
        st.info("💡 **Note:** This analysis uses changepoint detection (PELT algorithm), "
                "not traditional joinpoint regression. Segments represent periods with "
                "consistent linear trends.")
        
        changepoint_df = compute_changepoints_and_apc(df_filtered)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            plot_changepoints_comparative(df_filtered, 
                                        f"{cause_full} - Temporal Trends by Sex")
        with col2:
            st.markdown("#### Detected Segments")
            st.dataframe(
                changepoint_df.style.format({
                    "slope": "{:.2f}",
                    "APC_pct": "{:.2f}%"
                }),
                use_container_width=True
            )
        
        # Segmented fits
        st.subheader("📐 Segmented Linear Fits")
        for sc, sf in zip(sex_codes, sex_sel):
            plot_segmented_fit_series(
                df_filtered[df_filtered["Sex"] == sc],
                f"{cause_full} ({sf}) - Segmented Trend Analysis"
            )
        
        # Forecasting
        st.subheader(f"🔮 {forecast_yrs}-Year Forecast ({method} Method)")
        forecast_cols = st.columns(len(sex_sel))
        
        for idx, (sc, sf) in enumerate(zip(sex_codes, sex_sel)):
            fc = forecast_mortality(df_filtered[df_filtered["Sex"] == sc], 
                                  forecast_yrs, method)
            forecasts[sf] = fc
            
            with forecast_cols[idx]:
                fig = px.line(fc, x="Year", y=["History", "Forecast"],
                            title=f"{sf} - {method} Forecast")
                fig.update_traces(line=dict(width=3))
                fig.add_vline(x=year_range[1], line_dash="dash", 
                            line_color="gray", opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)
        
        # Benchmark comparison
        if country_code not in ["EU", "Europe"]:
            st.subheader("📊 Benchmark Comparison with EU Average")
            
            try:
                benchmark = compare_with_benchmark(df, country_code, cause_code, year_range)
                
                if not benchmark.empty and len(benchmark) > 0:
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Absolute Rates", "Percentage Difference from EU"),
                        vertical_spacing=0.15
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=benchmark["Year"], y=benchmark["Country"],
                                 mode='lines+markers', name=country_full,
                                 line=dict(width=3)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=benchmark["Year"], y=benchmark["EU Average"],
                                 mode='lines+markers', name="EU Average",
                                 line=dict(width=3, dash='dash')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(x=benchmark["Year"], y=benchmark["Pct_Difference"],
                              marker_color=np.where(benchmark["Pct_Difference"] > 0, 
                                                  'red', 'green'),
                              showlegend=False),
                        row=2, col=1
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="black", 
                                row=2, col=1)
                    
                    fig.update_xaxes(title_text="Year", row=2, col=1)
                    fig.update_yaxes(title_text="Rate", row=1, col=1)
                    fig.update_yaxes(title_text="% Difference", row=2, col=1)
                    fig.update_layout(height=700, showlegend=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Rate (Country)", 
                                f"{benchmark['Country'].mean():.1f}")
                    with col2:
                        st.metric("Average Rate (EU)", 
                                f"{benchmark['EU Average'].mean():.1f}")
                    with col3:
                        avg_diff = benchmark['Pct_Difference'].mean()
                        st.metric("Average Difference", 
                                f"{avg_diff:+.1f}%",
                                delta=f"{'Above' if avg_diff > 0 else 'Below'} EU avg")
                else:
                    st.info("Benchmark comparison data not available for selected period")
            except Exception as e:
                st.warning(f"Could not compute benchmark comparison: {str(e)}")
    
    # ======================================================================
    # SECTION 2: HEALTH FACTORS & REGRESSION
    # ======================================================================
    st.markdown("---")
    st.header("🔬 Health Factors - Exploratory Panel Regression")
    
    factors = st.multiselect(
        "Select health factors to analyze",
        list(FACTOR_IDS.keys()),
        help="Choose factors to include in the regression model"
    )
    
    panel_clean = pd.DataFrame()
    reg_coefs = pd.Series()
    
    if factors:
        reg_min, reg_max = st.slider(
            "Regression Period",
            min_value=year_range[0],
            max_value=year_range[1],
            value=(year_range[0], year_range[1]),
            help="Select at least 3 years for reliable regression"
        )
        
        if (reg_max - reg_min) < 2:
            st.warning("⚠️ Please select at least 3 years for regression analysis.")
        else:
            with st.spinner("Loading health factors data..."):
                all_factors = load_all_factors()
            
            # Prepare panel data
            pf = all_factors[
                (all_factors["FactorName"].isin(factors)) &
                (all_factors["Year"].between(reg_min, reg_max)) &
                (all_factors["Sex"].isin(sex_codes))
            ]
            
            pm = df[
                (df["Cause"] == cause_code) &
                (df["Year"].between(reg_min, reg_max)) &
                (df["Sex"].isin(sex_codes))
            ][["Country", "Year", "Rate"]].rename(columns={"Rate": "Mortality"})
            
            panel = pf.pivot_table(
                index=["Country", "Year"],
                columns="FactorName",
                values="Rate"
            ).reset_index().merge(pm, on=["Country", "Year"], how="inner")
            
            present = [f for f in factors if f in panel.columns]
            missing = set(factors) - set(present)
            
            if missing:
                st.warning(f"⚠️ Data unavailable for: {', '.join(missing)}")
            
            if present:
                before = panel.shape[0]
                panel_clean = panel.dropna(subset=present + ["Mortality"])
                dropped = before - panel_clean.shape[0]
                
                if dropped > 0:
                    st.info(f"ℹ️ Dropped {dropped} observations with missing data")
                
                if panel_clean.shape[0] < len(present) * 2:
                    st.warning("⚠️ Insufficient observations for reliable regression")
                else:
                    # Run regression
                    X = sm.add_constant(panel_clean[present])
                    y = panel_clean["Mortality"]
                    mdl = sm.OLS(y, X).fit()
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.subheader("📋 Regression Results")
                        st.text(mdl.summary())
                    
                    with col2:
                        st.subheader("📊 Coefficient Plot")
                        reg_coefs = mdl.params.drop("const")
                        
                        fig = go.Figure()
                        colors = ['green' if x < 0 else 'red' for x in reg_coefs.values]
                        fig.add_trace(go.Bar(
                            x=reg_coefs.values,
                            y=reg_coefs.index,
                            orientation='h',
                            marker_color=colors
                        ))
                        fig.update_layout(
                            title="Factor Coefficients",
                            xaxis_title="Coefficient",
                            yaxis_title="Factor",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model fit metrics
                        st.metric("R-squared", f"{mdl.rsquared:.3f}")
                        st.metric("Adj. R-squared", f"{mdl.rsquared_adj:.3f}")
                        st.metric("F-statistic p-value", f"{mdl.f_pvalue:.4f}")
    
    # ======================================================================
    # SECTION 3: CLUSTER ANALYSIS
    # ======================================================================
    st.markdown("---")
    st.header("🗺️ Geographic Cluster Analysis")
    st.info("Clustering countries based on mortality rate patterns over time")
    
    df_cluster = df[
        (df["Cause"] == cause_code) &
        (df["Sex"] == "T") &
        (df["Year"].between(*year_range))
    ]
    
    pivot = df_cluster.pivot(index="Country", columns="Year", values="Rate")\
                      .interpolate(axis=1, limit_direction="both")\
                      .ffill(axis=1).bfill(axis=1).dropna(axis=0, how="all")
    
    clust_df = pd.DataFrame()
    
    if pivot.shape[0] < 3:
        st.warning("⚠️ Insufficient countries for cluster analysis")
    else:
        X = pivot.values
        max_k = min(10, X.shape[0] - 1)
        
        # Compute silhouette scores
        sil_scores = {}
        for k in range(2, max_k + 1):
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            sil_scores[k] = silhouette_score(X, labels)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Optimal Cluster Selection")
            sil_df = pd.DataFrame.from_dict(
                sil_scores, 
                orient='index', 
                columns=['Silhouette Score']
            )
            st.dataframe(sil_df.style.format("{:.3f}"), use_container_width=True)
            
            best_k = max(sil_scores, key=sil_scores.get)
            st.success(f"**Optimal k:** {best_k}")
            st.metric("Best Silhouette Score", f"{sil_scores[best_k]:.3f}")
        
        with col2:
            fig = px.line(
                x=list(sil_scores.keys()),
                y=list(sil_scores.values()),
                markers=True,
                title="Silhouette Score by Number of Clusters"
            )
            fig.update_xaxes(title="Number of Clusters (k)")
            fig.update_yaxes(title="Silhouette Score")
            fig.add_vline(x=best_k, line_dash="dash", line_color="red",
                         annotation_text=f"Optimal k={best_k}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Perform clustering with optimal k
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)
        clust_df = pd.DataFrame({
            "Country": pivot.index,
            "Cluster": km.labels_.astype(str)
        })
        clust_df["CountryFull"] = clust_df["Country"].map(COUNTRY_NAME_MAP)
        clust_df["iso_alpha"] = clust_df["Country"].apply(alpha3_from_a2)
        
        # Choropleth map
        st.subheader("Geographic Distribution of Clusters")
        fig = px.choropleth(
            clust_df,
            locations="iso_alpha",
            color="Cluster",
            hover_name="CountryFull",
            locationmode="ISO-3",
            scope="europe",
            title=f"{cause_full} - {best_k} Clusters",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster characteristics
        st.subheader("Cluster Characteristics")
        for cluster in sorted(clust_df["Cluster"].unique()):
            with st.expander(f"Cluster {cluster}"):
                cluster_countries = clust_df[clust_df["Cluster"] == cluster]["CountryFull"].dropna().tolist()
                if cluster_countries:
                    st.write(f"**Countries:** {', '.join(cluster_countries)}")
                else:
                    st.write("**Countries:** No valid country names")
                
                # Plot average trajectory
                cluster_codes = clust_df[clust_df["Cluster"] == cluster]["Country"].tolist()
                cluster_data = df_cluster[df_cluster["Country"].isin(cluster_codes)]
                
                if not cluster_data.empty:
                    avg_trajectory = cluster_data.groupby("Year")["Rate"].mean().reset_index()
                    
                    fig = px.line(avg_trajectory, x="Year", y="Rate",
                                title=f"Average Mortality Trajectory - Cluster {cluster}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for this cluster")
    
    # ======================================================================
    # SECTION 4: GRANGER CAUSALITY NETWORK (CORRECTED)
    # ======================================================================
    st.markdown("---")
    st.header("🔗 Granger Causality Network Analysis")
    st.info("⚠️ **Methodology Note:** This analysis uses BIC-approximated Granger causality "
            "to identify potential predictive relationships between countries. High BF₁₀ values "
            "suggest that past values of one country help predict another, indicating potential "
            "cross-border health influences. This is **not** true Bayesian inference but a "
            "BIC-based approximation.")
    
    analysis_type = st.radio(
        "Analysis Type",
        ["Neighbor-Based Analysis", "Global Network Analysis"],
        horizontal=True
    )
    
    if analysis_type == "Neighbor-Based Analysis":
        st.subheader("🎯 Neighbor-Based Granger Causality")
        
        country_list = sorted(df["CountryFull"].dropna().unique())
        focal_full = st.selectbox(
            "Select focal country",
            country_list,
            index=country_list.index("Germany") if "Germany" in country_list else 0
        )
        focal_code = REV_COUNTRY_NAME_MAP.get(focal_full, focal_full)
        
        nbrs = NEIGHBORS.get(focal_code, [])
        
        if not nbrs:
            st.warning(f"No neighboring countries defined for {focal_full}")
        else:
            # Map visualization
            map_df = pd.DataFrame({
                "Country": [focal_code] + nbrs,
                "Role": ["Focal"] + ["Neighbor"] * len(nbrs)
            })
            map_df["CountryFull"] = map_df["Country"].map(COUNTRY_NAME_MAP)
            map_df["iso_alpha"] = map_df["Country"].apply(alpha3_from_a2)
            
            st.plotly_chart(
                px.choropleth(
                    map_df,
                    locations="iso_alpha",
                    color="Role",
                    hover_name="CountryFull",
                    locationmode="ISO-3",
                    scope="europe",
                    title=f"Focal Country: {focal_full} and Neighbors",
                    color_discrete_map={"Focal": "red", "Neighbor": "lightblue"}
                ),
                use_container_width=True
            )
            
            # Granger causality analysis
            nbr_lag = st.slider("Maximum lag (years)", 1, 5, 2, key="nbr_lag")
            nbr_bf = st.number_input("BF₁₀ threshold", 1.0, 100.0, 3.0, 0.5, key="nbr_bf")
            
            countries_to_analyze = [focal_code] + nbrs
            df_n = df[
                (df["Cause"] == cause_code) &
                (df["Country"].isin(countries_to_analyze)) &
                (df["Sex"] == "T") &
                (df["Year"].between(*year_range))
            ]
            
            pivot_n = df_n.pivot_table(index="Year", columns="Country", values="Rate")
            common_n = [c for c in countries_to_analyze if c in pivot_n.columns]
            
            if len(common_n) < 2:
                st.warning("Insufficient data for analysis")
            else:
                # Compute Granger causality
                bf_n = pd.DataFrame(np.nan, index=common_n, columns=common_n)
                pval_n = pd.DataFrame(np.nan, index=common_n, columns=common_n)
                
                with st.spinner("Computing Granger causality..."):
                    for src in common_n:
                        for dst in common_n:
                            if src == dst:
                                continue
                            pair = pivot_n[[dst, src]].dropna()
                            if len(pair) >= nbr_lag + 3:
                                result = compute_granger_causality_bic(pair, nbr_lag)
                                bf_n.loc[src, dst] = result["BF10"]
                                pval_n.loc[src, dst] = result["p_value"]
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### BF₁₀ Matrix")
                    st.dataframe(
                        bf_n.style.background_gradient(cmap='YlOrRd', axis=None)\
                            .format("{:.2f}"),
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("##### P-values")
                    st.dataframe(
                        pval_n.style.background_gradient(cmap='RdYlGn_r', vmin=0, vmax=0.1)\
                            .format("{:.4f}"),
                        use_container_width=True
                    )
                
                # Build network (CORRECTED: only between actual neighbors)
                edges_n = []
                edge_labels = {}
                
                for src in common_n:
                    for dst in common_n:
                        if src == dst:
                            continue
                        
                        # Only add edge if dst is actually a neighbor of src
                        if dst in NEIGHBORS.get(src, []):
                            bf = bf_n.loc[src, dst]
                            if pd.notna(bf) and bf >= nbr_bf:
                                src_full = COUNTRY_NAME_MAP[src]
                                dst_full = COUNTRY_NAME_MAP[dst]
                                edges_n.append((src_full, dst_full))
                                edge_labels[(src_full, dst_full)] = f"{bf:.1f}"
                
                nodes_n = [COUNTRY_NAME_MAP[c] for c in common_n]
                
                st.subheader("Network Visualization")
                if not edges_n:
                    st.info(f"No significant Granger causal relationships (BF₁₀ ≥ {nbr_bf}) "
                           "detected between neighboring countries.")
                else:
                    st.success(f"Found {len(edges_n)} significant relationships")
                    draw_directed_network(
                        nodes_n, edges_n,
                        f"Neighbor Granger Causality Network (BF₁₀ ≥ {nbr_bf})",
                        edge_labels
                    )
    
    else:  # Global Network Analysis
        st.subheader("🌍 Global Granger Causality Network")
        
        country_list = sorted(df["CountryFull"].dropna().unique())
        sel_countries = st.multiselect(
            "Select countries to include",
            country_list,
            default=[c for c in ["Germany", "France", "Italy", "Spain", "Poland"] 
                    if c in country_list]
        )
        
        if len(sel_countries) < 2:
            st.warning("Please select at least 2 countries")
        else:
            gl_maxlag = st.slider("Maximum lag (years)", 1, 5, 2, key="gl_maxlag")
            bf_thresh = st.number_input("BF₁₀ threshold", 1.0, 100.0, 3.0, 0.5, key="bf_thr")
            
            df_g = df[
                (df["Cause"] == cause_code) &
                (df["CountryFull"].isin(sel_countries)) &
                (df["Sex"] == "T") &
                (df["Year"].between(*year_range))
            ]
            
            pivot_gc = df_g.pivot_table(index="Year", columns="CountryFull", values="Rate")
            common = [c for c in sel_countries if c in pivot_gc.columns]
            
            if len(common) < 2:
                st.warning("Insufficient data for selected countries")
            else:
                bf_mat = pd.DataFrame(np.nan, index=common, columns=common)
                pval_mat = pd.DataFrame(np.nan, index=common, columns=common)
                
                with st.spinner("Computing global Granger causality..."):
                    for src in common:
                        for dst in common:
                            if src == dst:
                                continue
                            pair = pivot_gc[[dst, src]].dropna()
                            if len(pair) >= gl_maxlag + 3:
                                result = compute_granger_causality_bic(pair, gl_maxlag)
                                bf_mat.loc[src, dst] = result["BF10"]
                                pval_mat.loc[src, dst] = result["p_value"]
                
                # Display heatmap
                st.subheader("Granger Causality Heatmap (BF₁₀)")
                fig = px.imshow(
                    bf_mat.astype(float),
                    labels=dict(x="To Country", y="From Country", color="BF₁₀"),
                    title="BF₁₀ values (rows predict columns)",
                    color_continuous_scale="YlOrRd",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Build network (only between neighbors)
                edges_global = []
                edge_labels_global = {}
                
                for src in common:
                    for dst in common:
                        if src == dst:
                            continue
                        
                        src_code = REV_COUNTRY_NAME_MAP.get(src)
                        dst_code = REV_COUNTRY_NAME_MAP.get(dst)
                        
                        # Only add if dst is neighbor of src
                        if dst_code in NEIGHBORS.get(src_code, []):
                            bf = bf_mat.loc[src, dst]
                            if pd.notna(bf) and bf >= bf_thresh:
                                edges_global.append((src, dst))
                                edge_labels_global[(src, dst)] = f"{bf:.1f}"
                
                st.subheader("Global Network (Neighbors Only)")
                if not edges_global:
                    st.info(f"No significant relationships (BF₁₀ ≥ {bf_thresh}) "
                           "detected between neighboring countries.")
                else:
                    st.success(f"Found {len(edges_global)} significant neighbor relationships")
                    draw_directed_network(
                        common, edges_global,
                        f"Global Granger Network - Neighbors Only (BF₁₀ ≥ {bf_thresh})",
                        edge_labels_global
                    )
    
    # ======================================================================
    # SECTION 5: SPATIAL ANALYSIS
    # ======================================================================
    st.markdown("---")
    st.header("🗺️ Spatial Autocorrelation Analysis")
    st.info("Examining whether mortality rates in neighboring countries are more similar "
            "than would be expected by chance (Moran's I)")
    
    spatial_year = st.selectbox(
        "Select year for spatial analysis",
        sorted(df["Year"].unique(), reverse=True)
    )
    
    try:
        spatial_result = compute_spatial_autocorrelation(df, spatial_year, cause_code)
        
        if spatial_result:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Moran's I", f"{spatial_result['morans_i']:.3f}")
                if spatial_result['morans_i'] > 0.3:
                    st.success("Strong positive spatial autocorrelation")
                elif spatial_result['morans_i'] > 0:
                    st.info("Weak positive spatial autocorrelation")
                else:
                    st.warning("Negative or no spatial autocorrelation")
            with col2:
                st.metric("Countries Analyzed", spatial_result['n_countries'])
            
            st.markdown("""
            **Interpretation:**
            - **Moran's I > 0:** Similar values cluster together spatially
            - **Moran's I ≈ 0:** Random spatial pattern
            - **Moran's I < 0:** Dissimilar values cluster together
            """)
        else:
            st.warning("Insufficient data for spatial autocorrelation analysis")
    except Exception as e:
        st.warning(f"Could not compute spatial autocorrelation: {str(e)}")
    
    # ======================================================================
    # SECTION 6: DOWNLOAD REPORT
    # ======================================================================
    st.markdown("---")
    st.header("📥 Download Analysis Report")
    
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        # Changepoints
        if not changepoint_df.empty:
            zf.writestr("changepoints_apc.csv", changepoint_df.to_csv(index=False))
        
        # Forecasts
        for sf, fc in forecasts.items():
            zf.writestr(f"forecast_{sf}.csv", fc.to_csv(index=False))
        
        # Regression
        if not panel_clean.empty:
            zf.writestr("regression_panel_data.csv", panel_clean.to_csv(index=False))
            zf.writestr("regression_coefficients.csv", 
                       reg_coefs.to_frame("Coefficient").to_csv())
        
        # Clusters
        if not clust_df.empty:
            zf.writestr("cluster_assignments.csv", clust_df.to_csv(index=False))
        
        # Granger causality
        if 'bf_mat' in locals() and not bf_mat.empty:
            zf.writestr("global_granger_bf10.csv", bf_mat.to_csv())
            if edges_global:
                zf.writestr("global_network_edges.csv",
                           pd.DataFrame(edges_global, 
                                      columns=["source", "target"]).to_csv(index=False))
        
        if 'bf_n' in locals() and not bf_n.empty:
            zf.writestr("neighbor_granger_bf10.csv", bf_n.to_csv())
            if edges_n:
                zf.writestr("neighbor_network_edges.csv",
                           pd.DataFrame(edges_n,
                                      columns=["source", "target"]).to_csv(index=False))
        
        # Metadata
        metadata = f"""
European Public Health Dashboard - Analysis Report
Generated: {pd.Timestamp.now()}

Analysis Parameters:
- Country: {country_full}
- Cause: {cause_full}
- Period: {year_range[0]}-{year_range[1]}
- Forecast Method: {method}
- Forecast Horizon: {forecast_yrs} years
"""
        zf.writestr("README.txt", metadata)
    
    zip_buf.seek(0)
    st.download_button(
        label="📦 Download Complete Analysis Report (ZIP)",
        data=zip_buf,
        file_name=f"health_analysis_{country_code}_{cause_code}_{pd.Timestamp.now().strftime('%Y%m%d')}.zip",
        mime="application/zip",
        type="primary"
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
    <p>European Public Health Dashboard v2.0</p>
    <p>Data Source: Eurostat | Analysis Framework: Advanced Time Series & Spatial Methods</p>
    </div>
    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error("🚨 An unexpected error occurred. Please check your selections and try again.")
        st.error(f"Error details: {str(e)}")
        import traceback
        with st.expander("Show full error trace"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
