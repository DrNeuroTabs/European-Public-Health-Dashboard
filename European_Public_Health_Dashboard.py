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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pycountry
from requests.exceptions import HTTPError
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
from scipy import stats
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------------------------------
# CONSTANTS & MAPPINGS
# --------------------------------------------------------------------------
EU_CODES = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "EL", "HU", "IE",
    "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"
]

NEIGHBORS = {
    # EU Members
    "AT": ["DE", "CZ", "SK", "HU", "SI", "IT", "CH", "LI"],
    "BE": ["FR", "DE", "NL", "LU"],
    "BG": ["RO", "EL"],
    "HR": ["SI", "HU"],
    "CY": [],
    "CZ": ["DE", "PL", "SK", "AT"],
    "DK": ["DE"],
    "EE": ["LV"],
    "FI": ["SE", "NO"],
    "FR": ["BE", "LU", "DE", "IT", "ES", "CH"],
    "DE": ["DK", "PL", "CZ", "AT", "FR", "LU", "BE", "NL", "CH"],
    "EL": ["BG"],
    "HU": ["AT", "SK", "RO", "HR"],
    "IE": ["UK"],
    "IT": ["FR", "AT", "SI", "CH"],
    "LV": ["EE", "LT"],
    "LT": ["LV", "PL"],
    "LU": ["BE", "DE", "FR"],
    "MT": [],
    "NL": ["BE", "DE"],
    "PL": ["DE", "CZ", "SK", "LT"],
    "PT": ["ES"],
    "RO": ["BG", "HU"],
    "SK": ["CZ", "PL", "HU", "AT"],
    "SI": ["IT", "AT", "HU", "HR"],
    "ES": ["FR", "PT"],
    "SE": ["FI", "NO"],
    # EFTA & Former Members
    "CH": ["FR", "DE", "IT", "AT", "LI"],
    "LI": ["CH", "AT"],
    "NO": ["SE", "FI"],
    "IS": [],
    "UK": ["IE"]
}

SEX_NAME_MAP = {"T": "Total", "M": "Male", "F": "Female"}
REV_SEX_NAME = {v: k for k, v in SEX_NAME_MAP.items()}

AGE_NAME_MAP = {"TOTAL": "All Ages", "Y_LT65": "Under 65 (Premature)", "Y_GE65": "65 and Over"}
REV_AGE_NAME_MAP = {v: k for k, v in AGE_NAME_MAP.items()}

# Restored FULL cause map
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
    "GDP per capita (Volume)": "sdg_08_10",
    "Health care expenditure": "hlth_sha11_hp",
    "BMI by citizenship": "hlth_ehis_bm1c",
    "Phys activity by citizenship": "hlth_ehis_pe9c",
    "Smoking by citizenship": "hlth_ehis_sk1c",
    "Available beds in hospitals": "hlth_rs_bdsrg2",
    "Staff – physicians": "hlth_rs_prs2",
    "Consultations": "hlth_ehis_am1e"
}

def alpha3_from_a2(a2: str):
    c = pycountry.countries.get(alpha_2=a2)
    return c.alpha_3 if c else None

# --------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------
@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
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
        raise HTTPError(f"Could not fetch or find local file for {dataset_id}")

    first = raw.columns[0]
    dims = first.split("\\")[0].split(",")
    raw = raw.rename(columns={first: "series_keys"})
    keys = raw["series_keys"].str.split(",", expand=True)
    keys.columns = dims
    df = pd.concat([keys, raw.drop(columns=["series_keys"])], axis=1)

    years = [c for c in df.columns if c not in dims]
    long = df.melt(id_vars=dims, value_vars=years, var_name="Year", value_name="raw_rate")
    long["Year"] = long["Year"].astype(int)
    long["Rate"] = pd.to_numeric(long["raw_rate"].replace(":", np.nan), errors="coerce")

    mask = pd.Series(True, index=long.index)
    
    if "unit" in dims:
        uv = "RT" if "RT" in long["unit"].unique() else ("NR" if "NR" in long["unit"].unique() else None)
        if dataset_id == "sdg_08_10": uv = "CLV10_EUR_HAB"
        if uv: mask &= (long["unit"] == uv)
            
    if "freq" in dims:
        mask &= (long["freq"] == "A")
        
    if "age" in dims:
        mask &= (long["age"].isin(["TOTAL", "Y_LT65", "Y_GE65"]))
        
    if "resid" in dims:
        mask &= (long["resid"] == "TOT_IN")

    sub = long[mask].copy()
    
    rename = {"geo": "Country", "sex": "Sex", "age": "Age"}
    others = [d for d in dims if d not in ("geo", "sex", "age", "freq", "unit", "resid")]
    if len(others) >= 1:
        rename[others[0]] = "Cause" 
        
    out = sub.rename(columns=rename)
    
    if "Age" not in out.columns: out["Age"] = "TOTAL"
    if "Sex" not in out.columns: out["Sex"] = "T"
    if "Cause" not in out.columns: out["Cause"] = "TOTAL"
        
    cols = ["Country", "Year", "Cause", "Sex", "Age", "Rate"]
    return out[[c for c in cols if c in out.columns]]

@st.cache_data
def load_data() -> pd.DataFrame:
    def ld(ds):
        x = load_eurostat_series(ds)
        return x.dropna(subset=["Rate"])

    hist = ld("hlth_cd_asdr")
    mod = ld("hlth_cd_asdr2")
    mod = mod[mod["Country"].str.fullmatch(r"[A-Z]{2}")]
    df = pd.concat([hist, mod], ignore_index=True).sort_values(["Country", "Cause", "Sex", "Age", "Year"])

    df_eu = df[df["Country"].isin(EU_CODES)].groupby(["Year", "Cause", "Sex", "Age"], as_index=False)["Rate"].mean()
    df_eu["Country"] = "EU"

    df_eur = df.groupby(["Year", "Cause", "Sex", "Age"], as_index=False)["Rate"].mean()
    df_eur["Country"] = "Europe"

    return pd.concat([df, df_eu, df_eur], ignore_index=True)

@st.cache_data
def load_all_factors() -> pd.DataFrame:
    frames = []
    for name, ds in FACTOR_IDS.items():
        try:
            f = load_eurostat_series(ds)
        except HTTPError:
            continue
            
        f.loc[f["Country"].str.startswith("EU"), "Country"] = "EU"
        f = f[["Country", "Year", "Sex", "Age", "Rate"]].copy()
        f["FactorName"] = name
        frames.append(f)

    if not frames:
        return pd.DataFrame(columns=["Country", "Year", "Sex", "Age", "Rate", "FactorName"])
    return pd.concat(frames, ignore_index=True)

# --------------------------------------------------------------------------
# ANALYTICS ENGINE (Changepoints, Outliers, Forecasting)
# --------------------------------------------------------------------------
def detect_change_points(ts, pen: float = 3) -> list:
    ts = pd.Series(ts).dropna()
    if len(ts) < 2:
        return []
    algo = rpt.Pelt(model="l2").fit(ts.values)
    try:
        return algo.predict(pen=pen)
    except BadSegmentationParameters:
        return []

def compute_changepoints_and_apc(df_sub: pd.DataFrame) -> pd.DataFrame:
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
                continue
            
            slope = sm.OLS(sv, sm.add_constant(yrs[seg])).fit().params[1]
            apc = (slope / np.nanmean(sv)) * 100
            recs.append({
                "Sex": SEX_NAME_MAP.get(sex, str(sex)),
                "start_year": sy,
                "end_year": ey,
                "slope": slope,
                "APC_pct": apc
            })
    return pd.DataFrame(recs)

def detect_anomalies(df_sub: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    res = []
    for sex in df_sub["Sex"].unique():
        part = df_sub[df_sub["Sex"] == sex].copy()
        if len(part) > 3:
            mean = part["Rate"].mean()
            std = part["Rate"].std()
            part["Z_Score"] = (part["Rate"] - mean) / std
            part["Is_Anomaly"] = part["Z_Score"].abs() > threshold
            res.append(part)
    
    if res:
        return pd.concat(res)
    return df_sub.assign(Z_Score=0, Is_Anomaly=False)

def forecast_mortality(df_sub: pd.DataFrame, periods: int, method: str) -> pd.DataFrame:
    if df_sub.empty or len(df_sub) < 3:
        return pd.DataFrame(columns=["Year", "History", "Forecast"])
        
    dfp = df_sub[["Year", "Rate"]].rename(columns={"Year": "ds", "Rate": "y"}).copy()
    dfp["ds"] = pd.to_datetime(dfp["ds"].astype(str) + "-01-01", format="%Y-%m-%d")
    m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
    m.fit(dfp)
    fut = m.make_future_dataframe(periods=periods, freq="YS")
    fc_prophet = m.predict(fut)[["ds", "yhat"]].rename(columns={"yhat": "Prophet"})
    fc_prophet["Year"] = fc_prophet["ds"].dt.year
    
    ser = df_sub.set_index("Year")["Rate"]
    ari_preds = ARIMA(ser, order=(1, 1, 1)).fit().forecast(periods)
    yrs = np.arange(ser.index.max() + 1, ser.index.max() + 1 + periods)
    fc_arima = pd.DataFrame({"Year": yrs, "ARIMA": ari_preds.values})
    
    ets_preds = ExponentialSmoothing(ser, trend="add", seasonal=None).fit(optimized=True).forecast(periods)
    fc_ets = pd.DataFrame({"Year": yrs, "ETS": ets_preds.values})
    
    fc = fc_prophet.merge(fc_arima, on="Year", how="outer").merge(fc_ets, on="Year", how="outer")
    
    if method == "Prophet": fc["Forecast"] = fc["Prophet"]
    elif method == "ARIMA": fc["Forecast"] = fc["ARIMA"]
    elif method == "ETS": fc["Forecast"] = fc["ETS"]
    else: fc["Forecast"] = fc[["Prophet", "ARIMA", "ETS"]].mean(axis=1)

    hist = df_sub[["Year", "Rate"]].rename(columns={"Rate": "History"})
    return hist.merge(fc[["Year", "Forecast"]], on="Year", how="outer")

# --------------------------------------------------------------------------
# SPATIAL & NETWORK HELPERS
# --------------------------------------------------------------------------
def has_physical_border(src_code: str, dst_code: str) -> bool:
    if src_code is None or dst_code is None: return False
    return dst_code in NEIGHBORS.get(src_code, [])

def compute_spatial_autocorrelation(df, year, cause_code, sex_code, age_code):
    rates_df = df[
        (df["Country"].isin(NEIGHBORS.keys())) &
        (df["Cause"] == cause_code) &
        (df["Sex"] == sex_code) &
        (df["Age"] == age_code) &
        (df["Year"] == year)
    ][["Country", "Rate"]].set_index("Country")

    if len(rates_df) < 5: return None

    countries = rates_df.index.tolist()
    n = len(countries)
    W = np.zeros((n, n))

    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if has_physical_border(c1, c2): W[i, j] = 1

    row_sums = W.sum(axis=1)
    W = np.divide(W, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)

    rates = rates_df["Rate"].values
    mean_rate = rates.mean()
    z = rates - mean_rate

    if W.sum() == 0 or np.sum(z ** 2) == 0: return None

    numerator = np.sum(W * np.outer(z, z))
    denominator = np.sum(z ** 2)
    moran_i = (n / W.sum()) * (numerator / denominator)

    return {"morans_i": moran_i, "n_countries": n}

def make_lag_matrix(series: np.ndarray, maxlag: int) -> np.ndarray:
    return np.column_stack([series[maxlag - lag:-lag] for lag in range(1, maxlag + 1)])

def compute_granger_causality_bic(pair_df: pd.DataFrame, maxlag: int) -> dict:
    df_pair = pair_df.dropna().copy()
    if df_pair.shape[0] < (2 * maxlag + 5): return {"approx_bf10": np.nan, "p_value": np.nan}

    y_all, x_all = df_pair.iloc[:, 0].values.astype(float), df_pair.iloc[:, 1].values.astype(float)
    y = y_all[maxlag:]
    target_lags = make_lag_matrix(y_all, maxlag)
    cause_lags = make_lag_matrix(x_all, maxlag)

    X_null = sm.add_constant(target_lags, has_constant="add")
    X_alt = sm.add_constant(np.column_stack([target_lags, cause_lags]), has_constant="add")

    try:
        m0 = sm.OLS(y, X_null).fit()
        m1 = sm.OLS(y, X_alt).fit()

        approx_bf10 = float(np.exp((m0.bic - m1.bic) / 2.0))
        df_num, df_den = m0.df_resid - m1.df_resid, m1.df_resid

        if df_num <= 0 or df_den <= 0: return {"approx_bf10": approx_bf10, "p_value": np.nan}

        f_stat = ((m0.ssr - m1.ssr) / df_num) / (m1.ssr / df_den)
        p_value = np.nan if np.isnan(f_stat) or np.isinf(f_stat) else 1 - stats.f.cdf(f_stat, df_num, df_den)

        return {"approx_bf10": approx_bf10, "p_value": p_value}
    except Exception:
        return {"approx_bf10": np.nan, "p_value": np.nan}

# --------------------------------------------------------------------------
# MAIN APPLICATION & TABS
# --------------------------------------------------------------------------
def main():
    try:
        st.set_page_config(layout="wide", page_title="Euro Health Dashboard", page_icon="🏥")

        st.title("🏥 European Public Health Dashboard")
        st.markdown("### Advanced Mortality Trend Analysis, Spatial Mapping & Network Dynamics")
        st.markdown("---")

        with st.spinner("Loading epidemiological and economic datasets..."):
            df = load_data()
            df["CountryFull"] = df["Country"].map(COUNTRY_NAME_MAP).fillna(df["Country"])
            df["CauseFull"] = df["Cause"].map(CAUSE_NAME_MAP).fillna(df["Cause"])
            df["SexFull"] = df["Sex"].map(SEX_NAME_MAP).fillna(df["Sex"])

        # --- SIDEBAR FILTERS ---
        st.sidebar.header("📊 Global Data Filters")

        countries = sorted(df["CountryFull"].dropna().unique())
        default_country = "European Union" if "European Union" in countries else countries[0]
        country_full = st.sidebar.selectbox("Country", countries, index=countries.index(default_country))
        country_code = REV_COUNTRY_NAME_MAP.get(country_full, country_full)

        causes = sorted(df["CauseFull"].dropna().unique())
        cause_full = st.sidebar.selectbox("Cause of Death", causes)
        cause_code = REV_CAUSE_NAME_MAP.get(cause_full, cause_full)
        
        age_sel = st.sidebar.selectbox("Age Cohort", list(AGE_NAME_MAP.values()), index=0)
        age_code = REV_AGE_NAME_MAP[age_sel]

        sex_sel = st.sidebar.multiselect("Sex", ["Total", "Male", "Female"], default=["Total"])
        sex_codes = [REV_SEX_NAME[s] for s in sex_sel]

        yrs = sorted(df["Year"].dropna().unique())
        year_range = st.sidebar.slider("Analysis Period", int(yrs[0]), int(yrs[-1]), (int(yrs[0]), int(yrs[-1])))

        st.sidebar.markdown("---")
        st.sidebar.header("🔮 Forecast Settings")
        forecast_yrs = st.sidebar.slider("Forecast Horizon (years)", 1, 30, 10)
        method = st.sidebar.selectbox("Forecast Method", ["Ensemble", "Prophet", "ARIMA", "ETS"])

        df_filtered = df[
            (df["Country"] == country_code) &
            (df["Cause"] == cause_code) &
            (df["Age"] == age_code) &
            (df["Sex"].isin(sex_codes)) &
            (df["Year"].between(*year_range))
        ]

        # --- APP TABS ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Temporal Trends", 
            "⚖️ Cross-Cause Comparison", 
            "🔬 Health & Macro Factors", 
            "🗺️ Spatial & Clusters", 
            "🔗 Granger Network"
        ])

        # ======================================================================
        # TAB 1: TEMPORAL TRENDS
        # ======================================================================
        with tab1:
            st.header(f"📈 Temporal Trend Analysis: {cause_full}")
            st.markdown(f"**Cohort:** {age_sel} | **Country:** {country_full}")

            if df_filtered.empty:
                st.warning("⚠️ No data available for selected filters.")
            else:
                df_anom = detect_anomalies(df_filtered)
                changepoint_df = compute_changepoints_and_apc(df_filtered)

                st.subheader("🔍 Trend Segmentation & Outlier Detection")
                
                sub = df_filtered.sort_values("Year")
                fig = go.Figure()
                
                for sc, sf in zip(sex_codes, sex_sel):
                    part = sub[sub["Sex"] == sc]
                    yrs_arr, rates_arr = part["Year"].values, part["Rate"].values
                    bkps = detect_change_points(rates_arr)[:-1]
                    segs = np.split(np.arange(len(yrs_arr)), bkps) if bkps else [np.arange(len(yrs_arr))]

                    fig.add_trace(go.Scatter(x=yrs_arr, y=rates_arr, mode="markers+lines", name=f"Observed ({sf})", opacity=0.4))
                    
                    anoms = df_anom[(df_anom["Sex"] == sc) & (df_anom["Is_Anomaly"] == True)]
                    if not anoms.empty:
                        fig.add_trace(go.Scatter(
                            x=anoms["Year"], y=anoms["Rate"], mode="markers", 
                            marker=dict(color="red", size=10, symbol="x", line=dict(width=2, color="DarkRed")),
                            name=f"Anomaly ({sf})"
                        ))

                    palette = px.colors.qualitative.Dark24
                    for i, seg in enumerate(segs):
                        idx, vals = yrs_arr[seg], rates_arr[seg]
                        if len(vals) >= 2 and not np.all(np.isnan(vals)):
                            fit = sm.OLS(vals, sm.add_constant(idx)).fit().predict(sm.add_constant(idx))
                            fig.add_trace(go.Scatter(
                                x=idx, y=fit, mode="lines",
                                line=dict(color=palette[i % len(palette)], width=4),
                                name=f"Trend Seg {i+1} ({sf})"
                            ))

                fig.update_layout(height=500, hovermode="x unified", title="Segmented Trends with Z-Score Anomalies (>2.5σ)")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Segment Statistics")
                if not changepoint_df.empty:
                    st.dataframe(changepoint_df.style.format({"slope": "{:.2f}", "APC_pct": "{:.2f}%"}), use_container_width=True)
                else:
                    st.info("No significant segments detected.")

                st.subheader(f"🔮 {forecast_yrs}-Year Forecast ({method})")
                forecast_cols = st.columns(len(sex_sel))
                for idx, (sc, sf) in enumerate(zip(sex_codes, sex_sel)):
                    fc = forecast_mortality(df_filtered[df_filtered["Sex"] == sc], forecast_yrs, method)
                    with forecast_cols[idx]:
                        if not fc.empty:
                            fig_fc = px.line(fc, x="Year", y=["History", "Forecast"], title=f"{sf}")
                            fig_fc.add_vline(x=year_range[1], line_dash="dash", line_color="gray", opacity=0.5)
                            st.plotly_chart(fig_fc, use_container_width=True)
                        else:
                            st.warning(f"Not enough data to forecast for {sf}.")

        # ======================================================================
        # TAB 2: CROSS-CAUSE COMPARISON
        # ======================================================================
        with tab2:
            st.header("⚖️ Cross-Cause Trend Comparison (Displacement Analysis)")
            st.info("Compare two different causes of death to identify inverse relationships. Rates are normalized to 100 at the start year.")
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                c1_full = st.selectbox("Select Cause 1", causes, index=causes.index(cause_full) if cause_full in causes else 0)
                c1_code = REV_CAUSE_NAME_MAP.get(c1_full, c1_full)
            with col_c2:
                c2_full = st.selectbox("Select Cause 2", causes, index=0)
                c2_code = REV_CAUSE_NAME_MAP.get(c2_full, c2_full)

            df_compare = df[
                (df["Country"] == country_code) & 
                (df["Age"] == age_code) & 
                (df["Sex"] == "T") & 
                (df["Year"].between(*year_range)) &
                (df["Cause"].isin([c1_code, c2_code]))
            ]
            
            if not df_compare.empty and c1_code != c2_code:
                pivot_comp = df_compare.pivot(index="Year", columns="Cause", values="Rate").dropna()
                if len(pivot_comp) > 2 and c1_code in pivot_comp.columns and c2_code in pivot_comp.columns:
                    norm_comp = (pivot_comp / pivot_comp.iloc[0]) * 100
                    
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Scatter(x=norm_comp.index, y=norm_comp[c1_code], name=c1_full, line=dict(width=3)))
                    fig_comp.add_trace(go.Scatter(x=norm_comp.index, y=norm_comp[c2_code], name=c2_full, line=dict(width=3)))
                    fig_comp.update_layout(title="Indexed Growth Comparison (Start Year = 100)", yaxis_title="Indexed Value", hovermode="x unified")
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    corr = pivot_comp[c1_code].corr(pivot_comp[c2_code])
                    st.metric("Pearson Correlation Coefficient", f"{corr:.3f}", help="Close to -1 implies strong displacement")
                else:
                    st.warning("Not enough overlapping years between the two causes to compare.")

        # ======================================================================
        # TAB 3: HEALTH & MACRO FACTORS
        # ======================================================================
        with tab3:
            st.header("🔬 Socioeconomic & Health Factor Regression")
            st.info("Assess how macro-economic variables (like GDP per capita) and health system capacities impact mortality.")

            factors = st.multiselect("Select factors to include in OLS model", list(FACTOR_IDS.keys()), default=["GDP per capita (Volume)"])

            if factors:
                with st.spinner("Compiling cross-sectional panel data..."):
                    all_factors = load_all_factors()

                pf = all_factors[
                    (all_factors["FactorName"].isin(factors)) &
                    (all_factors["Year"].between(*year_range)) &
                    (all_factors["Age"] == age_code)
                ]

                pm = df[
                    (df["Cause"] == cause_code) &
                    (df["Age"] == age_code) &
                    (df["Year"].between(*year_range)) &
                    (df["Sex"] == "T")
                ][["Country", "Year", "Rate"]].rename(columns={"Rate": "Mortality"})

                if pf.empty or pm.empty:
                    st.warning("⚠️ No factor or mortality data available for the selected filters.")
                else:
                    panel = pf.pivot_table(index=["Country", "Year"], columns="FactorName", values="Rate").reset_index()
                    panel = panel.merge(pm, on=["Country", "Year"], how="inner")

                    present = [f for f in factors if f in panel.columns]
                    
                    if not present:
                        st.warning("⚠️ None of the selected factors have matching data for this cohort.")
                    else:
                        panel_clean = panel.dropna(subset=present + ["Mortality"])

                        # The safety check that fixes the previous crash
                        if panel_clean.empty or panel_clean.shape[0] < len(present) + 2:
                            st.warning(f"⚠️ Insufficient continuous observations (found {panel_clean.shape[0]}) for reliable regression.")
                        else:
                            try:
                                X = sm.add_constant(panel_clean[present])
                                y = panel_clean["Mortality"]
                                mdl = sm.OLS(y, X).fit()

                                col_r1, col_r2 = st.columns([3, 2])
                                with col_r1:
                                    st.text(mdl.summary())
                                with col_r2:
                                    reg_coefs = mdl.params.drop("const")
                                    fig_coef = go.Figure(go.Bar(
                                        x=reg_coefs.values, y=reg_coefs.index, orientation="h",
                                        marker_color=["green" if x < 0 else "red" for x in reg_coefs.values]
                                    ))
                                    fig_coef.update_layout(title="Factor Coefficients (Effect on Mortality)")
                                    st.plotly_chart(fig_coef, use_container_width=True)
                            except ValueError as e:
                                st.warning("⚠️ Could not compute regression. The data might be perfectly collinear or contain invalid mathematical shapes for OLS.")

        # ======================================================================
        # TAB 4: SPATIAL & CLUSTERS
        # ======================================================================
        with tab4:
            st.header("🗺️ Spatial Analysis & Animated Mapping")
            
            st.subheader("Time-Lapse: Spread of Mortality Rates")
            anim_df = df[
                (df["Cause"] == cause_code) & 
                (df["Sex"] == "T") & 
                (df["Age"] == age_code) &
                (df["Year"].between(*year_range)) &
                (df["Country"].str.len() == 2) 
            ].copy()
            
            anim_df["iso_alpha"] = anim_df["Country"].apply(alpha3_from_a2)
            
            if not anim_df.empty:
                fig_map = px.choropleth(
                    anim_df, locations="iso_alpha", color="Rate", hover_name="CountryFull",
                    animation_frame="Year", locationmode="ISO-3", scope="europe",
                    color_continuous_scale="Reds", range_color=[anim_df["Rate"].min(), anim_df["Rate"].max()]
                )
                fig_map.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)

            st.markdown("---")
            col_sp1, col_sp2 = st.columns(2)
            
            with col_sp1:
                st.subheader("Spatial Autocorrelation (Moran's I)")
                spatial_year = st.selectbox("Select Year", sorted(anim_df["Year"].unique(), reverse=True))
                spatial_result = compute_spatial_autocorrelation(df, spatial_year, cause_code, "T", age_code)
                if spatial_result:
                    mi = spatial_result['morans_i']
                    st.metric("Moran's I", f"{mi:.3f}")
                    if mi > 0.3: st.success("Strong clustering (High rates border high rates)")
                    elif mi > 0: st.info("Weak clustering")
                    else: st.warning("Random or Dispersed pattern")

            with col_sp2:
                st.subheader("Dynamic K-Means Clustering")
                pivot_clust = anim_df.pivot(index="Country", columns="Year", values="Rate").dropna(axis=0, how="any")
                if pivot_clust.shape[0] > 5:
                    best_k, best_score = 2, -1
                    X_clust = pivot_clust.values
                    for k in range(2, 6):
                        score = silhouette_score(X_clust, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_clust))
                        if score > best_score: best_k, best_score = k, score
                    
                    st.metric("Optimal Clusters Identified", best_k, f"Silhouette: {best_score:.2f}")
                    labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_clust)
                    pivot_clust["Cluster"] = [f"Profile {l+1}" for l in labels]
                    
                    st.dataframe(pivot_clust[["Cluster"]].reset_index(), use_container_width=True)

        # ======================================================================
        # TAB 5: GRANGER NETWORK
        # ======================================================================
        with tab5:
            st.header("🔗 Granger Causality Spillovers")
            st.info("Identifies if historical mortality rates in one country predict future rates in a neighboring country. Edges map directional influence.")

            gl_maxlag = st.slider("Max Lag (Years)", 1, 5, 2)
            bf_thresh = st.number_input("BF₁₀ threshold", 1.0, 100.0, 3.0, 0.5)
            
            df_g = df[
                (df["Cause"] == cause_code) &
                (df["Sex"] == "T") &
                (df["Age"] == age_code) &
                (df["Year"].between(*year_range)) &
                (df["Country"].str.len() == 2)
            ]
            
            pivot_gc = df_g.pivot_table(index="Year", columns="Country", values="Rate")
            valid_countries = [c for c in pivot_gc.columns if pivot_gc[c].count() > (2*gl_maxlag + 5)]

            if len(valid_countries) < 2:
                st.warning("Insufficient continuous data for network generation.")
            else:
                if st.button("🚀 Generate Global Spillover Network"):
                    with st.spinner("Running pairwise OLS models..."):
                        bf_mat = pd.DataFrame(np.nan, index=valid_countries, columns=valid_countries)
                        
                        for src in valid_countries:
                            for dst in valid_countries:
                                if src == dst or not has_physical_border(src, dst): continue
                                pair = pivot_gc[[dst, src]].dropna()
                                if len(pair) > 5:
                                    res = compute_granger_causality_bic(pair, gl_maxlag)
                                    bf_mat.loc[src, dst] = res["approx_bf10"]

                        edges = []
                        for src in valid_countries:
                            for dst in valid_countries:
                                bf = bf_mat.loc[src, dst]
                                if pd.notna(bf) and bf >= bf_thresh:
                                    edges.append({"source": src, "target": dst, "BF10": bf})
                        
                        edge_df = pd.DataFrame(edges)
                        
                        if not edge_df.empty:
                            st.success(f"Network generated: {len(edge_df)} significant edges found.")
                            fig_net = px.imshow(bf_mat.astype(float), color_continuous_scale="YlOrRd", title="Adjacency Matrix (Approx BF10)")
                            st.plotly_chart(fig_net, use_container_width=True)
                            st.dataframe(edge_df.sort_values("BF10", ascending=False), use_container_width=True)
                        else:
                            st.info("No significant predictive spillovers detected at current threshold.")

    except Exception as e:
        st.error("🚨 An unexpected error occurred. Please check your selections and try again.")
        import traceback
        with st.expander("Show full error trace"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
