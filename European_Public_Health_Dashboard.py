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
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
EU_CODES = ["AT","BE","BG","HR","CY","CZ","DK","EE",
            "FI","FR","DE","EL","HU","IE","IT","LV",
            "LT","LU","MT","NL","PL","PT","RO","SK",
            "SI","ES","SE"]

NEIGHBORS = {
    "AT":["DE","CZ","SK","HU","SI","IT"], "BE":["FR","DE","NL","LU"],
    "BG":["RO","EL"], "HR":["SI","HU"], "CY":[], "CZ":["DE","PL","SK","AT"],
    "DK":["DE"], "EE":["LV"], "FI":["SE"], "FR":["BE","LU","DE","IT","ES"],
    "DE":["DK","PL","CZ","AT","FR","LU","BE","NL"], "EL":["BG"],
    "HU":["AT","SK","RO","HR"], "IE":[], "IT":["FR","AT","SI"],
    "LV":["EE","LT"], "LT":["LV","PL"], "LU":["BE","DE","FR"], "MT":[],
    "NL":["BE","DE"], "PL":["DE","CZ","SK","LT"], "PT":["ES"], "RO":["BG","HU"],
    "SK":["CZ","PL","HU","AT"], "SI":["IT","AT","HU","HR"], "ES":["FR","PT"],
    "SE":["FI"]
}

SEX_NAME_MAP = {"T":"Total","M":"Male","F":"Female"}
REV_SEX_NAME = {v:k for k,v in SEX_NAME_MAP.items()}

CAUSE_NAME_MAP = {
    "TOTAL":"Total",
    "A_B":"Certain infectious and parasitic diseases (A00-B99)",
    "A15-A19_B90":"Tuberculosis",
    "B15-B19_B942":"Viral hepatitis and sequelae of viral hepatitis",
    "B180-B182":"Chronic viral hepatitis B and C",
    "B20-B24":"Human immunodeficiency virus [HIV] disease",
    "A_B_OTH":"Other infectious and parasitic diseases (A00-B99)",
    "C00-D48":"Neoplasms",
    "C":"Malignant neoplasms (C00-C97)",
    # ... include all mappings as before ...
    "U072":"COVID-19, virus not identified"
}
REV_CAUSE_NAME_MAP = {v:k for k,v in CAUSE_NAME_MAP.items()}

COUNTRY_NAME_MAP = {c.alpha_2:c.name for c in pycountry.countries}
COUNTRY_NAME_MAP.update({"FX":"France (Metropolitan)","EU":"European Union","Europe":"Europe"})
REV_COUNTRY_NAME_MAP = {v:k for k,v in COUNTRY_NAME_MAP.items()}

FACTOR_IDS = {
    "BMI by citizenship":"hlth_ehis_bm1c",
    # ... include all factor IDs as before ...
    "Unmet needs":"hlth_ehis_un1e"
}

def alpha3_from_a2(a2:str):
    c = pycountry.countries.get(alpha_2=a2)
    return c.alpha_3 if c else None

# --------------------------------------------------------------------------
# DATA LOADERS
# --------------------------------------------------------------------------
@st.cache_data
def load_eurostat_series(dataset_id:str)->pd.DataFrame:
    base="https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
    endpoints=[f"{base}/{dataset_id}?format=TSV&compressed=true",
               f"{base}/{dataset_id}?format=TSV"]
    raw=None
    for url in endpoints:
        try:
            r=requests.get(url,timeout=30); r.raise_for_status()
            if "compressed=true" in url:
                buf=BytesIO(r.content)
                with gzip.GzipFile(fileobj=buf) as gz:
                    raw=pd.read_csv(gz,sep="\t",low_memory=False)
            else:
                raw=pd.read_csv(BytesIO(r.content),sep="\t",low_memory=False)
            break
        except HTTPError:
            raw=None
    if raw is None:
        files=glob.glob(f"/mnt/data/estat_{dataset_id}*.tsv")
        if files:
            raw=pd.read_csv(files[0],sep="\t",low_memory=False)
    if raw is None:
        raise HTTPError(f"Could not fetch or find local file for {dataset_id}")
    first=raw.columns[0]
    dims=first.split("\\")[0].split(",")
    raw=raw.rename(columns={first:"series_keys"})
    keys=raw["series_keys"].str.split(",",expand=True); keys.columns=dims
    df=pd.concat([keys,raw.drop(columns=["series_keys"])],axis=1)
    years=[c for c in df.columns if c not in dims]
    long=df.melt(id_vars=dims,value_vars=years,
                 var_name="Year",value_name="raw_rate")
    long["Year"]=long["Year"].astype(int)
    long["Rate"]=pd.to_numeric(long["raw_rate"].replace(":",np.nan),errors="coerce")
    mask=pd.Series(True,index=long.index)
    if "unit" in dims:
        uv="RT" if "RT" in long["unit"].unique() else ("NR" if "NR" in long["unit"].unique() else None)
        if uv: mask&=(long["unit"]==uv)
    if "freq" in dims:  mask&=(long["freq"]=="A")
    if "age" in dims:   mask&=(long["age"]=="TOTAL")
    if "sex" in dims:   mask&=(long["sex"]=="T")
    if "resid" in dims: mask&=(long["resid"]=="TOT_IN")
    sub=long[mask].copy()
    rename={"geo":"Region","sex":"Sex"}
    others=[d for d in dims if d not in ("geo","sex","freq","unit","age","resid")]
    if len(others)==1: rename[others[0]]="Category"
    out=sub.rename(columns=rename)
    cols=["Region","Year","Category","Sex","Rate"]
    return out[[c for c in cols if c in out.columns]]

@st.cache_data
def load_data()->pd.DataFrame:
    def ld(ds):
        x=load_eurostat_series(ds).rename(columns={"Region":"Country","Category":"Cause"})
        return x.dropna(subset=["Rate"])
    hist=ld("hlth_cd_asdr"); mod=ld("hlth_cd_asdr2")
    mod=mod[mod["Country"].str.fullmatch(r"[A-Z]{2}")]
    df=pd.concat([hist,mod],ignore_index=True).sort_values(["Country","Cause","Sex","Year"])
    df_eu=df[df["Country"].isin(EU_CODES)].groupby(["Year","Cause","Sex"],as_index=False)["Rate"].mean()
    df_eu["Country"]="EU"
    df_eur=df.groupby(["Year","Cause","Sex"],as_index=False)["Rate"].mean()
    df_eur["Country"]="Europe"
    return pd.concat([df,df_eu,df_eur],ignore_index=True)

@st.cache_data
def load_all_factors()->pd.DataFrame:
    frames=[]
    for name,ds in FACTOR_IDS.items():
        try:
            f=load_eurostat_series(ds).rename(columns={"Region":"Country"})
        except HTTPError:
            continue
        f.loc[f["Country"].str.startswith("EU"),"Country"]="EU"
        if "Sex" not in f.columns: f["Sex"]="T"
        if "Category" in f.columns: f=f[f["Category"]=="TOTAL"]
        f=f[["Country","Year","Sex","Rate"]].copy()
        f["FactorName"]=name
        frames.append(f)
    if not frames: return pd.DataFrame(columns=["Country","Year","Sex","Rate","FactorName"])
    return pd.concat(frames,ignore_index=True)

# --------------------------------------------------------------------------
# ANALYSIS UTILITIES
# --------------------------------------------------------------------------
def detect_change_points(ts,pen:float=3)->list:
    ts=pd.Series(ts).dropna()
    if len(ts)<2: return []
    algo=rpt.Pelt(model="l2").fit(ts.values)
    try: return algo.predict(pen=pen)
    except BadSegmentationParameters: return []

def compute_joinpoints_and_apc(df_sub:pd.DataFrame)->pd.DataFrame:
    recs=[]
    for sex in df_sub["Sex"].unique():
        part=df_sub[df_sub["Sex"]==sex].sort_values("Year")
        yrs,vals=part["Year"].values,part["Rate"].values
        bkps=detect_change_points(vals)[:-1]
        segs=np.split(np.arange(len(yrs)),bkps) if bkps else [np.arange(len(yrs))]
        for seg in segs:
            sy,ey=int(yrs[seg].min()),int(yrs[seg].max())
            sv=vals[seg]
            if len(sv)<2 or np.all(np.isnan(sv)):
                recs.append({"Sex":SEX_NAME_MAP[sex],"start_year":sy,"end_year":ey,
                             "slope":np.nan,"APC_pct":np.nan})
            else:
                slope=sm.OLS(sv,sm.add_constant(yrs[seg])).fit().params[1]
                apc=(slope/np.nanmean(sv))*100
                recs.append({"Sex":SEX_NAME_MAP[sex],"start_year":sy,"end_year":ey,
                             "slope":slope,"APC_pct":apc})
    return pd.DataFrame(recs)

def plot_joinpoints_comparative(df_sub:pd.DataFrame,title:str):
    df_sub["SexFull"]=df_sub["Sex"].map(SEX_NAME_MAP)
    fig=px.line(df_sub,x="Year",y="Rate",color="SexFull",title=title,markers=True)
    st.plotly_chart(fig)

def plot_segmented_fit_series(df_sub:pd.DataFrame,title:str):
    sub=df_sub.sort_values("Year")
    yrs,rates=sub["Year"].values,sub["Rate"].values
    bkps=detect_change_points(rates)[:-1]
    segs=np.split(np.arange(len(yrs)),bkps) if bkps else [np.arange(len(yrs))]
    fig=go.Figure(); fig.add_trace(go.Scatter(x=yrs,y=rates,mode="markers+lines",name="Data"))
    palette=px.colors.qualitative.Dark24
    for i,seg in enumerate(segs):
        idx,vals=yrs[seg],rates[seg]
        if len(vals)>=2 and not np.all(np.isnan(vals)):
            fit=sm.OLS(vals,sm.add_constant(idx)).fit().predict(sm.add_constant(idx))
            fig.add_trace(go.Scatter(x=idx,y=fit,mode="lines",
                                     line=dict(color=palette[i%len(palette)],width=3),
                                     name=f"Segment {i+1}"))
    fig.update_layout(title=title,xaxis_title="Year",yaxis_title="Rate")
    st.plotly_chart(fig)

def get_prophet_forecast(df_sub:pd.DataFrame,periods:int)->pd.DataFrame:
    dfp=df_sub[["Year","Rate"]].rename(columns={"Year":"ds","Rate":"y"})
    dfp["ds"]=pd.to_datetime(dfp["ds"].astype(str),format="%Y")
    m=Prophet(yearly_seasonality=False,daily_seasonality=False); m.fit(dfp)
    fut=m.make_future_dataframe(periods=periods,freq="Y"); fc=m.predict(fut)
    return pd.DataFrame({"Year":fc["ds"].dt.year,"Prophet":fc["yhat"]})

def get_arima_forecast(df_sub:pd.DataFrame,periods:int)->pd.DataFrame:
    ser=df_sub.set_index("Year")["Rate"]
    res=ARIMA(ser,order=(1,1,1)).fit(); preds=res.forecast(periods)
    yrs=np.arange(ser.index.max()+1,ser.index.max()+1+periods)
    return pd.DataFrame({"Year":yrs,"ARIMA":preds.values})

def get_ets_forecast(df_sub:pd.DataFrame,periods:int)->pd.DataFrame:
    ser=df_sub.set_index("Year")["Rate"]
    m=ExponentialSmoothing(ser,trend="add",seasonal=None).fit(optimized=True)
    preds=m.forecast(periods); yrs=np.arange(ser.index.max()+1,ser.index.max()+1+periods)
    return pd.DataFrame({"Year":yrs,"ETS":preds.values})

def forecast_mortality(df_sub:pd.DataFrame,periods:int,method:str,title:str):
    n=df_sub["Rate"].dropna().shape[0]
    if n<3: st.warning(f"Not enough data ({n}) to forecast."); return
    prop=get_prophet_forecast(df_sub,periods)
    ari=get_arima_forecast(df_sub,periods)
    ets=get_ets_forecast(df_sub,periods)
    fc=prop.merge(ari,on="Year").merge(ets,on="Year")
    if method=="Prophet": fc["Forecast"]=fc["Prophet"]
    elif method=="ARIMA": fc["Forecast"]=fc["ARIMA"]
    elif method=="ETS":   fc["Forecast"]=fc["ETS"]
    else:                fc["Forecast"]=fc[["Prophet","ARIMA","ETS"]].mean(axis=1)
    hist=df_sub[["Year","Rate"]].rename(columns={"Rate":"History"})
    combined=hist.merge(fc[["Year","Forecast"]],on="Year",how="outer")
    fig=px.line(combined,x="Year",y=["History","Forecast"],title=title); st.plotly_chart(fig)

def compute_bayes_factor_bic(pair_df:pd.DataFrame,maxlag:int)->float:
    df_pair=pair_df.dropna()
    if df_pair.shape[0]<maxlag+3: return np.nan
    target,cause=df_pair.columns
    Y=df_pair[target].values[maxlag:]
    X_alt=np.column_stack([df_pair[cause].values[maxlag-lag:-lag] for lag in range(1,maxlag+1)])
    X_alt=sm.add_constant(X_alt); X_null=np.ones((len(Y),1))
    m0=sm.OLS(Y,X_null).fit(); m1=sm.OLS(Y,X_alt).fit()
    return float(np.exp((m0.bic-m1.bic)/2.0))

# --------------------------------------------------------------------------
# DRAW DIRECTED NETWORK (clock labels + arrows atop nodes)
# --------------------------------------------------------------------------
def draw_directed_network(nodes,edges,title):
    G=nx.DiGraph(); G.add_nodes_from(nodes); G.add_edges_from(edges)
    n=len(nodes)
    angles=np.linspace(0,2*np.pi,n,endpoint=False)
    pos={nodes[i]:(np.cos(angles[i]),np.sin(angles[i])) for i in range(n)}
    fig,ax=plt.subplots(figsize=(6,6))
    # draw blue circles first
    nx.draw_networkx_nodes(G,pos,node_size=500,node_color='skyblue',ax=ax)
    # then arrows
    if edges:
        nx.draw_networkx_edges(G,pos,arrowstyle='-|>',arrowsize=20,width=2,ax=ax)
    # labels like clock ticks, rotated
    for i,node in enumerate(nodes):
        angle=angles[i]
        x,y= np.cos(angle)*1.2, np.sin(angle)*1.2
        deg= np.degrees(angle)
        rotation=deg-90
        ax.text(x,y,node,ha='center',va='center',rotation=rotation,
                rotation_mode='anchor',fontsize=10,
                bbox=dict(facecolor='white',edgecolor='none',pad=0.3))
    ax.set_title(title,pad=20)
    ax.set_axis_off()
    st.pyplot(fig)

# --------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide",page_title="European Public Health Dashboard")
    st.title("European Public Health Dashboard")
    st.markdown("by Younes Adam Tabi")

    df=load_data()
    df["CountryFull"]=df["Country"].map(COUNTRY_NAME_MAP)
    df["CauseFull"]=df["Cause"].map(CAUSE_NAME_MAP)
    df["SexFull"]=df["Sex"].map(SEX_NAME_MAP)

    # Sidebar filters
    countries=sorted(df["CountryFull"].dropna().unique())
    country_full=st.sidebar.selectbox("Country",countries,index=countries.index("European Union"))
    country_code=REV_COUNTRY_NAME_MAP.get(country_full,country_full)
    causes=sorted(df[df["Country"]==country_code]["CauseFull"].dropna().unique())
    cause_full=st.sidebar.selectbox("Cause of Death",causes)
    cause_code=REV_CAUSE_NAME_MAP.get(cause_full,cause_full)
    sex_sel=st.sidebar.multiselect("Sex",["Total","Male","Female"],default=["Total"])
    sex_codes=[REV_SEX_NAME[s] for s in sex_sel]
    yrs=sorted(df["Year"].unique())
    year_range=st.sidebar.slider("Historical Years",yrs[0],yrs[-1],(yrs[0],yrs[-1]))
    forecast_yrs=st.sidebar.slider("Forecast Horizon (yrs)",1,30,10)
    method=st.sidebar.selectbox("Forecast Method",["Prophet","ARIMA","ETS","Ensemble"])

    # (mortality, regressions, clustering as before...)

    # Global BF (neighbors only)
    st.markdown("---")
    st.header("Global Bayesian Causality (Neighbors Only)")
    country_list=sorted(df["CountryFull"].dropna().unique())
    sel_countries=st.multiselect("Select countries (all)",country_list,default=country_list)
    gl_maxlag=st.slider("Global max lag (yrs)",1,5,2,key="gl_maxlag")
    bf_thresh=st.number_input("BF₁₀ cutoff",1.0,100.0,3.0,0.5,key="bf_thr")
    if len(sel_countries)>=2:
        df_g=df[(df["Cause"]==cause_code)&(df["CountryFull"].isin(sel_countries))&
                (df["Sex"]=="T")&(df["Year"].between(*year_range))]
        pivot_gc=df_g.pivot_table(index="Year",columns="CountryFull",values="Rate")
        common=[c for c in sel_countries if c in pivot_gc.columns]
        bf_mat=pd.DataFrame(np.nan,index=common,columns=common)
        for src in common:
            for dst in common:
                if src==dst: continue
                pair=pivot_gc[[dst,src]].dropna()
                if pair.shape[0]>=gl_maxlag+3:
                    bf_mat.loc[src,dst]=compute_bayes_factor_bic(pair,gl_maxlag)
        st.subheader("Bayes-factor matrix (both directions)")
        st.dataframe(bf_mat.style.format("{:.2f}"))
        edges=[]
        for src in common:
            for dst in common:
                if src==dst: continue
                bf=bf_mat.loc[src,dst]
                if pd.notna(bf) and bf>=bf_thresh:
                    sc=REV_COUNTRY_NAME_MAP.get(src); dc=REV_COUNTRY_NAME_MAP.get(dst)
                    if dc in NEIGHBORS.get(sc,[]): edges.append((src,dst))
        draw_directed_network(common,edges,
            f"Global Neighbor Network (BF₁₀ ≥ {bf_thresh})")

    # Neighbor-based BF
    st.markdown("---")
    st.header("Neighbor-Based Bayesian Causality")
    foc_full=st.selectbox("Focal country",country_list,index=country_list.index("Germany"))
    foc_code=REV_COUNTRY_NAME_MAP.get(foc_full,foc_full)
    nbrs=NEIGHBORS.get(foc_code,[])
    map_df=pd.DataFrame({"Country":[foc_code]+nbrs,"Role":["Focal"]+["Neighbor"]*len(nbrs)})
    map_df["CountryFull"]=map_df["Country"].map(COUNTRY_NAME_MAP)
    map_df["iso_alpha"]=map_df["Country"].apply(alpha3_from_a2)
    st.plotly_chart(px.choropleth(map_df,locations="iso_alpha",color="Role",
                                  hover_name="CountryFull",locationmode="ISO-3",
                                  scope="europe",title="Focal & Neighbors"))
    if nbrs:
        gb=[foc_code]+nbrs
        df_n=df[(df["Cause"]==cause_code)&(df["Country"].isin(gb))&
                (df["Sex"]=="T")&(df["Year"].between(*year_range))]
        pivot_n=df_n.pivot_table(index="Year",columns="Country",values="Rate")
        common_n=[c for c in gb if c in pivot_n.columns]
        nbr_lag=st.slider("Neighbor max lag (yrs)",1,5,2,key="nbr_lag")
        nbr_bf_cut=st.number_input("Neighbor BF₁₀ cutoff",1.0,100.0,3.0,0.5,key="nbr_bf_cut")
        bf_n=pd.DataFrame(np.nan,index=common_n,columns=common_n)
        for src in common_n:
            for dst in common_n:
                if src==dst: continue
                pair=pivot_n[[dst,src]].dropna()
                if pair.shape[0]>=nbr_lag+3:
                    bf_n.loc[src,dst]=compute_bayes_factor_bic(pair,nbr_lag)
        st.subheader("Neighbor Bayes-factor matrix (both directions)")
        st.dataframe(bf_n.style.format("{:.2f}"))
        edges_n=[]
        for src in common_n:
            for dst in common_n:
                if src==dst: continue
                bf=bf_n.loc[src,dst]
                if pd.notna(bf) and bf>=nbr_bf_cut:
                    edges_n.append((COUNTRY_NAME_MAP[src],COUNTRY_NAME_MAP[dst]))
        nodes_n=[COUNTRY_NAME_MAP[c] for c in common_n]
        draw_directed_network(nodes_n,edges_n,
            f"Neighbor Network (BF₁₀ ≥ {nbr_bf_cut})")

    st.markdown("---")
    st.info("Arrows drawn atop blue nodes; labels placed like clock ticks; matrices include both directions.")

if __name__=="__main__":
    main()
