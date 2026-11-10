from __future__ import annotations

import sys
import tempfile

__version__ = '4.8.0'
REQUIREMENTS_TEXT = 'numpy==2.3.5\npandas==2.2.3\nscipy==1.17.0\nstatsmodels==0.15.0\nprophet==1.5.0\ncmdstanpy==1.3.0\nruptures==1.1.10\nscikit-learn==1.8.0\nstreamlit==1.66.0\npycountry==26.2.16\nplotly==7.1.0\n'


# ============================================================================
# config.py: eurohealth
# ============================================================================

EU27 = ('AT','BE','BG','HR','CY','CZ','DK','EE','FI','FR','DE','EL','HU','IE','IT','LV','LT','LU','MT','NL','PL','PT','RO','SK','SI','ES','SE')
GEO_ALIASES = {'EL':'GR','UK':'GB','FX':'FR'}
ISO_TO_EUROSTAT = {'GR':'EL','GB':'UK'}
SEX_LABELS = {'T':'Total','F':'Female','M':'Male'}
AGE_LABELS = {'TOTAL':'All ages','Y_LT65':'Under 65','Y_GE65':'65 and older'}
MODERN_COLLECTION = 'hlth_cd_asdr2'
HISTORICAL_COLLECTION = 'hlth_cd_asdr'
COLLECTION_BREAK = 2011
COLLECTION_LABELS = {HISTORICAL_COLLECTION:'Historical', MODERN_COLLECTION:'Modern'}
COMBINED_COLLECTIONS = 'Historical + modern'
MORTALITY_SOURCES = {
    'Modern only (2011 onward)':(MODERN_COLLECTION,),
    'Historical only (through 2010)':(HISTORICAL_COLLECTION,),
    COMBINED_COLLECTIONS:(HISTORICAL_COLLECTION, MODERN_COLLECTION),
}

# Undirected edges. Membership is restricted to countries represented in a run.
BORDER_EDGES = {
    ('AT','DE'),('AT','CZ'),('AT','SK'),('AT','HU'),('AT','SI'),('AT','IT'),('AT','CH'),('AT','LI'),
    ('BE','FR'),('BE','DE'),('BE','NL'),('BE','LU'),('BG','RO'),('BG','EL'),('BG','TR'),('BG','RS'),('BG','MK'),
    ('HR','SI'),('HR','HU'),('HR','RS'),('HR','BA'),('HR','ME'),('CZ','DE'),('CZ','PL'),('CZ','SK'),
    ('DK','DE'),('EE','LV'),('EE','RU'),('FI','SE'),('FI','NO'),('FI','RU'),
    ('FR','LU'),('FR','DE'),('FR','IT'),('FR','ES'),('FR','CH'),('FR','AD'),('FR','MC'),
    ('DE','PL'),('DE','CH'),('DE','LU'),('DE','NL'),('EL','AL'),('EL','MK'),('EL','TR'),
    ('HU','SK'),('HU','RO'),('HU','SI'),('HU','RS'),('HU','UA'),('IE','UK'),('IT','SI'),('IT','CH'),('IT','SM'),('IT','VA'),
    ('LV','LT'),('LV','RU'),('LV','BY'),('LT','PL'),('LT','BY'),('LT','RU'),
    ('PL','SK'),('PL','BY'),('PL','UA'),('PL','RU'),('PT','ES'),('RO','UA'),('RO','MD'),('RO','RS'),
    ('SK','UA'),('ES','AD'),('SE','NO'),('CH','LI'),('NO','RU'),
    ('RS','BA'),('RS','ME'),('RS','MK'),('AL','ME'),('AL','MK'),('BA','ME')
}

FACTOR_SPECS = {
 'GDP per capita': dict(dataset='sdg_08_10', filters={'freq':'A','unit':'CLV20_EUR_HAB','na_item':'B1GQ'}, definition='Real GDP per inhabitant in chain-linked 2020 euros; a country-wide macroeconomic measure.'),
 'Health expenditure': dict(dataset='hlth_sha11_hp', filters={'freq':'A','unit':'EUR_HAB','icha11_hp':'TOTAL'}, definition='Current expenditure across all health-care providers, euros per inhabitant.'),
 'Obesity among citizens': dict(dataset='hlth_ehis_bm1c', filters={'freq':'A','unit':'PC','bmi':'BMI_GE30','sex':'T','age':'Y_GE18','citizen':'NAT'}, definition='Obesity percentage among reporting-country citizens aged 18 and older; survey years only.'),
 'Physical activity among citizens': dict(dataset='hlth_ehis_pe9c', filters={'freq':'A','unit':'PC','physact':'MV_AERO_MSC','sex':'T','age':'Y_GE18','citizen':'NAT'}, definition='Aerobic and muscle-strengthening activity among reporting-country citizens aged 18 and older; survey years only.'),
 'Daily smoking among citizens': dict(dataset='hlth_ehis_sk1c', filters={'freq':'A','unit':'PC','smoking':'SM_DAY','sex':'T','age':'Y_GE18','citizen':'NAT'}, definition='Daily smoking percentage among reporting-country citizens aged 18 and older; survey years only.'),
 'Hospital beds': dict(dataset='hlth_rs_bdsrg2', filters={'freq':'A','unit':'P_HTHAB'}, definition='Available hospital beds per 100,000 inhabitants.'),
 'Practising physicians': dict(dataset='hlth_rs_prs2', filters={'freq':'A','unit':'P_HTHAB','wstatus':'PRACT','med_spec':'PHYS'}, definition='Practising physicians per 100,000 inhabitants, excluding nurses and other professions.'),
 'Doctor visit in past year': dict(dataset='hlth_ehis_am1e', filters={'freq':'A','unit':'PC','duration':'Y_LT1','isco08':'OC221','isced11':'TOTAL','sex':'T','age':'TOTAL'}, definition='Percentage reporting a visit to a medical doctor within the preceding year; all educational levels; survey years only.')
}


# ============================================================================
# data.py: eurohealth
# ============================================================================

import gzip
import hashlib
import itertools
import json
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


API = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/'

class NoEurostatObservations(ValueError):
    """Eurostat explicitly returned an empty result for the query."""

def _empty_eurostat_response(payload):
    if not isinstance(payload,dict):return False
    errors=payload.get('error',[])
    if isinstance(errors,dict):errors=[errors]
    return any(isinstance(e,dict) and (
        str(e.get('id',e.get('code','')))=='100' or
        re.search(r'\bno[ _]+results?(?:[ _]+found)?\b',str(e.get('label','')),re.IGNORECASE)
    ) for e in errors) if isinstance(errors,list) else False

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def category_codes(category: dict) -> list[str]:
    index = category['index']
    if isinstance(index, list):
        return index
    return [k for k,v in sorted(index.items(), key=lambda kv:kv[1])]

def parse_numeric_flag(value):
    if value is None or pd.isna(value):
        return np.nan, ''
    if isinstance(value, (int,float,np.number)):
        return float(value), ''
    text = str(value).strip()
    match = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*(.*)', text)
    if match:
        return float(match.group(1)), match.group(2).strip()
    if text.startswith(':'):
        return np.nan, text[1:].strip()
    raise ValueError(f'Unrecognised Eurostat value: {text!r}')

def parse_tsv(content: bytes, dataset: str) -> tuple[pd.DataFrame,dict]:
    if content[:2] == b'\x1f\x8b':
        content = gzip.decompress(content)
    raw = pd.read_csv(BytesIO(content), sep='\t', dtype=str, keep_default_na=False)
    first = raw.columns[0]
    dims = first.split('\\')[0].split(',')
    keys = raw[first].str.split(',', expand=True)
    if keys.shape[1] != len(dims):
        raise ValueError('TSV dimension count does not match series keys')
    keys.columns = dims
    wide = pd.concat([keys,raw.drop(columns=first)],axis=1)
    values = [c for c in wide.columns if c not in dims]
    long = wide.melt(id_vars=dims,value_vars=values,var_name='time',value_name='raw_value')
    long['time'] = long['time'].str.strip()
    parsed = long['raw_value'].map(parse_numeric_flag)
    long['value'] = [v for v,f in parsed]
    long['flag'] = [f for v,f in parsed]
    long['dataset'] = dataset
    return long, {'dataset':dataset,'dimensions':dims,'format':'TSV','sha256_uncompressed':sha256(content)}

def parse_jsonstat(payload: dict, dataset: str) -> tuple[pd.DataFrame,dict]:
    if payload.get('error'):
        raise ValueError(payload['error'])
    if payload.get('warning') or 'id' not in payload:
        raise ValueError('Eurostat did not return a completed JSON-stat data cube; narrow the query or retry later')
    dims = payload['id']; sizes=payload['size']
    codes = [category_codes(payload['dimension'][d]['category']) for d in dims]
    # Materialise the entire selected cube, including explicitly missing cells.
    total = int(np.prod(sizes,dtype=np.int64))
    if total > 2_000_000:
        raise ValueError('Query contains over two million cells; select fewer strata')
    values=payload.get('value',{});status=payload.get('status',{})
    def cell(obj,i,default):
        if isinstance(obj,list): return obj[i] if i<len(obj) else default
        return obj.get(str(i),default)
    records=[]
    for i,coordinates in enumerate(itertools.product(*codes)):
        row=dict(zip(dims,coordinates));row['value']=cell(values,i,np.nan);row['flag']=cell(status,i,'') or ''
        row['dataset']=dataset;records.append(row)
    df=pd.DataFrame.from_records(records)
    df['value']=pd.to_numeric(df['value'],errors='raise')
    labels={d:payload['dimension'][d]['category'].get('label',{}) for d in dims}
    meta={'dataset':dataset,'label':payload.get('label'),'updated':payload.get('updated'),'dimensions':dims,'labels':labels,'format':'JSON-stat 2.0'}
    return df,meta

def query_url(dataset: str, filters: dict) -> str:
    if not re.fullmatch(r'[a-z0-9_]+',dataset):
        raise ValueError('Invalid dataset identifier')
    params=[('lang','EN')]
    for key,value in filters.items():
        for val in value if isinstance(value,(tuple,list)) else [value]:
            params.append((key,str(val)))
    return API+dataset+'?'+urlencode(params)

QUERY_CACHE_SECONDS = 3600


def _atomic_cache_write(path: Path, content: bytes):
    with NamedTemporaryFile(dir=path.parent,delete=False) as f:
        f.write(content)
        temporary=Path(f.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _cached_response(record_path: Path, dataset: str, url: str, refresh_after: float):
    try:
        record=json.loads(record_path.read_bytes())
        if record['url']!=url or not re.fullmatch(r'[0-9a-f]{64}',record['sha256']):
            return None
        retrieved=datetime.fromisoformat(record['retrieved_utc']).timestamp()
        if retrieved<max(time.time()-QUERY_CACHE_SECONDS,refresh_after):
            return None
        snapshot=record_path.parent/f"{dataset}_{record['sha256']}.json"
        content=snapshot.read_bytes()
    except (FileNotFoundError,KeyError,TypeError,ValueError):
        return None
    return (content,record,snapshot) if sha256(content)==record['sha256'] else None


def _request_eurostat(url: str, dataset: str) -> bytes:
    request=Request(url,headers={'User-Agent':f'EuropeanPublicHealthDashboard/{__version__} (research software)'})
    last=None
    for attempt in range(3):
        try:
            with urlopen(request,timeout=40) as response:
                return response.read()
        except HTTPError as exc:
            last=exc
            if exc.code==400:
                try:
                    error=json.loads(exc.read())
                except (OSError,ValueError):
                    error={}
                if _empty_eurostat_response(error):
                    raise NoEurostatObservations('Eurostat has no values for this selection.') from exc
            if exc.code not in (429,500,502,503,504):
                raise RuntimeError(f'Eurostat returned HTTP {exc.code} for {dataset}.') from exc
        except (URLError,TimeoutError) as exc:
            last=exc
        if attempt<2:time.sleep(1+attempt)
    raise RuntimeError(f'Eurostat retrieval failed for {dataset}.') from last


def fetch(dataset: str, filters: dict, cache_dir: Path|str, refresh_after: float=0.0):
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
    url=query_url(dataset,filters);request_hash=sha256(url.encode())[:20]
    record_path=cache_dir/f'{dataset}_{request_hash}.manifest.json'
    cached=_cached_response(record_path,dataset,url,refresh_after)
    content=cached[0] if cached is not None else _request_eurostat(url,dataset)
    payload=json.loads(content)
    if _empty_eurostat_response(payload):
        raise NoEurostatObservations('Eurostat has no values for this selection.')
    df,meta=parse_jsonstat(payload,dataset)
    if cached is None:
        digest=sha256(content)
        snapshot=cache_dir/f'{dataset}_{digest}.json'
        record={'dataset':dataset,'url':url,'filters':filters,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                'sha256':digest,'snapshot_file':snapshot.name,'source_updated':payload.get('updated')}
        # Publish a new query record only after its exact response is available.
        # Earlier saved analyses keep their own response even after a refresh.
        if not snapshot.is_file() or snapshot.read_bytes()!=content:
            _atomic_cache_write(snapshot,content)
        _atomic_cache_write(record_path,json.dumps(record,indent=2,ensure_ascii=False).encode())
    else:
        _,record,snapshot=cached
    meta.update(record);meta['snapshot_path']=str(snapshot)
    return df,meta

def assert_unique(df: pd.DataFrame, keys: list[str], label='data'):
    dup=df.duplicated(keys,keep=False)
    if dup.any():
        example=df.loc[dup,keys].head(5).to_dict('records')
        raise ValueError(f'Ambiguous duplicate keys in {label}; explicitly select all source dimensions. Examples: {example}')

def select_dimensions(df:pd.DataFrame, selectors:dict, keys=('geo','time')):
    out=df.copy()
    for dim,value in selectors.items():
        if dim not in out:raise ValueError(f'Required dimension {dim} is absent')
        if value not in set(out[dim]):raise ValueError(f'Required category {dim}={value} is absent; no automatic fallback applied')
        out=out[out[dim]==value]
    dimensions=[c for c in out.columns if c not in ('value','flag','dataset','raw_value')]
    ambiguous={d:out[d].unique().tolist() for d in dimensions if d not in keys and out[d].nunique()>1}
    if ambiguous:raise ValueError(f'Unresolved dimensions: {ambiguous}')
    assert_unique(out,list(keys),'selected indicator')
    return out

def prepare_mortality(df:pd.DataFrame) -> pd.DataFrame:
    required={'geo','time','sex','age','icd10','unit','freq','value','flag','dataset'}
    if not required.issubset(df):raise ValueError(f'Missing mortality columns: {required-set(df.columns)}')
    x=df[(df['freq']=='A')&(df['unit']=='RT')].copy()
    x['Year']=pd.to_numeric(x['time'],errors='raise').astype(int)
    if x['geo'].str.len().gt(2).any():
        legal=x['geo'].eq('EU27_2020')
        if (~legal & x['geo'].str.len().gt(2)).any():raise ValueError('Mortality selection contains non-country geographies other than the explicit official EU aggregate')
    x=x.rename(columns={'geo':'Country','sex':'Sex','age':'Age','icd10':'OriginalCause','value':'Rate','flag':'Flag'})
    x['Cause']=x['OriginalCause'].replace({'A-R_V-Y':'TOTAL'})
    keys=['dataset','Country','Year','Cause','Sex','Age']
    # TOTAL and A-R_V-Y are a documented code migration. Prefer a nonmissing TOTAL
    # value; reject conflicting nonmissing overlap instead of averaging.
    duplicate=x.duplicated(keys,keep=False)
    if duplicate.any():
        rest=x[~duplicate];resolved=[]
        for _,g in x[duplicate].groupby(keys,dropna=False):
            if not set(g['OriginalCause']).issubset({'TOTAL','A-R_V-Y'}):raise ValueError('Unexpected duplicate mortality stratum')
            vals=g['Rate'].dropna().to_numpy()
            if len(vals)>1 and not np.allclose(vals,vals[0],rtol=0,atol=1e-8):raise ValueError('Conflicting overlapping all-cause codes; inspect source before harmonisation')
            g=g.assign(_missing=g['Rate'].isna(),_priority=g['OriginalCause'].ne('TOTAL'))
            row=g.sort_values(['_missing','_priority']).iloc[0].copy()
            row['Flag']=' '.join(sorted(set(' '.join(g['Flag'].astype(str)).split())))
            resolved.append(row.drop(labels=['_missing','_priority']))
        x=pd.concat([rest,pd.DataFrame(resolved)],ignore_index=True)
    assert_unique(x,keys,'mortality')
    if (x['Rate'].dropna()<0).any():raise ValueError('Negative mortality rate')
    x['CountryISO2']=x['Country'].replace(GEO_ALIASES)
    return x.sort_values(keys).reset_index(drop=True)

def load_mortality(causes,cache_dir,source='hlth_cd_asdr2',geographies=None,refresh_after=0.0):
    requested=[]
    for c in causes:
        requested.extend(['TOTAL','A-R_V-Y'] if c=='TOTAL' else [c])
    filters={'freq':'A','unit':'RT','age':['TOTAL','Y_LT65','Y_GE65'],'sex':['T','F','M'],'icd10':list(dict.fromkeys(requested))}
    if geographies is None:filters['geo']=list(EU27)+['CH','LI','NO','IS','UK','RS','TR','EU27_2020']
    else:filters['geo']=list(geographies)
    df,meta=fetch(source,filters,cache_dir,refresh_after=refresh_after)
    data=prepare_mortality(df)
    if source==HISTORICAL_COLLECTION:
        data=data[data.Year<COLLECTION_BREAK]
    elif source==MODERN_COLLECTION:
        data=data[data.Year>=COLLECTION_BREAK]
    if not data.Rate.notna().any():raise NoEurostatObservations('Eurostat has no values for this selection.')
    return data,meta

def load_factor(name,cache_dir,sex='T',geographies=None,refresh_after=0.0):
    spec=FACTOR_SPECS[name];selectors=dict(spec['filters'])
    if 'sex' in selectors:selectors['sex']=sex
    filters=dict(selectors);filters['geo']=list(geographies or EU27)
    raw,meta=fetch(spec['dataset'],filters,cache_dir,refresh_after=refresh_after)
    selected=select_dimensions(raw,selectors)
    out=selected[['geo','time','value','flag']].rename(columns={'geo':'Country','time':'Year','value':name,'flag':'FactorFlag'})
    out['Year']=out['Year'].astype(int)
    meta.update({'indicator_name':name,'definition':spec['definition'],'selectors':selectors})
    return out,meta

def load_population(cache_dir,years,refresh_after=0.0):
    """Retrieve annual country population counts for the selected calendar range."""
    first,last=map(int,years)
    selectors={'freq':'A','age':'TOTAL','sex':'T'}
    filters=selectors|{'geo':list(EU27),'sinceTimePeriod':first,'untilTimePeriod':last}
    raw,meta=fetch('demo_pjan',filters,cache_dir,refresh_after=refresh_after)
    selected=select_dimensions(raw,selectors|{'unit':'NR'})
    out=selected[['geo','time','value','flag']].rename(
        columns={'geo':'Country','time':'Year','value':'Population','flag':'PopulationFlag'})
    out['Year']=out['Year'].astype(int)
    out=out[out.Country.isin(EU27)&out.Year.between(first,last)]
    if not out.Population.notna().any():
        raise NoEurostatObservations('Eurostat has no population values for these countries and years.')
    return out,meta

def consecutive_series(df:pd.DataFrame,year='Year',value='Rate',minimum=3):
    x=df[[year,value]].copy().sort_values(year)
    assert_unique(x,[year],'annual series')
    if x[value].isna().any():raise ValueError('Missing annual values; select a complete consecutive interval')
    if len(x)<minimum:raise ValueError(f'At least {minimum} annual values are required')
    if not np.all(np.diff(x[year].astype(int))==1):raise ValueError('Non-consecutive calendar years; annual lags and forecasts would be invalid')
    return x

def flag_contains(value,codes='bdu'):
    return bool(set(str(value).replace(' ','').lower()) & set(codes))


# ============================================================================
# forecasting.py: eurohealth
# ============================================================================

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


MODELS=('Naive','Drift','Damped drift','Log-linear','ARIMA','ETS','Prophet','Ensemble')
COMPONENTS=('ARIMA','ETS','Prophet')

# Deterministic Gaussian fits for the ARIMA(1,1,0) and ETS(A,A,N) models.
import math
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16384)
def _portable_log1p_value(rate):
    # Compute the same mathematical transform with scalar, correctly rounded
    # arithmetic instead of selecting a CPU-specific vector implementation.
    if not math.isfinite(rate) or rate < 0:
        raise ValueError('Forecast rates must be finite and nonnegative.')
    if rate <= 2.0**-54:
        return rate
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.rounding = ROUND_HALF_EVEN
        return float((Decimal(1) + Decimal.from_float(rate)).ln())


def _portable_log1p(values):
    return np.fromiter((_portable_log1p_value(float(x)) for x in values), dtype=float, count=len(values))


def _arima110_estimate(values):
    d = [b-a for a,b in zip(values,values[1:])]
    n = len(d)
    A = math.fsum(x*x for x in d)
    if A == 0:
        return 0.0,0.0,d[-1]
    B = math.fsum(a*b for a,b in zip(d,d[1:]))
    C = math.fsum(x*x for x in d[1:-1])
    # Exact stationary Gaussian likelihood for the differenced observations.
    # Profile the innovation variance, then locate every stationary point.
    p0,p1,p2,p3 = -n*B,n*C+A,(n-2)*B,(1-n)*C
    def derivative(phi):
        return ((p3*phi+p2)*phi+p1)*phi+p0
    limit = 1.0-1e-8
    breaks = [-limit,limit]
    if p3 != 0:
        q = (2*p2)**2-12*p3*p1
        if q >= 0:
            breaks.extend(r for r in ((-2*p2-math.sqrt(q))/(6*p3),(-2*p2+math.sqrt(q))/(6*p3)) if -limit<r<limit)
    elif p2 != 0:
        r = -p1/(2*p2)
        if -limit<r<limit:
            breaks.append(r)
    breaks.sort()
    options = breaks.copy()
    for a,b in zip(breaks,breaks[1:]):
        fa,fb = derivative(a),derivative(b)
        if fa == 0 or fb == 0 or (fa>0)==(fb>0):
            continue
        for _ in range(56):
            m = (a+b)*.5
            fm = derivative(m)
            if fm == 0:
                a=b=m
                break
            if (fm>0)==(fa>0):
                a,fa=m,fm
            else:
                b=m
        options.append((a+b)*.5)

    def cost(phi):
        stationarity=(1-phi)*(1+phi)
        ss=math.fsum([stationarity*d[0]*d[0]]+[(y-phi*x)**2 for x,y in zip(d,d[1:])])
        var=ss/n
        # Use the same high-precision scalar log for selecting a likelihood
        # optimum. The native logarithm does not enter the search itself.
        with localcontext() as ctx:
            ctx.prec=48
            value=n*Decimal.from_float(var).ln()-Decimal.from_float(stationarity).ln()
        return value,abs(phi),phi,var
    _,_,phi,var = min(map(cost,options))
    return phi,var,d[-1]


def _portable_arima110(values,horizon,q):
    values=tuple(map(float,values))
    phi,var,last_delta=_arima110_estimate(values)
    point=[];lower=[];upper=[]
    power=1.;step=1.;variance_factor=0.;level=values[-1]
    for _ in range(horizon):
        variance_factor+=step*step
        power*=phi
        level+=power*last_delta
        margin=q*math.sqrt(var*variance_factor)
        point.append(level);lower.append(level-margin);upper.append(level+margin)
        step+=power
    return point,lower,upper


def _ets_profile(centered,alpha,beta):
    # With smoothing weights fixed, the innovations are linear functions of
    # the initial level and slope. Profile both initial states by least squares.
    level=trend=0.
    ll,lt,tl,tt=1.,0.,0.,1.
    residual=[];col_l=[];col_t=[]
    for y in centered:
        pred=level+trend
        e=y-pred
        cl,ct=ll+tl,lt+tt
        residual.append(e);col_l.append(cl);col_t.append(ct)
        level=pred+alpha*e
        trend+=beta*e
        ll,lt=cl-alpha*cl,ct-alpha*ct
        tl,tt=tl-beta*cl,tt-beta*ct
    ll_norm=math.fsum(x*x for x in col_l)
    projection=math.fsum(x*y for x,y in zip(col_l,col_t))/ll_norm
    perpendicular=[y-projection*x for x,y in zip(col_l,col_t)]
    norm=math.fsum(x*x for x in perpendicular)
    slope=math.fsum(x*y for x,y in zip(perpendicular,residual))/norm
    intercept=math.fsum(x*(r-y*slope) for x,y,r in zip(col_l,col_t,residual))/ll_norm
    ss=math.fsum((r-x*intercept-y*slope)**2 for r,x,y in zip(residual,col_l,col_t))
    return ss,intercept,slope


def _ets_additive_estimate(values):
    offset=values[0]
    y=tuple(x-offset for x in values)
    lo,hi=.0001,.9999

    @lru_cache(maxsize=8192)
    def evaluate(u,v):
        a=lo+(hi-lo)*u
        b=lo+(a-lo)*v
        ss,l,t=_ets_profile(y,a,b)
        return ss,u,v,a,b,l,t

    def clamp(v):
        return min(1.,max(0.,v))

    # Search the whole constrained surface before local refinement. Retain
    # the best objective actually evaluated, including boundary solutions.
    grid=12
    points={(i,j):evaluate(i/grid,j/grid) for i in range(grid+1) for j in range(grid+1)}
    minima=[p for (i,j),p in points.items() if all(
        p <= points[ii,jj] for ii in range(max(0,i-1),min(grid,i+1)+1)
        for jj in range(max(0,j-1),min(grid,j+1)+1))]
    best=min(points.values())
    for seed in sorted(minima)[:6]:
        _,u,v,*_=seed
        move=1/grid
        simplex=[seed,evaluate(u+move if u<=1-move else u-move,v),
                 evaluate(u,v+move if v<=1-move else v-move)]
        for _ in range(240):
            simplex.sort()
            a,b,c=simplex
            best=min(best,a)
            if max(abs(c[1]-a[1]),abs(c[2]-a[2]),abs(b[1]-a[1]),abs(b[2]-a[2]))<2e-9:
                break
            cu,cv=(a[1]+b[1])*.5,(a[2]+b[2])*.5
            ru,rv=2*cu-c[1],2*cv-c[2]
            r=evaluate(clamp(ru),clamp(rv))
            if r<a:
                e=evaluate(clamp(3*cu-2*c[1]),clamp(3*cv-2*c[2]))
                simplex[2]=min(r,e)
            elif r<b:
                simplex[2]=r
            else:
                t=min(r,c)
                h=evaluate(clamp((cu+t[1])*.5),clamp((cv+t[2])*.5))
                if h<t:
                    simplex[2]=h
                else:
                    simplex=[a]+[evaluate((a[1]+p[1])*.5,(a[2]+p[2])*.5) for p in (b,c)]
        best=min(best,*simplex)
    # A clipped simplex can collapse onto an edge. Optimize all four edges
    # independently to ensure that such a collapse does not omit an optimum.
    for axis in (0,1):
        for boundary in (0.,1.):
            def edge(x):
                return evaluate(boundary,x) if axis==0 else evaluate(x,boundary)
            ranked=[edge(k/grid) for k in range(grid+1)]
            for i in range(grid+1):
                if ranked[i]!=min(ranked[max(0,i-1):i+2]):
                    continue
                left,right=max(0,i-1)/grid,min(grid,i+1)/grid
                for _ in range(44):
                    m1=(2*left+right)/3;m2=(left+2*right)/3
                    if edge(m1)<=edge(m2):right=m2
                    else:left=m1
                best=min(best,ranked[i],edge((left+right)*.5))
    ss,_,_,alpha,beta,initial_level,initial_trend=best
    l,b=initial_level,initial_trend
    for v in y:
        pred=l+b
        e=v-pred
        l=pred+alpha*e
        b+=beta*e
    return alpha,beta,initial_level+offset,initial_trend,l+offset,b,ss/len(y)


def _portable_ets(values,horizon,q):
    a,b,_,_,l,t,var=_ets_additive_estimate(tuple(map(float,values)))
    point=[];lower=[];upper=[];factor=1.
    for h in range(1,horizon+1):
        estimate=l+h*t
        margin=q*math.sqrt(var*factor)
        point.append(estimate);lower.append(estimate-margin);upper.append(estimate+margin)
        factor+=(a+h*b)**2
    return point,lower,upper


def _fit_method(z:np.ndarray,horizon:int,method:str,years:np.ndarray,alpha:float,seed:int):
    n=len(z);h=np.arange(1,horizon+1,dtype=float);q=norm.ppf(1-alpha/2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if method in ('Naive','Drift','Damped drift'):
            delta=np.diff(z);sigma=float(np.std(delta,ddof=1)) if len(delta)>1 else 0.0
            if method=='Naive':mean=np.repeat(z[-1],horizon);sd=sigma*np.sqrt(h)
            elif method=='Drift':mean=z[-1]+h*delta.mean();sd=sigma*np.sqrt(h*(1+h/(n-1)))
            else:
                # Fixed damping avoids estimating another trend parameter from 8 annual values.
                phi=.8;effective_h=phi*(1-phi**h)/(1-phi)
                mean=z[-1]+effective_h*delta.mean()
                sd=sigma*np.sqrt(h+effective_h**2/(n-1))
            lo=mean-q*sd;hi=mean+q*sd
        elif method=='Log-linear':
            fit=sm.OLS(z,sm.add_constant(np.arange(n,dtype=float))).fit()
            future=sm.add_constant(np.arange(n,n+horizon,dtype=float),has_constant='add')
            prediction=fit.get_prediction(future).summary_frame(alpha=alpha)
            mean=prediction['mean'].to_numpy();lo=prediction['obs_ci_lower'].to_numpy();hi=prediction['obs_ci_upper'].to_numpy()
        elif method=='ARIMA':
            mean,lo,hi=_portable_arima110(z,horizon,float(q))
        elif method=='ETS':
            mean,lo,hi=_portable_ets(z,horizon,float(q))
        elif method=='Prophet':
            from prophet import Prophet
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
            model=Prophet(yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False,
                          n_changepoints=min(3,max(0,n//4)),changepoint_prior_scale=.05,
                          interval_width=1-alpha,uncertainty_samples=1000)
            frame=pd.DataFrame({'ds':pd.to_datetime([f'{int(t)}-01-01' for t in years]),'y':z})
            model.fit(frame,seed=seed)
            future=pd.DataFrame({'ds':pd.to_datetime([f'{int(years[-1])+int(t)}-01-01' for t in h])})
            state=np.random.get_state()
            try:
                np.random.seed(seed)
                prediction=model.predict(future)
            finally:np.random.set_state(state)
            mean=prediction.yhat.to_numpy();lo=prediction.yhat_lower.to_numpy();hi=prediction.yhat_upper.to_numpy()
        else:raise ValueError(f'Unknown forecasting model {method}')
    mean=np.asarray(mean,float);lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    if not np.isfinite(np.r_[mean,lo,hi]).all() or np.max(np.r_[mean,hi])>20:
        raise ValueError('Nonfinite or unstable model prediction')
    if np.any(lo>hi):raise ValueError('Invalid prediction interval')
    return {'mean':mean,'low':lo,'high':hi,'warnings':sorted(set(str(w.message) for w in caught))}

def forecast(series:pd.DataFrame,horizon=3,models=('Ensemble',),alpha=.05,seed=20261016):
    if horizon<1 or horizon>10:raise ValueError('Forecast horizon must be 1–10 years; predictive validation may support only shorter horizons')
    x=consecutive_series(series,minimum=8)
    if (x.Rate<0).any():raise ValueError('Mortality rates cannot be negative')
    years=x.Year.to_numpy(int);z=_portable_log1p(x.Rate.to_numpy(float));requested=list(models)
    if any(m not in MODELS for m in requested):raise ValueError('Unknown forecasting model')
    needed=set(requested)-{'Ensemble'}
    if 'Ensemble' in requested:needed.update(COMPONENTS)
    fitted={};failures=[]
    for method in MODELS:
        if method not in needed:continue
        try:fitted[method]=_fit_method(z,horizon,method,years,alpha,seed)
        except Exception as exc:failures.append({'Model':method,'Error':f'{type(exc).__name__}: {exc}'})
    if 'Ensemble' in requested:
        if all(m in fitted for m in COMPONENTS):
            rng=np.random.default_rng(seed);samples=[]
            for m in COMPONENTS:
                fit=fitted[m];sd=(fit['high']-fit['low'])/(2*norm.ppf(1-alpha/2))
                samples.append(rng.normal(fit['mean'],sd,size=(12000,horizon)))
            mixture=np.concatenate(samples,axis=0)
            # The point and interval refer to the same empirical predictive mixture.
            centre=np.quantile(mixture,.5,axis=0)
            component_warnings=[m+': '+w for m in COMPONENTS for w in fitted[m]['warnings']]
            fitted['Ensemble']={'mean':centre,'low':np.quantile(mixture,alpha/2,axis=0),'high':np.quantile(mixture,1-alpha/2,axis=0),'warnings':component_warnings}
        else:failures.append({'Model':'Ensemble','Error':'A required component failed; ensemble weights were not silently changed'})
    rows=[]
    for method in requested:
        if method not in fitted:continue
        f=fitted[method]
        for i in range(horizon):
            rows.append({'Year':int(years[-1])+i+1,'Model':method,'Forecast':float(max(0,np.expm1(f['mean'][i]))),
                         'Lower':float(max(0,np.expm1(f['low'][i]))),'Upper':float(max(0,np.expm1(f['high'][i]))),
                         'MeanLog':float(f['mean'][i]),'LowLog':float(f['low'][i]),'HighLog':float(f['high'][i]),
                         'TrainingEnd':int(years[-1]),'TrainingN':len(x),'Horizon':i+1,'NominalCoverage':1-alpha,
                         'IntervalType':'Approximate equal-weight predictive mixture' if method=='Ensemble' else 'Conditional model prediction interval',
                         'Warnings':' | '.join(f['warnings'])})
    return pd.DataFrame(rows),pd.DataFrame(failures,columns=['Model','Error'])

def score_predictions(predictions:pd.DataFrame):
    x=predictions.copy();e=x.Forecast-x.Actual
    x['AbsoluteError']=abs(e);x['SquaredError']=e**2
    x['ScaledAbsoluteError']=np.divide(abs(e),x.MASEScale,out=np.full(len(x),np.nan),where=x.MASEScale.to_numpy()>0)
    x['Covered']=(x.Actual>=x.Lower)&(x.Actual<=x.Upper)
    alpha=1-x.NominalCoverage
    x['IntervalScore']=x.Upper-x.Lower+(2/alpha)*(x.Lower-x.Actual).clip(lower=0)+(2/alpha)*(x.Actual-x.Upper).clip(lower=0)
    return x

def forecast_analysis_tables(series,settings):
    """Fit the selected forecast and optional comparator on the same observations."""
    primary=settings['model'];comparison=settings.get('comparison_model')
    methods=list(dict.fromkeys([primary,comparison] if comparison else [primary]))
    predicted,failed=forecast(series,settings['horizon'],methods)
    tables={'forecasts':predicted,'failures':failed}
    if len(methods)>1:
        if predicted.empty:
            tables['comparison_forecasts']=predicted.copy()
        else:
            tables['forecasts']=predicted[predicted.Model.eq(primary)].reset_index(drop=True)
            tables['comparison_forecasts']=predicted[predicted.Model.eq(comparison)].reset_index(drop=True)
    return tables

def rolling_series(task):
    country,cause,records,origins,horizons,models=task
    frame=pd.DataFrame(records).sort_values('Year');rows=[];failures=[]
    for origin in origins:
        train=frame[frame.Year<=origin]
        available=[h for h in horizons if origin+h in set(frame.Year)]
        if not available:continue
        try:
            pred,failed=forecast(train,horizon=max(available),models=models)
            failed=failed.assign(Country=country,Cause=cause,Origin=origin)
            failures.extend(failed.to_dict('records'))
            scale=float(np.mean(np.abs(np.diff(train.Rate))))
            for rec in pred.to_dict('records'):
                if rec['Horizon'] not in horizons:continue
                actual=frame[frame.Year==rec['Year']].Rate
                if len(actual)!=1 or pd.isna(actual.iloc[0]):continue
                rec.update(Country=country,Cause=cause,Origin=origin,Actual=float(actual.iloc[0]),MASEScale=scale)
                rows.append(rec)
        except Exception as exc:failures.append({'Country':country,'Cause':cause,'Origin':origin,'Model':'All','Error':str(exc)})
    return rows,failures


# ============================================================================
# analysis.py: eurohealth
# ============================================================================

import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


def loglinear_trend(series:pd.DataFrame,minimum=6):
    x=consecutive_series(series,minimum=minimum)
    if (x.Rate<=0).any():raise ValueError('Log-linear APC requires strictly positive rates; zeros are not replaced by invented values')
    t=x.Year.to_numpy(float);X=sm.add_constant(t-t.mean())
    model=sm.OLS(np.log(x.Rate.to_numpy(float)),X).fit(cov_type='HAC',cov_kwds={'maxlags':1,'use_correction':True},use_t=True)
    lo,hi=model.conf_int()[1]
    return {'start_year':int(t.min()),'end_year':int(t.max()),'n':len(x),'APC':100*np.expm1(model.params[1]),'APC_low':100*np.expm1(lo),'APC_high':100*np.expm1(hi),'log_slope':float(model.params[1]),'covariance':'HAC lag 1; interval conditional on a single log-linear trend'}

def temporal_summary(series:pd.DataFrame,penalty_multiplier=3.0,min_segment=4):
    x=consecutive_series(series,minimum=min_segment)
    z=np.log1p(x.Rate.to_numpy(float))
    spread=float(np.std(z,ddof=1))
    scaled=(z-z.mean())/spread if spread>0 else np.zeros_like(z)
    breaks=rpt.Pelt(model='l2',min_size=min_segment,jump=1).fit(scaled).predict(pen=penalty_multiplier*np.log(len(x)))
    rows=[];start=0
    for stop in breaks:
        s=x.iloc[start:stop]
        result={'start_year':int(s.Year.min()),'end_year':int(s.Year.max()),'n':len(s),'mean_rate':float(s.Rate.mean()),'APC':np.nan,'APC_low':np.nan,'APC_high':np.nan}
        if len(s)>=4 and (s.Rate>0).all():
            result.update(loglinear_trend(s,minimum=4))
        result['selection']='Exploratory PELT mean shifts in standardised log1p rates; intervals conditional on the selected segment'
        rows.append(result);start=stop
    # Anomalies are defined relative to a fitted time trend, not the raw-series mean.
    fit=sm.OLS(z,sm.add_constant(x.Year.to_numpy()-x.Year.mean())).fit()
    resid=fit.resid
    mad=np.median(np.abs(resid-np.median(resid)))
    score=(resid-np.median(resid))/(1.4826*mad) if mad>1e-12 else np.zeros_like(resid)
    detail=x.copy();detail['ResidualScore']=score;detail['PotentialAnomaly']=np.abs(score)>3.5
    return pd.DataFrame(rows),detail

def compare_causes(a:pd.DataFrame,b:pd.DataFrame):
    assert_unique(a,['Year'],'first cause');assert_unique(b,['Year'],'second cause')
    pair=a[['Year','Rate']].merge(b[['Year','Rate']],on='Year',suffixes=('_A','_B'),validate='one_to_one').dropna()
    if len(pair)<3:raise ValueError('At least three common years are required')
    if pair.iloc[0].Rate_A<=0 or pair.iloc[0].Rate_B<=0:raise ValueError('Cannot index a zero or negative baseline to 100')
    pair['Index_A']=100*pair.Rate_A/pair.iloc[0].Rate_A;pair['Index_B']=100*pair.Rate_B/pair.iloc[0].Rate_B
    raw=pair.Rate_A.corr(pair.Rate_B)
    consecutive=np.diff(pair.Year)==1
    da=np.diff(pair.Rate_A)[consecutive];db=np.diff(pair.Rate_B)[consecutive]
    differenced=float(np.corrcoef(da,db)[0,1]) if len(da)>=3 and np.std(da)>0 and np.std(db)>0 else np.nan
    return pair,{'n_years':len(pair),'level_correlation':raw,'annual_change_correlation':differenced,'n_consecutive_changes':int(consecutive.sum()),'interpretation':'Descriptive co-movement; no displacement or causal inference'}

def fixed_panel_dispersion(mortality:pd.DataFrame,years,countries=EU27):
    """Describe the same eligible countries in every year, with membership retained.

    The input must already identify one cause, sex and age. A missing year excludes
    that country for the entire interval, so changing coverage cannot change a gap.
    """
    first,last=map(int,years)
    if first>=last:raise ValueError('Dispersion requires at least two calendar years')
    members=sorted(set(countries))
    selected=mortality[mortality.Country.isin(members)&mortality.Year.between(first,last)]
    assert_unique(selected,['Country','Year'],'fixed-panel dispersion input')
    if (selected.Rate.dropna()<0).any():raise ValueError('Negative mortality in dispersion input')
    panel=selected.pivot(index='Country',columns='Year',values='Rate').reindex(index=members,columns=range(first,last+1))
    excluded=panel.index[panel.isna().any(axis=1)].tolist()
    panel=panel.dropna(axis=0,how='any')
    if len(panel)<5:raise ValueError('At least five countries with complete annual rates are required')
    summary=panel.quantile([.1,.25,.5,.75,.9]).T
    summary.columns=['P10','P25','Median','P75','P90']
    summary['IQR']=summary.P75-summary.P25
    summary['LogSD']=np.log(panel.where(panel>0)).std(axis=0,ddof=1).where((panel>0).all(axis=0))
    summary['CV']=panel.std(axis=0,ddof=1)/panel.mean(axis=0).where(panel.mean(axis=0)>0)
    summary['P90_P10']=np.divide(summary.P90,summary.P10,out=np.full(len(summary),np.nan),where=summary.P10.to_numpy()>0)
    summary['N_countries']=len(panel)
    summary['IncludedCountries']=';'.join(panel.index)
    summary['ExcludedCountries']=';'.join(excluded)
    summary=summary.rename_axis('Year').reset_index()
    return summary

def benchmark(mortality:pd.DataFrame,country:str,cause:str,sex:str,age:str,years,kind='Official EU aggregate'):
    if kind=='Official EU aggregate' and years[0]<2013:
        raise ValueError('Official EU benchmark comparisons require 2013 onwards because of the early aggregate discontinuity')
    sub=mortality[(mortality.Cause==cause)&(mortality.Sex==sex)&(mortality.Age==age)&mortality.Year.between(*years)]
    national=sub[sub.Country==country][['Year','Rate']].rename(columns={'Rate':'CountryRate'})
    assert_unique(national,['Year'],'national benchmark input')
    if kind=='Official EU aggregate':
        ref=sub[sub.Country=='EU27_2020'][['Year','Rate']].rename(columns={'Rate':'ReferenceRate'})
        ref['N_countries']=len(EU27)
        ref['ReferenceCountries']=';'.join(sorted(EU27))
        ref['ExcludedCountries']=''
    elif kind=='Unweighted fixed EU27 panel':
        member=sub[sub.Country.isin(EU27)]
        p=member.pivot(index='Country',columns='Year',values='Rate').reindex(columns=range(years[0],years[1]+1))
        p=p.dropna(axis=0,how='any')
        if p.empty:raise ValueError('No complete fixed EU27 panel for this interval')
        ref=pd.DataFrame({'Year':p.columns.astype(int),'ReferenceRate':p.mean(axis=0).to_numpy(),'N_countries':len(p)})
        ref['ReferenceCountries']=';'.join(sorted(p.index))
        ref['ExcludedCountries']=';'.join(sorted(set(EU27)-set(p.index)))
    else:raise ValueError('Unknown benchmark definition')
    out=national.merge(ref,on='Year',validate='one_to_one')
    out['GapPercent']=np.where(out.ReferenceRate>0,100*(out.CountryRate-out.ReferenceRate)/out.ReferenceRate,np.nan)
    out['ReferenceDefinition']=kind
    return out

def fit_panel(panel:pd.DataFrame,predictors:list[str],fixed_effects=True,log_outcome=False,log_predictors=()):
    assert_unique(panel,['Country','Year'],'regression panel')
    required=['Mortality']+predictors
    x=panel.dropna(subset=required).copy()
    if not len(x):raise ValueError('No complete observations for the chosen indicator definitions')
    if (x.Mortality<0).any():raise ValueError('Negative mortality')
    if log_outcome:
        if (x.Mortality<=0).any():raise ValueError('Log mortality model requires strictly positive observations')
        y=np.log(x.Mortality)
    else:y=x.Mortality
    cols={}
    for name in predictors:
        val=x[name].astype(float)
        if name in log_predictors:
            if (val<=0).any():raise ValueError(f'Log predictor {name} must be positive')
            val=np.log(val)
        cols[name]=val.to_numpy()
    X=pd.DataFrame(cols,index=x.index)
    if fixed_effects:
        if x.Country.nunique()<10:raise ValueError('Country-clustered fixed-effects analysis requires at least 10 represented countries')
        if x.Year.nunique()<3:raise ValueError('Fewer than three survey years; a longitudinal fixed-effects interpretation is not supported')
        X=pd.concat([X,pd.get_dummies(x.Country,prefix='Country',drop_first=True,dtype=float),pd.get_dummies(x.Year,prefix='Year',drop_first=True,dtype=float)],axis=1)
    X=sm.add_constant(X,has_constant='add').astype(float)
    rank=np.linalg.matrix_rank(X)
    if rank<X.shape[1]:raise ValueError('Design matrix is rank deficient; remove redundant predictors or definitions')
    if len(x)<X.shape[1]+10:raise ValueError('Insufficient residual information for the selected model')
    fit=sm.OLS(y.to_numpy(float),X.to_numpy()).fit()
    if fixed_effects:
        fit=fit.get_robustcov_results(cov_type='cluster',groups=x.Country.to_numpy(),use_correction=True,df_correction=True,use_t=True)
    else:
        fit=fit.get_robustcov_results(cov_type='HC3',use_t=True)
    ci=fit.conf_int();rows=[]
    for name in predictors:
        idx=X.columns.get_loc(name)
        rows.append({'Predictor':name,'Coefficient':fit.params[idx],'SE':fit.bse[idx],'CI_low':ci[idx,0],'CI_high':ci[idx,1],'p':fit.pvalues[idx]})
    info={'n':len(x),'n_countries':int(x.Country.nunique()),'n_years':int(x.Year.nunique()),'rank':int(rank),'residual_df':int(fit.df_resid),'r_squared':float(fit.rsquared),'covariance':'Country-clustered with country and year effects' if fixed_effects else 'Pooled descriptive OLS with HC3 covariance','log_outcome':bool(log_outcome),'log_predictors':list(log_predictors),'interpretation':'Ecological association, not individual risk or a causal effect'}
    return pd.DataFrame(rows),info,x

def adjust_pvalues(frame:pd.DataFrame,column='p',output='q',method='fdr_bh'):
    out=frame.copy();out[output]=np.nan
    eligible=out[column].notna()
    if eligible.any():out.loc[eligible,output]=multipletests(out.loc[eligible,column],method=method)[1]
    return out


# ============================================================================
# spatial.py: eurohealth
# ============================================================================

import numpy as np
import pandas as pd
import pycountry
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,silhouette_score


def alpha3(code):
    iso=GEO_ALIASES.get(code,code)
    if len(iso)!=2:return None
    found=pycountry.countries.get(alpha_2=iso)
    return found.alpha_3 if found else None

def weight_matrix(countries):
    countries=list(countries)
    if len(set(countries))!=len(countries):raise ValueError('Repeated country in spatial input')
    n=len(countries);A=np.zeros((n,n),dtype=float)
    edges={frozenset((GEO_ALIASES.get(a,a),GEO_ALIASES.get(b,b))) for a,b in BORDER_EDGES}
    for i,a in enumerate(countries):
        for j,b in enumerate(countries):
            if i!=j and frozenset((GEO_ALIASES.get(a,a),GEO_ALIASES.get(b,b))) in edges:A[i,j]=1
    sums=A.sum(axis=1,keepdims=True)
    W=np.divide(A,sums,out=np.zeros_like(A),where=sums!=0)
    return A,W

def moran(values:pd.Series,permutations=9999,seed=20261016):
    values=values.dropna()
    A,W=weight_matrix(values.index)
    active=A.sum(axis=1)>0
    isolates=values.index[~active].tolist()
    values=values.iloc[np.flatnonzero(active)]
    if len(values)<5:raise ValueError('At least five connected countries are required after excluding isolates')
    A,W=weight_matrix(values.index);n=len(values);z=values.to_numpy(float)-values.mean()
    den=float(z@z)
    if den<=0:raise ValueError('No variation in spatial rates')
    def stat(v):return float((n/W.sum())*(v@W@v)/den)
    observed=stat(z);expected=-1/(n-1)
    rng=np.random.default_rng(seed)
    simulated=np.array([stat(rng.permutation(z)) for _ in range(permutations)])
    p=(1+np.sum(np.abs(simulated-expected)>=abs(observed-expected)))/(permutations+1)
    summary={'Moran_I':observed,'expected_I':expected,'p':float(p),'n_countries':n,'isolates_excluded':isolates,'permutations':permutations,'seed':seed,'weights':'Symmetric physical border adjacency, row-standardised after excluding isolates','permutation_null_low':float(np.quantile(simulated,.025)),'permutation_null_high':float(np.quantile(simulated,.975))}
    weights=pd.DataFrame(W,index=values.index,columns=values.index)
    return summary,weights

def cluster_trajectories(panel:pd.DataFrame,mode='shape',stability_runs=100,seed=20261016):
    p=panel.sort_index(axis=1).copy()
    if np.any(np.diff(np.asarray(p.columns,dtype=int))!=1):raise ValueError('Clustering requires a defined consecutive set of years')
    excluded=p.index[p.isna().any(axis=1)].tolist();p=p.dropna(axis=0,how='any')
    if len(p)<6 or p.shape[1]<4:raise ValueError('Clustering requires at least six complete countries and four years')
    X=np.log1p(p.to_numpy(float))
    if mode=='shape':
        scale=X.std(axis=1)
        flat=scale<1e-10;excluded+=p.index[flat].tolist();p=p.iloc[np.flatnonzero(~flat)];X=X[~flat]
        X=(X-X.mean(axis=1,keepdims=True))/X.std(axis=1,keepdims=True)
    elif mode!='level':raise ValueError('Choose shape or level clustering')
    if len(X)<6:raise ValueError('Too few nonconstant complete trajectories')
    scores=[]
    for k in range(2,min(6,len(X)-1)+1):
        labels=KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(X)
        if 1<len(set(labels))<len(X):scores.append((k,float(silhouette_score(X,labels))))
    if not scores:raise ValueError('No identifiable clustering solution')
    k,score=max(scores,key=lambda item:item[1]);labels=KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(X)
    rng=np.random.default_rng(seed);stability=[]
    for b in range(stability_runs):
        cols=rng.integers(0,X.shape[1],size=X.shape[1])
        alternative=KMeans(n_clusters=k,n_init=10,random_state=seed+b+1).fit_predict(X[:,cols])
        stability.append(adjusted_rand_score(labels,alternative))
    result=pd.DataFrame({'Country':p.index,'Cluster':labels+1})
    info={'k':k,'silhouette':score,'representation':mode,'n_countries':len(p),'excluded_countries':excluded,'stability_runs':stability_runs,'median_year_resampling_ARI':float(np.median(stability)) if stability else None,'seed':seed,'interpretation':'Empirical clusters; resampling years measures sensitivity and does not identify stable disease mechanisms'}
    return result,pd.DataFrame(scores,columns=['k','silhouette']),info


# ============================================================================
# network.py: eurohealth
# ============================================================================

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller,kpss


def predictive_network(panel:pd.DataFrame,lag=1,dmax=1,min_residual_df=8):
    if lag not in (1,2,3):raise ValueError('Lag must be 1–3 years')
    if dmax!=1:raise ValueError('This experimental implementation assumes integration order no greater than one')
    panel=panel.sort_index()
    if panel.index.duplicated().any() or np.any(np.diff(panel.index.to_numpy(int))!=1):raise ValueError('Network years must be unique and consecutive')
    countries=panel.columns.tolist();A,_=weight_matrix(countries)
    tests=[];excluded=[];diagnostics=[]
    for country in countries:
        series=panel[country].dropna().to_numpy(float)
        if len(series)>=8 and np.std(series)>0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:a=float(adfuller(series,maxlag=1,regression='ct',autolag=None)[1])
                except Exception:a=np.nan
                try:k=float(kpss(series,regression='ct',nlags=1)[1])
                except Exception:k=np.nan
            diagnostics.append({'Country':country,'ADF_trend_p':a,'KPSS_trend_p':k,'n':len(series)})
    for i,src in enumerate(countries):
        for j,dst in enumerate(countries):
            if i==j or A[i,j]!=1:continue
            pair=panel[[dst,src]]
            if pair.isna().any().any():
                excluded.append({'Source':src,'Target':dst,'Reason':'Missing annual observations; calendar years were not compressed'});continue
            y=pair.iloc[:,0].to_numpy(float);x=pair.iloc[:,1].to_numpy(float)
            augmented=lag+dmax;n=len(y)-augmented
            cols=[np.ones(n),np.arange(n,dtype=float)]
            cols.extend(y[augmented-k:len(y)-k] for k in range(1,augmented+1))
            cols.extend(x[augmented-k:len(x)-k] for k in range(1,augmented+1))
            X=np.column_stack(cols)
            if n-X.shape[1]<min_residual_df:
                excluded.append({'Source':src,'Target':dst,'Reason':f'Only {len(y)} annual values; augmented model needs at least {augmented+X.shape[1]+min_residual_df} for this lag and residual-df rule'});continue
            if np.linalg.matrix_rank(X)<X.shape[1]:
                excluded.append({'Source':src,'Target':dst,'Reason':'Rank-deficient augmented lag model'});continue
            fit=sm.OLS(y[augmented:],X).fit()
            R=np.zeros((lag,X.shape[1]))
            for k in range(lag):R[k,2+augmented+k]=1
            contrast=R@fit.params;cov=R@fit.cov_params()@R.T
            if np.linalg.matrix_rank(cov)<lag:
                excluded.append({'Source':src,'Target':dst,'Reason':'Singular Wald covariance'});continue
            wald=float(contrast@np.linalg.solve(cov,contrast));p=float(chi2.sf(wald,lag))
            tests.append({'Source':src,'Target':dst,'Wald':wald,'p':p,'n_annual':len(y),'residual_df':fit.df_resid,'lag':lag,'augmented_lag':augmented})
    result=pd.DataFrame(tests)
    if len(result):
        result['q_BY']=multipletests(result.p,method='fdr_by')[1]
        result['Selected']=result.q_BY<.05
    return result,pd.DataFrame(excluded),pd.DataFrame(diagnostics)


# ============================================================================
# robustness.py: eurohealth
# ============================================================================

import numpy as np
import pandas as pd


def dispersion_audit(mortality,years,countries=EU27,population=None,resamples=10000,seed=20261018,*,fixed_weight_year=None):
    """Return endpoint bootstrap stability and leave-one-country-out tables.

    Whole country histories are paired across endpoints. These ranges concern
    geographical composition, not uncertainty of national mortality rates.
    Population weights, when supplied, weight national standardised indicators;
    they do not reconstruct the official age-standardised EU mortality rate.
    """
    if not 1000<=resamples<=100000:raise ValueError('Use 1000 to 100000 bootstrap repetitions')
    annual=fixed_panel_dispersion(mortality,years,countries)
    included=annual.IncludedCountries.iloc[0].split(';')
    start,end=int(years[0]),int(years[1])
    matrix=mortality.pivot(index='Country',columns='Year',values='Rate').loc[included,[start,end]].to_numpy(float)
    n=len(matrix)

    def measures(values):
        values=np.atleast_2d(values)
        q=np.quantile(values,[.1,.25,.75,.9],axis=1,method='linear')
        positive=(values>0).all(axis=1)
        logsd=np.full(len(values),np.nan)
        logsd[positive]=np.std(np.log(values[positive]),axis=1,ddof=1)
        return {'P90_P10':np.divide(q[3],q[0],out=np.full(len(values),np.nan),where=q[0]>0),
                'IQR':q[2]-q[1],'LogSD':logsd,
                'CV':np.divide(np.std(values,axis=1,ddof=1),values.mean(axis=1),out=np.full(len(values),np.nan),where=values.mean(axis=1)>0)}

    rng=np.random.default_rng(seed)
    indices=rng.integers(0,n,size=(resamples,n))
    point0=measures(matrix[:,0]);point1=measures(matrix[:,1])
    sample0=measures(matrix[indices,0]);sample1=measures(matrix[indices,1])
    if population is not None:
        assert_unique(population,['Country','Year'],'population weights')
        base=start if fixed_weight_year is None else int(fixed_weight_year)
        weights=population.pivot(index='Country',columns='Year',values='Population').reindex(index=included,columns=sorted({start,end,base}))
        if weights.isna().any().any() or (weights<=0).any().any():
            raise ValueError('Complete positive population weights are required for the endpoints and the fixed weight year')
        def weighted(values,weight):
            values=np.atleast_2d(values);weight=np.broadcast_to(weight,values.shape)
            weight=weight/weight.sum(axis=1,keepdims=True)
            out=np.full(len(values),np.nan);positive=(values>0).all(axis=1)
            logs=np.log(values[positive]);w=weight[positive]
            centre=np.sum(w*logs,axis=1,keepdims=True)
            out[positive]=np.sqrt(np.sum(w*(logs-centre)**2,axis=1))
            return out
        for name,w0,w1 in [
            ('LogSD_CurrentPopulation',weights[start].to_numpy(),weights[end].to_numpy()),
            ('LogSD_FixedStartPopulation',weights[base].to_numpy(),weights[base].to_numpy())
        ]:
            point0[name]=weighted(matrix[:,0],w0);point1[name]=weighted(matrix[:,1],w1)
            sample0[name]=weighted(matrix[indices,0],w0[indices]);sample1[name]=weighted(matrix[indices,1],w1[indices])

    rows=[]
    for metric in point0:
        differences=sample1[metric]-sample0[metric]
        finite=differences[np.isfinite(differences)] if np.isfinite(point0[metric][0]) and np.isfinite(point1[metric][0]) else np.array([])
        pct=100*np.divide(differences,sample0[metric],out=np.full(len(differences),np.nan),where=sample0[metric]>0)
        pct=pct[np.isfinite(pct)]
        relative=100*(point1[metric][0]/point0[metric][0]-1) if point0[metric][0]>0 else np.nan
        rows.append({'Metric':metric,'StartYear':start,'EndYear':end,'N_countries':n,
                     'StartValue':point0[metric][0],'EndValue':point1[metric][0],
                     'Difference':point1[metric][0]-point0[metric][0],
                     'StabilityLow':np.quantile(finite,.025) if len(finite) else np.nan,
                     'StabilityHigh':np.quantile(finite,.975) if len(finite) else np.nan,
                     'PercentChange':relative,
                     'PercentChangeLow':np.quantile(pct,.025) if len(pct) and np.isfinite(relative) else np.nan,
                     'PercentChangeHigh':np.quantile(pct,.975) if len(pct) and np.isfinite(relative) else np.nan,
                     'DefinedBootstrapRepetitions':len(finite),'BootstrapRepetitions':resamples,
                     'Interpretation':'Country-composition stability; not national-rate confidence limits'})
    leave=[]
    for i,country in enumerate(included):
        retained=np.delete(matrix,i,axis=0)
        before=measures(retained[:,0]);after=measures(retained[:,1])
        for metric in before:
            leave.append({'OmittedCountry':country,'Metric':metric,'StartValue':before[metric][0],
                          'EndValue':after[metric][0],'Difference':after[metric][0]-before[metric][0],
                          'PercentChange':100*(after[metric][0]/before[metric][0]-1) if before[metric][0]>0 else np.nan})
    return pd.DataFrame(rows),pd.DataFrame(leave)


# ============================================================================
# export.py: eurohealth
# ============================================================================

import hashlib
import importlib.metadata
import io
import json
import platform
import zipfile
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd



def _json_default(obj):
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(np.floating,)):return float(obj)
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,Path):return str(obj)
    raise TypeError(type(obj).__name__)

def operation_id(kind,settings,input_hashes):
    payload=json.dumps({'kind':kind,'settings':settings,'inputs':input_hashes},sort_keys=True,default=_json_default)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def environment():
    deps={}
    for name in ['numpy','pandas','scipy','statsmodels','prophet','ruptures','scikit-learn','streamlit','pycountry','plotly']:
        try:deps[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:deps[name]=None
    return {'python':platform.python_version(),'platform':platform.platform(),'dependencies':deps}

def make_archive(analyses:dict,snapshot_metadata:list[dict]):
    if not analyses:raise ValueError('No completed analyses to export')
    buffer=io.BytesIO();files={};operations=[];sources={}
    source_path=Path(__file__).resolve()
    sources[source_path.name]=source_path.read_bytes()
    sources['requirements.txt']=REQUIREMENTS_TEXT.encode()
    for identifier,item in analyses.items():
        op={'id':identifier,'kind':item['kind'],'settings':item['settings'],'metadata':item.get('metadata',{}),'tables':{},'inputs':{}}
        for label,df in item.get('tables',{}).items():
            name=f'analyses/{identifier}/{label}.csv';content=df.to_csv(index=False).encode();files[name]=content
            op['tables'][label]={'file':name,'sha256':hashlib.sha256(content).hexdigest(),'rows':len(df)}
        for label,df in item.get('inputs',{}).items():
            name=f'inputs/{identifier}/{label}.csv';content=df.to_csv(index=False).encode();files[name]=content
            op['inputs'][label]={'file':name,'sha256':hashlib.sha256(content).hexdigest(),'rows':len(df)}
        operations.append(op)
    snapshots=[]
    for meta in snapshot_metadata:
        record={k:v for k,v in meta.items() if k not in ('snapshot_path','labels')}
        path=Path(meta.get('snapshot_path',''))
        if not path.is_file():raise FileNotFoundError('A saved Eurostat response is no longer available for export: '+path.name)
        if path.is_file():
            name='snapshots/'+path.name;content=path.read_bytes()
            if hashlib.sha256(content).hexdigest()!=meta['sha256']:raise ValueError('Source snapshot checksum mismatch during export')
            files[name]=content;record['archive_file']=name
        snapshots.append(record)
    manifest={'software_version':__version__,'generated_utc':datetime.now(timezone.utc).isoformat(),'environment':environment(),
              'operations':operations,'source_snapshots':snapshots,
              'source_sha256':{name:hashlib.sha256(content).hexdigest() for name,content in sources.items()},
              'reproduction':f'Use python source/{source_path.name} --archive FILE.zip to rerun the saved operations with the included source and input tables. Floating results compared with numerical tolerances; model warnings retained.'}
    files['manifest.json']=json.dumps(manifest,indent=2,ensure_ascii=False,default=_json_default).encode()
    for name,content in sources.items():files['source/'+name]=content
    files['README.txt']=(f'European Public Health Dashboard {__version__}\n\nEach saved analysis has its own complete settings and inputs in manifest.json.\nRaw Eurostat snapshots, flags, source code and checksums are included.\nUse python source/{source_path.name} with --archive and the original archive path.\nForecast intervals are model dependent; descriptive and ecological outputs do not establish causality.\n').encode()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,content in sorted(files.items()):z.writestr(name,content)
    return buffer.getvalue(),manifest


# ============================================================================
# reproduce_export.py: dashboard_option_a
# ============================================================================

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# Panel analyses shared by the dashboard and the archive runner.

def mortality_panel(mortality, settings):
    """Compare the official aggregate and the distribution of national rates."""
    first,last=settings['years']
    data=mortality[mortality.Year.between(first,last)].copy()
    assert_unique(data,['Country','Cause','Year'],'mortality panel')
    if settings.get('combine_dementia'):
        pair=data[data.Cause.isin(['G30','F01_F03'])]
        if set(pair.Cause)!={'G30','F01_F03'}:
            raise ValueError('Select both Alzheimer disease and F01/F03 dementia to show their sum.')
        combined=pair.groupby(['Country','Year'],as_index=False).agg(
            Rate=('Rate',lambda s:s.sum(min_count=2)),
            Flag=('Flag',lambda s:' '.join(sorted(set(' '.join(s.astype(str)).split())))))
        combined['Cause']='G30 + F01_F03'
        data=pd.concat([data,combined],ignore_index=True)
    national=data[data.Country.isin(EU27)].copy()
    official=data[data.Country.eq('EU27_2020')].copy()
    annual=national.groupby(['Cause','Year'],as_index=False).agg(
        Median=('Rate','median'),Mean=('Rate','mean'),Countries=('Rate','count'))
    annual=annual.merge(official[['Cause','Year','Rate']].rename(columns={'Rate':'OfficialEU'}),
                        on=['Cause','Year'],how='outer',validate='one_to_one').sort_values(['Cause','Year'])
    endpoint=lambda x: x[x.Year==first][['Country','Cause','Rate','Flag']].rename(columns={'Rate':'StartRate','Flag':'StartFlag'}).merge(
        x[x.Year==last][['Country','Cause','Rate','Flag']].rename(columns={'Rate':'EndRate','Flag':'EndFlag'}),
        on=['Country','Cause'],validate='one_to_one').dropna(subset=['StartRate','EndRate'])
    paired=endpoint(data)
    paired['AbsoluteChange']=paired.EndRate-paired.StartRate
    paired['PercentChange']=100*paired.AbsoluteChange/paired.StartRate.where(paired.StartRate>0)
    flags=data.groupby(['Country','Cause']).Flag.agg(AnyFlag=lambda s:s.astype(str).str.strip().ne('').any(),
                                                  ComparabilityFlag=lambda s:any(flag_contains(v) for v in s)).reset_index()
    paired=paired.merge(flags,on=['Country','Cause'],validate='one_to_one')
    paired['Direction']=np.select([paired.AbsoluteChange>0,paired.AbsoluteChange<0],['Increased','Decreased'],default='Unchanged')
    summary=[]
    for cause,g in paired[paired.Country.isin(EU27)].groupby('Cause'):
        for label,sub in [('All eligible',g),('Exclude b/d/u series',g[~g.ComparabilityFlag]),('Exclude any flagged series',g[~g.AnyFlag])]:
            change=sub.PercentChange
            summary.append({'Cause':cause,'Sensitivity':label,'Countries':len(sub),'PositiveBaseline':int(change.count()),
                            'Increased':int(sub.Direction.eq('Increased').sum()),'Decreased':int(sub.Direction.eq('Decreased').sum()),
                            'Unchanged':int(sub.Direction.eq('Unchanged').sum()),'MedianPercentChange':change.median(),
                            'Q25':change.quantile(.25),'Q75':change.quantile(.75)})
    coverage=data.groupby('Cause',as_index=False).agg(Cells=('Rate','size'),Observed=('Rate','count'),
        Zero=('Rate',lambda s:int(s.eq(0).sum())),Flagged=('Flag',lambda s:int(s.astype(str).str.strip().ne('').sum())))
    return {'annual':annual,'country_endpoints':paired[paired.Country.isin(EU27)].reset_index(drop=True),
            'official_endpoints':paired[paired.Country.eq('EU27_2020')].reset_index(drop=True),
            'country_summary':pd.DataFrame(summary,columns=['Cause','Sensitivity','Countries','PositiveBaseline','Increased','Decreased','Unchanged','MedianPercentChange','Q25','Q75']),
            'coverage':coverage,'flagged':data[data.Flag.astype(str).str.strip().ne('')].reset_index(drop=True)}


def dispersion_tables(mortality,settings,population=None):
    years=settings['years'];countries=settings['countries'];base=settings.get('fixed_weight_year',years[0])
    annual=fixed_panel_dispersion(mortality,years,countries)
    stability,leave=dispersion_audit(mortality,years,countries,population,
        settings.get('bootstrap_repetitions',10000),settings.get('bootstrap_seed',20261018),fixed_weight_year=base)
    if population is not None:
        members=annual.IncludedCountries.iloc[0].split(';');idx=annual.Year.tolist()
        rates=mortality.pivot(index='Country',columns='Year',values='Rate').loc[members,idx].to_numpy(float)
        pop=population.pivot(index='Country',columns='Year',values='Population').reindex(index=members,columns=list(dict.fromkeys(idx+[base])))
        def weighted(w):
            valid=(rates>0).all(axis=0)&np.isfinite(w).all(axis=0)&(w>0).all(axis=0)
            result=np.full(len(idx),np.nan)
            if valid.any():
                v=np.log(rates[:,valid]);weights=w[:,valid]/w[:,valid].sum(axis=0)
                result[valid]=np.sqrt((weights*(v-(v*weights).sum(axis=0))**2).sum(axis=0))
            return result
        annual['LogSD_CurrentPopulation']=weighted(pop[idx].to_numpy(float))
        annual['LogSD_FixedStartPopulation']=weighted(np.broadcast_to(pop[[base]].to_numpy(float),rates.shape))
    return {'dispersion':annual,'stability':stability,'leave_one_out':leave}


def dispersion_sensitivities(mortality,settings,population=None):
    outputs=[];failures=[]
    specs=[(a,b,False) for a,b in settings['periods']]
    if settings['exclude_flags']:
        specs.append((*settings['years'],True))
    for first,last,exclude in specs:
        panel=mortality[mortality.Year.between(first,last)]
        members=list(settings['countries'])
        if exclude:
            flagged=set(panel.loc[panel.Flag.map(flag_contains),'Country']);members=[c for c in members if c not in flagged]
        op=settings|{'years':[first,last],'countries':members}
        try:
            annual=fixed_panel_dispersion(panel,[first,last],members)
            eligible=annual.IncludedCountries.iloc[0].split(';')
            pop=covered_population(population,eligible,[first,last,settings['fixed_weight_year']])
            summary,_=dispersion_audit(panel,[first,last],members,pop,10000,20261018,
                                       fixed_weight_year=settings['fixed_weight_year'])
            outputs.append(summary.assign(ExcludedFlags=exclude,IncludedCountries=';'.join(eligible)))
        except ValueError as exc:
            failures.append({'StartYear':first,'EndYear':last,'ExcludedFlags':exclude,'Reason':str(exc)})
    return {'sensitivity':pd.concat(outputs,ignore_index=True) if outputs else pd.DataFrame(columns=['Metric','StartYear','EndYear','Difference','StabilityLow','StabilityHigh','ExcludedFlags']),
            'ineligible':pd.DataFrame(failures,columns=['StartYear','EndYear','ExcludedFlags','Reason'])}


def covered_population(population, countries, years):
    if population is None:return None
    values=population.pivot(index='Country',columns='Year',values='Population').reindex(index=countries,columns=sorted(set(years)))
    return population if values.notna().all(axis=None) and (values>0).all(axis=None) else None


def gdp_sensitivity(mortality,gdp,settings):
    first,last=settings['years'];before=settings.get('earlier_end')
    data=mortality[mortality.Country.isin(EU27)&mortality.Year.between(first,last)]
    rows=[];exclusions=[]
    for cause,g in data.groupby('Cause'):
        label=f'{first}–{last}'
        panel=g[['Country','Year','Rate']].rename(columns={'Rate':'Mortality'}).merge(
            gdp[['Country','Year','GDP per capita']],on=['Country','Year'],validate='one_to_one')
        flagged=set(g.loc[g.Flag.map(flag_contains),'Country'])
        variants=[(label,panel)]
        if before is not None and first<=before<last:variants.append((f'{first}–{before}',panel[panel.Year<=before]))
        variants.append(('Exclude b/d/u series',panel[~panel.Country.isin(flagged)]))
        for spec,sub in variants:
            discarded=int(sub.Mortality.le(0).sum())
            try:
                co,info,_=fit_panel(sub[sub.Mortality>0],['GDP per capita'],fixed_effects=True,
                                    log_outcome=True,log_predictors=['GDP per capita'])
                row=co.iloc[0].to_dict()|info|{'Cause':cause,'Specification':spec,'NonpositiveExcluded':discarded}
                for a,b in [('Coefficient','PercentChangeFor10PercentGDP'),('CI_low','Effect10Low'),('CI_high','Effect10High')]:
                    row[b]=100*np.expm1(row[a]*np.log(1.1))
                rows.append(row)
            except ValueError as exc:exclusions.append({'Cause':cause,'Specification':spec,'Error':str(exc)})
    output=pd.DataFrame(rows)
    if not output.empty:
        output['q']=np.nan
        for _,idx in output.groupby('Specification').groups.items():output.loc[idx,'q']=adjust_pvalues(output.loc[idx]).q
    else:output=pd.DataFrame(columns=['Cause','Specification','Coefficient','CI_low','CI_high','p','q'])
    return {'associations':output,'ineligible':pd.DataFrame(exclusions,columns=['Cause','Specification','Error'])}


def compare_moran(mortality,settings):
    rows=[];failures=[];result={}
    for cause,g in mortality[mortality.Year.eq(settings['map_year'])&mortality.Country.isin(EU27)].groupby('Cause'):
        try:
            info,weights=moran(g.set_index('Country').Rate,permutations=settings['permutations'])
            info['isolates_excluded']=';'.join(info['isolates_excluded'])
            rows.append(info|{'Cause':cause})
            result[f"weights_{cause}"]=weights.reset_index().rename(columns={'index':'Country'})
        except ValueError as exc:failures.append({'Cause':cause,'Reason':str(exc)})
    stats=pd.DataFrame(rows)
    result['associations']=adjust_pvalues(stats) if len(stats) else pd.DataFrame(columns=['Cause','Moran_I','p','q'])
    result['ineligible']=pd.DataFrame(failures,columns=['Cause','Reason'])
    return result


def rolling_benchmark(mortality,settings,progress=None):
    """Expanding-window validation on common observed targets and training-only scales."""
    first,last=settings['years'];causes=settings['causes'];horizons=(1,3)
    data=mortality[mortality.Year.between(first,last)&mortality.Country.isin(EU27)]
    assert_unique(data,['Country','Cause','Year'],'forecast panel')
    tasks=[];eligibility=[]
    for (country,cause),g in data.groupby(['Country','Cause']):
        try:
            x=consecutive_series(g[['Year','Rate']],minimum=9)
            if (int(x.Year.min()),int(x.Year.max()))!=(first,last):raise ValueError('Incomplete selected calendar interval')
            tasks.append((country,cause,x.to_dict('records'),range(first+7,last),horizons,list(MODELS)))
            eligibility.append({'Country':country,'Cause':cause,'Eligible':True,'Reason':''})
        except ValueError as exc:eligibility.append({'Country':country,'Cause':cause,'Eligible':False,'Reason':str(exc)})
    if not tasks:raise ValueError('Select at least nine consecutive annual observations, including eight training years and a held-out target.')
    rows=[];failures=[]
    for i,task in enumerate(tasks,1):
        good,bad=rolling_series(task);rows.extend(good);failures.extend(bad)
        if progress:progress(i,len(tasks))
    if not rows:raise ValueError('No forecast targets could be scored.')
    pred=score_predictions(pd.DataFrame(rows)).sort_values(['Country','Cause','Origin','Horizon','Model']).reset_index(drop=True)
    keys=['Country','Cause','Origin','Horizon'];assert_unique(pred,keys+['Model'],'rolling forecasts')
    available=pred.groupby(keys).Model.transform('nunique').eq(len(MODELS))
    usable=pred.loc[available&pred.MASEScale.gt(0)].copy()
    if usable.empty:raise ValueError('No targets have predictions from all models and a positive training error scale.')
    flagged=data.groupby(['Country','Cause']).Flag.agg(lambda s:any(flag_contains(v) for v in s))
    unflagged=~pd.MultiIndex.from_frame(usable[['Country','Cause']]).isin(flagged[flagged].index)
    groups={
        'All targets':np.ones(len(usable),dtype=bool),
        'Exclude 2020-2021 target years':~usable.Year.isin([2020,2021]),
        'Entire history and prediction path excludes 2020-2021':(usable.Year<2020)|(first>2021),
        'Exclude b/d/u series':unflagged,
    }
    def scores(df,columns):
        return df.groupby(columns,as_index=False).agg(N=('Actual','size'),MASE=('ScaledAbsoluteError','mean'),
               MAE=('AbsoluteError','mean'),RMSE=('SquaredError',lambda x:np.sqrt(x.mean())),
               Coverage95=('Covered','mean'),MeanIntervalScore95=('IntervalScore','mean'))
    summaries=[];counts=[];by_cause=[];contrasts=[]
    for label,mask in groups.items():
        subset=usable.loc[mask]
        summaries.append(scores(subset,['Horizon','Model']).assign(Sensitivity=label))
        by_cause.append(scores(subset,['Cause','Horizon','Model']).assign(Sensitivity=label))
        for h in horizons:
            part=subset[subset.Horizon.eq(h)]
            counts.append({'Sensitivity':label,'Horizon':h,'TasksPerMethod':len(part)//len(MODELS),
                'CountryOutcomeSeries':len(part[['Country','Cause']].drop_duplicates()),
                'Origins':';'.join(map(str,sorted(part.Origin.unique()))),'TargetYears':';'.join(map(str,sorted(part.Year.unique())))})
            if part.empty:continue
            pivot=part.pivot(index=['Country','Cause','Origin'],columns='Model',values='ScaledAbsoluteError')
            paired=pivot.drop(columns='Naive').subtract(pivot.Naive,axis=0).groupby(['Country','Cause']).mean()
            country=paired.groupby('Country').mean().sort_index()
            n=len(country)
            if n<2:continue
            indices=np.random.default_rng(20261018).integers(0,n,size=(10000,n))
            for model in country:
                boot=country[model].to_numpy()[indices].mean(axis=1)
                contrasts.append({'Sensitivity':label,'Horizon':h,'Model':model,'Countries':n,'Series':len(paired),
                    'MeanDifference':country[model].mean(),'StabilityLow':np.quantile(boot,.025),'StabilityHigh':np.quantile(boot,.975)})
    series=[];labels=[]
    for (cause,h),g in usable.groupby(['Cause','Horizon']):
        pivot=g.pivot(index=['Country','Origin'],columns='Model',values='ScaledAbsoluteError')
        for model in MODELS:
            if model=='Naive':continue
            series.append((pivot[model]-pivot.Naive).groupby('Country').mean())
            labels.append((cause,h,model))
    simultaneous=[]
    # Use one country intersection for the full contrast family and report its size.
    matrix=pd.concat(series,axis=1,join='inner').sort_index().to_numpy(float)
    if len(matrix)>=2:
        n=len(matrix);indices=np.random.default_rng(20261018).integers(0,n,size=(10000,n))
        est=matrix.mean(axis=0);boot=matrix[indices].mean(axis=1);se=boot.std(axis=0,ddof=1)
        scaled=np.divide(boot-est,se,out=np.zeros_like(boot),where=se>0)
        critical=float(np.quantile(np.max(np.abs(scaled),axis=1),.95))
        for j,(cause,h,model) in enumerate(labels):
            simultaneous.append({'Cause':cause,'Horizon':h,'Model':model,'MeanDifference':est[j],
                'PointwiseLow':np.quantile(boot[:,j],.025),'PointwiseHigh':np.quantile(boot[:,j],.975),
                'SimultaneousLow':est[j]-critical*se[j],'SimultaneousHigh':est[j]+critical*se[j],
                'FamilySize':len(labels),'MaxTCritical':critical,'Countries':n})
    return {'predictions':pred,'scores':pd.concat(summaries,ignore_index=True),'by_cause':pd.concat(by_cause,ignore_index=True),
        'by_year':scores(usable,['Horizon','Year','Model']),'targets':pd.DataFrame(counts),
        'paired_differences':pd.DataFrame(contrasts,columns=['Sensitivity','Horizon','Model','Countries','Series','MeanDifference','StabilityLow','StabilityHigh']),
        'simultaneous_bands':pd.DataFrame(simultaneous,columns=['Cause','Horizon','Model','MeanDifference','PointwiseLow','PointwiseHigh','SimultaneousLow','SimultaneousHigh','FamilySize','MaxTCritical','Countries']),
        'eligibility':pd.DataFrame(eligibility,columns=['Country','Cause','Eligible','Reason']),
        'failed_fits':pd.DataFrame(failures,columns=['Country','Cause','Origin','Model','Error'])}


def _execute_saved_operation(kind,settings,inputs):
    if kind=='mortality_panel':return mortality_panel(inputs['mortality'],settings)
    if kind=='forecast_benchmark':return rolling_benchmark(inputs['mortality'],settings)
    if kind=='dispersion_sensitivity':return dispersion_sensitivities(inputs['mortality'],settings,inputs.get('population'))
    if kind=='gdp_sensitivity':return gdp_sensitivity(inputs['mortality'],inputs['gdp'],settings)
    if kind=='moran_panel':return compare_moran(inputs['mortality'],settings)
    if kind=='temporal':
        segments,detail=temporal_summary(inputs['series'],settings['penalty_multiplier'])
        return {'segments':segments,'anomaly_detail':detail}
    if kind=='forecast':
        return forecast_analysis_tables(inputs['series'],settings)
    if kind=='comparison':
        table,info=compare_causes(inputs['cause_A'],inputs['cause_B']);return {'comparison':table}
    if kind=='benchmark':
        return {'benchmark':benchmark(inputs['mortality'],settings['country'],settings['cause'],settings['sex'],settings['age'],settings['years'],settings['reference'])}
    if kind=='dispersion':
        return dispersion_tables(inputs['mortality'],settings,inputs.get('population'))
    if kind=='regression':
        table,info,used=fit_panel(inputs['panel'],settings['predictors'],fixed_effects=settings['fixed_effects'],log_outcome=settings.get('log_outcome',False),log_predictors=settings.get('log_predictors',[]))
        return {'coefficients':table,'complete_cases':used}
    if kind=='spatial':
        data=inputs['spatial_values'];info,W=moran(data.set_index('Country').Rate,settings['permutations'])
        return {'weights':W.reset_index().rename(columns={'index':'Country'}),'moran':pd.DataFrame([{k:v for k,v in info.items() if not isinstance(v,list)}])}
    if kind=='clustering':
        data=inputs['trajectories'].set_index('Country');data.columns=data.columns.astype(int)
        data=data.apply(pd.to_numeric,errors='coerce')
        memberships,scores,info=cluster_trajectories(data,settings['representation'],stability_runs=100)
        return {'clusters':memberships,'silhouettes':scores}
    if kind=='network':
        data=inputs['network_panel'].set_index('Year');data.index=data.index.astype(int)
        data=data.apply(pd.to_numeric,errors='coerce')
        tests,excluded,diag=predictive_network(data,settings['lag'])
        return {'tests':tests,'excluded':excluded,'diagnostics':diag}
    raise ValueError(f'Unknown exported operation {kind}')

def _reproduce_cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('reproduced_export'))
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    with zipfile.ZipFile(args.archive) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        for name,expected in manifest['source_sha256'].items():
            local=Path(__file__).resolve().parent/name
            if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Source differs from captured release: {name}; run using the source included in this archive')
        results=[]
        for op in manifest['operations']:
            inputs={}
            for label,item in op['inputs'].items():
                raw=archive.read(item['file'])
                if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Input checksum mismatch')
                import io
                inputs[label]=pd.read_csv(io.BytesIO(raw),keep_default_na=False)
                for col in inputs[label].columns:
                    if col in ('Rate','Mortality','Population','GDP per capita') or col in op['settings'].get('predictors',[]):
                        inputs[label][col]=pd.to_numeric(inputs[label][col],errors='coerce')
            actual=_execute_saved_operation(op['kind'],op['settings'],inputs)
            for label,table in actual.items():
                if label not in op['tables']:continue
                item=op['tables'][label];raw=archive.read(item['file'])
                if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Output checksum mismatch')
                import io
                expected=pd.read_csv(io.BytesIO(raw),keep_default_na=False) if raw.strip() else pd.DataFrame()
                output=args.output/f"{op['id']}_{label}.csv";table.to_csv(output,index=False)
                a=table.astype(str).reset_index(drop=True);e=expected.astype(str).reset_index(drop=True)
                if list(a.columns)!=list(e.columns) or a.shape!=e.shape:raise AssertionError(f'Reproduced output shape differs: {label}')
                for col in a.columns:
                    av=pd.to_numeric(a[col],errors='coerce');ev=pd.to_numeric(e[col],errors='coerce')
                    numeric=av.notna()&ev.notna()
                    if numeric.any() and not np.allclose(av[numeric],ev[numeric],rtol=1e-6,atol=1e-8):raise AssertionError(f'Numeric output differs: {label}.{col}')
                    if not (a.loc[~numeric,col].replace({'nan':''})==e.loc[~numeric,col].replace({'nan':''})).all():raise AssertionError(f'Text output differs: {label}.{col}')
                results.append({'operation':op['id'],'table':label,'matched':True})
        (args.output/'verification.json').write_text(json.dumps(results,indent=2))
        print(f'Regenerated and verified {len(results)} saved tables')


CAUSES = {
    "TOTAL": "All causes of death",
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
    "U071": "COVID-19, virus identified",
    "U072": "COVID-19, virus not identified"
}


# ============================================================================
# app.py: dashboard_option_a
# ============================================================================

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


_cache_user=str(os.getuid()) if hasattr(os,'getuid') else 'default'
CACHE=Path(os.environ.get('EUROHEALTH_CACHE',str(
    Path(tempfile.gettempdir())/f'european_public_health_{_cache_user}_live')))


def live_inputs(cause,source,refresh_after=0.0):
    return load_mortality([cause],CACHE,source=source,refresh_after=refresh_after)


def contextual_input(name,sex,countries,refresh_after=0.0):
    return load_factor(name,CACHE,sex=sex,geographies=countries,refresh_after=refresh_after)

def remember(kind,settings,tables,inputs,metadata=None):
    key=operation_id(kind,settings,[m['sha256'] for m in st.session_state.snapshots.values()])
    st.session_state.analyses[key]={'kind':kind,'settings':copy.deepcopy(settings),'tables':{k:v.copy() for k,v in tables.items()},'inputs':{k:v.copy() for k,v in inputs.items()},'metadata':metadata or {}}
    st.session_state.analyses[key]['metadata']['available_source_snapshots']=list(st.session_state.snapshots)
    st.session_state.pop('export_archive',None)
    return key

def load_mortality_view(cause,choice,refresh_after=0.0):
    """Keep the two collections and their source manifests distinct."""
    parts=[];sources=[];issues=[]
    for code in MORTALITY_SOURCES[choice]:
        try:
            frame,meta=live_inputs(cause,code,refresh_after)
        except NoEurostatObservations:
            issues.append(('empty',code,'No rates for this cause in the selected countries.'))
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append(('error',code,str(exc)))
        else:
            parts.append(frame);sources.append(meta)
    if not parts:return pd.DataFrame(),sources,issues
    data=pd.concat(parts,ignore_index=True)
    assert_unique(data,['Country','Year','Cause','Sex','Age'],'mortality timeline')
    return data,sources,issues


# Dashboard presentation: the global sidebar and five workspaces are shared by
# the Eurostat collections. Only the active tab runs its analysis UI.
from dataclasses import dataclass
from plotly.subplots import make_subplots


WORKSPACES = (
    '📈 Temporal Trends',
    '⚖️ Cross-Cause Comparison',
    '🔬 Health & Macro Factors',
    '🗺️ Spatial & Clusters',
    '🔗 Temporal Network',
)
INK = '#28599d'
TEAL = '#117d77'
AMBER = '#b86a35'
PURPLE = '#795ab3'
RATE_LABEL = 'Standardised deaths per 100,000'

DASHBOARD_HEADER = '''
<style>
[data-testid="stMainBlockContainer"] {max-width: 1520px; padding-top: 3rem;}
[data-testid="stSidebarContent"] {border-inline-end: 1px solid rgba(90,120,145,.20);}
[data-testid="stSidebar"] h2 {font-size: 1.05rem; letter-spacing: -.015em;}
.eph-intro {
  padding: 1.35rem clamp(1rem, 3vw, 2.3rem) 1.45rem;
  border: 1px solid #cde4e5; border-radius: 16px;
  border-inline-start: 4px solid #117d77;
  background: linear-gradient(112deg, #edf7fa 2%, #eff8f4 85%);
  color: #15314b; margin: .1rem 0 .25rem;
}
.eph-intro h1 {font-size: clamp(1.66rem, 3.2vw, 2.55rem); line-height: 1.13;
  letter-spacing: -.033em; margin: 0 0 .42rem; padding: 0; color: #15314b;}
.eph-intro p {font-size: .98rem; line-height: 1.48; margin: 0; color: #36546c;}
.st-key-analysis_tabs [role="tablist"] {gap: .35rem; scrollbar-width: thin;}
.st-key-analysis_tabs [role="tab"] {white-space: nowrap; padding-inline: .7rem; border-radius: .5rem .5rem 0 0;}
.st-key-analysis_tabs [role="tab"][aria-selected="true"] {background: rgba(17,125,119,.10);}
.st-key-analysis_tabs [data-baseweb="tab-highlight"] {background-color: #117d77;}
[data-testid="stMetric"] {border-radius: 12px;}
[data-testid="stMetricLabel"] {min-height: 1.25rem;}
[data-testid="stMetricValue"] {font-size: clamp(1.42rem, 2.1vw, 1.92rem); letter-spacing: -.025em;}
@media (max-width: 700px) {
  [data-testid="stMainBlockContainer"] {padding-top: 2.65rem;}
  .eph-intro {border-radius: 12px; padding-block: 1.1rem;}
  .eph-intro p {font-size: .91rem;}
  .st-key-analysis_tabs [role="tab"] {padding-inline: .48rem;}
}
</style>
<header class="eph-intro">
  <h1>European Public Health Dashboard</h1>
  <p>Explore mortality trends, forecasts and geographic patterns across Europe.</p>
</header>
'''


@dataclass
class DashboardSelection:
    data: pd.DataFrame
    cohort: pd.DataFrame
    observed: pd.DataFrame
    selected: pd.DataFrame
    settings: dict
    available: list
    country_labels: dict
    view_years: tuple
    combined: bool
    spans_change: bool
    refresh_after: float

    @property
    def title(self):
        cause=self.settings['cause'];country=self.settings['country']
        return f"{CAUSES.get(cause,cause)} · {self.country_labels.get(country,country)}"

    @property
    def subtitle(self):
        s=self.settings
        return f"{AGE_LABELS[s['age']]} · {SEX_LABELS[s['sex']]} · {self.view_years[0]}–{self.view_years[1]}"


def plot_chart(fig, *, height=350, alt, key=None):
    fig.update_layout(height=height,margin=dict(l=16,r=16,t=30,b=40),
                      legend=dict(orientation='h',yanchor='bottom',y=1.02,x=0,title_text=''),
                      font=dict(size=13))
    fig.update_xaxes(gridcolor='rgba(95,122,147,.15)',zeroline=False)
    fig.update_yaxes(gridcolor='rgba(95,122,147,.15)',zeroline=False)
    st.plotly_chart(fig,width='stretch',key=key,alt=alt,
                    config={'displaylogo':False,'responsive':True,'scrollZoom':False})


def current_result(kind, settings):
    """Show only results that belong to the filters and options on screen."""
    for item in reversed(st.session_state.analyses.values()):
        if item['kind']==kind and all(item['settings'].get(k)==v for k,v in settings.items()):
            return item
    return None


def show_analysis_exports():
    with st.container(border=True,key='export_panel'):
        st.subheader('Download analysis report',anchor=False)
        analyses=st.session_state.analyses
        n=len(analyses)
        if not n:
            st.caption('Run an analysis to download its results, settings and source data together.')
            return
        st.caption(f'{n} saved {"analysis" if n==1 else "analyses"}. Each keeps its own filters, Eurostat inputs and calculation code.')
        left,right=st.columns(2)
        if left.button('Prepare complete analysis archive',width='stretch'):
            try:
                archive,_=make_archive(analyses,list(st.session_state.snapshots.values()))
                st.session_state.export_archive=archive
            except (OSError,ValueError) as exc:st.error(f'The archive could not be prepared: {exc}')
        if 'export_archive' in st.session_state:
            right.download_button('Download analysis archive',st.session_state.export_archive,
                                  file_name=f'European_Health_Analysis_{__version__}.zip',mime='application/zip',width='stretch')
        with st.expander('Saved analyses'):
            names={'temporal':'Trend segments','forecast':'Forecast','benchmark':'EU comparison',
                   'comparison':'Cause comparison','regression':'Contextual factors','dispersion':'Geographical dispersion',
                   'mortality_panel':'EU and national trends','forecast_benchmark':'Rolling forecast comparison',
                   'dispersion_sensitivity':'Dispersion sensitivities','gdp_sensitivity':'GDP associations','moran_panel':'Moran’s I across causes',
                   'spatial':'Spatial association','clustering':'Trajectory groups','network':'Temporal network'}
            rows=[]
            for item in analyses.values():
                s=item['settings'];first,last=s['years']
                rows.append({'Analysis':names.get(item['kind'],item['kind']),
                             'Country':s['country'],'Cause':CAUSES.get(s['cause'],s['cause']),
                             'Age':AGE_LABELS[s['age']],'Sex':SEX_LABELS[s['sex']],
                             'Collection':COLLECTION_LABELS[s['dataset']],'Years':f'{first}–{last}'})
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')


def read_selection():
    with st.sidebar:
        st.header('📊 Global data filters',anchor=False)
        mode=st.selectbox('Mortality data',list(MORTALITY_SOURCES),
                          help='Choose a Eurostat collection or view both on one timeline.')
        combined=mode==COMBINED_COLLECTIONS
        country_slot=st.empty()
        available=list(CAUSES)
        cause=st.selectbox('Cause of death',available,index=available.index('I60-I69'),format_func=lambda c:CAUSES.get(c,c))
        if st.button('Refresh Eurostat data',width='stretch',
                     help='Fetch the latest available Eurostat data for this selection and the analyses you run next.'):
            st.session_state.refresh_after=time.time()
    refresh_after=st.session_state.get('refresh_after',0.0)
    try:
        with st.spinner('Loading Eurostat mortality rates…'):
            data,sources,issues=load_mortality_view(cause,mode,refresh_after)
    except Exception as exc:
        st.error(f'The selected mortality data could not be loaded: {exc}')
        return None,[]
    country_labels={}
    for meta in sources:
        st.session_state.snapshots[meta['sha256']]=meta
        country_labels.update(meta.get('labels',{}).get('geo',{}))
    for status,code,detail in issues:
        label=COLLECTION_LABELS[code]
        if status=='empty':st.info(f'{label} collection: {detail}')
        else:st.warning(f'{label} collection could not be loaded. Try refreshing the data. ({detail})')
    if data.empty:
        st.info('Choose another cause or mortality collection.')
        return None,sources
    available_rates=data[(data.Cause==cause)&data.Rate.notna()]
    if available_rates.empty:
        st.info('No rates are available for this cause. Choose another cause or collection.')
        return None,sources
    countries=sorted(available_rates.Country.unique(),key=lambda c:country_labels.get(c,c))
    with st.sidebar:
        default='DE' if 'DE' in countries else countries[0]
        country=country_slot.selectbox('Country or official aggregate',countries,index=countries.index(default),
                                       format_func=lambda c:country_labels.get(c,c))
        by_country=available_rates[available_rates.Country==country]
        age=st.selectbox('Age group',[a for a in AGE_LABELS if a in set(by_country.Age)],format_func=AGE_LABELS.get)
        sex=st.selectbox('Sex',[s for s in SEX_LABELS if s in set(by_country.loc[by_country.Age==age,'Sex'])],format_func=SEX_LABELS.get)
        first,last=int(available_rates.Year.min()),int(available_rates.Year.max())
        view_years=(first,last) if first==last else st.slider(
            'Years',first,last,(first,last),key=f'years_{mode}_{first}_{last}',persist_state='page')
        if first==last:st.caption(f'Available year: {first}')
    view_cohort=data[(data.Cause==cause)&(data.Sex==sex)&(data.Age==age)&data.Year.between(*view_years)]
    observed=view_cohort[view_cohort.Country==country].sort_values('Year')
    available_periods={code:(int(g.Year.min()),int(g.Year.max()))
                       for code,g in observed[observed.Rate.notna()].groupby('dataset')}
    if not available_periods:
        st.info('No rates match these years. Adjust the year range or the population selection.')
        return None,sources
    spans_change=combined and view_years[0]<COLLECTION_BREAK<=view_years[1]
    with st.sidebar:
        if len(available_periods)>1:
            source=st.radio('Calculate using',[MODERN_COLLECTION,HISTORICAL_COLLECTION],
                            format_func=COLLECTION_LABELS.get,horizontal=True,key='calculation_collection',
                            persist_state='page',help='Both collections stay on the chart. Calculations use the chosen collection.')
        else:
            source=next(iter(available_periods))
        years=available_periods[source]
        if combined or years!=view_years:
            st.caption(f'{COLLECTION_LABELS[source]} calculation period: {years[0]}–{years[1]}')
    if spans_change:
        for code in (HISTORICAL_COLLECTION,MODERN_COLLECTION):
            if code not in available_periods and code not in {item[1] for item in issues}:
                st.caption(f'{COLLECTION_LABELS[code]} collection: no rates for this selection.')
    source_meta=next(m for m in sources if m['dataset']==source)
    settings={'country':country,'cause':cause,'age':age,'sex':sex,'years':list(years),'dataset':source,
              'mortality_sha256':source_meta['sha256']}
    if combined:settings['display_years']=list(view_years)
    if country=='EU27_2020' and source==MODERN_COLLECTION and years[0]<2013:
        st.caption('Inspect early official EU aggregate values with care. Official reference comparisons use 2013 onward.')
    cohort=view_cohort[(view_cohort.dataset==source)&view_cohort.Year.between(*years)]
    selected=observed[(observed.dataset==source)&observed.Year.between(*years)]
    return DashboardSelection(data,cohort,observed,selected,settings,available,country_labels,view_years,
                              combined,spans_change,refresh_after),sources


def show_timeline(selection, trend):
    s=selection.settings
    with st.container(border=True,key='mortality_overview'):
        st.subheader(selection.title,anchor=False)
        st.caption(selection.subtitle)
        raw=selection.data
        extra=raw[(raw.Country==s['country'])&(raw.Cause==s['cause'])&(raw.Age==s['age'])&
                  raw.Year.between(*selection.view_years)&raw.Sex.ne(s['sex'])]
        comparison=[]
        with st.expander('Compare sex groups'):
            options=[code for code in SEX_LABELS if code in set(extra.loc[extra.Rate.notna(),'Sex'])]
            comparison=st.multiselect('Also show on the chart',options,format_func=SEX_LABELS.get,
                                      key='plot_comparison_sexes',persist_state='page')
            st.caption(f"Calculations use the selected group: {SEX_LABELS[s['sex']] }.")
        observed=pd.concat([selection.observed,extra[extra.Sex.isin(comparison)]],ignore_index=True)
        observed=observed.sort_values(['dataset','Sex','Year'])
        fig=go.Figure()
        for (code,sex),g in observed.groupby(['dataset','Sex'],sort=False):
            title=COLLECTION_LABELS[code] if selection.combined else 'Observed'
            if comparison:title+=f' · {SEX_LABELS[sex]}'
            colour=(AMBER if code==HISTORICAL_COLLECTION else INK) if sex==s['sex'] else {'T':TEAL,'F':PURPLE,'M':TEAL}[sex]
            hover=g[['Flag','OriginalCause','dataset']].to_numpy()
            fig.add_trace(go.Scatter(x=g.Year,y=g.Rate,name=title,mode='lines+markers',connectgaps=False,
                line=dict(color=colour,width=2.6,dash='solid' if sex==s['sex'] else 'dot'),marker=dict(size=6),
                customdata=hover,hovertemplate='%{y:.2f}<br>Flag: %{customdata[0]}<br>%{customdata[2]} · %{customdata[1]}'))
        if trend:
            model_data=trend['inputs']['series']
            for i,row in enumerate(trend['tables']['segments'].itertuples(index=False),1):
                if not np.isfinite(getattr(row,'log_slope',np.nan)):continue
                seg=model_data[model_data.Year.between(row.start_year,row.end_year)]
                fit=np.exp(np.log(seg.Rate).mean()+row.log_slope*(seg.Year-seg.Year.mean()))
                fig.add_trace(go.Scatter(x=seg.Year,y=fit,name=f'Trend segment {i}',mode='lines',
                                         line=dict(color=TEAL,width=3.2,dash='dash'),hovertemplate='%{y:.2f}'))
            detail=trend['tables']['anomaly_detail'];marked=detail[detail.PotentialAnomaly]
            if not marked.empty:
                fig.add_trace(go.Scatter(x=marked.Year,y=marked.Rate,name='Potential outlier',mode='markers',
                    marker=dict(color=AMBER,size=10,symbol='diamond',line=dict(width=1,color='white'))))
        if selection.spans_change:
            fig.add_vline(x=COLLECTION_BREAK-.5,line_dash='dot',line_color='#72808b',
                          annotation_text='2011 collection change',annotation_position='top left')
        fig.update_layout(xaxis_title='Year',yaxis_title=RATE_LABEL,hovermode='x unified')
        plot_chart(fig,height=370,alt='Observed mortality rates by year and Eurostat collection',key='observed_rates')
        if selection.spans_change:
            st.caption('Eurostat changed the age standard and some cause groupings in 2011. The chart keeps the collections separate.')
        with st.expander('Values and source flags'):
            if observed.Flag.astype(str).str.len().gt(0).any():
                st.caption('Eurostat flags mark observations that require care when interpreting trends.')
            chart=observed.assign(Collection=observed.dataset.map(COLLECTION_LABELS),SexLabel=observed.Sex.map(SEX_LABELS))
            st.dataframe(chart[['Year','Rate','Collection','SexLabel','Flag','OriginalCause','dataset']],hide_index=True,width='stretch',
                         column_config={'SexLabel':'Sex','dataset':'Eurostat dataset','OriginalCause':'Cause code',
                                        'Rate':st.column_config.NumberColumn('Rate',format='%.2f')})
            values=observed[['Country','Year','Rate','Sex','Age','OriginalCause','Flag','dataset']]
            st.download_button('Download selected rates (CSV)',values.to_csv(index=False),
                               file_name='eurostat_mortality.csv',mime='text/csv')


def forecast_model_name(model):
    return 'Naive persistence (flat baseline)' if model=='Naive' else model

def show_forecast(observed,tables,primary):
    frames=[tables[k] for k in ('forecasts','comparison_forecasts') if k in tables and not tables[k].empty]
    if not frames:return
    observed=observed.sort_values('Year')
    last_year=int(observed.Year.iloc[-1]);last_rate=float(observed.Rate.iloc[-1])
    prediction=pd.concat(frames,ignore_index=True)
    view=st.radio('Forecast chart',['History and forecast','Forecast detail'],horizontal=True,key='forecast_chart_view',
                  help='Forecast detail zooms in on the annual point forecasts. The table always shows their intervals.')
    close_up=view=='Forecast detail'
    history=observed.tail(1) if close_up else observed
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=history.Year,y=history.Rate,name='Observed',mode='lines+markers',line=dict(color=INK,width=2.5),
                             hovertemplate='%{y:.3f}'))
    if not close_up:
        band=prediction[prediction.Model.eq(primary)].sort_values('Year')
        if band.empty:band=frames[0].sort_values('Year')
        band_name=f'95% interval · {band.Model.iloc[0]}'
        if len(band)==1:
            fig.add_trace(go.Scatter(x=[int(band.Year.iloc[0])]*2,y=[band.Lower.iloc[0],band.Upper.iloc[0]],
                mode='lines',name=band_name,line=dict(color='rgba(17,125,119,.35)',width=5),hoverinfo='skip'))
        else:
            fig.add_trace(go.Scatter(x=band.Year,y=band.Upper,showlegend=False,line=dict(width=0),hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=band.Year,y=band.Lower,name=band_name,fill='tonexty',
                fillcolor='rgba(17,125,119,.16)',line=dict(width=0),hoverinfo='skip'))
    for model,part in prediction.groupby('Model',sort=False):
        part=part.sort_values('Year');baseline=model=='Naive'
        fig.add_trace(go.Scatter(x=part.Year,y=part.Forecast,name=forecast_model_name(model),mode='lines+markers',
            line=dict(color='#667785' if baseline else TEAL,dash='dot' if baseline else 'dash',width=2.5),
            marker=dict(symbol='diamond-open' if baseline else 'circle',size=8),
            customdata=part[['Lower','Upper']].to_numpy(),
            hovertemplate='%{y:.3f}<br>95% interval: %{customdata[0]:.3f} to %{customdata[1]:.3f}'))
    fig.add_vline(x=last_year+.5,line_dash='dot',line_color='#72808b')
    if close_up:fig.update_xaxes(dtick=1,tickformat='d')
    fig.update_layout(xaxis_title='Year',yaxis_title=RATE_LABEL,hovermode='x unified')
    plot_chart(fig,alt='Observed rate and annual forecast points by model',key='forecast_chart')
    st.caption(f'Last observed: {last_rate:.3f} in {last_year}. Rates are deaths per 100,000. '
               + ('Prediction intervals are listed below.' if close_up else f'The chart shows the {band.Model.iloc[0]} interval. All intervals are listed below.'))
    display=prediction[['Year','Model','Forecast','Lower','Upper']].copy()
    change=display.Forecast-last_rate
    display['Change from last observed']=change.mask(change.abs()<1e-12,0.0)
    display['Model']=display.Model.map(forecast_model_name)
    display=display.rename(columns={'Lower':'95% lower','Upper':'95% upper'}).sort_values('Year',kind='stable')
    with st.expander('Forecast values',expanded=True):
        st.dataframe(display,hide_index=True,width='stretch',column_config={
            name:st.column_config.NumberColumn(format='%.3f')
            for name in ['Forecast','95% lower','95% upper','Change from last observed']})


def panel_causes(selection,key):
    return st.multiselect('Causes',selection.available,default=[selection.settings['cause']],
        format_func=lambda c:CAUSES.get(c,c),key=key,max_selections=4,persist_state='page')


def panel_inputs(selection,causes):
    s=selection.settings;data=[]
    if not causes:raise ValueError('Select at least one cause.')
    for cause in causes:
        try:
            if cause==s['cause']:
                frame=selection.data[selection.data.dataset.eq(s['dataset'])]
            else:
                frame,meta=live_inputs(cause,s['dataset'],selection.refresh_after)
                st.session_state.snapshots[meta['sha256']]=meta
        except (OSError,RuntimeError,ValueError) as exc:
            raise ValueError(f'{CAUSES.get(cause,cause)}: {exc}') from exc
        population=frame[frame.Cause.eq(cause)&frame.Sex.eq(s['sex'])&frame.Age.eq(s['age'])]
        if not population.Rate.notna().any():raise ValueError(f'No matching rates for {CAUSES.get(cause,cause)}.')
        data.append(population)
    return pd.concat(data,ignore_index=True)


def render_mortality_panel(selection):
    with st.expander('EU and national trends across causes'):
        s=selection.settings
        st.caption('Compare the published EU aggregate with national trends and endpoint changes for the current age, sex and years.')
        causes=panel_causes(selection,'mortality_panel_causes')
        can_sum={'G30','F01_F03'}.issubset(causes)
        combine=st.checkbox('Show the combined Alzheimer and F01/F03 dementia rate',disabled=not can_sum,key='combined_dementia') and can_sum
        op=s|{'causes':causes,'combine_dementia':combine}
        if st.button('Calculate EU and national changes',disabled=not causes):
            try:
                with st.spinner('Loading the selected causes…'):
                    data=panel_inputs(selection,causes)
                    data=data[data.Country.isin((*EU27,'EU27_2020'))&data.Year.between(*s['years'])]
                    tables=mortality_panel(data,op)
                remember('mortality_panel',op,tables,{'mortality':data})
            except ValueError as exc:st.warning(str(exc))
        item=current_result('mortality_panel',op)
        if item is None:return
        tables=item['tables'];a=tables['annual']
        view=st.selectbox('Series to plot',['Official EU aggregate','National median','Show both'],key='mortality_panel_series')
        fields={'Official EU aggregate':'OfficialEU','National median':'Median'}
        cols=[fields[view]] if view in fields else list(fields.values())
        labels={v:k for k,v in fields.items()}
        shown=a[a[cols].notna().any(axis=1)]
        if shown.empty:
            st.info('This collection has no values for the selected series. Choose another series to plot.')
        else:
            order=shown.Cause.unique().tolist();ncol=min(2,len(order));nrow=(len(order)+ncol-1)//ncol
            titles=[CAUSES.get(c,'Alzheimer + F01/F03 dementia' if c=='G30 + F01_F03' else c) for c in order]
            fig=make_subplots(rows=nrow,cols=ncol,subplot_titles=titles,vertical_spacing=min(.18,.6/nrow))
            for i,cause in enumerate(order):
                g=shown[shown.Cause.eq(cause)].sort_values('Year')
                for field in cols:
                    fig.add_trace(go.Scatter(x=g.Year,y=g[field],name=labels[field],mode='lines+markers',connectgaps=False,
                        showlegend=i==0,legendgroup=field,line=dict(color=INK if field=='OfficialEU' else TEAL)),row=i//ncol+1,col=i%ncol+1)
            fig.update_yaxes(title_text='Deaths per 100,000');plot_chart(fig,height=max(320,nrow*250),alt='EU aggregate and national median mortality trajectories',key='eu_cause_trends')
        st.caption('National medians are unweighted. The combined line sums two disjoint underlying-cause groups to examine attribution.')
        st.markdown('**Official EU endpoint changes**')
        st.dataframe(tables['official_endpoints'][['Cause','StartRate','EndRate','PercentChange']],hide_index=True,width='stretch')
        summary=tables['country_summary'];st.markdown('**National direction counts and median change**')
        st.dataframe(summary[summary.Sensitivity.eq('All eligible')].drop(columns='Sensitivity'),hide_index=True,width='stretch')
        detail=tables['country_endpoints']
        if not detail.empty:
            selected=st.selectbox('Cause for country changes',list(detail.Cause.unique()),format_func=lambda c:CAUSES.get(c,c),key='country_change_cause')
            c=detail[detail.Cause.eq(selected)].sort_values('AbsoluteChange')
            plot_chart(px.bar(c,x='Country',y='AbsoluteChange',color='Direction',color_discrete_map={'Decreased':INK,'Increased':AMBER,'Unchanged':TEAL},
                hover_data=['StartRate','EndRate','PercentChange','StartFlag','EndFlag'],labels={'AbsoluteChange':'Rate change per 100,000'}),
                alt='National endpoint changes',key='national_changes')
        with st.expander('Country values, source flags and coverage'):
            for name in ('country_endpoints','country_summary','coverage','flagged'):
                st.dataframe(tables[name],hide_index=True,width='stretch')


def render_forecast_benchmark(selection):
    with st.expander('Forecast accuracy and coverage'):
        s=selection.settings
        st.caption('Test all eight models against held-out years for the EU27 country series. Each origin trains on at least eight annual observations.')
        causes=panel_causes(selection,'rolling_causes')
        op=s|{'causes':causes,'horizons':[1,3],'models':list(MODELS),'training_minimum':8,'bootstrap_seed':20261018,'bootstrap_repetitions':10000}
        short=s['years'][1]-s['years'][0]<8
        if short:st.caption('Select at least nine consecutive years for a one-year backtest and eleven for a three-year backtest.')
        if st.button('Run rolling forecast comparison',disabled=short or not causes):
            meter=st.progress(0,text='Loading the selected mortality series…')
            try:
                data=panel_inputs(selection,causes)
                data=data[data.Country.isin(EU27)&data.Year.between(*s['years'])]
                out=rolling_benchmark(data,op,lambda n,total:meter.progress(n/total,text=f'Fitting and scoring series {n} of {total}'))
                remember('forecast_benchmark',op,out,{'mortality':data})
            except ValueError as exc:st.warning(str(exc))
            finally:meter.empty()
        item=current_result('forecast_benchmark',op)
        if item is None:return
        t=item['tables'];sc=t['scores']
        conditions=t['targets'].Sensitivity.drop_duplicates().tolist()
        sensitivity=st.selectbox('Target and source-flag restrictions',conditions,key='forecast_restriction')
        sub=sc[sc.Sensitivity.eq(sensitivity)].copy();counts=t['targets'][t['targets'].Sensitivity.eq(sensitivity)]
        st.dataframe(counts.drop(columns='Sensitivity'),hide_index=True,width='stretch')
        if sub.empty:st.info('No eligible targets for this restriction.')
        else:
            a,b=st.columns(2);sub['Coverage (%)']=100*sub.Coverage95;sub['Horizon label']=sub.Horizon.map(lambda h:f'{h} year' if h==1 else f'{h} years')
            with a:
                st.markdown('**Scaled forecast error**')
                fig=px.bar(sub,x='MASE',y='Model',color='Horizon label',barmode='group',orientation='h',color_discrete_sequence=[INK,TEAL])
                plot_chart(fig,height=400,alt='Mean absolute scaled forecast error at one and three years',key='rolling_mase')
            with b:
                st.markdown('**Prediction interval coverage**')
                fig=px.bar(sub,x='Coverage (%)',y='Model',color='Horizon label',barmode='group',orientation='h',color_discrete_sequence=[INK,TEAL])
                fig.add_vline(x=95,line_dash='dot',line_color='#72808b');fig.update_xaxes(range=[0,100]);plot_chart(fig,height=400,alt='Observed coverage of nominal 95 percent intervals',key='rolling_coverage')
            st.dataframe(sub.drop(columns=['Sensitivity','Horizon label','Coverage95']),hide_index=True,width='stretch')
        with st.expander('By target year and cause'):
            by=t['by_year'].copy();by['Coverage (%)']=100*by.Coverage95
            measure=st.selectbox('Measure by year',['MASE','Coverage (%)'],key='forecast_year_measure')
            m=st.multiselect('Models shown',list(MODELS),default=['Naive','Damped drift','ARIMA','Ensemble'],key='forecast_year_models')
            if m:
                fig=px.line(by[by.Model.isin(m)],x='Year',y=measure,color='Model',facet_col='Horizon',markers=True)
                fig.for_each_annotation(lambda a:a.update(text=a.text.replace('Horizon=','Horizon: ')+' year(s)'))
                plot_chart(fig,height=320,alt='Forecast performance by target year and horizon',key='rolling_target_year')
            st.dataframe(t['by_cause'][t['by_cause'].Sensitivity.eq(sensitivity)],hide_index=True,width='stretch')
        with st.expander('Paired comparisons and eligibility'):
            st.caption('Country-block resampling keeps years fixed. Simultaneous bands cover the displayed cause × horizon × model family on its shared countries.')
            st.dataframe(t['paired_differences'][t['paired_differences'].Sensitivity.eq(sensitivity)],hide_index=True,width='stretch')
            st.dataframe(t['simultaneous_bands'],hide_index=True,width='stretch')
            st.dataframe(t['eligibility'],hide_index=True,width='stretch')
            if not t['failed_fits'].empty:st.dataframe(t['failed_fits'],hide_index=True,width='stretch')
        st.caption('Errors and coverage use common targets with a positive training error scale. Forecast details and all sensitivity tables are retained in the analysis archive.')


METRIC_NAMES={'IQR':'Interquartile range','P90_P10':'90th / 10th percentile ratio','LogSD':'Equal-country log SD',
              'CV':'Coefficient of variation','LogSD_CurrentPopulation':'Contemporary population log SD',
              'LogSD_FixedStartPopulation':'Fixed population log SD'}


def show_dispersion_details(item):
    t=item['tables'];a=t['dispersion'];s=t['stability'];l=t['leave_one_out']
    logs=[c for c in a if c.startswith('LogSD')]
    x=a.melt(id_vars='Year',value_vars=logs,var_name='Measure',value_name='Log SD')
    labels=METRIC_NAMES|{'LogSD_FixedStartPopulation':f"Fixed {item['settings'].get('fixed_weight_year',item['settings']['years'][0])} population log SD"}
    x['Measure']=x.Measure.map(labels)
    left,right=st.columns(2)
    with left:
        plot_chart(px.line(x,x='Year',y='Log SD',color='Measure',markers=True),height=290,alt='Log-rate dispersion under equal and population weighting',key='annual_log_dispersion')
    with right:
        plot_chart(px.line(a,x='Year',y='CV',markers=True,labels={'CV':'Coefficient of variation'}),height=290,alt='Coefficient of variation by year',key='annual_cv')
    with st.expander('Endpoint stability and country influence'):
        percent=s.dropna(subset=['PercentChange','PercentChangeLow','PercentChangeHigh'])
        if not percent.empty:
            fig=go.Figure()
            for i,r in enumerate(percent.itertuples(index=False)):
                label=labels.get(r.Metric,r.Metric)
                fig.add_trace(go.Scatter(x=[r.PercentChangeLow,r.PercentChangeHigh],y=[label,label],mode='lines',
                    name='95% stability range',showlegend=i==0,legendgroup='range',line=dict(color=TEAL,width=3)))
                fig.add_trace(go.Scatter(x=[r.PercentChange],y=[label],mode='markers',name='Endpoint change',
                    showlegend=i==0,legendgroup='point',marker=dict(color=INK,size=9,symbol='diamond')))
            fig.add_vline(x=0,line_dash='dot',line_color='#72808b');fig.update_layout(xaxis_title='Endpoint change (%)')
            plot_chart(fig,height=315,alt='Endpoint percentage changes with paired country bootstrap stability ranges',key='dispersion_percent_change')
        metric=st.selectbox('Dispersion measure',s.Metric.tolist(),format_func=lambda v:labels.get(v,v),key='dispersion_metric')
        row=s[s.Metric.eq(metric)].iloc[0];p=l[l.Metric.eq(metric)].sort_values('Difference')
        if pd.notna(row.StabilityLow):
            fig=go.Figure([go.Scatter(x=[row.StabilityLow,row.StabilityHigh],y=['Country resampling']*2,mode='lines+markers',name='95% stability range',line=dict(color=TEAL,width=4)),
                go.Scatter(x=[row.Difference],y=['Country resampling'],mode='markers',name='Endpoint difference',marker=dict(color=INK,size=12,symbol='diamond'))])
            fig.add_vline(x=0,line_dash='dot',line_color='#72808b');fig.update_layout(xaxis_title=labels.get(metric,metric)+' change')
            plot_chart(fig,height=205,alt='Paired bootstrap change and stability range',key='dispersion_stability')
        if not p.empty:
            pview=st.radio('Country influence plot',['Distribution','By omitted country'],horizontal=True,key='influence_view')
            fig=(px.histogram(p,x='Difference',nbins=9,labels={'Difference':'Endpoint difference'}) if pview=='Distribution' else
                 px.bar(p,x='OmittedCountry',y='Difference',labels={'Difference':'Endpoint difference','OmittedCountry':'Omitted country'}))
            plot_chart(fig,height=290,alt='Endpoint changes leaving out one country at a time',key='dispersion_influence')
        st.dataframe(s,hide_index=True,width='stretch');st.dataframe(p,hide_index=True,width='stretch')


def population_weight_years(selection):
    s=selection.settings
    if s['dataset']!=MODERN_COLLECTION or s['age']!='TOTAL' or s['sex']!='T':
        return []
    data=selection.data
    return sorted(int(y) for y in data.loc[
        data.dataset.eq(MODERN_COLLECTION)&data.Country.isin(EU27)&data.Rate.notna(),'Year'].unique())


def retrieve_population_weights(years,refresh_after):
    try:
        with st.spinner('Loading Eurostat population weights…'):
            population,meta=load_population(CACHE,(years[0],years[-1]),refresh_after=refresh_after)
    except (ValueError,OSError,RuntimeError):
        st.warning('Eurostat population weights are unavailable. Showing equal-country measures.')
        return None
    st.session_state.snapshots[meta['sha256']]=meta
    return population


def render_dispersion_sensitivity(selection):
    with st.expander('Compare periods and source flags'):
        s=selection.settings;first,last=s['years']
        data=selection.data[selection.data.dataset.eq(s['dataset'])&selection.data.Sex.eq(s['sex'])&
            selection.data.Age.eq(s['age'])&selection.data.Country.isin(EU27)][['Country','Year','Rate','Flag']]
        choices=sorted(int(y) for y in data.Year.unique() if y<last)
        starts=st.multiselect('Additional start years',choices,key='dispersion_starts')
        end=st.selectbox('Additional earlier end year',[None]+[y for y in choices if y>first],format_func=lambda y:'None' if y is None else str(y),key='dispersion_earlier_end')
        flags=st.checkbox('Also exclude countries with b, d or u flags in the selected period',value=True,key='dispersion_exclude_flags')
        periods=list(dict.fromkeys([(first,last),*[(y,last) for y in starts],*([(first,end)] if end else [])]))
        years=population_weight_years(selection)
        base=st.selectbox('Fixed population weight year',years,index=years.index(first) if first in years else 0,key='dispersion_fixed_weights') if years else first
        op=s|{'periods':[list(pair) for pair in periods],'exclude_flags':flags,'fixed_weight_year':int(base),'countries':list(EU27)}
        if st.button('Run dispersion sensitivities',disabled=first>=last):
            pop=retrieve_population_weights(years,selection.refresh_after) if years else None
            tables=dispersion_sensitivities(data,op,pop)
            remember('dispersion_sensitivity',op,tables,{'mortality':data}|({'population':pop} if pop is not None else {}))
        item=current_result('dispersion_sensitivity',op)
        if item:
            t=item['tables'];table=t['sensitivity']
            if not table.empty:
                if years:
                    specs=['StartYear','EndYear','ExcludedFlags']
                    weighted=table.Metric.eq('LogSD_FixedStartPopulation')
                    if len(table.loc[weighted,specs].drop_duplicates())<len(table[specs].drop_duplicates()):
                        st.caption('Population-weighted estimates require complete weights for the countries and years in each comparison.')
                metric=st.selectbox('Measure to compare',list(table.Metric.unique()),format_func=METRIC_NAMES.get,key='sensitivity_metric')
                sub=table[table.Metric.eq(metric)].copy();sub['Period']=sub.apply(lambda r:f"{r.StartYear}–{r.EndYear}"+(' · flags excluded' if r.ExcludedFlags else ''),axis=1)
                fig=px.bar(sub,x='Period',y='Difference',labels={'Difference':METRIC_NAMES[metric]+' change'})
                plot_chart(fig,height=300,alt='Dispersion changes under alternative periods and source flags',key='dispersion_sensitivity_plot')
                st.dataframe(table,hide_index=True,width='stretch')
            if not t['ineligible'].empty:st.dataframe(t['ineligible'],hide_index=True,width='stretch')


def render_gdp_panel(selection):
    with st.expander('GDP associations across causes'):
        s=selection.settings;first,last=s['years'];causes=panel_causes(selection,'gdp_panel_causes')
        end=st.selectbox('Additional earlier end year',[None]+list(range(first+2,last)),format_func=lambda y:'None' if y is None else str(y),key='gdp_earlier_end')
        op=s|{'causes':causes,'earlier_end':end}
        if st.button('Compare GDP associations',disabled=not causes):
            try:
                data=panel_inputs(selection,causes)
                gdp,meta=contextual_input('GDP per capita',s['sex'],tuple(EU27),selection.refresh_after)
                st.session_state.snapshots[meta['sha256']]=meta
                remember('gdp_sensitivity',op,gdp_sensitivity(data,gdp,op),{'mortality':data,'gdp':gdp})
            except (ValueError,OSError,RuntimeError) as exc:st.warning(str(exc))
        item=current_result('gdp_sensitivity',op)
        if item:
            t=item['tables'];a=t['associations']
            if not a.empty:
                a=a.assign(Outcome=a.Cause.map(lambda c:CAUSES.get(c,c)),Upper=a.CI_high-a.Coefficient,Lower=a.Coefficient-a.CI_low)
                fig=px.scatter(a,x='Coefficient',y='Outcome',color='Specification',error_x='Upper',error_x_minus='Lower',hover_data=['p','q','n'],labels={'Coefficient':'GDP elasticity (95% interval)'})
                fig.add_vline(x=0,line_dash='dot',line_color='#72808b');plot_chart(fig,alt='GDP associations by cause and specification',key='gdp_panel_plot')
                st.dataframe(t['associations'],hide_index=True,width='stretch')
            if not t['ineligible'].empty:st.dataframe(t['ineligible'],hide_index=True,width='stretch')


def render_moran_panel(selection):
    with st.expander('Spatial association across causes'):
        s=selection.settings;causes=panel_causes(selection,'moran_panel_causes')
        year=st.selectbox('Map year',sorted(selection.cohort.Year.unique(),reverse=True),key='moran_panel_year')
        op=s|{'causes':causes,'map_year':int(year),'permutations':9999}
        if st.button('Compare Moran’s I across causes',disabled=not causes):
            try:
                data=panel_inputs(selection,causes)
                data=data[data.Year.eq(year)&data.Country.isin(EU27)]
                remember('moran_panel',op,compare_moran(data,op),{'mortality':data})
            except ValueError as exc:st.warning(str(exc))
        item=current_result('moran_panel',op)
        if item:
            t=item['tables'];table=t['associations']
            if not table.empty:
                fig=px.bar(table.assign(Outcome=table.Cause.map(lambda c:CAUSES.get(c,c))),x='Outcome',y='Moran_I',hover_data=['p','q','n_countries'])
                plot_chart(fig,alt='Moran’s I for each selected cause',key='moran_panel_plot')
                st.dataframe(table,hide_index=True,width='stretch')
            if not t['ineligible'].empty:st.dataframe(t['ineligible'],hide_index=True,width='stretch')


def render_trends(selection):
    s=selection.settings;selected=selection.selected
    with st.sidebar:
        st.divider()
        st.header('🔮 Forecast settings',anchor=False)
        method=st.selectbox('Forecast model',list(MODELS),key='forecast_model',persist_state='page',format_func=forecast_model_name,
                            help='Persistence repeats the most recent rate. Other models can project change based on the selected history.')
        horizon=st.slider('Forecast horizon in years',1,3,3,key='forecast_horizon',persist_state='page')
    # A placeholder keeps the chart above the controls while letting a click
    # add the newly calculated trend to the chart on the same rerun.
    overview=st.container()
    with st.container(border=True,key='trend_segments'):
        st.subheader('Trend segments and outliers',anchor=False)
        with st.expander('Trend settings'):
            penalty=st.slider('Segmentation penalty multiplier',1.0,8.0,3.0,.5,key='trend_penalty',persist_state='page')
        op=s|{'penalty_multiplier':penalty}
        if st.button('Calculate trend segments'):
            try:
                series=selected[['Year','Rate']]
                segments,detail=temporal_summary(series,penalty)
                remember('temporal',op,{'segments':segments,'anomaly_detail':detail},{'series':series})
            except ValueError as exc:st.warning(str(exc))
        trend=current_result('temporal',op)
        if trend:
            cols=['start_year','end_year','n','mean_rate','APC','APC_low','APC_high']
            labels=['Start','End','Years','Mean rate','Annual change (%)','Lower 95%','Upper 95%']
            st.dataframe(trend['tables']['segments'][cols].set_axis(labels,axis=1),hide_index=True,width='stretch')
            st.caption('Intervals describe the fitted trend within each selected segment.')
        else:
            st.caption('Find changes in the trajectory and flag unusual annual values.')
    with overview:
        finite=selected[selected.Rate.notna()].sort_values('Year')
        first,last=finite.iloc[0],finite.iloc[-1]
        change=100*(last.Rate/first.Rate-1) if len(finite)>1 and first.Rate>0 else None
        a,b,c=st.columns(3)
        a.metric(f'Observed rate · {int(last.Year)}',f'{last.Rate:.1f}',border=True,help=RATE_LABEL)
        b.metric(f'Change · {int(first.Year)}–{int(last.Year)}',f'{change:+.1f}%' if change is not None else '—',border=True,
                 help='First to last observed rate within the calculation period.')
        c.metric('Years with data',f'{len(finite)} of {int(last.Year-first.Year+1)}',border=True,
                 help=f"Reporting years in the {COLLECTION_LABELS[s['dataset']].lower()} calculation period.")
        show_timeline(selection,trend)
    with st.container(border=True,key='forecast_panel'):
        st.subheader('Forecast',anchor=False)
        period=s['years']
        st.caption(f"{forecast_model_name(method)} · {horizon}-year horizon · {COLLECTION_LABELS[s['dataset']]} series, {period[0]}–{period[1]}")
        if method=='Naive':st.caption('Persistence repeats the last observed rate at every horizon. Its point forecast is flat by design.')
        if selection.spans_change and s['dataset']==HISTORICAL_COLLECTION:
            st.caption('Projections from the historical series use the historical definition.')
        other='Damped drift' if method=='Naive' else 'Naive'
        compare=st.checkbox('Compare with '+forecast_model_name(other),value=True,key='forecast_compare',
                            help='Both models use the same observed years and forecast the same future years.')
        op=s|{'model':method,'horizon':horizon,'comparison_model':other if compare else None}
        if st.button('Generate forecast',type='primary'):
            try:
                series=selected[['Year','Rate']]
                with st.spinner('Fitting the selected forecast…'):
                    tables=forecast_analysis_tables(series,op)
                remember('forecast',op,tables,{'series':series})
            except ValueError as exc:st.warning(str(exc))
        result=current_result('forecast',op)
        if result:
            if not result['tables']['failures'].empty:st.warning(result['tables']['failures'].to_string(index=False))
            show_forecast(result['inputs']['series'],result['tables'],method)
    render_forecast_benchmark(selection)
    if s['country']!='EU27_2020':render_benchmark(selection)


def render_benchmark(selection):
    s=selection.settings;country=s['country'];cohort=selection.cohort;years=s['years']
    with st.container(border=True,key='eu_benchmark'):
        st.subheader('Compare with Europe',anchor=False)
        options=['Unweighted fixed EU27 panel']
        if years[1]>=2013 and cohort.loc[(cohort.Country=='EU27_2020')&(cohort.Year>=2013),'Rate'].notna().any():
            options.insert(0,'Official EU aggregate')
        reference=st.selectbox('Reference population',options,key='eu_reference',persist_state='page')
        interval=[max(2013,years[0]),years[1]] if reference=='Official EU aggregate' else list(years)
        op=s|{'years':interval,'reference':reference}
        if st.button('Calculate EU comparison'):
            try:
                source_data=selection.data[selection.data.dataset==s['dataset']]
                table=benchmark(source_data,country,s['cause'],s['sex'],s['age'],interval,reference)
                if not table[['CountryRate','ReferenceRate']].dropna().shape[0]:
                    raise ValueError('No matching reference values are available.')
                remember('benchmark',op,{'benchmark':table},{'mortality':source_data})
            except ValueError as exc:st.warning(str(exc))
        item=current_result('benchmark',op)
        if item:
            table=item['tables']['benchmark']
            paired=table.dropna(subset=['CountryRate','ReferenceRate'])
            country_name=selection.country_labels.get(country,country)
            a,b,c=st.columns(3)
            a.metric('Mean country rate',f'{paired.CountryRate.mean():.1f}',border=True)
            b.metric('Mean reference rate',f'{paired.ReferenceRate.mean():.1f}',border=True)
            gap=paired.GapPercent.mean()
            c.metric('Mean gap',f'{gap:+.1f}%' if np.isfinite(gap) else '—',border=True)
            fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.13,row_heights=[.62,.38])
            for col,label,colour,dash in [('CountryRate',country_name,INK,'solid'),('ReferenceRate',reference,TEAL,'dash')]:
                fig.add_trace(go.Scatter(x=table.Year,y=table[col],name=label,mode='lines+markers',
                                         line=dict(color=colour,width=2.5,dash=dash)),row=1,col=1)
            fig.add_trace(go.Bar(x=table.Year,y=table.GapPercent,marker_color=np.where(table.GapPercent>0,AMBER,INK),
                                 name='Gap',showlegend=False),row=2,col=1)
            fig.add_hline(y=0,line_color='#72808b',line_dash='dot',row=2,col=1)
            fig.update_yaxes(title_text='Rate per 100,000',row=1,col=1)
            fig.update_yaxes(title_text='Gap (%)',row=2,col=1)
            fig.update_xaxes(title_text='Year',row=2,col=1)
            plot_chart(fig,height=480,alt='Country and EU reference rates and the annual percentage gap',key='benchmark_chart')
            with st.expander('Reference years and countries'):
                st.dataframe(table[['Year','CountryRate','ReferenceRate','GapPercent','N_countries']],hide_index=True,width='stretch')
                st.caption('Reference countries: '+table.ReferenceCountries.iloc[0].replace(';',', '))
                if table.ExcludedCountries.iloc[0]:st.caption('Excluded from the fixed reference: '+table.ExcludedCountries.iloc[0].replace(';',', '))


def render_comparison(selection):
    s=selection.settings
    with st.container(border=True,key='compare_panel'):
        st.subheader('Compare two causes of death',anchor=False)
        st.caption(selection.title+' · '+selection.subtitle)
        choices=[c for c in selection.available if c!=s['cause']]
        if not choices:
            st.info('Choose a mortality collection with another cause to compare.')
            return
        left,right=st.columns(2)
        left.selectbox('First cause', [s['cause']],format_func=lambda c:CAUSES.get(c,c),disabled=True)
        second=right.selectbox('Second cause',choices,index=choices.index('G20') if 'G20' in choices else 0,
                               format_func=lambda c:CAUSES.get(c,c),key='second_cause',persist_state='page')
        op=s|{'second_cause':second}
        if st.button('Compare causes',type='primary'):
            try:
                other,meta=live_inputs(second,s['dataset'],selection.refresh_after)
                st.session_state.snapshots[meta['sha256']]=meta
                b=other[(other.Country==s['country'])&(other.Cause==second)&(other.Sex==s['sex'])&
                        (other.Age==s['age'])&other.Year.between(*s['years'])&other.dataset.eq(s['dataset'])][['Year','Rate']]
                a=selection.selected[['Year','Rate']]
                table,info=compare_causes(a,b)
                remember('comparison',op,{'comparison':table},{'cause_A':a,'cause_B':b},info)
            except Exception as exc:st.warning(str(exc))
        item=current_result('comparison',op)
        if item:
            table=item['tables']['comparison'];info=item['metadata']
            c1,c2=st.columns(2)
            c1.metric('Shared years',info['n_years'],border=True)
            r=info['annual_change_correlation']
            c2.metric('Correlation of annual changes',f'{r:.2f}' if np.isfinite(r) else '—',border=True)
            labels={'Index_A':CAUSES.get(s['cause'],s['cause']),'Index_B':CAUSES.get(second,second)}
            chart=table[['Year','Index_A','Index_B']].melt(id_vars='Year',var_name='Cause',value_name='Index')
            chart['Cause']=chart.Cause.map(labels)
            fig=px.line(chart,x='Year',y='Index',color='Cause',markers=True,color_discrete_sequence=[INK,TEAL],
                        labels={'Index':f'Index ({int(table.Year.min())} = 100)'})
            fig.update_layout(hovermode='x unified')
            plot_chart(fig,alt='Two mortality trajectories indexed to their first shared year',key='comparison_chart')
            st.caption('The correlation describes co-movement; it does not establish a causal link between causes.')
            with st.expander('Indexed rates and original values'):
                st.dataframe(table,hide_index=True,width='stretch')
        else:
            st.caption('Both causes are indexed to 100 in their first shared year, so you can compare how they change.')

    render_mortality_panel(selection)


def render_context(selection):
    s=selection.settings
    with st.container(border=True,key='context_panel'):
        st.subheader('Health and economic factors',anchor=False)
        st.caption(selection.title+' · '+selection.subtitle)
        factors=st.multiselect('Indicators',list(FACTOR_SPECS),default=['GDP per capita'],key='factor_indicators',persist_state='page')
        eligible=sorted(set(selection.cohort.Country)&set(EU27))
        with st.expander('Model settings and indicator definitions'):
            chosen=st.multiselect('Countries in the ecological model',eligible,default=eligible,
                                  format_func=lambda c:selection.country_labels.get(c,c),key='factor_countries',persist_state='page')
            fixed=st.checkbox('Country and year fixed effects with country-clustered uncertainty',value=True,
                              key='factor_fixed',persist_state='page')
            log_model=st.checkbox('Log mortality and predictors',value=True,key='factor_log',persist_state='page')
            for name in factors:st.caption(f"{name}: {FACTOR_SPECS[name]['definition']}")
            st.caption('Survey measures cover the population in each indicator definition. Log models require positive values.')
            if not fixed:st.caption('Pooled OLS is descriptive. Its HC3 covariance does not account for repeated observations within countries.')
        op=s|{'predictors':factors,'fixed_effects':fixed,'log_outcome':log_model,
              'log_predictors':factors if log_model else [],'countries':chosen}
        if st.button('Fit ecological association model',type='primary'):
            try:
                if not factors:raise ValueError('Select at least one indicator.')
                panel=selection.cohort[selection.cohort.Country.isin(chosen)][['Country','Year','Rate']].rename(columns={'Rate':'Mortality'})
                definitions=[]
                with st.spinner('Matching indicators by country and year…'):
                    for name in factors:
                        factor,meta=contextual_input(name,s['sex'],tuple(EU27),selection.refresh_after)
                        st.session_state.snapshots[meta['sha256']]=meta
                        definitions.append({k:v for k,v in meta.items() if k in ('definition','selectors','dataset','indicator_name')})
                        panel=panel.merge(factor[['Country','Year',name]],on=['Country','Year'],how='left',validate='one_to_one')
                    co,info,used=fit_panel(panel,factors,fixed_effects=fixed,log_outcome=log_model,
                                           log_predictors=op['log_predictors'])
                remember('regression',op|{'definitions':definitions},{'coefficients':co,'complete_cases':used},{'panel':panel},info)
            except Exception as exc:st.warning(str(exc))
        item=current_result('regression',op)
        if item:
            info=item['metadata'];co=item['tables']['coefficients']
            a,b,c=st.columns(3)
            a.metric('Country-year observations',info['n'],border=True)
            b.metric('Countries',info['n_countries'],border=True)
            c.metric('Years',info['n_years'],border=True)
            fig=go.Figure(go.Scatter(x=co.Coefficient,y=co.Predictor,mode='markers',marker=dict(color=TEAL,size=10),
                error_x=dict(type='data',symmetric=False,array=co.CI_high-co.Coefficient,arrayminus=co.Coefficient-co.CI_low),
                hovertemplate='%{y}: %{x:.3f}<extra></extra>'))
            fig.add_vline(x=0,line_color='#72808b',line_dash='dot')
            fig.update_layout(xaxis_title='Elasticity (95% interval)' if log_model else 'Coefficient (95% interval)')
            plot_chart(fig,height=max(230,48*len(co)+110),alt='Contextual factor coefficients with 95 percent intervals',key='factor_chart')
            st.caption('Country-level associations do not establish individual risk or a causal effect.')
            with st.expander('Coefficients and model details'):
                st.dataframe(co,hide_index=True,width='stretch')
                st.caption(info['covariance'])
        else:
            st.caption('Compare mortality with country-level indicators using matched years and countries.')

    render_gdp_panel(selection)


def render_spatial(selection):
    s=selection.settings;cohort=selection.cohort;years=s['years']
    map_data=cohort[cohort.Country.isin(EU27)].copy()
    map_data['iso3']=map_data.Country.map(alpha3)
    map_data['CountryName']=map_data.Country.map(selection.country_labels).fillna(map_data.Country)
    map_data=map_data.sort_values(['Year','Country'])
    with st.container(border=True,key='map_panel'):
        st.subheader('Mortality across Europe',anchor=False)
        st.caption(f"{CAUSES.get(s['cause'],s['cause'])} · {AGE_LABELS[s['age']]} · {SEX_LABELS[s['sex']]} · {COLLECTION_LABELS[s['dataset']]} collection")
        seen=map_data[map_data.Rate.notna()]
        if not seen.empty:
            plot=px.choropleth(map_data,locations='iso3',color='Rate',hover_name='CountryName',scope='europe',
                animation_frame='Year' if seen.Year.nunique()>1 else None,
                color_continuous_scale='Teal',labels={'Rate':'Per 100,000'})
            if plot.frames:
                # Start at the most recent year. Playback restarts from the first year.
                plot.data[0].update(plot.frames[-1].data[0])
                plot.layout.sliders[0].active=len(plot.frames)-1
                for menu in plot.layout.updatemenus:
                    if menu.buttons and isinstance(menu.buttons[0].args[1],dict):
                        menu.buttons[0].args[1].update(fromcurrent=False)
            plot.update_geos(showframe=False,showcoastlines=False)
            plot_chart(plot,height=470,alt='Map of EU27 mortality rates with time-lapse controls',key='mortality_map')
        else:st.info('No EU27 national rates match this selection.')
    with st.container(border=True,key='spatial_panel'):
        st.subheader('Spatial association',anchor=False)
        st.caption('Moran’s I measures whether neighbouring countries tend to have similar rates.')
        a,b=st.columns(2)
        year=a.selectbox('Year for spatial association',sorted(cohort.Year.unique(),reverse=True),key='spatial_year',persist_state='page')
        permutations=b.selectbox('Spatial permutations',[999,9999],index=1,key='spatial_permutations',persist_state='page')
        op=s|{'map_year':int(year),'permutations':permutations}
        if st.button('Calculate spatial association'):
            try:
                spatial=map_data[map_data.Year==year][['Country','Rate']]
                info,W=moran(spatial.set_index('Country').Rate,permutations)
                tables={'weights':W.reset_index().rename(columns={'index':'Country'}),
                        'moran':pd.DataFrame([{k:v for k,v in info.items() if not isinstance(v,list)}])}
                remember('spatial',op,tables,{'spatial_values':spatial},info)
            except ValueError as exc:st.warning(str(exc))
        item=current_result('spatial',op)
        if item:
            m=item['metadata'];a,b,c=st.columns(3)
            a.metric('Moran’s I',f"{m['Moran_I']:.3f}",border=True)
            b.metric('Permutation p value',f"{m['p']:.3g}",border=True)
            c.metric('Connected countries',m['n_countries'],border=True)
            if m['isolates_excluded']:st.caption('Without a connected neighbour: '+', '.join(m['isolates_excluded']))
    with st.container(border=True,key='dispersion_panel'):
        st.subheader('Geographical dispersion over time',anchor=False)
        st.caption('Track how widely rates vary across the same countries in each year.')
        op=s|{'countries':list(EU27),'bootstrap_repetitions':10000,'bootstrap_seed':20261018}
        if st.button('Calculate fixed-panel dispersion'):
            try:
                inputs=cohort[cohort.Country.isin(EU27)][['Country','Year','Rate']]
                table=fixed_panel_dispersion(inputs,years)
                captured={'mortality':inputs};population=None
                weight_years=population_weight_years(selection)
                if weight_years:
                    candidate=retrieve_population_weights(weight_years,selection.refresh_after)
                    included=table.IncludedCountries.iloc[0].split(';')
                    population=covered_population(candidate,included,years)
                    if population is not None:captured['population']=population
                with st.spinner('Resampling country histories…'):
                    tables=dispersion_tables(inputs,op,population)
                remember('dispersion',op,tables,captured)
            except ValueError as exc:st.warning(str(exc))
        item=current_result('dispersion',op)
        if item:
            table=item['tables']['dispersion'];left,right=st.columns(2)
            with left:
                st.markdown('**Relative spread**')
                plot_chart(px.line(table,x='Year',y='P90_P10',markers=True,color_discrete_sequence=[INK],
                                   labels={'P90_P10':'90th / 10th percentile ratio'}),height=270,
                           alt='Annual ratio of the 90th to the 10th percentile',key='relative_dispersion')
            with right:
                st.markdown('**Absolute spread**')
                plot_chart(px.line(table,x='Year',y='IQR',markers=True,color_discrete_sequence=[TEAL],
                                   labels={'IQR':'Interquartile range per 100,000'}),height=270,
                           alt='Annual interquartile range across the fixed panel',key='absolute_dispersion')
            st.caption(f"{int(table.N_countries.iloc[0])} countries with complete annual coverage.")
            if population_weight_years(selection) and 'population' not in item['inputs']:
                st.caption('Population weights were unavailable or incomplete for this calculation.')
            with st.expander('Rates and endpoint sensitivity'):
                st.dataframe(table[['Year','N_countries','P10','P90','P90_P10','IQR','LogSD','CV']],hide_index=True,width='stretch')
                st.caption('Included: '+table.IncludedCountries.iloc[0].replace(';',', '))
                if table.ExcludedCountries.iloc[0]:st.caption('Incomplete: '+table.ExcludedCountries.iloc[0].replace(';',', '))
                if table.P90_P10.isna().any():st.caption('The percentile ratio is undefined when the 10th percentile is zero.')
                st.dataframe(item['tables']['stability'],hide_index=True,width='stretch')
                st.caption('The country bootstrap resamples whole histories to assess sensitivity to country composition.')
            show_dispersion_details(item)
    render_dispersion_sensitivity(selection)
    with st.container(border=True,key='clusters_panel'):
        st.subheader('Country trajectories',anchor=False)
        representation=st.selectbox('Group countries by',['shape','level'],format_func=lambda m:'Trajectory shape' if m=='shape' else 'Log-rate level',
                                     key='cluster_representation',persist_state='page')
        op=s|{'representation':representation}
        if st.button('Cluster complete trajectories'):
            try:
                panel=map_data.pivot(index='Country',columns='Year',values='Rate').reindex(columns=range(years[0],years[1]+1))
                with st.spinner('Grouping country trajectories…'):
                    membership,scores,info=cluster_trajectories(panel,representation)
                remember('clustering',op,{'clusters':membership,'silhouettes':scores},{'trajectories':panel.reset_index()},info)
            except ValueError as exc:st.warning(str(exc))
        item=current_result('clustering',op)
        if item:
            info=item['metadata'];a,b,c=st.columns(3)
            a.metric('Trajectory groups',info['k'],border=True)
            b.metric('Countries',info['n_countries'],border=True)
            c.metric('Silhouette score',f"{info['silhouette']:.2f}",border=True)
            membership=item['tables']['clusters'].copy();membership['iso3']=membership.Country.map(alpha3)
            membership['Cluster']=membership.Cluster.map(lambda x:f'Group {x}')
            colours=[INK,TEAL,AMBER,PURPLE,'#789443','#ae4b74']
            scale=dict(zip(sorted(membership.Cluster.unique()),colours))
            left,right=st.columns(2)
            with left:
                plot_chart(px.choropleth(membership,locations='iso3',color='Cluster',scope='europe',hover_name='Country',color_discrete_map=scale),
                           height=330,alt='Map of country trajectory groups',key='cluster_map')
            with right:
                panel=item['inputs']['trajectories'].merge(membership[['Country','Cluster']],on='Country',validate='one_to_one')
                means=panel.melt(id_vars=['Country','Cluster'],var_name='Year',value_name='Rate').groupby(['Cluster','Year'],as_index=False).Rate.mean()
                plot_chart(px.line(means,x='Year',y='Rate',color='Cluster',color_discrete_map=scale,labels={'Rate':RATE_LABEL}),
                           height=330,alt='Mean observed mortality trajectory in each group',key='cluster_curves')
            with st.expander('Group membership and sensitivity'):
                st.dataframe(item['tables']['clusters'],hide_index=True,width='stretch')
                st.dataframe(item['tables']['silhouettes'],hide_index=True,width='stretch')
                st.caption('Median agreement after resampling years: '+f"{info['median_year_resampling_ARI']:.2f}")

    render_moran_panel(selection)


def render_network(selection):
    s=selection.settings;cohort=selection.cohort;years=s['years']
    with st.container(border=True,key='network_panel'):
        st.subheader('Temporal prediction between neighbours',anchor=False)
        st.caption(f"{CAUSES.get(s['cause'],s['cause'])} · {AGE_LABELS[s['age']]} · {SEX_LABELS[s['sex']]} · {years[0]}–{years[1]}")
        eligible=sorted(c for c in cohort.Country.unique() if len(c)==2)
        a,b=st.columns([3,1])
        chosen=a.multiselect('Network countries',eligible,default=[c for c in ['DE','FR','BE','NL'] if c in eligible],
                             format_func=lambda c:selection.country_labels.get(c,c),key='network_countries',persist_state='page')
        lag=b.selectbox('Predictive lag in years',[1,2,3],key='network_lag',persist_state='page')
        op=s|{'lag':lag,'countries':chosen}
        if st.button('Run directed prediction tests',type='primary'):
            try:
                panel=cohort[cohort.Country.isin(chosen)].pivot(index='Year',columns='Country',values='Rate').reindex(range(years[0],years[1]+1))
                panel.index.name='Year'
                tests,excluded,diagnostics=predictive_network(panel,lag)
                remember('network',op,{'tests':tests,'excluded':excluded,'diagnostics':diagnostics},{'network_panel':panel.reset_index()})
            except ValueError as exc:st.warning(str(exc))
        item=current_result('network',op)
        if item:
            tests=item['tables']['tests'];excluded=item['tables']['excluded']
            a,b,c=st.columns(3)
            a.metric('Pairs tested',len(tests),border=True)
            b.metric('Pairs excluded',len(excluded),border=True)
            c.metric('Adjusted p < 0.05',int(tests.Selected.sum()) if not tests.empty else 0,border=True)
            if tests.empty:st.info('No neighbouring pair met the model requirements for the selected years and lag.')
            else:
                q=tests.pivot(index='Source',columns='Target',values='q_BY').reindex(index=chosen,columns=chosen)
                fig=px.imshow(q,aspect='auto',color_continuous_scale='Teal_r',zmin=0,zmax=1,
                              labels={'x':'Target country','y':'Source country','color':'Adjusted p'})
                plot_chart(fig,height=340,alt='Adjusted p values for eligible directed pairs',key='network_heatmap')
                with st.expander('Tested pairs'):
                    st.dataframe(tests,hide_index=True,width='stretch')
            if not excluded.empty:
                with st.expander('Why pairs were excluded'):
                    st.dataframe(excluded,hide_index=True,width='stretch')
            with st.expander('Time-series diagnostics'):
                st.dataframe(item['tables']['diagnostics'],hide_index=True,width='stretch')
        st.caption('These tests assess predictive associations. They cannot establish transmission or a causal effect.')


def main():
    st.set_page_config(page_title='European Public Health Dashboard',page_icon='📊',layout='wide',initial_sidebar_state='auto')
    st.html(DASHBOARD_HEADER)
    if 'analyses' not in st.session_state:st.session_state.analyses={}
    if 'snapshots' not in st.session_state:st.session_state.snapshots={}
    selection,sources=read_selection()
    if selection is not None:
        tabs=st.tabs(WORKSPACES,key='analysis_tabs',on_change='rerun')
        for tab,render in zip(tabs,(render_trends,render_comparison,render_context,render_spatial,render_network)):
            if tab.open:
                with tab:render(selection)
    show_analysis_exports()
    with st.expander('Eurostat sources'):
        latest={meta['dataset']:meta for meta in st.session_state.snapshots.values()}
        latest.update({meta['dataset']:meta for meta in sources})
        for code,meta in latest.items():
            label=COLLECTION_LABELS.get(code,meta.get('indicator_name','Population weights' if code=='demo_pjan' else code))
            st.caption(f"{label} · {code} · Source update: {meta.get('updated',meta.get('source_updated','not provided'))} · Retrieved: {meta.get('retrieved_utc','not recorded')}")
        st.markdown('[Eurostat causes-of-death metadata](https://ec.europa.eu/eurostat/cache/metadata/en/hlth_cdeath_sims.htm)')
        st.caption(f'Dashboard version {__version__}')


if __name__ == '__main__':
    if '--archive' in sys.argv[1:]:
        _reproduce_cli()
    else:
        main()
