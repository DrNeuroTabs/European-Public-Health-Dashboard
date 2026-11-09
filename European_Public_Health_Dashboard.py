from __future__ import annotations

import base64
import sys
import tempfile

__version__ = '4.6.1'
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

def fetch(dataset: str, filters: dict, cache_dir: Path|str, offline=False, refresh=False):
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    url=query_url(dataset,filters);request_hash=sha256(url.encode())[:20]
    path=cache_dir/f'{dataset}_{request_hash}.json';meta_path=path.with_suffix('.manifest.json')
    if not path.exists() or refresh:
        if offline:
            raise FileNotFoundError(f'Offline snapshot not found for {dataset}: {path.name}')
        last=None
        for attempt in range(3):
            try:
                with urlopen(Request(url,headers={'User-Agent':f'EuropeanPublicHealthDashboard/{__version__} (research software)'}),timeout=40) as response:
                    content=response.read()
                payload=json.loads(content)
                if _empty_eurostat_response(payload):
                    raise NoEurostatObservations('Eurostat has no values for this selection.')
                if payload.get('warning') or payload.get('error'):
                    raise ValueError(payload.get('warning') or payload.get('error'))
                parse_jsonstat(payload,dataset)
                path.write_bytes(content)
                record={'dataset':dataset,'url':url,'filters':filters,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'sha256':sha256(content),'snapshot_file':path.name,'source_updated':payload.get('updated')}
                meta_path.write_text(json.dumps(record,indent=2,ensure_ascii=False));break
            except HTTPError as exc:
                last=exc
                if exc.code==400:
                    try:
                        error=json.loads(exc.read())
                    except (OSError, ValueError):
                        error={}
                    if _empty_eurostat_response(error):
                        raise NoEurostatObservations('Eurostat has no values for this selection.') from exc
                if exc.code not in (429,500,502,503,504): raise
            except (URLError,TimeoutError) as exc:
                last=exc
            if attempt<2:time.sleep(1+attempt)
        else: raise RuntimeError(f'Eurostat retrieval failed for {dataset}') from last
    content=path.read_bytes();manifest=json.loads(meta_path.read_text())
    if manifest['sha256']!=sha256(content):raise ValueError('Input snapshot checksum mismatch')
    # Keep immutable content-addressed bytes in addition to the mutable query
    # cache. A refresh must not invalidate an earlier completed analysis.
    snapshot=cache_dir/f'{dataset}_{sha256(content)}.json'
    if snapshot.is_file():
        if sha256(snapshot.read_bytes())!=manifest['sha256']:
            raise ValueError('Archived source snapshot checksum mismatch')
    else:
        with NamedTemporaryFile(dir=cache_dir,delete=False) as temporary:
            temporary.write(content)
            temporary_path=Path(temporary.name)
        temporary_path.replace(snapshot)
    if manifest.get('snapshot_file')!=snapshot.name:
        manifest['snapshot_file']=snapshot.name
        meta_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    df,meta=parse_jsonstat(json.loads(content),dataset)
    meta.update(manifest);meta['snapshot_path']=str(snapshot)
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

def load_mortality(causes,cache_dir,source='hlth_cd_asdr2',geographies=None,offline=False,refresh=False):
    requested=[]
    for c in causes:
        requested.extend(['TOTAL','A-R_V-Y'] if c=='TOTAL' else [c])
    filters={'freq':'A','unit':'RT','age':['TOTAL','Y_LT65','Y_GE65'],'sex':['T','F','M'],'icd10':list(dict.fromkeys(requested))}
    if geographies is None:filters['geo']=list(EU27)+['CH','LI','NO','IS','UK','RS','TR','EU27_2020']
    else:filters['geo']=list(geographies)
    df,meta=fetch(source,filters,cache_dir,offline=offline,refresh=refresh)
    data=prepare_mortality(df)
    if source==HISTORICAL_COLLECTION:
        data=data[data.Year<COLLECTION_BREAK]
    elif source==MODERN_COLLECTION:
        data=data[data.Year>=COLLECTION_BREAK]
    if not data.Rate.notna().any():raise NoEurostatObservations('Eurostat has no values for this selection.')
    return data,meta

def load_factor(name,cache_dir,sex='T',geographies=None,offline=False):
    spec=FACTOR_SPECS[name];selectors=dict(spec['filters'])
    if 'sex' in selectors:selectors['sex']=sex
    filters=dict(selectors);filters['geo']=list(geographies or EU27)
    raw,meta=fetch(spec['dataset'],filters,cache_dir,offline=offline)
    selected=select_dimensions(raw,selectors)
    out=selected[['geo','time','value','flag']].rename(columns={'geo':'Country','time':'Year','value':name,'flag':'FactorFlag'})
    out['Year']=out['Year'].astype(int)
    meta.update({'indicator_name':name,'definition':spec['definition'],'selectors':selectors})
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

"""Deterministic Gaussian fits for the ARIMA(1,1,0) and ETS(A,A,N) models."""
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


def dispersion_audit(mortality,years,countries=EU27,population=None,resamples=10000,seed=20261018):
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
        weights=population.pivot(index='Country',columns='Year',values='Population').reindex(index=included,columns=[start,end])
        if weights.isna().any().any() or (weights<=0).any().any():
            raise ValueError('Complete positive population weights are required at both endpoints')
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
            ('LogSD_FixedStartPopulation',weights[start].to_numpy(),weights[start].to_numpy())
        ]:
            point0[name]=weighted(matrix[:,0],w0);point1[name]=weighted(matrix[:,1],w1)
            sample0[name]=weighted(matrix[indices,0],w0[indices]);sample1[name]=weighted(matrix[indices,1],w1[indices])

    rows=[]
    for metric in point0:
        differences=sample1[metric]-sample0[metric]
        finite=differences[np.isfinite(differences)] if np.isfinite(point0[metric][0]) and np.isfinite(point1[metric][0]) else np.array([])
        rows.append({'Metric':metric,'StartYear':start,'EndYear':end,'N_countries':n,
                     'StartValue':point0[metric][0],'EndValue':point1[metric][0],
                     'Difference':point1[metric][0]-point0[metric][0],
                     'StabilityLow':np.quantile(finite,.025) if len(finite) else np.nan,
                     'StabilityHigh':np.quantile(finite,.975) if len(finite) else np.nan,
                     'DefinedBootstrapRepetitions':len(finite),'BootstrapRepetitions':resamples,
                     'Interpretation':'Country-composition stability; not national-rate confidence limits'})
    leave=[]
    for i,country in enumerate(included):
        retained=np.delete(matrix,i,axis=0)
        before=measures(retained[:,0]);after=measures(retained[:,1])
        for metric in before:
            leave.append({'OmittedCountry':country,'Metric':metric,'StartValue':before[metric][0],
                          'EndValue':after[metric][0],'Difference':after[metric][0]-before[metric][0]})
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
        if not path.is_file():raise FileNotFoundError('A source snapshot is unavailable; restore the bundled data or reload the source before exporting: '+path.name)
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


def _execute_saved_operation(kind,settings,inputs):
    if kind=='temporal':
        segments,detail=temporal_summary(inputs['series'],settings['penalty_multiplier'])
        return {'segments':segments,'anomaly_detail':detail}
    if kind=='forecast':
        predictions,failures=forecast(inputs['series'],settings['horizon'],[settings['model']])
        return {'forecasts':predictions,'failures':failures}
    if kind=='comparison':
        table,info=compare_causes(inputs['cause_A'],inputs['cause_B']);return {'comparison':table}
    if kind=='benchmark':
        return {'benchmark':benchmark(inputs['mortality'],settings['country'],settings['cause'],settings['sex'],settings['age'],settings['years'],settings['reference'])}
    if kind=='dispersion':
        stability,influence=dispersion_audit(inputs['mortality'],settings['years'],settings['countries'],inputs.get('population'),
                                           settings.get('bootstrap_repetitions',10000),settings.get('bootstrap_seed',20261018))
        return {'dispersion':fixed_panel_dispersion(inputs['mortality'],settings['years'],settings['countries']),
                'stability':stability,'leave_one_out':influence}
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
                    if col in ('Rate','Mortality') or col in op['settings'].get('predictors',[]):inputs[label][col]=pd.to_numeric(inputs[label][col],errors='coerce')
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


# ============================================================================
# Eurostat query cache and population reference series
# ============================================================================

_BUNDLED_ASSETS_SHA256 = '7fc6226af65a03f5473d725cfc21d6f7f6ad4585f406c68c744f82b8fd708e53'
_BUNDLED_ASSETS_BASE85 = (
    'P)h>@6aWAK2mqI%X<ZiPQ|-|M008C*0012T003lRbYU-WZ*X;NVRUJ4ZeML>bYWy+bYU)Pb8l{qmRoPyNECqI^D7V!yflIFfE|N`R)j!8LkI#|sjVtB*'
    'vH`3*xvCt+icbUzH=j-$!b@Xkk0t?@tKQfKK%2d(P+j@unI(TAe(S3w&gdLH!szljO`#(d|8xr#zb1=4f*pL?^#|m{?gdJG}s;*ERP!%{4fvIr7Q@sdW'
    ';epT#^12^*6LTAVmJ&5lV<jjMF3Jm8@CG+%J+ymT>q}-&KY3RGiwfms9i9k!<dvI5mYV)|Ab!=YgxTKT+A!mAoFQ(T#FBiL9yFFWe_eT&z^)qCW0`yP0'
    'Z4C~@<&8O>GhrVww7suAPV|437_L1{M=m3IT#KCO>-nhM`(TxWa6Q+DR#&x()#hUkFt?SsO%kRKWUj<7C%2mHwLmouJdJgG#H<03PSXB;y5rct(@@Er$'
    ';0MWIg>yoAU8BX;&PC3hQ>b<68UdXM5yVEN)lM94~HvXh@UrJtAXFC6==$wgXTwtyX%u{iwnKP032YsXd0mofY^F11hzH?vaa#>NYkGj$$%LJ;j(DPn^'
    'H~?2Gg2pq_`J=iL&dXWO%_(x%Z@Dp)i+V3N!z*7sDZVO>?%>xmy|7vF3@(2-arPbasm`CKx!dbnzs&Vt5JT>6Kz)YL4y45M%rTbn5>aX?b!gYpu0wm4_'
    '8jV4>N_;BG;rv(rPmG(Ee##ox3rHbC00hnp>}T6nH#lpqt4u@otuni#V+isHIqUfA??oh>!-z4C75W?wl!SMwLK3)l+~cqfv62NFG?muN3~RThm~n3PO'
    'A!zX|5I9Q8=wc8dV*yO*+cYomVseA>ZbHI9arm(}ts#^w3btrzJHr+_a=ch9eVhCMMkJaKF-F{K<sRI@~_!Ffp0anF%KI*1*Z!JpMt4nSnFY!~CNO_R6'
    'F@nyZC*ZfS#ot0x`SrrWjYZ?ZO5lU9{VKWyYSTmnyvc&K(>tIkE5Igpwy)=bh&0ROk=rQaP(TRS^CmKt`v`vo20OXOfIUMwywCM;oWUhGwTEU^}`lCfE'
    '_nXrhlC9wvv{IKt^%CNz(o3N3vg|K3<hOq0gS9J61X_?)3`9>toH*Xc6FLSauNQc5R#M@d#uU``$x$?q|M+bf|=+btt?}w!SPJddMJy9JP^|eUienMFs'
    'vM}rc4f@QZ-H=2;{2=m45+^?KJ(5JR7m{v|g<cTDuIu>$BawWt@7sDgE2~2xwsNz9^NqAk5|->}&x`zPKM2@$*k`@=x2mvTPnG<KiD3KN-$NB-5Q*Y{_'
    'p|*zQQsxMv%}S5Ui|t8P)h>@6aWAK2mqI%X<dI!{LtkO0012#001%o003lRbYU-ZZeeh9Xm50LFJxtHZ(nd~VQyb$W?^PxGBhz`H8(XjHeok7Ibk#|YI'
    'ARHty*nw8_Ci9E5znYf=K4P?Ci`4&OMNA-kmP8WQcT-Ja-3SEv{waD3T$Uwsi;p_f*x8vgBCE3j!P7ZO%+jPj_{759!~k_uYCkTP~`j$~`gFL3K56Hyc'
    'RPwr@9Gk5A_9n{JLzUM*MKdD~<10DtY^hjy`T*SCW=w}bY&8?=k*VAFlXg3WTfzUr`IG@cw!@WFNkZQYc67}p${eCWc&`6FZS_m9RIZv4KTZ?WjN%GI$'
    'tN<IeL)CZ6fq}n9wETv2-*wh$E!9WTo#?(-<lv16YBnK%PQ%9}~o>FwF36yF`$vM|VN`X?qu*Mol*;u3918vFPHASK=Ii~C~CD@WJHi+(^f+f}(V;Y|h'
    'Dk~v|lsJ!q^)W}N@Dh^2i4Llog7eXFBv9}ew6Lj>5G;lcswg4DLTF1A>JWm3kSPS`YHG7GA%r^eR8}@LHa5;fk`b|XF~c_&Rz%<A+QT)rrr;v9K!6nyK'
    '?;G>U`6DrKn<`W8kd{QtzboyeTo4_SXj}(>j(-C3nQ8sVl4qSH1M2_bTdq7LX(}m>?wGs!#Y^d)CrN6k14>Xmis6eJ_8FB7>mXkD<MfpK~@tLjWOQQY9'
    'k@lIs!4#_{i|thLC7-iVRUBL|`JdHFXJk)-;%qjjI`C+cXR^43aijfY|S}!Gc_8BdZxOQ>P}<rH&CZ!GyZzS(7A~;wa@Jxtam=Mv_nRino#=*6_G@k|K'
    'Z#S9l`FG=Ny(%vw^Nli^s9ls*1HDz@Z0dD!gZZue;5YA;*38xN-xF-QZ$tWb);g)vj7Qp5zHlYUf+7yzqoV40#~EyCiVQ(<v7hK8O{T#$V5B4;u%(C9o'
    'PsK|JD*;$M!G#(C)45(t`0bfH8DmWgSTcT3YL24k?G3KEY_%sMUQpiw9JRk&U#DFU!jskG)akko12sv5`P#541N1N(Ep-!=ZL+XYCkH+l8xu6+H;<QA;'
    'xybXAgupZ47P^fVr08o#i4+t_7lb8oWIa-bTtyr~fMb~6QG&7DlsH1-heSX$SpuSfO`|CzxrVlAi_q{q(MibG!<^`)trkvVEg^||qAc)WB{(F?0&hhDQ'
    '5G~{D5%cj3NVq+NTBQrd%<<sDGPVRnZdFw+@*}%;5y+hfzE1Jn}xgRLro`V;SOF7q_8a9A=I9j%*p{AL^uGX3$#I~%tf=yg5Y^<Ym^3cgX|@(8QD`CQ~'
    '+kNfkAac3`I>26k(C0Goh3sK-2^gWTXhDibZph2#$B;4U;9>B%fH)DJMr{6-BF!YE&DlBB9nzwkeC6$fz}PA3)S7t%?9h%rug$3K169Qbns!0)lS{XEG'
    'iV34UU_l1T!Nr_$ufB%wz2QSBg;1Qb6<6Fiv&WRJunS&&HrVC+qzL>wZOz*Ur(3W2Z^k$6cli-5=^Apn;Qy(g2v@hDc(Nm~M$nRb#%kau8X9Wj!pQ46D'
    '!16UDwgq(E5EV2h9%O9!=h+qJYD5@^hK)fo8Dno<9=@oy~h9&`X%xSCY5Si2}P^u5n<x<xz-38~Mnkf$SSHm(Ngbq54nrt-MNR`TAT*ZYht6f52C9vo+'
    ')JK@eZRoNZok-F`x~xX#HYzZ58DIjnff9qO!YzrhHL=i%rOOb@+S7rS<m+>TBB2_bDZ%3^XXG)DM#Hdou_Q8&>gEk~lF2-{FEQsFQJZNqLF)+GoWTUDE'
    'yQdFBpto(2w6-tq8vxW!leOi1SOm(M^Hha7UGpamnR(I3Z{fhrIQ5^eUyuu$qLxOP@cgt>QG^80@*R>xnXj8mV4R~a#pTIrW!amkxL^J4b6;8*exl|3~'
    'hm+B=a%Rs-zT)QF)(~U}Sdeo0MJ|{^tn&pF)VF!bJCH7(k~c0+yM?dX|_wV4y@~C#H@?VzDvF%pCBo&vG`iBcpiMOdZQwfffPvC25rCgDa3`j4$g&V|)'
    '$b(&VVo9CmAy);6qS6p20~5>(G(w2_YmDy4EHmO~<rr8#7<QJo0nAgi2j3S+W3hip(vy+mehf~%dH89ENgLAT1K^c*UHZ-ha*4x~UQQwo%T@`eV|d7uG'
    'gChep5kQ{U@r9=sPMy<<~5Dj|P(t!^B`A)Kw46NZy788u2b>QwSCK$u9X&H-&BaoxINFD+_qE?o*B%q>+A6QC2kj878I1mujw5%GR1!<O%0JPNXxsBXO'
    'XH#&Bi(xi77LW$HSdxk1b-=6?Ft{FE!sKD;07dyM1q>`20R^2U8lM2PL<DOz)YM6VVUfmm=;t)X!yDwk$Vg>6qv2Ewt~$+368Wf5XMk0dr%IiD?d2GiY'
    'LIm<T3AUlTQTjxO3M3*ntbLG;ufNGGMQMP9R23Vxj<9ajf9Mgt~{`{bFPqEts4&67My5<MwnVYCHaj^Tf^~Q0*MWcj(T~rEX$i$m(S!zk4$5oijrk^Qe'
    'a5-U9d_LqQtu9RFTj>k;0u?+Xmc&1gA*PASaMz(%49haGTW7H=uNIi6+xIxS^nVq5Y&-h^Tbzj2t-@@xtMZZXNVbgqunSPnJc*&^fThbL0V7$4eYHA`d'
    '`}rMYqB0r(F96DyJjEc-^s3g#M-IU3ZcTTo+*h;VXp<94G#u4JnV5zFOs_PP*FXsqrP*a+*e4YB9gMe-&I|7tWA5wyk=v6Iwjvf3z1M49EC)Dfx3BJiY'
    'Go>27-+C-(MBUXLGj-DNw6Icy%Q`c^x6+Ic?L{AE?XmxzK;Kp*}z%5h<6kw69p<LO8Snv-}K$`_wL=#skS|!XH+QhcP3A7M>%`9<A%^zSUa(2lV&9(X('
    'mt+;NTdt8Q?5;YE_7$DsvXKpMsZM-m3`>$UCyDqg2$In-l|3BSlG2c?5mzi~R)Y*jnF0oN$cO+8p=Pzn8C_GIv{spoT&qP(YLYWMaz#O1a%SsK&UI>&('
    'afNpXtZQE8ZxIgPFGCuyBa>i8HLHEasyO-Wk}en5zMd%JtiAPVmY%bDl@(M05sNY28raH5gfIR0y1hOue%f#N#w1{ov%wI5pTxX*CmpGan`6JS%YAhZ9'
    'bH{EZPSFIg2JiX<BMTlOZUq-iRhqpBN}_M3W?8Y9&8AUhdTLlAs~6E2k7B5r&i$U|^rMz#GLE6s-o>D7?VFT5NVnj`tSUB+o7h5|Wea*d+l$pb0vFT@t'
    '{Ce5ajO#DwTLjj`(1P@{V_yCkxKFyO7)c2wsg?b>!1v`!_<<32-=Tea=A7bkhAwmlHUl&mDQkGlUmN$9aPCR-2^=C*gT1y0KD(?PZA+kU&@muFB{y{Ys'
    'Kl%C{Ey|AIs3kjx7p<U(8-7TZuYE|u;Wxc7#ETdtsRs~vbs<h=TN=v6TWS469>h4~d3sqO`aKC#?dc!GZ)9UEQ>g~GwJHInr%=qPFd(BDZd&TQ+iGTaq'
    'O;`Pl&t|`Nm~RitA2f;9<KroY7v&qx--<3g_g`U7FgW0c+ir1n%Wpf|zPnzoZ#ikUm|`2e99Ku?)7|%;lB?szV%yHEPoF;ROuMJ`HRQqa?ck<sH`{ghn'
    'A)?8FZA+*>TG-Srdt=qJnTKa$JgDUd>#9U#>x5Q`1A{<yU(i0vftagDYN%#$8zS^6@IDgp&|z0o}%x%<!U*f^|PyXKIj(xtiK)ne%-B>>wdPl{=;LNJR'
    '4z`gX-CLpo@d*`TryR4yqS_M)|=5V@UfE9?0+$JdkQ5oCkS38dpb1?Qh{V<k<HYWE;$x@Y@RL`5T^4bov*5v4JN)hpN-pctY3rd_vhvehzJCr|gans$<'
    'H~_v(~{f#-yDa8O;G;|YD^6Fi}B3<G%q{gyKHjbWQ>?mLU;s-AW8>)H0ED(9=7ZRgkRde&Cupw;tVyQ_B+dEn|rx43E7zf|SmaOnD`T@aUaKsDO*%Y`P'
    '>5!K1Mo41Q;RUA@%x9++tf{lKu#;bO=sET*0Z`bWYLv&L0e7$V@qK5H2K{&Yu9acTRU9Gpo7u{B!w*7ler1PrNS^sX^76a(T>U8_DyLq$RuCJ@&&+7Z_'
    '0>`5^y0v=Q&U?<LZ>zJef7h*PG4<2U)$w-I7iZAh)vM+1JoGuFb-%p^)Qaz`i{;I3bB3TA&zJAJo&4vW-2Nh&A*sHdEq2-%nQHu@n|6x=TQSBLEbtgO`'
    'R%gci5Dk;;>8Q1c&P<ayad6NRa`)2BP*!lB?+u-JOo!%RwKaT#TLlo%@xq{szBJ@hf%ShJoaXk$KICm*c((Hd#lQ0Z(4cm?JJMHk>#<sv^@6amdD=a^4'
    'Qy59(&u%<CFZ{ee6F|FM@GR44+kn*n#ku=a;X3I6f<^4*orv-usQ{^Y2hrzeDNwr+&S?>Pe{#(SM{QdT1e~E*9XgULXGWTh-sL_`}rmi_!69^rCFj=@c'
    'd(8h_~G#7109aP9e{;TMyi4(=AcI2n&eFHg>nCnx7;moLswF8})SR0XM@bNp#Lz1bYirewmmvu-{3cr#yYj`-=HeyZN}{px5q{P5w!lMnvMa(z8qY=#{'
    'B-A~nS=H{2>*Jr}IdT88`6NXpIb@$|5e>1;ZH@)gVFI#Qbb1j-)4SVp(W_UMmJ$?IC_2_#u)jM~HDTkZeMc;lrT+p&_%Arm^9||*$T9Ckg%MEl*m$RbC'
    'JVEE2hIdh-0vdg^4}z96wDjp$jNYu;^^oINv%aHVX!s|sPlpe5e)AL#IJ#PQZNDt`t-JCQ<oyla%u%K|ee<N-u9vG8kKumeu-*3WmTMF+x)%ius5pGHo'
    'Zb$Fte>i3*<`laY`f_rhj<t;I|Mq$SMR!;_ICss4#HG9H1^Qur^C<MzIi(Q!pHkhK11o7$8mW6?1$0w$>rp4uSWN<_~LkSJRVIRUp6^;$)mk|b;+ss6X'
    'B5`X#WjO-<_U6J3js2=z4W}e0KIQL;tn1(fH-~^83kytQ>t@fr<!?+d&V8+6;bw)AcRG#}4J)x_gURdn9*P_t3#NcGknI7I7TzmkqncaN6B0FIUUeWw*'
    'K6FdP5LjPZ+?fBheu(XM90DL~WR%oc+3@a=5gZT^q{{^`!Tho)_&Hy@w;I$Pbh@BG>L^7;Aev&lXCBJVi%%UQ2R7boX0F3<lmx;Q>Py*veJj_<GHFU9|'
    '2&H3qzhih<kco4Ml`Rj}4qsx=C@#JLknlKo1Hhe_4?aqFG!RuFm?C8J1?QY?Ca`F24<n_hq%hm4C`iZss)uR8O?_aL&YV;dkbU--23}%Y~+l#@bThBT+'
    'IxF6q*30)Z79w>syb*nU`oW;vK;-|8I2b%!q&1I^-XE-3dsoZNOzd{}0d3piVtGg~e7*zy`1AhW!Z%$s_@b);U)0+3r8G-F{)~$L4^T@31QY-O00;n=p'
    '=n*hd&wCM0ssI?1ONa#0001FVRT_Hb8ca9b7*gLb1!6NZEs(2YGH0)W@ce#Vlp%_V>LH5Ha1~5I5}Z7E^T3MX=Y_}bS`RhZ*E;vQBSKd6n^(tm_6{~-D'
    't~-KtgV&LnY2_8EB&BW~orNnhm(Ln41{?`z`GzoloC)zH@p``40U&J3k-fL=aBJAdr;qt9AWO%E7NL(>A+K-3n3jz_w{NqHR;vM1tKhE{L#6&Fq|UPWP'
    '-Nf>ov6$&3q@aT|_+-SxM>JOAF1auY^#t7z!oaAJ`S4bw$BnOU6v3sY-Dt8k*>D$;PP;BKYh`dPz^g6X}2NLSv(8g%7Ui_uIUe^d}_G1e1iPZ|!9E_u+'
    '=T%TJWK#Qw|f<%u^^t>q1QDkveW;A7cx}*)O@(=+Ct+o_wcOoK3`&+P`pbeKEf6A~Eq$i!@6P1^Obf=4zK}Svcq_kr!XUq6i1u~o*6Ovy=2EXn3-Kyu;'
    '&jx?#`SiZ$k<sTSHn`DeYSctCW7cEOW1}WE?m2rhxTzaw9}InN^k16Xs97y~o*3Vf82677LyrcE{yExwy35Y;Zk<Qhe{5B!<zi*_OD(sdqG9xd=Id5uh'
    'qOC{U<kbwq5!*rgGOT?d%(W{AOJxB6MRb$cGY9_3q*2{c;1*I*C!A;9>^%bZiYdT7Z_j&iY$j7aK^-gZcg7Eh+P6QdC<p)ekG~-wi0Xkv(b|`zAg$<yr'
    'D77@FjL#a_RYGJbLHV;q~*XZ8ExUYx#g_{{02OA0nJ0Cs3EZ07%~a?CU>JO9KQH000080GFX@UCs$VSEE}10GQzb0Ac_D0AyiwVJ~xTVQ_P3Z*+4nXl!'
    '(7Ut?rnVRK}1GG8(_W@BbBGBY`0GcYtXH!wJ5W;Zr8VK_1}WivN0HD)<7Wn*SxWM(;KH8W&kWHe!7Gh{GjG%_$bFl1&lI5;k9b8l|7y={{o$91OpR}ej'
    '4JYf>3e5tG|hQ=Ed30quKq=JAZ#%ntmO`t_$O#<BRhAr9Qf41NDk8PgoI`uq|#+r@U*bPzW%BSA4GEbiObLYR^{PW%Y!<%<+Z$7!HPnVmIZeIO(`|vPZ'
    'uWui3AMPIKFF)RXclYD`<+I!O4|fkw-u>>$>$}^>AD(>t<l*u5?d#k7*KZ!~UgwYZw~u#EzWc+IuYUXT#gqEv{_gwv@VMvx?%~brySK0I<`W;@y}zFap'
    'WHlu@$%D`^N068&)40(j!rCfe7wy6_LsGNvMlqzzg+4PJ^Sa|AK%ZL{_Un7$8z(@Sf1|tN6{#=(Vs3WjXE3U>E39xn@`s5=`m>Z+1Q_M2aR<$mZw{xF='
    'nGa?Vj0gKIzBP?$dS}mE&U`Z%@m7SX*X?KI!YzwbPg#=>6%aG-d~S@yT0nbgl0+I*n3ktn_oNGzMK8UbfL#eZcI%I-XV=2mOq0ZtLv8I$y(ap=Z8wt@g'
    '~T8a*q#W$iRJ8}t3WJ}q80-(Qb$wq^%fS)U$WxW~Evbfw$t5$F8Yx{W2y?e=2KUev8TZFFPQXzfpTFKp48U2L>^w8qmhXw9C~eK`Evm|bbh@pR?oV|Jy'
    '+bb2nj(#F$ewpwY;i7GVW+s4ylqp>m`tCtPFx_V~gt39zhQ5?r-9enZpy7^UYf7%C)`BiPsdZST2V`IL*&AH`xzP{CbeN1hfh;i71YW8d#K{b1ID373;'
    'ZSg~opqi~eZJkC?O=DD#C5UFepW~pJ>X05w&`gr%5+t+Iui1$2nsegq2MyLqSr{<p=O~4d$U&c1&GE<?-<D{VB$RDlSZd=D+cvvd=Ik8I>eg%xW_+7HJ'
    '=XKo?CP;RT|6_py3gnNh>b7p(9Aaby0xcXH9IkmB-}j~xHCN)g-?yK*<#_2^LQ{ebTgLiSf6f?R4m)c9wgbAfq6Gyv&Rffj=*#Gn1S`2sF;DZJzX~%4f'
    'fEyr!#KF#>#*$-ZMC}HX0jL*Ldk3^I2dPv(fmN0`c#iN!<Lhm6>jg%sx2dEiwTnaoc4AosZt*bNu9mPRccc3C85wsP@Z7vtN!(r~BQnoX<TvNAscEoJf'
    '8SI+=6VD~({TyPRP!m0ndFjqdVQd+AW%u(8sBgY9L&_2}|mHn?7%+4=av@xdoDDtld+ik+^-%AV_1?zYF$o(uHp%3ZOxOFg9GJr;K<hrDTz<y|HhYd$3'
    '?e3=s;L$b#TFU#bMoQoxXB*)%ki63(%^Tb%<^Gx0}JF+KZ-D8b!a)Q|y%rAo%+<TH<rGIem15Pp<2h_$mz`duu<(6RIla=o_s`SMxVc)|MO70DL?a94;'
    '){u|UFL?JP1Ns%Ld#>u?Wx?WB1>@EhIV3dE3UA3qm0v-{!D9C%sW2vD(g<`L(-7>+6SD(b&4nM6FxTg1V|NupxG@E@(Uo|Mf>Bvm<ZC=FCSb16V6jUu{'
    'K+^Lq--#$y}`SeU{O<??8xHaP;)NIMx)Z1o#V#w@D`;nkJ@CUi?=I$4{zUidl2Ys>^8jL={xAyE`?U6Y^V!V#yq6pY`DNe2=?Z}-9rY>d8*88h`>q4%x'
    'nn2IcF{;<lp9y?;-xCypwgsV9W_Tyyrm1$e?00S}H($j7Cc-zsG2VT%rT8>tmB)>@gl?X^MkAMx;!UEZxVDlsUdG+8C2)+4C5bGMB2nB1UCO!sHJzEU`'
    'Ladwa0$a_SNLj_8=<%=>~R#yJ;I@WiP`z_0h<i6Ik^Y%9+w_Unkn5iM5N5vwEQ6a#mx@QHl)5euZ$l$)_a$}*sk*@8{f6v-=WVsVI$SR}M~T}*7WSU^Y'
    '2?Cc~oe+<5+IO#`B?WsQPwOj?<UMnAi{<hbKP;hE$M_{oY_o^HiUZ!MUnY6mWDP#^+b=1{e50>pBe_J4~g)=PZ<*972ssVbUL22lzhs!yZ<7~JfJXQ?t'
    '5PFE!;7zP2#A-(wn~g&mgO-FkysK9&DTz>w7FR&AfmG$FBf8Yk5UB^Edp4y1T?!ho+ylZ3kvKV%BM~BTOQE)hNSuRwaOOzXzz1U*9C^<nYfyBDRGh;vp'
    'N+8?sv$c=an1{4u_SLj6kg6{jK!33)ennx+Rm|XorbnE8wa1$Pz*z`&f#A;gJ4|{0kaY7bdjHx;w&-DrKo)DXr>`pn<~a^#7LF1VGn#*sL7j?QCOey*('
    '0cI+E?(f*@{(h91hzNbXF#LBNC4wv@%Ct`zR=_98C@A2vRGP5f9c#&{~;9?P>~YD^p!S^EiUs%IdhxRxHqy%^wKz5R~)`K|YT24bhj$wuV<a)voJyT-3'
    'j#tIJjH2^kXe!Q+r2lNHdX5aPj}=~4*sNspR`A;jl8*}P|H0(hB>msf@mpU14{BZT<8Wsx~8&NOFY3Tv(Tu`!2ru1-4_lUP;sV+w0`5zST@UTZGmBPOw'
    'izM#4u)41kRIAS7e5P~xjhF9J5zTh}@&VJ5obfl*ydj!v^Wmhpig6Wj0+c~lW5m2?mju1n0TAjKOL;LwLA%-Ta(AplV?AkCPhNca}Dmp?8&2QnuHm3v4'
    '_y{R9Ph_Jpf{t?tp?=QVb~W2(Bh5DN*)E##5%MS0PQH9or-1PuyE=soI)u+$(rB8;u2VppK19!y_q&a*i04?fb$H8w@uAclyG*$Gnq4cL4?r6)2+CGsR'
    '_16lR$i6$u%fH)Uu^jPysD#BRPSGTST?jRWEQbHgJ?XX;d*#R%XNnfwB@2n%!X1x{tyduat`iU7KHCW6wg-IitG9^Rt1}0j@gH~?$fam{P#HhA8KaJd('
    '_N46HIQ?LYxg{;o+GIP5HE<YFUvbu|9Uyl-UURKhDe0@GFYWY^8=@y>M5H-_YD*flO{z_@<bNsp%kgmY9k86)@x_refb%9JA4oFAVDvQxTs;k76ozogT'
    'BHeqU^C5SE`5QxUY}dt!;s5voYen2V`(XNO`k>PXQWlTjyM#NDv8CP!GjFj&Rpea=R3$vS!YPIrS#28(gjTXJkV5==5!d-L7VlX5D!SUXcasSH3c$`ZN'
    ';k}DJ*Tq;No{5ey3gtGAGOp~d5DmeT6ICSwcP$f&sb@)iox-B^SnvR)ZBxUmA4!H*>!F;o|Yuf40(Fsv=Vd#WtNTAt54b)(n+P75I_@1=1)ljgsci8i6'
    'XlYB9Ebc95gT=k(?Cz>KUX=<K!yRjT&Wke<YQ<!#L#;^X$xtiO;ir)jYQ;3U>`kxoR|%rGRDhZF5JVw-xhn+GB-hO|17)%~cUzd~sP?n5vjXiyNTPkD2'
    'Qnm4&>HuI&b+5HY6+Ej8b{SeD4^UO+Hx#RMgV1bvSHtkwmd&*_~e+eDQSj}jv1?KE(NsZCi|cES<G1NsZkDQtRqflt7>dio)XkE*Pz2PbQ`&dgLaN1XE'
    'sP@j*7<L#>5V2am%NnqLvUS^8rj#IjtI9bXYaBC9A}$nXIiyU>i@@FNjeydEjR2aNcNyR5`S#W<yrON0p$w<D}4_ynRo09hA4DfVUFUIpt;7wYX`Ka!X'
    '9-INg*nooSwyRY3-Ml~w1L>h!84wVN@UYtja;#i^0FLhvn4<<o_Lb1~es@O!+Ab#my9MzEV*w?f0g7^(wY84qKfFBrZcl#@_>Y~hN{^@NwP*5SC|XRIY'
    'n3mlEL=-O;(NuhM|z7by#-GtUh(BzzH7taWyj1S`N0}u7CT|a^*>l|MudElT9owL)5uB<w?+kl~k8fL4gH9;x^4|P$0oUIs@pnyWF(`gP-npGxei^WlM'
    '6%Y8?+H{f3=FZto0X5)g>-@gTy|A<%@|m;nw7H0|x&}1ex?Zo@#I_Anns0=xTd+zmUh&tCr#_cH9m$Ud=6a1l11;>^$*khe4il=W^LCg}3`i)W9WGSEo'
    '~ALJw$%$`#3%0xvDx86by2VHFrsp3%6B+XnG3-eV7||3i)GQ_MP)AhyfQv<Tv}-z3|q*M!BZOE$zFB6HQbYZ3OD3A{>c^^6b-bT95fu1?c@b`C|gTa4H'
    'xB9o|K9BD1*9&S8f@eD75yZ`;FG&i1~8V@CXRdG1Ty>k4S8&rn?$w8|tR#)T5lv;Y7z7!AI8l2H4p^Jq-ofRw&3*yur*z2(jLEtIrk|%b}DXJ0^=h#o5'
    '9LO-4OJlC^26NK7HhT4+&_OUSZ1Ck3&)1qX;LS*;t!c&HzR7IC57b-lp>Cee8+RUd@l1_Rhu$$o<YWJqqds#Xm{wrM6^&Y}CPSb?*(I@E_(4i%N^xtOA'
    '=eK9{Frai(JwqPpbY{fI@AR+5V823REi|2?nEy-ez&4^WTSauaQFO1I{DU&x??Wv7#OR>|uV1v~*jqeRkJK5slV0M>R^;<*J=&Ac`K~|H_y)CG!S1pX$'
    'D$W+{ZNs=rhtg~{d8C6G3T&Me0KIP*`%};Jb(K%;`GSp*9*6q}qG3aWof8Km!ATryDyIluZmE12y$m<G=R!XYBZT20emQl+%77TaG)B@gf|<IBG0e$xn'
    'GK_uT{E*~7?Tk;h?&Z`QOrq&s3(Lm+pgRsjM-{S;zddvI%!*mF%9W7!kED#SxAO4*YiH4#8e20{~5<jUqmyGIdvTARVh)k#|nxj&mxYAcg1s?T(R21yV'
    'A2SNQud`WCnea@4DbS!gJAF8KF=1MOVaZp=$0)*+d|x3fQ4uLXc~!BLmmQnv^&2ZNw6XP2t^`QV{)a^dnh=0y!APWK+t-n4cDwp)ClpD`zpv+tiZq!DE'
    '$2KcN83@;UQ3(HeI79Df%NhIyMK&pcq6M?dKcrg<m_h*pgA2vJw>gJCy)WSHkcwyb#U^XS>|?T5>NZy;|Vv;(GSW2lGxlQ?IzTuUL2Yv>|0ng$dkfGD1'
    '1g?U4qvxFdKD)pNm$+GmDr*1{4GmNC&bBE&21%%&x7}aWoJHtYA9pg7&PqNzaG`D0aNQZrMg~kZ&;pL#4xK2kJNrV&kNn_G9&eM^45CO$4v}rzp%!>V_'
    '=@1U<yw5ds5F$;hpSa>`1iBel-1hUfRiTWeASANRk!{G+SWs(OUGxM+#$*V{DWW2yqyQi;vd$$0HzF|7XAAv^$Vlj2#b@G@Y3CbYG8xS$II>Pr!Ga<>('
    '!=dKXvKosa3?mxBRw25d|fLatRSLdh*QP^H~N}R3F2a{F8S#V#iBgjd<uybMU{ZqSa6nME3+A7WSHzVCDW$Ko+c9Uv2tV}ivU?Irj9f+Ot8z*#X`o!TB'
    '~SLsPl)sx8uwSm8YP?sm||s7CP>n$hs)#X-M<Pj-gdj+=XoKcyz)ev7$1T)S(ZQ%W`@$=!;HI-C=#v?`gEU>(-f6=CTPn90Jd!K>|)ZlW#wWVWxRaEWe'
    'GG1hOnLf2fmaekjmR^pZ1%d?lVAbA%8jpdVk0ydt9C=G%wQWX3jm7cu=9K~=#eb%NEAV0l;G&~--5_t`jfj|>JP<oUExc}T|_6--rqyjGna8&iZlpHkY'
    'vt$K}9B@THW&vZC(-cLHtXJ%IzTYUF(9Gr!ku@OEsi1Qf9t%#l<kqp701aTf~ax0?e`_a#H;ai)kn5#61bKlduM)Z6Sw+#(kd`bO;&$sX^*b0}!lu$TV'
    '44{V`pn<n}%SM>`-BgZ72z|=PWEaS%*5q+S(QCwNxN<G3MqY`0GDO~pqlZwnBWsH7W^0w7``zmYL6Jr@Jw&q>1kR6B=nzj2_uDSdSu$eEmCWiAo_ro!@'
    'waBWKqK;3!fasW6Z<P=lDKok%*M*32UTNk(f>dIu!N4MDMVnVMu?SWGhVr-+0=Bwj2s_gGx4Pk5W$%x^0gTC;9^Jk9c7cm=QiES1c&=#Osx?dj+URgk#'
    'MjoSqKidlnKO!L(y;v?u%j>K}>8oT3Ta!YPBqRW5dU3f9;y$`2y5{`Aj7XYCthS*a#4^SQ%E-fWr`BBSaVsfUna4TJ)+CBW%l-rr=RC6^^cijH{;+N0='
    '}?Cr<gXrc2lRhVdC^;z&-9wnD5h94V{s%ScX#0pPL^3y6h`)QQk7E4iiLMhh7~@os8L*H0V)Oz&IewTJhewh|lwsZ&g#>FFrG+?R~1=^R*;Gu2jcVN{('
    '1RpMtP2yd-xNv(?HpOQ?52v?{AO=r;1WQ9i);p%f5y3QbYdvnZ)aJ56e)rfGh?$P34nrnJEh;Ys8T?9n9+NpXF;fkl|Zi=<l=ITKeAmTZuy_e@mu}Y1Z'
    'i02xK1~kZ;lcyc#%9uLTh<EnMG+`ygJHy|@Iw#&4rpe}g^kjSLsTh=!jEit*P$T*Z;ZDfW+jc#2de)er7IQ+A8EPZL?gTwcJ(&<;&k_cwL)c|Iht3K@p'
    'M^|5&%$;x8r%qfPIj<nFYr5$G=d0$9`l?Z)^(4zoQ)L+X~wuo*w`{M(8<{;k)Z`e;Q%YCwRQ4#j)>Brf`l%1M1~f`j#Ia|3_H0y`y9O;@(y;i9R?UW={'
    'wWd&E-U-a68>d#0u9X7Z4tq@KQq1cx0Lc)(I)BBiSDr#Vx$CPy_CTHFZ>Sid#6nA%LUnP;j<j30oRdq!mX(EVL&_Sd!Y{2<=D;h9gWimb%!YuWjdpa4&'
    '?-M>m&rpE`iV@uu;C7oX&qb<U?N6z6ax1vy7#5ionCj-9zvq{j5ktST~N`nWs+4~1TE5@UkUu{%hN>C;T)*<?Ije9eOt*|Z=@jKL|z^FzIFLoZ?@-W+x'
    'ZS~LOY5k6hsjk~|56O54anz;#$nDYotnHK7OtTOH+>^%D{Eu!8ZX}cSEo}*013_NGlz{qnr?CA;uF3cl72mv=uep%~YwYa)8)O6xV!CGS@Ra|^^$CAu+'
    'A2P*!%BHE&StJIO*PL*oPU)!VhMgk7%zeb2rfRhCJ_1iuym#0<X`XY))H$KcaJ=eH9h%v6niN6%PlDk5wsg84YNm~MjYTE7aSo@*UXny{ibR$lPr)g&4'
    '!C6}r$QB@2Tw5(Y94PK7mkCn^pSTGDhyop;3QfP^Ej5EzKtG~ok59C$UsY|6EJ{2JAR;iJYOpjcRp5CrI5_;VksmA%YQ4$OKN<T<Rx7rmE0xO&>BlEBh'
    '8h3VBHIq+$9ORVqi(Km1JPu8kK}>J@=J-AeFgFI*==>8YR(3Qnjii+lo=ONc&ZriX*(E(r?8%x}108tCx>gyw{8Cp=y@oW%#ZanOenlT@{lR?e#d%;HB'
    '5?x8haS!@S~Hfp4tnQ}%F$bvpkmx|D;qqDje6SM(_LrWKayoUbrU`N#^tG@QM{E%nQ;F>yulH{g^ic#Zghlhp_vNjnXNUZ+9BBPi{(ApO-zY!JTuqz#|'
    'RH8?Khpp6KoOtTHYrEwg(4>G<DY0%5Zg=A}}B+9jU9F9{tOGO>A`xRwG^R%LhsDM^9k(|p=L`*ZC$K^uH6(s~-T$^O-u{~qvQ9qLGD2ULLQC)GU&lUe`'
    '5)|H7mGg@GHM@))k=IrPi7d0O<Fbk?QsjI-1mn^|geVCr0>l}u!zIUIy*`FS;vB)oh)BeZm1{T20Fy{+cSQWXUM9Da{-ttQagwSmRa~SR{Z&TgyKs+Ym'
    'oYw_^eUs`k;Jc6jVdlHcu*yHoma>FZ~0Mvh{sYL9l%YMQ;rnVms&%H4b9O?=G6InxXO@Tb(8V&){tFma-dj~BgY0uUS5M6Pjjc@#7ho`3ol<q4rGocIg'
    'l>OigPA81KFT{75R_y_DU+KuFNJgZ5}fvFyDqh#-VR`W6t0Eg&{UPG3UMG!v7nAC#S#Rsmiq{Gli!#nJJ&%hzZ)aR+Jq)pMaqHMk5?31r{=nW!8lvjLa'
    '<9nK|=Vg7u|GyO8uHNfnAPGV{e@M(~szW-`7UW>W(WSEbi5b*6h!gO=Gfc);XkE7@XFz)G^1Hu*}fn8Ie2k;p^I6gz*Hk$63GgkxZB*JZQHNaV{h5~y='
    '+_Sx%M^^Duvc<<tvGhe&oIW9u3k&K$W5*ttP(XjB;c^md!&M(%TWZ1Cna(-+3;2P$OacfmZ%b{e#RM+jI8Z;dxj!DB!lA)iblf)5ix=1907N38lIB70%'
    'e}{iJkHg8H?`TY!hfs=NM!u>u&`4KJY0|H1Mqdcor>e7%?y5_pFLs>n^#%Wqc2+No4OFWj@h}-ta!9JkfLkwEea=J=D#!SmhvdRYVIBYLB8l~wE~{qd@'
    'y>V4S@^!>GTUW|SDZX4WVYj~+w99l@2Y#b=wo$PkK&LS#ZORoSD?Q$qwt|!$>}9(Z$2F*0N>k6?vtOg5=3-yt+)Uso>dXX<5!HKB>oU#OD-aOf2rH7EF'
    'xsj8X|0Y>LcT0IY8fWA&%>pu4u-|{FFg5s_y%pdOjNhQUoEmn<7Z6Ax4)3rwEdlDT2J5<-@XVgyv55rCfT`RF%7GP`fOvH6-i8fl|cJ=;x1WCH^?hAED'
    '{|;QjpZ$b*nr9+dOod_@Q<5Aykq_|~4>!SI&3LXGHFu8M}^&sv4q2!$@WSKCQ=QDtmFAte*vTad_^LDY8AxXycAJZP)!$W_mAf!;X`e|0YH<Er8}xwKq'
    'JVYqoK8mhy4NSqvM(0IAZBny)Og~F|eFg(|&mIYj=hcG->BB{74x1B`XQ^_RNGqu>EN)NkeI9P^`nTCJG+=E7TQKrl^l8Z7Ap>2+`q_#QAel@|&^RtO%'
    'TBkT0vMA?llSTQt=E@~oD{upiwz^ydT9?7^9$y)JuMBzfdW9Q&ucBHOuXo6Ke!U8oUR@ZBp~8(0b+Q)Z2c4|j<e2TMN_B;sMyJ9JHLR;9qZ4-3<nA?aK'
    'M$E|yHC|{UQW@lvanM#tjg4715UfDXz_RWfU<V;wXm~oT<qs{lfqz0GF?2SD~|PCnYh&*l2xX=X{up+q1mtd2ewyoU6{hI?ZtaPV7n`hblGLMK1w?3mP'
    'h5F5b{{$M}4@a__V>If+KcSW!(Y6Zgy3NPO(AGhPtvsj#2~3u@42QFFW|1=TrQ59-}NH|9en*dLh)69cc>xl2BBZan^Ok+wd60R*wW#e>r1{H#EJzBfl'
    'l$y<F_0zF&=&jzcKlMc|^QAl}nl=czC-y=NW%l@hnZzqTCG-KEX#urGCuez>xE-$D2=)qul=eR!1%TczS4!B@X`KV+Y*ic!dB$P6e{giYZzP!UFELlNs'
    'dv&jr~X5I-th2#vQs-k$TYGvwURYNh61eZe9gFXy~>FXGT!jd9KEc|p<tZB1yxLp^5i*Z#k)w800uv|33)x`+C_bcmJ70K00@JAe8vtHMwGFctCyC!&*'
    'GFcX{NraW%wTWbf#|QfMqK&{>)F$G!>0k1=T|Q8Y2s&GJFD+g&F2ZCbE-gP_iAwv{b@^ZritCe+T}vvue0xFBg%(%T#YJOmg&HomV9shuq`nACTZ0((^'
    '8{<ukuGRukkU#i%xQW=NFd)pCCo`Jif+=7s*>Bh+0@tJIivX*sFI_}X1HB-ee)PLb9fA!*}NGxW3qA3(3$(&_yn_@)YnbZY_J)#rWUO^?p<^!=ykdp&4'
    'W5da|9Pyi`@j9AzuYHgL>##Adt}d3$(rwpDvL4<~$dMC7=T~8bmPQE`@O!TVB?LKv{S|jLT{WtS~UEbXXXfl_HVC$gGwJ6^3Tjsw<4ms@_-_oVB1*6@>'
    'jJ+banBtz{$l;igNhpwxwDpW{loOtX=7mIsZj!S?=D3>S`vuna!S_pdgrw#v60bTd2z1(hxeq-+qeF-^Xp*0sY_&VphWK?t8grqZ&qcnZp01P>3N5=*R'
    '`ms?cuBHP|8_e?$UnN-tnj+lDhK{<Cp(Tm_naPiX3Lm`@9(DedYB^GfZ4RF=vQpf^ajR6!g(^eIlLSkB1J{I!QB7?|=m5sp*3wdb~im<VArunoX=t-p^'
    '0!k<wg`702&<h!9%{WTg%-NF!F#mTYXu<y9#ULrL<;|c#ficg?t=_Y9COCP4H4hy)1>U@$XS&cmK04U*&>LCc&#UQv1qQt;BnupRHF=`IqPxE2qWT<T-'
    'D$117SCaE=MqViJI66??mgIV(dM3lk1y2dE+##ew7S*33ugCWSg>GrA05_NFuQk)stShpp2Lsfy&1_WsH%I(dzvT)bSPGE#J+h=tl(jSS3z6dtWPhft7'
    'FTaHo;IJx}1u`w+M!Ec;*nEf?IH(3J6nrf-o#xh67b>oq~U0U(fH{EsWzbIQ}AS!|{i+hr++DIcix0-@VMQfwIu~H>*qwN`I_UI-ND}EC<y7^~gcXk~='
    '#P9C5X}si0m*;uB+Iw7cg`2D|lu7vfTf>Rur*wVyLWpxaPGAv6`}A3QuOtU9lytYJCwV<~G`hV)H<>Nt|>!r8E^^A=)MN6&G@aO)*WDN93;TBQrQvnO?'
    '8ycIpd1<f`p;UNERxzpxC@0L3WxKPk<4=tX8f_oU;FKD>OI3G`LSH3yc{l4WVQ)V}w@}k{2goUEzZq|Di)Z75x;W%Lo=;<!-%i$SeQ5==sDeN{xez9?g'
    'tFiI`+^H?U^hm~MLwKHS$oMvR_ON=Ni6ID+N!~>pddca;h(2=4Frp8mG6f_0SW?EsOzeoRn1y{!(G`=hYs?o+XmdqMGc_Tl37H97ijPgH&RVB1ASzc`;'
    'M?*+&03)XBk0Fd7T`ATvH+KdQ+}~;DM=RBp${*~Rp9DHuEOY2d$DPa6-qBQRk}ju#byd*q3|Nhm<y)ESkCDg7gVC6<|`gaQS~j@o(0`ElY@%NZ*daTes'
    '{)7_18qwqW)_hO+f?Biv6MroJtcqa5s;ypaggA)S?#L)Q<~#a0y5^T4b<R0jSDqsMTz?vRbxZ90#@G!~CS8I=qQ0R#vT}>QJZ=cQHi@Dscd46`!&vegS'
    '>PLP{H}pco%!z!Z$`IeQ##CUka&LqqTsa#(^}@^hlCk!t8Ut-<A~jVd+^%#o05S@j{6RfF)9!dQfBt_=dOB~yG0q1X7B#ryb}ll_O6G%N24!Szh+FE|R'
    'eHRZ=j@sq+V`~;DkiYyok{LO~djI!R*T<zRH`UPdH;4A1oCHF|QBQM)1%C!Trq`P_Yg-VvFA7c?T>nUMABzCWtGZlu_Z77<U(Jgi5ylmy3#WU)vkO(Q='
    'Yw~n5y1NAng@o>I+_Y>bq#Qly<pYYwOkU23FmW_jGO?fIBx%aiHzhyOpTko6LilH|NzYYb-F%$T87?t0frq0&2xt>Yq!0tzlD85Bn$uO)>D1m;H@m1}s'
    '*dPqX(Uh_RMs3hGC+n(p~!~04yg=UIM@X&LRal-hbp_PiYGHlMbM^^UML6}W)>Gpf{LlDg^D1{hRSNer@`5xZ@yVJUhsj}Kzg#(vb5e-w^WNEppvKQlr'
    '1>KYgQE%yyC$xT&E+;?tS->N&DPJb;+_N6R^6Z`YpK1XTLEz3;y!s>?Gc*<9x>I{6PR)(w$F84UakBG$is(4yW|A3y~r<><I`VPZ91;=D9nW>eoPG)T9'
    '%#m{L#H2Xf@<*9#JEEL(@wK|avRp^<~TW`?btlMqE)vxfvJ=ICOS5}Y9w(x<GJ7)s`XRPOmPB=#)Y=Y=RufGr?34b`ki_WDR%0az|MBcU;|sIc+(<7@~'
    'Gp%J{O&$A3L^E3UiVCbJ>084)}JYF>QiSraq{nWn3B5)^MZxoF?809E(BV_FeSu{*Uo{e03JQ?|1)L3!T4TF#cH{F)12yVKOl3^sg-s)wC+C7gB8I-k0'
    '1wY*+kzqsdHG5Ds!94CCaKM<Csnv4JmJ%Ts>4xgCxi0rh&3>WcZdjCAD7ahqG+c*JF)BQWgqoaWNS4KW`UjTpkjET06bT-y6I5LsR6?XpjBh72Zrvwj('
    'pJNsg*v;C(OyX&ZL#p8jNU#__J?*wYQogn6$iB^x<K=Rn+69n(Gy(&il@ddOiI=vHFjW=@jMKRo^qcWyAa^cjc{jMQUv)<HFc9A+*!+sBj>4_^;Cv<XQ'
    '}QF{rRb4Q!Rh}Ha+@BJ~QSekWzREKK&&fOffIpxr9jST-RJ)<UpNfHr4Vi@fi!n@~x!ZK&^aKq&f_P0^-y;sFaTae}@6yIkjqDKw;ixPJ?wuBI4m~-HK'
    '1Blp^_h`V%OUuMswbvdBOiTPoIPCco-0F}`++9rDm3%mI0kiIz!`g=!><|Kdujk*H}Jw2DPPd=S-0QSR3~hhmoR+x1!)`YKc-l>r<{*_YJI6o;|YcXNi'
    'zmPWb47%%L8HV&ATx<*10p%88fNu-OXs`_L@#M$aSdRx$CDI;FjT=-QtAzw%paZe+aI^q!ePS~c(DDyO2Sd;p<BWaTvBQRXk^cj1iytsM^4Hvz%We6CU'
    'zDLU9gQlHnqQj@Op)9^gM+S#;(y57`_B=iXO5%@`j8Lf$)0PSq@c~mHPerF5J-H=1^_D6p#pSf<6gu@V>!DCw9+S-5P^qIHghslXWG{**15Sn7asYB@l'
    '*GW%{7blz4rOc(m5tr+htuvlungmrT?dw+8f7mD$%u^GOTsB4Nq|d;c|vgmE+MYme99_k*u8Ja+>m;^piFv!b}^YOw=JM09I3fe)*5-|5GnqI^{L10Do'
    '`nDy-1K<uwQM|T@)te#4aLO@v{8_g^Yi)btA472W2~<WDOm)gaF4y8R^adY8pPOLhx0blz~Z|tzzX3XluABgLiDUnyd=AN+U_TLZITQtm&(&xGJOec-Y'
    '#kl?0cq3fIuYWh0HR<<hDo@<p$Ut&%C*+=rvPxPB_>eCSy;vNEB0;i`fJ5MVtk6Y{o1b^)M{#J+kW(AKk9DU6j$a$hLM;MS7f7t2jJPBQ!^|7#9CNq!+'
    'jP+BT^ekBT$vxzYsbwQ1$i)9EaN5Wr7RPjy8`3tEdBqnKp5wZ}algz*1qJr{~{8x^4<fBT>M|m$q-qZT3@FuF*H(J0S$fYa1sV0Z4@FqTerB$<z+$)>7'
    'Tnx)F?whFd@TMp$U%hW=6hJ$$r)cFfcX3pF-Jy{WZm0Zdk`<qXYqQq`F*M<Rv5uwU+HBp3T*bE;BtB?hbR{e^&dn%~1Jn=iW;?<E%)Kw_o)!0IV0&zyY'
    'l2MVIV$27*B1^>z$xeoTF9Q`K)&Tv935LS2e(zHKqYr@TLcBFWDib#^Sl!8wE?6`24U2lf&NDhVH8e)JxLZ}&*)SokFYo6@|8@&sAXX#noB5hS~Zz4k('
    ';Xdgu}d#+BDm97Mf7-34KU@S33bL&G4#3>#I3lEx~Qg@~Vl2Zk|_>`ByW&gaE3fdIi(5amaIg8I`Wv$PpsVZQ5bXWD|)YDU2k@z9h;fX^;UC<L7`CxSN'
    'ttNra4mmdmDPg&hd8q(TN60iI1VWRyJdI?CF*E}Tj}<cUC1$%u?{$!cqp8G7MvX;k;!M`ASa{FPl2G)60#{Oi)aZDpSZ?WjsJe*gjbd`!Sem>_eK(BBu'
    '#fJ#PxUlNv>oc<OPh6XOkT-K!Z_XHZMB=-04)6nQ*S~xyf90Acv|HYl$^0O%=+Y+(YQ0OAoVr_!^tim1>4PDm+DJC4cE=#gR3x>osszgM48%bUi4D|#>'
    'CMMdWKHvMYG>esNRCGxIYQv)IrNyis$-a>m=~t^wak0x(2#W@Hpqs+vm^4_4iUPhxSCAH==K7FB<^XJHU(y$|+bXs|OYP$1n1oP?i$+!N;+0(&3&cl$J'
    '2Zu8M2(+{L-8C1Ylc56d8s|4p4H6M^g~s0Qv+0h!6iF2Af%LrfRJQhrkbL<nGse^Qr*Nns-~$9!v@t%)xZ?=9Cc!Q3W1O_eNRyDA&9U3s*3#{y*_9#&u'
    'dBq>ti+AD;9hdjc&FMy8@6cHhgewqy_nP)6@oS79&35-P}bEb?fI{U0nh_gA{~dS_~%f(wJEJJU9rO8L5?cX?Rq4LD$!wdJO^7aI(SC2$-IGbkr_dL~a'
    '+-t!pAnpb{_*H@t#`!luQ@cqL>SI0*2VZgrD^ph`57vSesQeKYD!z46GENQ-o&rEMvmbhXwkzgJ&tZ%L70fgYw>R3ejmP00xuXfPqhf^7TUfKw$lIU>6'
    '5l_9N)`SjK3WO|2d7ouALhe{}AOC-!nEM*)r9%3oe3U!|kf+<@9FIA!`+p(nz8*6F|yq1-C$~aEVmpw!_%ZMrwm8qBr#ZE{iz5wNR8TkP|1dMBwFDgNm'
    'U}dt(A;oR-)J@_CJ-UfauY0EwPYbhO2`)Olvra&*WpQgIs8-WWUx}&(-bL|TaslNGTwpnp0g{_~<e(E;3)(4OnJ!rujcR?W9XNeLCq}UrTxrwITXFgXd'
    'H~E=aHVEtU&ZMYj)m28S>?<8;P<HsXjJk0#6sca;^Yn2XirR5R36<vnIKVd`#}BBNm1k>rAY;da!_fO`mWVXq-yGFT~jS2dA0B7xuG~j3YoxKa-$J40>'
    'EIh1*S^kXq(5nVX0OVS-U&YkP{I4<4}KWS{|F9Q`4DFqsanPfLGngEOS8}%gE~nZWj)X3{>4yiNIQMPld4~U&B3h<a)(D1tb7(q3pPfWSk=*mKoe3*1U'
    'Yi67(gvVKCCZtUVQ9L+$701IF5yYD#6zV0)ID*NnF3c0jd@F4~r4Q=8%D6yS#3>z-^eWYfOu9I9P}(WF?S?#C{MVI}UqF9xNm>7|$HM>f^mbfqhsR_Tl'
    '2J5UvLbx+URVJnTE>iZ7Y1Phq}eB5WxjuUu@(V?7k)p1{m!PmqhtOVf$$S;{&V%7LXLdg1j=XXu(%!J|~H^+q;2GY<b{*NQ2HsQj^IZ9p&265lgEGJ}$'
    'Wv6`u3UNS@D5QX6n`)bCUr!w3+!vt>M7&rhTg}Y~d#lJ3fo3)IRtYArHFW@j$-QL{2_{n)4+};Ih!`0xn1ZvBb{fItV3@;aVzrHQDF_4K-7Je>GCa~N6'
    'W-Kw)bOrJRfH0$%7)3K1w`pDjV3A;++aPm07U4j7LYje1QXjyj6@S-0XFa3wN_YH#1lhDga8OAM*IhLi-=+{MH@AZL|rmX-^0))lQ1T(ca-dU@?Zr{?j'
    'NG}xb4Z25jf47#yS)@CC}+SZpnY}9*<OWbsw*~#;fk*o?98{KJIBO={~N4AyhF|4^vF*P~wQfQwfBPP^5Jh1+@eS2(v(SQs!t?0%3ESk_H595DC5#2n)'
    'M}&0<!KAVyZe25Sx~QLw;MSv|LLne2fmSU;TxM8N`%0NsCelbVw>2YZjDtwCUPE2l?`3Zc_VATSyrY-|%lWOU8Umr7)G)r(gNjD~3g=O8c|P;u24kPwo'
    '4rw|!ki-EREWHeSHMm&+xfPHbzY-Da)pH~TuMs}+z8o|*BpmB`<$aqU7I2!3-sWah#j#o&<oAE${M0M2#z3({|_@G0&V@0ewq3b!Oc%cIh(jMW44)h*G'
    '62a0^3mItD<c?*@{Uyxu31VY_Nb>>M$qn)r;a75#1yX$isB5uBvXU?j)nuRb+%J!8pO#jE={`rQ%I5o=I}4fcQ`0v|#?J^&Ym1m>Zk<J-VXUbwlAIhCO'
    '4ouwlStK^^kldy%L)I&SW+a5Y*ce;<6mes7asnFCH*nvC<pIw`N9$+B^ULoWfTXQOP+E)HEI%-1Fl|c1`h)rA(<v1#Hw>hx|xxzN-|B5u!KXx*&%#<C7'
    'EU&Egik2oQ0#829~=?gOy~OxDb7pWSW2<)Xz(j)KYUHQ_H)X@?J@%>E|ANt2;`XyI+Y+?|aRWB{IF5UR{UV16s6EiA`@Ahpz^wH%%3T(=9i_Fgjhg<YD'
    'Oc0nFr;Y;HAj>X2~>0$B-B549(@sVU|q)uDi^xlv#t-kmE)MBCj&|4I}pV2s(?&kfy#dai__0=rRSBjj9ixClh8Tg?qGiA1%uz70i{#l8OwMWtO!EGiJ'
    'F><PiB{e1V4J{>;iNJ|;Vkify~o*$<}Bq@W~PC(7ETm%jd93#(-v|<=#Jt9Zi5qd<7Q&drg)Sti@O3nt3OuDz-w1Cb6A(I-mI~L^6l)u0!#Ey~V-L62('
    'ZP_n~CRNlJX?n3Jz52Vb4#Z|+(QO7fcPpF;w1E+*<F9}aI5Jq2CC(gD3-1*aIFq^pE?}lIfC{LWrtG3I5YYh=P_%*qARskO1vJ2{asUi)wXUoH2DplOR'
    '=@)6d-g3iLp64_E|)DD3qXJoJK(a_{jaitj5TWHhI(6$#_G0?1mOSyT;2As?1Ca?*bNDryQbDGPba=&c{(zz)=LRdty!PWA%#new^Bi*gR36bTboscIH'
    '@dNMCk<LBTYvzK3YbDaK=iaV0^fY3aGa^(TBu91o)5`2%#M!6!4LUwXdtc_sf(5XqAC};H;X{3z}7NgUte%Rc7h~WhK^01;{G*i37%J<TS4CNk!yY;6$'
    'CQWmn-hn+~AZQj-pbHDeMQs(vwb1C-eCsz6ex`3pxBwM)`tzmyOa&|$M+xBw1IpGfpIs_O=H#<f8h-15$y+C;l89E8Lt>INwYd!lZ5Tu%6i0<7Q}pk?c'
    '^TzQGQ;qMk2LuhO~me0pIWd0<GwHcZRuy!{m2*Ae85bU;|C0PZm#y#^4$Ux4q-JCKG={>0fyWFb#<&cVBvI}P6@2+4DE!kxxVS8B@0>oj-{3~~+SWM|*'
    'V~U~!zpyVwqWu=OrI<=#@t&++ga|F<FrLk(6oH2B)2xe@hK&*$Y-UG_Vwqq$5;;Ngrr<CM4t&6p$(L@3R`byUZ%RN%qDkzYv98>XHXqF-xEI>_fOai;E'
    'bfH1m_y)3XqnXmFreGOl*0-&pDV3#xL+8u+n-4ahwRondpTi+3Mhae(i9K?RTK;Ge<S+_e7jHYf)Mf2zMWOG1<1cl?gsF0P2$C;Du_y9$AfcR1U;?EX@'
    'xvB%R|7^A`24kRHVux*r^I**|Q!n-;<^Mp2>nh1(5}TDaFLWSd>dt1j3NZ>isC@>~fTL9eW(H`AKDmqhhjmWrL$MdWu!c-fPwJ>apk_Ue9EqeiWGvs&u'
    'hxx%e(BB9XmpXr!;G0Ga6CDFwj9Oh*R7&sbCez)$0?0N<C~1N^>`)Q6xlAqeokOhp2^?{dv?3Tt5dQb2>+cQWl!KR717n<9d&STPX_czu<yTmh{w;mXQ'
    'kN^CAXc<mwIILwHZ=YZ4~(DF8lwaVPH$p>6sINh>551789ZCF6*%M{uQI%&&2r2!|6_@rmZ$Z&%_cQ`Q}xVpiKMF!OkPPAR!-oy&G>h31#`DJsH+EXJV'
    '3Y%N2vb9M~In2%`H3cynn<N@rVf&kNm6m#0c4T*^U@iSD4=#YG3U*w8PEB12;MB~)2R5}Q6F|+*$BVciD=Q~TPXR48GwzQ>6M|@gE7Xz#0V#JOEGnB=H'
    'i>}L{N2~G5ccCl5vU*_WzGZ$Na`B3sBF4G4nZ!wzMqIdwJ7p1R*E7|?z4%)#G=@?Q*AD8{+HjS;38Mdt^X9G$9}2XEr)`Oj4${7%jYP$6bOEMQEw9R#^'
    'dcoRVwcJ_ta0^PM%tenh&ZQh=LjRsup}>$bl%ha8S>Kb|TGGumt^aJrBl-T~<L6^vC@?5PzkDr{2@u2Ap~`pRj^U9y#yOVlmTx^D^9In&6<4w+xy9N?t'
    'Rn1SWY+SC^3F+j$x3*}6pmB(J&2IC$h0@P!H*c@@X6F3HDt??*c%z60>_rXN$-V@YDUf>6~(Rag5)oQ~nK8a9XH#LB6QFdc`#Gh_5kp;=w^%UQf#0aMN'
    '0wSiLYEtfX+NA=WP*Gq+I1w+4ELsNm!?`blFpx;j$ZE>&d!})M`=He6-HxuP)V3QsC|CZeKmG9P9tmn?H+H%Zg{>@{H%w<LprSGE<Cue*WERRkJTI2u<'
    '*Wsj&da{-ZlxKQ1*qF_7-pamg-H{^O8u`hlY?dF_MPxf^W?YQIx`=E?_5*;wYtEI(cJ_V~YmB#oJf4FkT1WmqPGrOCCf&w(f%rXH@^+jsffaB~jp|x1q'
    'OJn1sVZ4(xl&h^2j?|ndU7~UV3}GI)yzlkCu&-)qMDglz}7uVn_4euEVT-H(=^DHuu?m5ll449-APXSp@_x*!#2#7tkB9X%q0UT?801g16Fon&fPI9+c'
    '1|S6%o=c7C92zFejQ&WhdrLi>~a%T+`Q65~ED+hNj)m%LtdGev}wx%tlE~yV($P0`FESS9Q1+Q%!Y+6f;fD{a}aVbA#`4qLk4cQOcV4?~pHt2gOwOr7o'
    'A<rYjq;_L^E>&83C}5-~hQ89XgrWJSDi{Z#D0+H{drc3^EiMPzkLU`H@_V1-QGfi+hXJFvEr<lxpiJ^<X!nyM<mn^oUKt!8tjNpG@Urf`yL*W@sOZ=R`'
    '^pl?=PAr<(|YJq<Rf3wOd!QZSspKP5JM<UOrl3Gr#SzMs10WQ$p<DU9tEs0y?<!~b>{)(nki)cFP<f7>eHHhWJ6s$!w9chbXoYW5XD>o{qIc1|^-1CAa'
    'W{C=VT2nlg?JdrWBpzl`?-d-Wi1&dAD~bl+)~_t1Xivc=ita0^Q-jL&m71lCpeqZRn_@!1Q;*dwyA4v{a@?VZ6vpiM7!|2D)GT>$Dwf>tCKkD#R@d#J<'
    'wM>jqPP%x9hv83{PZM%c3s_on3fDH17lh;%3Iy<kbp)tXqt{Kw>nhFSGR7>CvOP?1mkm)L-|{Ek3&79z?qhe003xON_GHL2^VCd!1^o=VTjgw(+Ad<6S'
    'JiP0=wm^h<VH0+NvRJoM3T(cO1E*mgxQ!wM2J#qt<Yk;6bLj`3O{QK_bH~WW7}<WjFoT5(z!Z{H`xs59MlL#>xrY(?WPquWoe)Wv*6EQQIb%3Ka&zXi$'
    'AYr}VW_psjy^>MgZ6;Y6SayqO$UUrd6H6}H5`IW(Ag!3}}?)jb15Zp;PqO3ZM#B{v5HETF{z54)L~QXp)xk!O&Wh(e#Rm;Mgqk-l}oe==d8t$+**v6+a'
    'WZFd!D?#2x;c_w$b+a$;CS26gIo!|A~x*LW?!tS<lC@tc5H<d&$BWbO<=MO1s-Rd7Q)^f{J^3|H&({3z3OP&tPa;QsK+!ouaoGVyP%=wm14}i)>*7SFd'
    'v0OH5@bSN343(@o5=X%zXQ6RcYnULr#d;763@yR1MjoVoYs&<rXQR@fYD(WS)5*d&w*q(f2>pu`y)9-@b@2%psd0fy07XM#%Pz&>A#_sLu!-<^n0ST5!'
    'yUNU0>JbdK;WRBku`+#y@SX)#4wHvILv1GV(5s#zqz>x@@ig)tdfK`H&>5Uf-2dfQ=1DnMq$g~!t9fPD<L@CBbi0CT8k3mVO=ka6QEFoUaM8OK~1aM(g'
    '*8Vt+k9EQ0S;tNSR}27KgRB@wOo?$9xiLH!s;uTNlJ&;&TLr6RKWJBh6}>P_Bq2>19b|j#%KkJS6z*E)NOmv|FtioyVPRD(G&Vcql|Tvt4NTvXzdgiPR'
    '<0SVb$r6^@$NvM6}T4cc)@p(Ym64Al}BiBBzYQE5X<JhC_+L>)KQED5LsMT4hfDOIBBP%KxZ83-MYDE2k#;n4m@KV0$mp&hPnbXv0Nm7krB=q2uvSX!_'
    '|Rb+I+F1?}>c9Q|n2z&Zrs%!;!GXY0=rA(O5bn-(&P=p3tn+5&BqAWZP+7Y`W$icGOd9+EJ^Le}5cZ<TsW5wX2Zx;9ANVK$12x5!g1hE&;W{5B99^Alo'
    '3K<~8ru4z@gqejYwElQ=PX$&*73)#ZA@LV40+dpQVX%lwhEWr7SI{z{y#xlq3DcBO;(sZmAcB?%R{*Vu3WDF)Tt~f~O$ewi<_crK3NQxi-o@*JrB@N2z'
    '+w;_Nmef$QWf(`&?PsW5*f)FMS)IMi*!Ub8kP;N;_c+FRpRBC-nomRA`n#(wS;O<fI+N4u7WsxP!>t0Z|#W~(sSE*D%&9s?SbUmDm&Fhh-An2-ct=W+_'
    '9zI%EHAy9)6U1nKx(tJNYlEmmO9rS_WdGOxBqc%L=WVNQ5GDfsq?(J%{KsAu$gN{z4r)*cL7nIig=1((kaSJysTuK$V-sm<j%w2s6PF&6|+wm*wJ-DI%'
    '{WTQcqowcy+-=N6}|rs4*tyvXfThiG(enhkSs$8R@juWldADs~&jMcs}KA<G3lkCbMUoylkJxn=jJj<vbdt%S_xuhBEWBd$O{Aytl&$Ogh~;rI=?a)Jw'
    'mVA;izC5*AHi6Kxe%@zodg=+D&(!gSIq1mWD9db(QQm^Z`xZ{r4tD9>I;7PnKxviPia>8a}&6$f!{mYu$hwS<t+-<X6z*<^jOyo9x7}F?-_?~kscQ?s#'
    '_;xXmha0Gg1#QG6lf(^Im|Q$`8+%6gd2U#4nC@_YCCp`mlc}%kGAY@pZmgNEMX`S}_8>Tqj=$9{Uv<Zf+~yZ=l)yaIObw=MF$+V06&*8-Kju+UBxxOr!'
    '=S00C<t1OA}eVEs=;H_lI^W4uB2O%W95?X5|9RC7w50#99Y~@m_y{YSpn&ZZW5=V^7XpqPcgfA&nn-|(*14-**a4%%PKFUh>NLP25U*mTr#Fh6~La+4N'
    '@|VMGz{88WH4XOGG!Tr3Aq71Y6yn>Hn8?fxWn&^{2{Avwz8M%9QHD1enq7pkooabV-QpADBm76T=jEY1Mq|jmK28v@d8wIDLkBDfoOw=0>CrY%-a!hnX'
    'F#s!t-ROVG?tmz2qrFd?RQ6@AFOO#w$TbxjKZt$(@bc9er{xjO_I3Fmfimj5qq?*=1uC+9SH-OKBO%+XA0%e}g+{CBTsopA*VEPZio7$swNi#9DP^S0V'
    'mchm<^vv_bzz&<|5)cW+hY$uKzjoq4fZb=lk7;94M<<1m)<tAyAdL4qfQ|NWb6Dae-d%`Blhps^qIGZSKly^lfKE6dtb&F6mRBJ~SJ&T)4-6hW)Ru@Pv'
    'G2W3!8xK1~{6P{QUCtlfi}60v4NZa2c`SGwGWk#-`L@H&yO|U~NY6blsI~*H7uWBW8?r5lmjO;WuHVc-ry!dhE4geW$Oe~fB4^mxErW*ZQ~A~P#5Y0KL'
    '`g4+puk>aHA~2YLPTfpvP<FF!`uv8vE|lwHt=-GJSlUDT4*%owkG64&{E4?aM^yiCBtPKwG;Zq_Z(@jgw3kyOQ@>n{)xgd7r2GHSS3&%ILM+CfL6UKMc'
    'JWd73<KAC$!Lk)MrhO#BX*@3VnRQI=Z;Q7uI1t>XvxIDlNhiBT*gNn&_4IR+_*J>>$&%6PX$@8cddKWn4**it3X2&&`Ut+yoPVh9h_Sp$sK=11ef6?m?'
    'xU;=WT3k~<~AYTadtTx<c#&#j4bgOEfpahnf{KSb27>5j;)f5hgoQF2dYH+~V7Vs-NtKg7Kc+|M%i3tCeJb%QRkLfqD6Q)>!J!oRT`P`q%Vi+e%~7rK~'
    '!En4V;(b#ie%$O6(=>fM0qC&+lW=Qp(<}F^4HO+MPrxYp_oaY1zRZNQ$njWN8yjXT<Ew20E@6<YOddaDD+;IM8m7~s<s#J+tqrfq@Z=||$WoPrAI8H+|'
    'd9k*iWE|X5;d;aH!Z8)>B(ID$c!*(Ma889YuG?BI^u<8NK^2G?HnxPhz@VoZ4|4|>RV;l{dT>%vB*tZ;E0lB(ppj%C%k5n)K6i81+{@j)U5k!La<^JmW'
    'RSZRd-!f#mR0qg2OL(l2Jw@-6#x%0)Z}gj%c~eDl(QYMA9A;VhNIE*4EIpYrX^1DR=^{}a>(Bb;3w8BDO>@~hlF)c*Rh(s;+C%Kuyo2DU3=Br=Z3C5!S'
    'USBwQt5c+|IS9S?O-BYr<i>nQH)M@STut&#f9r?4s~M^is-j)2Jyhl+eC3(A~3LM^%BPT#)bJDK$eD1*Q_kjRI#=q_pTkEzOMv6Eqpl5+!ALFPd8orrQ'
    'SYQnH$$z+cL#amrz~R3C}-_L}`%_N=T$n<UM8a%!JRqIHy3k>N(Fb(DTA`lq^OLnK*|Zym*sK4eHjhY#7Lp~VY#MdjwX!y&_8v8d%HSoUK5oP}0o(QUa'
    '`3mITz7kJAQu+{t}w{gvSQg?CP1Y9|+2@=U9w{dlYTz7G8*0Ylc7CRtoRqrB`Z#?Zt#D{nq=>Xk~cCx`vEI;zBA}gLV+wihrHtvNQHCQxuHwfiDpogC%'
    '#@e2PhYNtEy^BFeK2Fq_L(h9ci9~O(HC7SF$;gSla#`8T#L%@tPEJi^9JiVs?vChAvza#TMzc+wn!L4^Iber~WHL1p*8<pJ;c5E^od%BVFeXdl8Wng9P'
    'NC#3j;dX*G@3NQ@}rO}bmi#rZZ+@VTMo<nxr)5(q40Lc;g!4!_Zu$OsO5IU=}r-(0j};|!wD2@ZuU^&J;bdyDT3temT-=T`eORQmL<1AhY^r-5~cIF6i'
    '6=`N5X+GMxGL~js&B1+j2=2z$Y_(kQ5_oyEuo(bx9yFl8j=vT$eDhs8vsBSp;ZdRo1L(_GNJSV?8g+>O|xfiAI5?Qgrz2%7fvRhudhowqf;KIt`YyI*}'
    'g3>(@+2ca!ygu3xx)>&aMP_cfg^Zn9pD?w8zTJu~8Sll754aW`2vg@|0_z<ol10FTh{7Lgp-oAwClfl;r`+5$bI=v^T}u<dTLM26rA2|<cr6w5K~=6wm'
    'P>CWo8QGgq(Hz|!2m1_1Wd4lXVzv2Mj#m`|_-2Ae+jf9(qrr<4Z8=93v1!Fd9HuaDL?zu1NI|tk|f@yfz*|9gXBMMe+P$oI6*tIQp9V#X<OiI_gP8!3J'
    '^kyEzp4tZ4T~URMi;NV}z!)JHl^oGpt=TEd>EPYHvMauzIAZxrCTA}Ccn|Xom!9AgxJ1Yl4Hpqf@1%-G(XevI0)JJ_LbB*odHe*jMa#Lx2kD}*RCM5x?'
    'zN7ds}jukR+8rQ5@zuNNncSnK0HU*t*KCv__eL)S_6VPGG2HCQSX`<)TV!JW-k{Kz}DlWV=*@sGKD0tfDeKsblD$S-+<G8-F3(m-1Y$sh94fseLzvDI='
    'X2=ZsLLSK9rxs)@C){=6$=*^ft=&ltaZ_ja^7cocL=^$0%<6wbiqQ^DHxDsX}R~P#5z5V!<4?5FqDdvI$SsDMkulXvqqE_TgC)xt}D!5MtGHyX-^H^r4'
    'C*dtHQhNETX*928sVyCvDA800y1ZuL6cC$<QNb0luFU^s`VWCh1a6jm3n+|_h;pOVOCur<6R0aC$Iz&}zmO`+f+3B7#v$|e>jYZMvX5`wx4Y?h$5`bv&'
    'OKQ%{iU2}DT$=bJEhro1~a#H3Y`j+#vr6T9BY-g<=18rq+ZV5rkV1O4%wOh)2)w=D30Q4ydxTPd>H}@qe8tY^~xsGsb0vYP!IXAk{$J2ED6`edsVo{37'
    'moDUjn`hT^Uhwn8#8Mkn1s%<D6?{D<-Ihf}zRcP!BJyRjSi$4Nwhq2PXmvi^NbHiPQZW`ACC3OJqp?fNoo|T89^I`P2*@5Mcsmi<qmQ!{*8gH%Y9T1QA'
    '5H5nh;u)IAdSm5lRJu`*=9y?DWa8{09gpn2E>~NBhu;ElUpdBdftb8S`xpMIBj5v*)c@amI&TNYs38nrzB(>=4JEB)e&qyWqe?Q5V!@Fg|}lLX7o8BTA'
    'CbZ(TP=5<@LdkEl|keEHRg=EZ~5Ktt=}!tnC&<s*uSUhE@i`w*;n8WC^d|6Nq>b4RqPgKshnD#e|h&PG`-0ltPS)UJR!c&(P40D2D7qu0VZ+F@w_@bO$'
    'wXUGg>VqLVRHBAH<>Iw-&f9=RP8u}suS=24jM<OoIBz>;D5s*dH53kkC%)tbq8vU!r%dKNkm2m=btD^n2|UYYn4`X{hvT5hR9K&;hMkc>ph^n*q!)Wf8'
    'K^lK1mB`_8&0KQCYEW#K1D5!Edw+A3T7KVsI4Rc{64m4>$_1qalPwbWjKDc6m<025fq>qJ3aVP90i`I0=o;%LyksVpSrAzh%!`3I8!QY}&wq_=fUfELx'
    '*Dbpw>}rBz?x)pW<zt6Q6HVdEun?eI^C1L(LJwe0h|z6X&|8esrPgSTNKJ*A_)mbNxFnB+VB`5P&CE+EmWs43ok0|%o{k1$K%t24o?F#6<s3wcd(UhTD'
    'l@vaZXxEkCRSS^=ofet@?PpQYFa*o{lczBLR7aU$r42Ze|?-{fWm;D717n5NzyAo>@T497N3cwe57b*)AEuoI0AsN366rM=#K_us}KZ??a=ZS%n(KmPc'
    '?6PJC$bKzt|Z*X-(J~)>Rdy_k@|QI)P7SU3CL*`}rt>7!lLqMIxuIi7QX8!M>U5TSy}bhhXmTh65{rug%QBgSE6|ek7C}Ew`N@rR3=6-f|uBk+DAVO7@'
    'bw9gtbFts^PCBR))QDrT4LS!!s0$(mbAnPIY}B*Upc7QxL5>7MOO-XP%<g~D^q6h{Xd2f~?~LEdMqQYAcG;kYCOZCaMF$2t<M#|eQe^otI*s7u)>0%rM'
    '0TnV|QW+5sD&o?`X-TYcAWcYGpr$6U~lre*Bh3t-iq1e1~h^x?SZCc`LBBU#wP}pR7^j)lavON081S`@!K$_$sWahLbFKj}KRuiYX2CD-GhCqM^+5+lM'
    'K#yWZ??n|Q8T$l(CQ3W@HNTEH?Wsi?sV%v=l(E`TlX?x;Ci^8|8(1)|XF|69awSpBba4~5Z5jA1?Zo4e0M}gkyXWw|9qGhe6N$q7BO<a(+GxaodgO4~3'
    'NF->M;HT2sBD8k+v)ohmNX&n4Fqk;nT-YYEEy+*<plgK30Qls1rv2smWuw`+UY?}gloO4r0ZrGNg?DCG7-9MEklM389|qz>5z#4#h60AS)SO)!X)88x&'
    'R7@CEc65LGR>dib23(Ik(H*tcp4K)t#W@sAu)fcHIRUv=8}>u=9A(fbF8jr(1DEd~aLMGQytqMx>|Kv#PFd7YK@OvC+zHZr{nT<1AwAZhKEGX1i2fp1T'
    '6a!z^P6is|~DjX2_UHsKu5wkGQmvW?t)@!ACJ0IjX*tP!it3=ELknqDAr+M2%$2@bh;5lx6y;W!jBSa>$;nL(}4;0Al4F*>{TU@?E-F6bNsb-{#gFc;j'
    'n5X6NFy8v%N(*U#u**<U!Dj=XO=#T+xK}ZUa7I3hLK^n6r{%B0~5GJGZqh(f+9bES;XdQ0w9d0UMg{)2hQ%G<?0EP5ZfF@)Z6DXm~zHQc$@#{D@fp(W<'
    'k<SIK(=r5|Q?(jsPOFIl<}{)MDyKjqKysQU4IZZ`H=uDEr3Z@BG+J;t&BYUhlkXIQ6WRmOZ7@}<HM}Gz*Ar)9wSYH7>sl{0tU$PFF@fKv!xiv0$$!m}j'
    '096aNzGGsC_72dE|qwj8>REG<$`!26zEfaJWe2i)grsN-x@oTrmb>roWJ%q3+&>x@{n-h0({5qg4%HVeijw27B>fT1~sk-bpvKxRF1?YZN1c`0rnrr0$'
    'd}OK;tvr!hGRaBmpL_FPA-s*PhS~WPN9;5y#>fFh>1?1OT!>NDb?n4~&QBj_3@2R#?UnkR5!RxB>^)vT2pKt$>iMA`0;Q0nr7&14(Unw1g9g#k;x(`ob'
    'YYALqC0%*o{9%eG5p<GP)QLb=O4>FyS5L8!90So?sa#o65SJdVzdt5-KXkHxXN-+9Qa)pGKPH?O&n;?O<3V;!xQV8sz%bWls+meKL+a2dr)Wak;?3(Uv'
    'O{Ep!=if0X%5x1?^!%N*zZX+L~yOC|m+t~GGLkMig<u>+vG19YH+f9vurQJbG+i^ii*@d)$`H5%)`viHHRlQr=9m+2Ix+s%&*Ltw8Ra7TfgvI8KU$@I7'
    'S!&LF2<a3LTNBYKeznG)WN}OEd9DaPUeK?<B0__2152hAB_^5c1owkH&5g43ExJvXo7@fedC1@E1&ak3vSOM!AY@tE1qN9`&mB6X;Ag|y06u3}8sH@^A'
    'h>uDBO`uxzaG|-aEk>Oyt~Pm&f$IyOO%ON_!?91CNjeJxP=Vj*8O8NMcqBd8FlX%k%`<nMt7=P$7p`IGlR~AHStp52isZNxQdp-{B@Os_?18j)DQAwcW'
    'Hu1g-jRo5e*%{M{OnnAJKb6dqi0R>d{1CaUNM8?64w8!7ltmaK+uy<6t?vvIEhs?5O-2BLJa`$C8KRRPMlH(QFclMcI#o3|R;YOZ=n^UMX-XX9tF5)=U'
    '_oED1&#0F78<9nTj59S%NH48$}l1}YaYqACZ=NzX$)l4}#;g321+hQ>kV8BYM~%xJ_eV%9>}FQ69yoh|Vf<s(!8K+et!1ziB(jK3gr#R3D{jGtQm70&^'
    'D=elQ+6UnX+J6^FI0)1jqWY+eEPBOBzH|BI?E$7B#HjWud&}}vcVzmT3mVQDcOToj1z#(}gRvF1ggFZn<9$v7Xb(5m5qBjD$+Vha<pntgRv6?|zmCl4&'
    'g&l&is<IEjs$fB&t7^~yt(w{%18xS!a6=wfE*nJbEREr2Jm}kEb}Ji1?A|S-O3)Wny)J4^tT3^<dtN^*hK^6qvRpVumgVv@AsYuu?#p+A5^2ZezOW7f)'
    'Z$VRy%?yYpIAv^2CB)DJnMAHd^?Ab7z>8XSr!aiKv^-%22et5v-^`!KkOc?0RyP0Iv{|e?wJdisL~xEQPo~hL|r97h`OS{4|Tx+9_qF$Act!AY_8vqLD'
    'x^zg9PlO5Ur0ByAQO!o(qF-J2*Vfr3_A=`E#K38Lb4PkN5%reOLsb^EF-zHlN9xAoJyU(eyAQuQCj@$IJ*|+t+00?D|@qZ#I1`GGq39EzSX3zUIz;?D$'
    '$l^kl==A}3+L*Gx-hyVv5^u-j|K>)Grz<>IC{48zvsLsU<8PgO5s?^HAKH?>OM*#&Onap;1<w%IpT{{*|HTIaEeCKZJ0j;Vf`SW*y-P*U8+uqJgeu8LT'
    '8am@G@n>bcIy6oXt<qT}$7{C>AJS<petcLuJF(37^#@N2G>TP59#%fmVmOxSvS^`Ny9OeL7$GFP`SB|{~QZYt{xZblUB;37V@D(e$cQ;55ehUO&F_gc%'
    'k()BjZQLYJY(V#r2ZHasbyzKrRyR&y88%+)sR3$K#Vgt1Gc?}a^~8^^C*-Z`DI6qwuBQ+&6cO_>#9{n}zUi)TBglA`5|gT35MQ~8Gv?qPCt%^WU1@=mW'
    '7MsUieo1zy&oLHJRsjhTH&^F>Yv+o`PNO-0Jn}PD_nrywc~GQd$&x9XLq+$s<u?TAn@Ix9vnN=P*y5l9&c)x+?y*bzF`+NlHU(k82stba%1%yq=S<fX4'
    'C)ShI`J00X8ejjINnXW@ya>D&uN+RfQ!qwOdp|qq2e$TFH2%mdhg(!!B}g+?LN|FT$;m{I0zgQ@qs^!3r!#)yiFf4S!v)hw7a{4+3HZu?eoRumAqh&BN'
    'pE<NJr3PyQ`R73LFeUf)o)u%3rx8J>srdf4)?k^Xxge(-C*9<JBJaXs9whx_&LxEuzRoQLIlczxS`ecOJ0Tfe+wyZ#E<c!<~PJgnEl^I?6tyrM7r2fy~'
    '~;d(tB*Te04xL*&C%i*zH56kuN`nKcxw&VJ?^?Zf<)8}DXE{FT|f=eIV3*JMR&w2R4ua}4I*N4G<&ck{=Y}dnnJzTGc<9c{~Te-fiT;8@_X48##@~|$K'
    '!(&_zx9j2gFcGBB!w-JFJZ!r@49D?#Sg(iedf2ar>-BJ453g^l*SFQn+s0)!iI(Tl<x$rTT;G3rJ+%dSbp7oz`Io2k3t3Z6kVh9X>lZ3pFGO}(xc#zj`'
    '(@cuJK)iE>XyroW{`PE%GP;UuZI_sXqV?3{x}c&^>Do&j_cueJ>0K{7eXP_aUPcI;q`6f`nGX>TTb2NZf{=SeDe1<zq`NtKjw+|Z{Ix5KM!}mpa0%|e>'
    'eYq^Xl~^{f}<Gzk4_TeSGuN-OWGFE8hI;-Mp`SR6c6+fAin<qtfPy*Ykq64{zSR4SvZ}=ieW1zq|W!zVPKd`6PaPfA{v)ALip;-9FxZ|L*<|anYN%ujg'
    'y$!Kc&Cn;pxWXy1JL_U-%IA8-Es-~T;3`N30vJKIm*{qD(6cefAk@9+NHW50R%C(n%^ZoZk9`q&3AJD>W3G58Ok^b$k#*HZ+2bo19!QTp(q@ywU+9%o;'
    '<;^x=4Ki)muaNW&c-~B|3*Tv|5@VZaGzkBlP$J>X8Kl{?JUw--}FaEo4zkCUjP#pfvb9BBx>Vj8Zx^Z5A^X1*c!;{A!Zr?r`>ytm+-QGXkTpw|B{`TbE'
    '{gZe9e0Og*o&FCx`o-JFyZg7dk8_G|e|+*{%EmVjk8fVh<Ike|Z+<tA<7CXp=Wib7oZLM;xqbWk$v1aD&hL8t<X3mMKR*8O<Tv;4zWed+r-vth@x^DKe'
    '_WRT{O3OWi?MwCMd)FFG<la_Z$7)bzx(d~-9O(xyn6rR?fsM2`BM8IWcxwy|K{%9&+mSGV~6kFKE8SU!;`<bzx(;!{o|Xr-~ZV@JpXOoz7^PUPQWjpGx'
    ')#!n>p|w-F)`HO&#*1o6r9iJ@fPDIB%ZMKT{+5V*Z(`&ht4PQ_Fe&0&nzRe<PCiuj21~&KEC3Cz=oXyU?FLy7}_IhB`L?#_#i4U;Z{!xOsj2olpC9{5~'
    'J|)tBLs_~_=-m-^~{^JOUW^KZ20^S=2y6#b8GUVI@S*^9poBVfKi+VgoYo`(T4-~S?vlKGzaeLnB2uft%O{rOcm5I(y3?catmGoSY&44jW{UValvWgkw'
    ')2g7so{QdpApYJA(&1E^M;NvHCd-Cet`?rtxZzdu9#qaLl{q!U*jj@h_y!qwbkKey}|I>{Fdh^ToKYo9E|K|3_k-hosU+-T1Kr=@8=JUI^Ki%H{(~ToN'
    '=kxxj+qc0=V#IHrKRmvB%f&JBH(%UOntK}@FJ|E8Z|?8zUIl-MiMV<3^X;3rH%`aRU*F%p<y1_{&1d)TCJpnLSkGU~bri3N3A*|04?o|(kA)M{bo1ry<'
    '3Dp@OxDepZytYme`^O~)^5Ih|NFb2zI*rn{`(u}@8(zU-_FmAr^Pe|dw7hiV=iyLx_kWL?ml`P&yQ)n`Skt6qk|E%d-I!j`8_emvvvRY{(IJV%=OJT?|'
    '#ac$CTf^_;D`dZ2xVxZ~uvQkigAfzj>Qa3re_o@h^9;?_xlL9&WyPbq-yS#r(243y+htKi)BDK_fR`y}SRHTL$J=-2dkBU+w;lL<47@EB<fi(t7<aX80'
    'Qh{_g%edJ%+l^YZ`tKkxtP%^#8pBVYbRV*I&9DdgvJ!hMz#>a$*OJ}+plCz9(0#dVwrt{3#y{X}j(PSn=2oRF;v(MpQYhjecb{Bw50KWAV3b9Tr-XRrK'
    'ocFjL$|NL`y(m!WU{d0ELKWCr)bN1UmXW#wv?~L=^?;o#XdQRE*v3S0km%hGxcy<5g=SNQ6{Nm@o3IQ3s{QmCy$>=Aqef{M1-R<KKb1lDp|I>GO_i-|R'
    '`FXa=vOHPl|KGg*;r6>XkGF3hA3l0=`}%*MggkG3eD~z`$uy|mzP`PG{bb6(VAqqvzk45V{LfFm`@@r0x9>w_oKO8Iew*v!$ydL9`J&Pdh0Fcj!<$L4u'
    'kJoVyv(bn1b_PE7tenFtJ&?({`e`A1--s|eLgBIy1OU0uU@^oe?4h4dKEvHWnOsq`pqf6;<x|b#NE^A=#Bc<&9A<E`Kxa~`~2HaUwr;eRrQ(*(id^+`H'
    'Pp66MS?tC9?B%GtK0fsN2WeIdbn`J%-K`>*9a<K>@89D&Xzgx9=Wro$|l`x0}a5{CpRT?wjYIzI^_<uewX(ojw1!ZZAv!WL>Aw`O78pX#f4A%T1qu@#4'
    'kvUw`q{r!T+w`m1k0|N4t>|J$#>L~nfjG0y+u_VrH>pTxX9&Y_#+^825DeEaZ8{Q6)1;pT_O$DcoW_UvE&<zJruOMCk6{`+TdAD+dz|MU+xdF8`Dz59='
    '^3~b?NOMMm>Jew1H_w<LypMJdD_xjc2^|qhi-~Y%>uU|cz)BDrIv&(h&uYdO+K0fx}h{xCZ@x0{Yhd;c1y#4*hZ=+{_<wt#S{iu1xC%2RBpRatF&wBmt'
    'jSmVKy=8y)M>7=<iH?7AP6UsLDUNad4?6zv^X>h!IR9#L*m(AQ!hiVWUp)Jx=l<0%=0JS%>VB??clL8GqbZ+W{n1Ol`!T8d;ptt7^V|97*?Hl!+xL$@y'
    't|*8Qw;OeoF;XA{N20Pe|QFk{lm>OU-IVR;r*ol|M)}xc;d>lNhL2{{V?_8{}jYFKPGt9$4mLR9KU#WecE6B;@O{!_4&({r2gv9)8p5_{9m7c_VU}8fA'
    '^c`AJF9IpT7L`#q*bc{<fE2{5n4E*T4BTF8y#p)ba5$|Jz^I_Q|r$|Ne5R|Dy$QebV22`SmY9{qj$rb$ndP|NOIl^W~>sef7sH_5b-{&tLrd#kaqD`A4'
    'hs`R{+8vLHxf^2sS8Cb#<O?(sIJFVwXk?(cp#uR1CE?$rm>5l6vRpS`-B^Z4w%?b+SiXFp8-_w6g#@V5_detL+7`QP>A#pl2NZ~q_n<o4$`&!%*Fxclj'
    'i4E)*e-c0lBe~Ewp%cX<=UGE-V|MdH(|N7?V54-sFFJFB7+1J1Q>g5Mr>}zikzy6cUe)IgBFTVc#+pquY^KU-=^2=|(oILc!hqoz&|A+T{{pII>de2la'
    'KDg(_*T4Pdv*+J_@zsl$U%dQnP{WJ3dX7^n;PL$*-tgPsOv-)!|AKt;#uqQY`R!*ffBVhzKfSwsT-wW0&byHY|M;QH^r9<;2zq_@yEktYudm<y?sroVJ'
    '^71=ySpdXsSVxa=XVcW^YHQ1D1Z3)?YocT+|{4(_y2hM0OAO<Jq|Fa;~-7eIK=vk1NF+k|HrBP{NGSZ0|XQR000O8m!WB0(3wVPN@xH81o!{|HUIzsWM'
    'On+FLQ2TaC2yHbaO9gY;<T}V`N`pb7XQdUo&JhGBY(eFlIP5I5=cuHfA+pG%jj$Z*H``ZI5Qhb*}eU2AwZ9&ahSQFN9-&mS~F*B^e}j5a*mgXw9^!iH9'
    '>hJw29WL;oz_^2bv5b+6q$G3*5S5Lo6s)q6ktsa>^dy{~)y+s(h+KYV!e{@u;zH{~|nyu5k!_U^-nGkSgZc=zG{@%-iO-S6(-p1=I`?&F914=>*T{>AJ'
    '2yT?Dg`0T}p$Gdm0?;c*i`EdU_{`heBc>m&ee}3_+-+cG&i}K>({`>QAedpo+!<*Ol?_S-XCw_SU@!@><{N~GVzx(35^M{W=o}asatx2RZK1=6+>$_4v'
    'PwD*czoe2hvwyjJ`|*6~-)=@;<>vFb+U=MxHL7Pc`t4YZ+Rf)H@7uH)^)pKIw!OURyLNqT+}6x!oKfGmtumVD#+i8J87*B)x2tQ^&M24j#lmCS)^Oz8x'
    ';5<S3|m^aiO2SQhBJ-Zt?ATqesj!m+X|!en`6w|;nDfcS<>xnjFzsQJM~g&XT3(g?F`QQuG(&wFP-;YE&FR}pZDD*-wyrwIN#d)Jdt-A=WFLrw}WYq^R'
    '=8-zwN$wzL#_Dx0&J6`}ca=d8c*evd%*bPhDF-KW{UmopfirEqrfkxzDT<D>b#;W#0PI&?^jPeMb6$mTuF~tdkzH=j~>ockSbL@_I=xd$zMKMpbV#Z)e'
    's>>o@1RorRIsZ|~bx8EMWn-4>RtXwJJauaW8*jm87p`Q>rwgYUGg9q@aeQPExXwy_39^EmssJQ9(e#jAyp-ny4qO6~C)haC0HGLp!#+?K*fzi?KWA1~S'
    'wjk@pDGg}aD>iYGw$`Ezx=h`UymObuGq_J)r)9fOtvs_-u(b?8~+gPD7G(6kWJ)GKQy%P@_>mpq(%lW&mo|*Hg+#rcFp~Bf|dU82)NR6SMTs0+@G*_R&'
    '&VzH)t2941WgWEhE8dvQrX7OOi=AT>qrqtU5PCrt$L`R1X0-LpayxnE)bUs=2w`eN&vq1keQC+(^qwwOd_$jS1@(~JZl`{(itNUjw08K6M0Y9Q?#Zacv'
    'cgod7B01U3scEURM>A9>y*U+a!!8tT*LrM+nm*0HJ5V)c4nl>q-#ZM@zl0QTESFq2cyK|m&2X-{lvB&#xeNGG~y>cyENI0bz8aGhe?i|$$bWqG0>9E>a'
    'ECqn~E1dIdn#X-pAotI<7&*d1`I+J)SZ-lYYaR^%jTt;jk4tsMRD?p@Z`f$h|@bhhj26u+M0-eR-b=@592pfSa{GWXM|Y1yOdG*eGe4&gN#c&Fk6Dk~Q'
    'VC3cOs&TI)k08R{wG!+CDF))^Mq-Hdy6oX=t(%IB2oJm<E~frb>|;q`13Z(nsX&%W+>M;*;`I1cB8i|cTn1zNX-BR;hrN030ZMeI@3z!H}k0?2a^e-!i'
    'KF{SFxb(k!J?p^bdk)&F?#u6Z@N;4&t#@d||<P||wr9*JTLs9AxA7-WnQFUI&dt4;~ju_-8tN6&4z+nZ<M(XnVRI`!8feQClSZ&vS^;wc-g?REAiEhqW'
    '7#@jk4o&yYNK120kp$NniH9EPw<{yw*u3V9v^0n7BXDO$7M_je_c%m{f$+UH5r&5iEk@0<V|vs*7qOEw8kQDuZTc0EcWpvs{>TwIFTj^p^NwkC&q$Lz3'
    ')XnVf+0#a6^2@)md%sSa0Ss!9xG;cmcAky+h!vUhvY{#<d#x=b3<-tMHfHH>8Rb8I2+9r-xmTzls6c`gDafpjF_wU^NjTJ=NEh<dPm6YHDIiRV{=BqSd'
    'nu}>nWz2c*sQXl-s4(ACj<J{6aGuVtYqq@&f~|cZ4s$J+xs<QpF=}*bV|2qWX1axjhmMre-1-4IN=YWdlh)4x91%nXaq1y^F0g)KNOa4DT#3_A@M;iD^'
    'ZoC-k76q2)8wAvlC5*?LBr!TC*Q-o;;}d_Zbl{6(|dnq5ceuXK%cL_->$k%&ub7}G&9X`xJZ6q6ZBXxFkywL+Eb5c*lE%CGy_=UZQz){$7YE*88cad@O'
    'y@IIOkwA{sl_mLEX7ZM9zGZwf@6zj+1h4jp$jV3=u)9g7GO=op=i?K6mK7n4O3X{nNrS=R(2*I9u*RE}m6OYlhNd=8-OnX?#?$SQC^h4H6%Evy!m~+<~'
    '#AQXY_h9p#Umq>z$&U}-&kDpDh02+I=d=drO&QD5u0k@%zBFQ~k$vYxaN;_AL)@3|!8eY61>fl68)ILz1m8H6g$~_ON)C-OG3g{yC%j}*N_5CP>4l=Xu'
    'u8hjm(dpr+ZpLJb}LzDq$eipguO|#Y&ur(1!<OT$NRK_iEvJ;N7}u^ia1D(71Of&T9`yoQuB(1Zr_mlN977`TKwj)yD8oy8i1Qloj_^&6CR=ih$ar#uw'
    'E52TPdCb3tb7dvE2H`X1xU#vLxGO@<>-Y3x-ZtV*SK02Xdo}TaL2|u)r>6IgYr6d3Tvc<4A+%YpBf@jb|k8b<T{pKpKcfS@R#7vF?QS2by5%rker3I9i'
    'F1!bm){-THgd(vIqaKs`fiY2|tjQuA<y>Z=@^GxO>1qD3Ahh*LB&`kw?3Bh<e`B3xV9T!{@F%lvG`a~AVEY?;|EKPH)@jD-$=c*F98nBXCW?k9PIGaii'
    'KIf|1BY-}FfVpn@dBP_wVq!S(S5L`wo(K-tFfNFd5NsI4^-g=0|E>(3&1@=>>DaA)wgtABQ3S`@nm+;VaDGU7gg7-=phWOyU#Y*iNNw|%4jWA@7O!S>W'
    'VX(i2rhbGQ{Abl}-iwm4a1Na1uihJ61sOf|jKXH|3X5&dYy<U-bYLa3_?}dw*@9`00^tTl52;4_9uHyCXA7rqL?YXTai%h@kW7#Gx#HDaV!iWlX()ABU'
    '>>o<Kmv327+`oDLiBw)C5J@VttFN5C~B4-ON_TmtU(bS`j!>MH2Vl88YVsZyj%znVu!w0=oO<w-_<6|;mnrNDUQmpj9wU~XE@?!xZn`y-mP6<U8W!_32'
    '^ROylrE`c@`$s8D%Ryh1t(=6U8+Ql~A1qiuvKXlV5;y?`Bm4&Mi8$7UDyZLJh%gXxI8+F&FuM_=(oTfaX*TEaogNkC$rKa!p@Fmpj%G+S};1#e+)^tP0'
    'tmezI8En%7r1x<@r*9($NESC)NGn|R&?F>Ni6X%%Dm`iTkCKH>^z(M&y<{bV$}kXFtUibA%~o8n^<j}Xj+YwI*{XAVJ%^q)C<;~?Fc(F0=xX08@B%zC1'
    'H$i6_~&lslD@3$=eZAG3@7#PrXFMd7farH30IBrWTC>yH#A>HD{N8bi>=s^n~%?#POE+36wy`n9O+z@+}Z8=v9)2^`T9E#W!*VuA~Vf^Ue?xVk<VoV)e'
    'nY_s(={Ltv@dz?XN16^s(m_iIxY>sNnI24YR$EC8iCldAX2Wo)Hp*Bt$t5oJY;y|kIa##BtkH324{ip>nz##+oo{D^(Q+6q+Q{Sx*Tb_sYmRU^uY?Lb2'
    'M6Ua86GrzP%3V)Gej)yP=Y5;t9*E&X>Z~|IfhmDFo&lSPPs6hj!+3T0o$p|rNgFUJhkxmoS|mX4##JM7BZ}fka?7j(XqRVsYl1?7;x1zzp*0Phz;9V5y'
    'vbJNdhe&Hp#E#(6&54*{Ws<D_<6)M<ToP+P2V;BP(cWAxBobq*$<M&l^)YUwef6Mb-u92$S~+1n&_hZx1Op?SQq-FlWfx?9sH)w4FYeC+&wwhdA0;XnG'
    'jE&F-x;@?KT*kZBcShek2i<zV!ISSz;9(_!nfykXqT)}>>On4Y%x+4`haryb4#aj|`5GmLbjLLOd$#kNr)HNr8A4tdBPWoRdq+wOJS;$lbty!JO1fP=j'
    'g=P{)5&K&c0dCpPBlFS2}!NbmRTfx)1j_HkcJj=g*ZJRCzBqKqx;$!v`uz}YKPAKo)!r^`Fd;75$9`D-5x`t;eRZc^#(;1!@tm7~%F}btd8n)@>Kf^Q8'
    'D*BehnVjvqbr=@IrK`oJ#-WZglpuyJoS`KDp-PO6$(cNgPsK8roXM$!qyxP04c8=wy&39vv8gA+$qyyDH_k{un~pRW9iC0kdZA+(G3BkJR1s|+>g}b(Y'
    '~Os|zKO^1dRoQSj=m})Im7eSR-1L*2@$s~MVPl06(5^@AerXU?pN7Fk`-sDNlE1wp-ya%GpX*k#rUWzc<`KJ8+FNQVH)BaRi3-eN{MvOmSlE)Lb`Qs2M'
    '5j^^{9Q+iO!P?Z@;c-=+vK^GfZZ3>|3k$j-NLeP(~)SbcTBJQ8Mr*P87F#_KiK5Xl@M>=DXtE1@l^~vBbLzK5`E=jWT@PCNi(AWpDFgDNy9HPwxz&NL_'
    '_c8ml4qH?I#VtS15$(bsm4Jfwg(zSeBPdxizdu#tC$dUEYE-bj3|93f}oUSf3RC<>g>i8HF~w2|l&Z&Z=S4!th1skY0El@yz5l6j3siB0uVaOa<qC|?|'
    'o-yt&AlQusy6wU3!QpEqx@I1KJVRifrhfd8QP%otPgWALJKEuWfWgjt&irOTinWOZB+PXaw3#WcPHV|t0!l{3*T17a+#di~%e}<yaLr4gL!&r6o=%u$F'
    '=^}<FQ<fyF$HL^)Ajqnmjr>aXb0({}2wt3TR@UhZbuUp6j=9LQNQV(yoCHbF^jDl!sWhBFFO+neCPfW;+_FZx1hZae5+*%_&|x5xFlo(4<=(zppqk{p-'
    'O}U^|1i}|!fj3QHaXxY@wO%d-LsPY*Ukw>**QaP^*p9TW>%9m&yIji!jP~gkF<(CRH!rDtlRW<5%;XJ5IKGXJ`c-CK2clJtL}?^ZYdsB{!r05kseM#X3'
    '?RN!_<RaLM25?7o#2*r7|pnXNTqCVORvuns&AzE2J~DW0|AEpP>j@L8HZCD=P@9bJ%3=9)jB5%p`la5<TU+ayFiuhdDZ-(4L1S;35{<ww!|=>WK7YEB*'
    '}S1ZkrYbX)|B`d$t?E}}(!MKtgXampMHA2*|yADc;<;Ub!Sw!ogaML4@2$zpm;u-cI!^E1+37O7JVrQx6L8Ze?oT=X2pgsyjntXy}j^9*6jtHsT-FxxR'
    'uVlgBOo0iI4Ye34!49?5~S(pwDWwvi2l!`5<MUKHarxWTtL;dbxL$cXUH<d4faaESRh!0O`mG3Hhn2|XXZDrysMaUpGaF(9k!W%wumR7By99?I~o1aC?'
    '9wLykJ=GJ(O`V}#PdqngnL^+e)?ahljNt91To;ja)mG>k!mx^6R%eJjT4v^F2*Wxe26j#4wYG4|pYzCOWjoBzP){xkP5g}HjVp-&;~B}vT8kaiXDA|HV'
    'b!0Z^pve=I9D@!)2xr4p-6F-U^B!jJ81`*;e=4r^<6zVDKw##!IO68<Y4{8l~yqT81>*^gOOGuFT1NJj<l*^Ph$jcmV?eim~##Bi79`cW6#ILmaj>UZ#'
    '}pn#YO)x!X~cSjwrV7)V!75&W|&M=X4v<XDIEVq>8qEh7FFCIl^ZMUpdQ&(6cjHCK&7sZ}K@y`_`+t@hv-vVb+4<`3O1qVwI499WM{XDny!_ktQKWf_&'
    '*`sIww_%=-cB?&rZMDOD`#yoK518Rj@i9Dxd1D+(|@9K7x1e4io8?V+IIt7t>r?xr)`e0NzLs7Rk@vhX?{R)`R<unL`T_MpNl6k_0XRq^t(QZp7qyu3;'
    'B`eyO+CK_UQE6iNJ5;m+N@sB#oXCB*f90*Czs$c~p&bOjV9O3E%etyKO*kMoA?CA_4lEYyAa%^P{f9M%z*wgi{gXtfo3YDZv*=Z>`5oZzuKH5No{hlF8'
    '<YB33es#j4sza}{F6+V7<X2&Qso1}HhI&`|CVel=;T+TMi@dEIZu~4~;z$sb#$(;EJ<`T`A<o1RzwtX}J3X|J3fqy%iLPE@J0&^deun71)4}?q<4Ny5$'
    '_IS@3{g8GA>$0uZ$x1n5t&{^Zkgt*i$~=+USd2&!F)EpNYTwQuF(976x}nlC-Mwo-^mf5Gla_~N2bn@?H$fwJ45)l;&PcG{JUD_A4Y0ot*D7-h-$H|!Q'
    'g(b#F9Eg*7aB$5DE_H^Wf#59oFuS^23yIK_dy`s#V!>>#sI6m|=d0Dxo|!zKA9v*3NTUW{vf7EE@)1I==!}ZMno%%!QUw>d*dc7Byc=JxnR1(<ghQ&Tu'
    'e{L(w$-qBFfc#9ByUSv5`nNWz5-X_|gfMpE%mO8il#@FAVyK;yUWbA}QNEv2A}f>=mSemFx3oN~UK_^uATg8!zsrUT#P)PXk=dv58x9>R*)bCXtODV}N'
    '=DHg?&i9H>AcKIrn#-T<_bqU2|&t_9pmsIQu$5=l=vdJugR*3<hZ&Wkki7p|d(?jG$xz7BA*YvPx>f}_P*EL_7=3ud375!P(9$Ifje^#1W^C>UTb$A%6'
    '?Kve%=~3fJO+otN9zQegA^G6va^#@5;~qc8M7@S9{2s}VLKn|=ILrY*mz@H2qWLMEkcAueg2>49hK;azOrEge)_e)xaj~IRk<`Pv5Qlg$S$3SBhJu!B*'
    'f$LYO(wJn+0$>A;AZ++t{~nOk<xqIOsM3rq3w+)RD!9;P7jObr&mWbw;iuD#Lo|r1hEeB^YbVtO-y1XXBF*ZoCHgbp<zXKk`dGJ-81Kj<s|0M8)<KwQa'
    'gu-5;{8bakF*eZHC^&{+hB#8f%4XMnfM)lwgUK=HW!JuUp@XEQoxU*=%NY`@bf+F;yx81V?fV2Ep)Ec-D?1)3yyO=ir=Si%sykMg=WORPceq)>YukDvw'
    'T3k=|~FsK$Y^EEu?tW|-7O&CjppV0m}}Qn4L&hI&6GT{J&RE2oSt|Ip-Y%JAaJI&Vi`>bNr}`5IO32G}JD8;4lqW->Oa;&{V1Lr0nw9OH@mCttO}=g9R'
    '@6JgUe)LSxE@G~Y~<yb7|hD~x-s-g*<5nFU-0Rtgvr7Dif*(7V_NQ3SzB5eh4nOPCGuPiP0zGM5&b(uLFxm_H2pCpe|61X_JC>#nv@P5O5tc74CpvUGj'
    'D<XPF@2i%Rg!C}xea5DZqGGW*;A4jb;wpYAiQH;^9=}u%t$}!U(~;XYnb@)9$o(XueHci-^+oJV#e98Hq;R-TY#5r`F<dR5JDOWLOT{4~m!-*M^$tN~@'
    '{fBv2YSE?sv@1GURfQJe3r8bPKlZ%w4B{_>eB4i+<FVrXCY-rFJEOLC3Sf+OdTbiC*u{7>v#FP8*kb2LXy~`)D6@G5?hWrT0^2~9gihD<LwtR`6z)VS6'
    's3>ann!g;gp-ujShZihZw<7B42M3W}GR9zlb>FAnOoMuvCsA+0R-KX*@Ejh9@}VfXL1vJwrJ!(pur%)qzt6sO2i@jE7wlaGRtvj$E7h3yE!2DqsxpUFU'
    'dtQroIrgI2z33oo2mF_uB!EsV{2IsZ(L)dw5*(In`M={Y^bFrAluZ&%ZBU}uu_a+H^5q1$c3pP#M1+0JU`gH7^Ys<JFFle$fKN50Ppk~*irLW@Yma9;H'
    '!@dat<!)IkhiYIR;W#t_89@5*&$~k=5--RKbL(%=<4fpJ!!XioL2w|A9E{iibCb<{q=!Nx9^2HI%JK`Uxu%(2sAj7nl@I1{KI&AT2n`VF|%nFiBkDv+n'
    '@`M4(`DZfFjE{Ed*CPzvl#XHtIYu9TYhKTbdbZ1RWKqTbc`z~fs<aco(6%O%BA9?sT9Mg!hESNI^87BbA*Hh*XD9^8hmc`4$q724nc+FhO5oj$x<_PsF'
    'A@`~6|X^9As5H222q79?@D3h3IV5*FLAG;5#J-EkfRL65rw0HWv)4(kjzlXU)~{DNnh5Xz_uYM9#XUr6StR|B9I*(20~(IP<+K9cyx#4G<gbo#+r`(I-'
    'Qy50cPQ1MC8|*ID^=MbS7$MmQ!knGuUyZvjDzE%-fkp!!hCN(DPU%S5-SQ$8v6!mBm58As>>i%!RP1;zvru-kHeqbQJ&rFpVk$O}a<bkHe6|*#(0NB`J'
    'wKqfT!&bsf0g72Bea2+zCTKrc>t(k8cX;B?;(Hw;%Mg?DJ+anH$$;aIi!J*P+2YhVJY@zzu=5Yo^pqZUy#kT0_eKd^`nWfM2BrjPcF7g%wI#QdEsU>jr'
    'XNESeM9Ii}K`W>%Bsz8Z^b_$GS0}U2PGM(%s-cM32_ehM#wVxx;c2Z75W}8Pc4ztT}CVt}*IV%&lagkKzDbxvM+iQb3M7D!t5QR{{8UiB-!7J#I)HkrW'
    '3`o(RR->c5A-r&TlRN7g1*)!gR2dL_*fXv;a9&Mn+i`Oz4opfSR6D_m%HD7`fPx5-46?Y8GaOa}rBg&-SPwC~yu8Mfuz`HB$R45w%IZM<rg|!;?2}E*K'
    'uq2PKOg>2yg&><WI=I|+~wv|*!YKJkbA^8O19ZbbfYMGtpO=v<(e=R2xn60Ag*JI!!YsIYg-^RN`pkaa3aEyA|GGFNpnX2Z4xP~^H-)TMC5B=P`!a(;z'
    'EvicxSZly9bSv0IJj>=%4)0>3ovuBAw-&tw}hqBzya)C(>a+Iz~NFwa{`oQ4YsNjxrXjfRq!710xLxGL;<a1~De((D*1;{2xxenpC0bd-m#&_+Imf?}f'
    '2h2H%U#r?fIhHOVCIq3uQ^BkaNEGf%O<z2cG!$(#T>T}LLD8xKMFq_8ow%U7I;HK|<6XVM%S(g_uI<&mT!&5(Q<*>hQ66xqmJMk%AnMmdVm21Pbv3fZI'
    'S?PAU@mTAFuaB7JfC_6&alZ>PS2SrbG6Qde3vM_>OIdN{F<>hSW{y@poUQ*&k4TFw8+}(+V;Sn5oPV%wbD4@g9imQhbD^|K&6&bi$b4#wpa?#0fHC=Vh'
    'iIcKNb;(Kd3suyENO2!^2lHZLI9N-yvNrIq%6;sTRq-p4PWMQ>p~f`rT$MMAfoQCdu{`WpEf&*F2cc9-UxuBks-h4RC6`Hl0V@8;Wg>2}83o-S<49)Om'
    'W8A;>2$Ikq%v)%QJw*X1LoJHfGDGyE0h9<^mENZu}E{tf&==aJ8#ya&jkxnF6DX!9F>CrTx7Zq!f#>^Jd7Hvl&(?cM=1@%fh@BMA&b{k$)Qxl;WW*W3N'
    'Y}QO0ZEGipvxSzP|@;I9XxfG0l<2bVw>$I471G-XTkyr&w^ivS;8F6BY3W_{8#6QjAwj>JjA7tX%AP2LwQkdR$y$A~Ceg<_b|#jz_!H%FGDnIvlqexN3'
    'P>Pr8xN=iS6pi%e&lB_zD3S%TV!uQAQo(dI*$_~<%Tmtf+F%gb@DL3~XkuMnHHY;TX|((Ne7Be<rKzejp*R0s@0YkC1s8QvsQTZ0k0o^PC&y2|1~(b<m'
    '3f%i?*Y#uo~I378j9E@lhxZcE6JQYNmL?)}Zg5jWBx+X1?tUO$imh250E=WuE*$>yFrKnPNJz7fMIja}crVadU+X-z0Z=22_PuM<o%@5FdYQaEI3fl^n'
    'VI@6|aeY$Ksd?x4S3C4az4-X@wQee;l5_@Xp`%yCcy-;#zy}R;hHIpt3d(448|Y356t4k1kU=)zM$yA%ki$hV@EAZraE=_Ch-yDAtwS@LB{rkMB5W>HP'
    'sMd;HL6@)hIURP=g(DW#g#*YK!8T^y)Z1M?;M}rnL~qtga`s4d*DO19;SB~zase@nGrH1Bnn<d(XSaBwa%kn-&;Op%isHiwtJMxx<tK~uml*0XiAR~WI'
    'Pn5S>z8K6vy!MF5{n&G+4eSZLrzcA9yF^ikW`qv=D)nQk5gUQ7Xp3H&Id!!Xy=08UQqlad9Ze2L$8dvTxgETufgJ<6`<Lm&OTih?-Zg5w%b`wARrsV}{'
    '>zQg}%r#A;aH*(0aRny-@*t~=S;X(Mi4-`h45;93)z)XY8Sr9r_HXA4U+c^BfuGikiIQj@)vaLDt$80|RMY!T|tk;5j(N`;XX2OPM2dtYma+U(In_eG<'
    'au4iyWcaV&(WrhGqM&~$j)OwgN9Bi`DvnyW)$>`F-EXq5Tj2JA#x#-r42dU`tok1p}o4g>(5z32C;Fva7vac=Ej4$E7aNW+pK`JYU<;FiMdx$rq<nNef'
    'mQUSE<8@fz%&mBJV9dpe?!YrDx<iWM86}s&qv9pt)w=a+r)`aHy*f}qsQFv%IxL9=x1OQ3D*Zb9n6x@MJRg&`IW|4IP1<1Fc&Jh6{FG@^*k5CTW~^lO0'
    'ws?$s~tRPm_vp6?z*>ghOO7ap(qr0$y>52KBD3+D)OV?t@|k8_2YOkh%=KsqVN^^z~3ivicB8nO-}h71iPiRSSv3sZZ}^c)J@KfM<uito>IcS!Re*i3T'
    '_Ku;gD!`piWmjl;Hd>c&PsMsCRJH?Qp$=Ji#1MmpeG?XhgL`HdmYDGJveF=1^rh8Z(?SIC)%bv_rdct<er}43`>h`vr$9jgD4>D&#_=<&WV7@r!dRkA3'
    'CG8@OQmsJPTil@Y!+yeP%jreECqh+&qWz85DQZsaO2B^~p7ELaZ7OyE)@Z%`nUU^@`CyuzWwK})&Wu*5j1HSsZ9Jiv^I22q&h+k$x64q`!~YtMsJ;iq{'
    'lPQKH72Z120bT-cRO*Wffwr|2cvuxnhTVE<M6&z+rqvC|SL0Kz5?yq%!&Hhz#B?Q-ln=ofv;t+;SeexH^G+xCF>bl%erjio+t>s3!&JHwsiG%J(&NzTP'
    '6TTjMgGYNC-ozE9QH4HSP+D71f_<Jryd#Q1u4_=qMRqO)!Fw~vffLF2(hf5ZmmV2#d&D8e5?v#R9LL$RhK3(q<+DEDIL@ZQ4hCNItV|@<0KH3SC06>f$'
    '=sS9+-YnpeefrR;qejM%p#6ADB}^w@HVw@4Y1icOec;kRXj4Cw(p#uD6}hw>4c_EIzP|n2TB~^ZgX^4mqLG@mmTlzX7O6R4A&h-9oKf4cnBGn&@?+V&d'
    '9}2tYx(UBMopcf`n|JR~s-QZ5^IUZeW&`FHNBqmo=-BJu`G|jwSsW!VQj?mWfjVXvyT+@?BOy#md>kjTCW}e=UW>EM2-Y%s5*~9jxerN;%e1tlbzfP6?'
    '|$oT$yeLZoU(tiD$BnZq}w^w(-eQw;7>Yp~`Chp*KdpnN9!Zo$!`ax&(JqlXF;-W;P+x#WIx6g?p(@M`9Yqr{8D7E5FtD!UskZL#`*LY+0UcS#W{PREL'
    '`x^8rmd8Kp|M*cjb_-ZpfEUUwF(GJd#xvv~QiS!t$#N<&@Nn+-zE3A?&#Y*yShNrL-fx2UZu#&uc<{Mj1&h1eo4Z%>!xjmX>Orz!tYDdaVCFd6S4QW0#'
    'yQ;j|b$5msKH15YssKQfaHV3+F3D>ySglr4m82?5ucD=9?3T>X0>wqVl|%ziLyxNV=MG1gYqdvLp#$bzF8DJjz6KL5eiT`pS->X?3POn&XjtdF)e2!O9'
    'f_QI4_7u<dux{MqH1rAaD6)G4o)Jhj>ZwS_q$jvti@{Ka&1N})(hK+?X|--@*XQ$f@hg+-dB{oy==-<lsx4*Ghwv6-EDGJauAfa$T6!%3<^!DRwD*QvU'
    'MeIHp8@7M62%MZK!C}S+Uu{iG?~-Av2&_rPfvs?iF!vW^SkD4#cEtucb<Xz1Yj)2g{#IYG@Jph-KFnP%oA_wHfL~h!rrobe;U_t5&mDFaneUy=rFPa4_'
    'sh&ab!_jz#IZgi4}YsoSy7YDZLBj<ZyIqEfgSs$EeMyVJv`bzxT3{(W0R)h@KOT`*PcLo2TEt<B};$=tq%+2G3H8fHVpfhu-`SgQ}mf_q-fADwfYg-J{'
    'wrL(Vib=$ULWWE;yUJm6~`<{w`3b<+;r#l0w+WRC7sm`l9qOzTTTJ1X`g^a75eN|yu%-{%6GEC=D_5_H6^>$X-l{wDYx6Xd0g;kXHBbe;nsHCSBp)E;M'
    'tN3Pt5KF9#3#yaz%4;`&jWU_q9fHUX5!7(LtQ>@D?@I}?u=_<xRIsmZ<h7YYNZEBVoYYb6JL!S(Q0+Vs5ME&#2tT=ipaTaEl<>$j966}dZnNspfeIAa-'
    'C>?{Cdo0dWu-@*QE_;%bhDA-_#lF*GBg7&*a~L|n_SL|Izwv$1y$z^dq4szoN<ODbO@LXLHa|hgU{x^F5D^8$Oy~X`NbveHG(9?9*hzJq7p6)`SvTrc_'
    '9t@Q7f2E@^|h1?CejOy#;QHxNo5X7B)EfU2I*!A$AXSighq~SzHQ7IdK|P!KqMGJJPtBs8TvF+n6LlGZ9e?rSYc#Gdn}f=s?da6w@AI>#~QM4I7}m%f+'
    'x=LtnkX{UFcmIKQeseY0a?$U_`1#}-Br%SKx2@e~k~mfZn08#TbWe#Kb~+Q!W`!Zqa(NFKt*8!6;vdn2orsj}^v3<*&76E2}xe%U@9$q6^bnGLuEp`L7'
    'OHYksh;w-$y${L0C;@XH}(^qIc-ka%6h=%T592rt(@nX604J&ttoq~1u>Kgbm;Ymu}F~KaSg$y`(nbs*};q%}<0$#21VCRCA9=p8l4X^=KW@Lj{Eh@|F'
    '*qwu2mXE_A&VNQcxH@W6mepicrt>7*Sj}FTGHqnZec)jW2)N9`IkiNh-dCMm&|@8)Tp~@=Z@)sL8h(RI!5V(Qb}4p)S(#34N*3y#^G??E3)X2CL|@`&w'
    'o~Qwi<?pGE;PepgXtJsNXTN7X-uobX9(}B82E)MXygwB=#-LTFIsU*i6jg=zw9#QurFu5^cEAc$8;HQrK~5xnjrd?E@s7J&2#38ce7gE4N_E`_9Rmfj^'
    'w;2rG6I=4KQbCMOk)WPD{wLLJit6VW-jzX8{yoNY2gJU1bF-j@=D~94bM|WoQxjw<8aI0)T&8D=6M6Iq70B5p>>o9EyF;y2`@LSi+0@)^X7(-*uQST_h'
    '1YMhm76>#EHW?CNZaqdi-@O~&R}F5^4Hd?Cmp1g6n(rWLL<YY=MockLLrYLh~dm>xMXBN<b!tZ*FVSYcaqvwIC~(A8`C1Ic*!*xA>XW7(GRa5qMMS$4L'
    'q6g1!jrN|yE&KOGq^?+b(Fy$T;?AeFeU#K_FLNncLBtGJqcMy!7#Nk~j%h^X~#k;R^7;+8TTP=<mKg(es=U%f^M<JU-N@YhF62C;DFggSGF&JWy2GZdf'
    '+9~K1%`?QA(rf{pAxhtL)S@^AYv&NU&cRX@2qJkocHHf8FW?kp<&!L5^*+aw?b*Oe;N{DO7~-Sv<7{oKfX|f`hUh-gB>+R>yw}j2b+W2+8dDV4)MNzNX'
    '`z~&WVJ$_O7;+3reoT+Yy21k_GT;9*)~MfX4Qb&trTz89L{W)))uGr<p@I*=k*aIW?$C|v(dEYpz3stuNbKYr#Y|cxT>;i+4fz6F(lt?d;D7PePJx(A#'
    'g4c%9h{2fgSs2E7$;>Gwktfp{)1gs)cfKNLnZXu^-Dr`2o`oEdm^-Y2@ZgVv=xYOd$m?YU|e!+f|MeM1MikDuzruL#KJzWIjVQt?8V#vPmivO7(^^vFl'
    'Zk#DnVy4-!(xiH|w}rg0}HMAdY0>Z6>538J<LP^g@rb;gZKXFM#yx08o<3^~vJ3|n}SkoAUn7nQ)Ool2>sSt~mBJ<s@AirRwF6+lah6e?KtcEVeBp{gJ'
    '`utHsZrQ$<Y#^5%pZbJv1@m;K*9H^Yd35aqvWRAaGS%ZZb4#^n(q-+DK9Tvlp+~I96rrYiOxgKI@+m5d;X)W7Ho#pybXDyT&kD8aZiZbG?OL???aRok_'
    'x7{p1U%bqj3lUpB!yJ3(DX$V<N9&TtHLN>+)<><oy1#5A3*}3NE|-GqQ4V)z<A6V5KciCHm!88%kV^}PLg6?RtLD`lC)@0AJB|{BngTYul7J_yQ)C}{v'
    'p~fx3elT|IEF7LrAvBPGTJHKoTzk$n2|DlZDx`Briw#fb>7g$6zx(fPPMPek^N@eo=QUHnWyDaC8ZK`efztdp5oN>Gh~03LoKS=j7A68x~W{a#2bzpGQ'
    'g3H>J(CNF+E(L*a?*+(!3*<Tgf8L>P+C9Sw}FeJS+@dOLm6Dy^;|M&JZEj!xNKGb(ZliD+#Ig>3E3nlw+9fvw;_8XLZ`3*d|W0JRJ1BXb6EQz+qDj0ue'
    'wD$cD2f6x!k;v#3^$tMj0Ne#20kc4d$$T%C1FDH*S4I-l431f3f-?hMU^oH~An{Zr3UveR$bQ=FXOBbNbBvP8)%_RHBiIX0v@&>kX#1l`;mg|Erhdh~`'
    'k!yfvgcPo(pXUyN3T>;g5hATkgaA%QeeaPUV&VIp~W?yC5Ru65`8cMW}sL2{iX`y<eItzExx~5Y=$Btuw#wg&5&|=+pqnj*-GjSt&;Ot@H3~->FnJ%lM'
    'dbgbI6aKkuy|1_^{0w`rwAk7asCRh&ve{o>Au*jDkY|W(XLWAd-PH-SLhG{P5sv<djpXq4JG-}LTO<(_$dORth~^>e<QPGC(QS|f4A~j_{cL)6;sRYf4'
    'fIIj0&QQo=w#1!S<B$pkv`i6*aW^=a^Z9jjPYcwo#D__PsZAj!s(kSoX(EQ^;C^d*bD|o6OM(RB~Z2G43|I;={~uKa015?PkF0RZPM|SH@UdLfNR;Ms@'
    'urRErUc<>`Cl-85%iychD}T=_GJ-Hcr?=9CY@i_ZCR-sZ&X|1K2j2a%^X5c8FUQe};*JxQ}`4`l7SK^Qwj8X3;_yX#hBf`O^jO1Y9HbPw&&&VLef9O6l'
    'fe*r`4ZU`fciD}v1&Ma`Y1(a6xqdBrz@M{ePE@9+dAV|ppz2`Vx&Twi8Y<$@`U02ew2D3pLM*aXR)Sk%n?TR^5}G5Ch7Q!h)~i`J4$pgMw5`{1`jP`1$'
    'nDK=5Nt`HnRlC%=(;&J>g@~w|;f!2|4f;DB11SC4K>c%irs&C!4)2RUnI5v|`4Lyessb?^aR1O9hB;8M1?a>eicUn_p0X7qV+L^Mm;(7)@R2q{hOpHra'
    'MF9A_<A#9Q{I(sW@f-++1Q7*d!DsqoL_Dv%&jY=DIEo;?3&#f>#td4(6K*FRLYnMEuXySK#$_T;h8^0zN;IQY5zt^rRqAZ9PLy_wiGfmCDeYE8Bq-pS@'
    '@2&lfzzi=NK@e==;(;Lu4RJ^CP!`9QPK(Cw3!^H?F4U{9N_GGHEp_J<cug|3<R1^SXrfkMr3r~4<T2B`bm|`F)UN86gaaAY4Kpf%1teJ0RASd-1-XidM'
    'Z1*nqq*ySd3890>G9C<fIo7Z@cMTGp~)_?nDvXky@UuCOE7kq1rYhL%b8M<v2B8<p3$;g=qloBMblti~w<$`3eGZ32^s-%BNyOEeAX+N;bsi3{Op*tBp'
    't<?8LdA>XXlq>cnlV#xsPI8rP!}MI5RVW|gH+sP#=;0}FPJSSY$cDdYmfCPduZw!^s-BJTBqf2FlfpBIKYy>-+;A<ikiS@0h)Ds-bT3_+)b%#wq)LPIn'
    '#=!Ha>PIyeHF5YS4W0RT##6kl__#A=^()|_padVn(up5DmfOuS^+&VzMX?fDq)r*0iC#_GH6?7uRf^;?xFA?H?1%A3ac{R8weZ165%6x_J@BjpvSFa%Y'
    'XV<!&;ig|skCz3rTNrb0B0{Gpg-{_1@-jN812ho<wipq~%fohNq)&H(kjQ3<`Gzm|ysJA9G<>;dfk+K+E6_uXbkmkOu<D&E!xlg!%R{eIVdL9!MecOHR'
    'l`%H#Y5q1%ZF__&99MDsD2%=rAAVrkma!<8i|}j*vE6)NagI(YPzq2p0-`3(n#j)lyr86n_-DZfvTY3h!&&K!=VXlWb(TpxT;B;zYB(IUxpNK<a916Is'
    'PF~TvZY!s%fNkE}_FD9<|(=%{EgbLVGsbpvYo6gkkIkY26*IrgO&|?mIyw@)=OW>O_c!@2*YfDK%VoNY~+eL92_@_%mEl1TB-{3^#JkT39oFB|7M22s~'
    'YP>SiN-U7QNB(a2x#DuQOo$w8CS;0+-Qty5II7=i*=U5V7lLvv|%XUGXzG@fd>wtEhw`i5`&Ov?2jjU-o<ETb+o(p;xtwFgVQ>6#T~LCrlLM(D52O&(-'
    'JKr=LX=))io&0QXb>WegsLBlUGXlT`a9u&Q*T(xR3ouJxWYCfi;JB`%S!wwps;eFy*JcmW>(V(c2dMc*uB_@!qqsC`FbA^Mv0rcNRc+JxTp01Gp)W~TU'
    '{C7+znE{P!?q)!a@HH3Q3t_VPW^}@2p5KTMpP@q%XTF@m)UWq8wu!MG8yT=d$}$a*0xP(ecrtAg<Z0onwDtk0pHZQ$i6va4Fz&2hMXhKAxQ9S>7cWq~='
    'k^UX+WV;N1;*aUMnxy|#=-e&vc1#DM?JczqgaiE)N^)d9MMQfZSpn1%*aS>VvEq0(R2h_wwWU=r)}p+G#U@+K*h=!$Yr@!p^=x`FL1#|VrmmCH@uC^)T'
    'SmPiASLdt>M5)BRy55VJ1p`Dk|{w%?l;O{h1mK2wH7+!#0Y@l%f^P47j?o+~ul~2Y$GJLWDFD!6ih!>STheRTWl9BNe>syV;Kdry*OPIZPBW-JQzRXu;'
    '8Q#D1G`az}H2iH3Vytx3=e8}98c(*knEy?u_qQ-vDt?Jhq+_03GOxn!yV#10Y&#7wyQY>GYBjc|1bT8Z5@)KGvkH!^mc!Wx66!@845@d?AalSbgP2v+A'
    'PK`9{CNx`muk~Y815nBZ0$O!nHMmv-y&hA-YI*sD+ZX=1c(I<c>Z@0ReegpDYt?&3Mx^3BUPPTR18ZSg!hO(V2A?;aWJ==%XXw0CxMn8(ExypcoLr=F7'
    'nm!oQO)esA(}<c+S)}|fVe?5397gnb*e)b%6jch32ZOs!_gLK_a)?)-tIoY{l(Kd~?+jj7T3M&7Ca-I>SXCJNq_f#$#7S@VFyuIL4WYl9cL)_)uE6Z0T'
    '~OdyHH2rM06{b(gUdQaUK%uaIWf86V?P%i<7nH4Z6uF2eC%R#J|P=sPKyVv!^V~nQNT?CY`D1srWtG)6_OQ&3(KBk9j0MVB6qD(i`I4x!9_7zn~~nvs7'
    '7n_Hbr29dorVvr%{jwWXiDEHe+fg8Tw@$Lau&$qzWbbogus>pcfml7q!2O!-b{3qq$%s_9ANb=4^3v^%t`yxPi10T2o=Ej9J-7tgFGB6hDPD8_f<>;sy'
    'mZx*Y@<S0l-1?#OX&B>4zh+*IK16*wDs`%Z739EZ&fEaU@#DoFAX{4aKLBgscO__rQO-&Pin7Y;K^?p4&_X=zX#;cDdg^b4Dyk>jI14y9$-Q{>E;HVq7'
    '(L+v8FrO77qxlDod=Y!~`cX`LwSGOrN27V)Yzw*U8iQX?9biV2~5sryAipCZhPQDps^(?w)fdwQiP<EHRs|iH6Pcb?~F$8oYujFV<-BVu1?8(6Z04q+B'
    'an?vH0YBe*BbL1sc|U2VP`8m^<k)nj)D2c@<BU)mAm80M#(Irq#gOc61Pe3sFP9uiDSp>N2Sc((+lEHnMJKd_97#4@>@#g-w{%5o<58S!7n!sRklZyhj'
    'o9FkS1XPMEG!{X2n=%As=sJfV>kO50-~35=-2iERHy$o!uPU~F5SEYa%*Kq(x8v5=w4$2=Tl&;5Ey~;p*9iQR|owTOT0Q}rkXfHtF*RKIlWB_dRq-dn?'
    '|T!%5C*BRWPp72>VV1rp|}nT9Rzkxzp^LBpW=F%im>t@{U<}6{j=g8*Qf44ADv*ji$P?X8l49l2ZB(Ispz5MXutzbgUUJ)NUl%OrcalQY6_3@6S}?RzG'
    'c|P$2Ay4Je1^W<OOSBVVQYWwMq^Zpl&#v0s;VKZywvpj_)(nC*@o(N45&n|jGT6xy~(Jg*)@w=->FqtP`*zWxc7&?|IQV~{m>%@$k|ng=PG0*UmAAR({'
    'Iy9K?O#-jmPBOy=DWgEIA<calKM<XHcyhjOPmPUBOJjaAMZ3bLwgso*O;F@TJt()53+RhjE8*)0wP_&u?O}>$5qqZq_AfQIeDcIN@cSTsP|7~;_#dm_a'
    'wA^VVx?D8!Y}6ppN(y;4w)z>dZo{efjWiqKp;<Q~&1OouMb{<GX6ER&+eovSB*yWxZsgUR<KO7PGL^sy70`!T<zqfR3(WCL`jRs%!D6kMoLLALlZc$zo'
    '+4woku%!`+tj_ccGky}H_T5+VzKi&wmFVdCu)r4OWzx!#)H~1+wQnpmLXT#wnHQ(KCkd?n^X}-`$@>-qy>}SWNtTkD7qWdWG`~0S~VO@xEgSomFSt%@R'
    '&t`TEkzaNIx%<oz@UelI%(b#{?UZ1f@{NXu153vzd$CINlV(2t7Hx)QBReo<#x7Mi4;>g?!I>*?ok|#%bB1?>Q^G+bMWP%Xjj!E|Qs{Shz;!QkMyZmFQ'
    '~Fit7Vf%oDg*PqMiUhQ%{+0|nSSg9d<}nSUlJxY_C6@S`hm$kTD7E2R@Agcn^Yj~*G2%E@DmSVM_^0*z3E!tDM9LZp{lD`OdPG92?##|&nSQB<7h(j4)'
    'c#`<cMbCvFjzJ}Xfh5LwgyzV6$lr)^~D)fSdQOYIC@HKJ;0zs;|kV~hBLGe05nVCw-05EODG-L|+O{tVzIo^g!$sMj6$|Op(9%#6MNy;ybiZP2g7C3>g'
    '?D0kk7c0vPfr=DO=NjP`1rhy1l8V$*^F5;z6VYin_SKSuGl5^9hzD;DwC}d>y&3QR;`)ZWpW$3H+WmH9uHiD4D-agxI;iD*qUJKF<syUTDyWE=H$o64Q'
    'iGc-IR@25&2=2zC`HY696dZjjijDF<Lw+v5AtwvnrviF0g?1&Sd~WPK_gCxt3MPeXXFGPlCWEpez-<bQp5;*tP0g-J$l49M-D2;ijBB|T)z4iQ3C+U_0'
    'u)dkIgt;Ru-MTnyboWCHl@9HTk@NM6c3ixJHBn4cOOMbpv}eM>z6`{i#uHiZ0-?u5QHU=Bln0>MmY;jTpF$I%wD>Qp53y<_f9Hh3L-lD-M^dVfqXjh5)'
    '*&yBoDI;%C)&UInL2Wx8pCf1JQxU>>#_kj*~Xg!Ot3a-goNOy}n{msO1(t4USW3Z;R#oD!uo#|dSmx$SQw)Fe=N_LWGJ0&6BQq*V*Und-cDsc#hLscgn'
    '!fQk+idSC}aJ#vU!K<m^v_J-DRNWHl*b%yax`~wl7fl=k0#N3P=Us6mOX%u)Wjbs7wVHhQ)TP?uJ8bK39mr(pE+`4eM&mtp44fNiqDZ-EDQ4f4IylSwE'
    'NOL<Qx`8+ir%@9m^aUkq;Mr1MHknPRebV}<=Roa~fS|$$tq;&Fe~(frduS^<A4-Dn3<y2)Q}$I-PED2b6qC{WDjt+_RYc-(RWK>8^?>vAXm~Eocg_%9R'
    '3?ELQ6tp}Z!1epT@w02ipnZ{tso=&8XjvaXBAJYsEj%)J6g+C3PotAu9zpfOfkG|#mm~GqhQ6!+9OFsP_=j@o#KRxRrt@iE}ZZbEsB~Go>V@*8Fj-zH3'
    'q|+viimuxO0WkGN#V*wZQX&=A7x&!-jXFQ+PM3%PbDHkVWV!3mw|ZDA*hchf!k@6|OF^P#=uN$SH*`d2><vB4@U`zG6k~n3K1sq3A2VWq~1+y@fMXK-X'
    'm683wyu>$ePP?{dy)hhr16x*JY_4OVx<$zCKu9!ph1ove6KpQ`EbULK{(*<iTKo^r$RmX#5O!Shl*Ab)&iDl;Id3|Co!qH&Wcd`0=;<e+!U#yWk_vriZ'
    'PTWTN}gdRW|9nM=lq<RFLj0vTV3k<LNA=A3W@>A9`gv^;^4PDGLCO0=DH!BXht7S=b&@IrBs$*^`4v<4``DujP&xwP?+uDe8KHW+u#R9>qYLr6~ZEGp2'
    'c5wn}(f#9QRW%m94KF&`qB!MLDd;T_d6%exskr1++_ZV2&K6iY)@g2GZ+>;rUA?vrZc(EKavDN7iuzq~mt7R^db?=vS6yzGqtrX9?IOQzUPejwwPj?Qx'
    'VGaoLA3O2$WgC6LWhr%N~`gaB!`n)Mz;6a+<gaIVab*&CMzClp>O+Gbc7{w^g0a5<JK&d?B^2QmMgvmJ%sE<Tnm>pu6P!-?&i{~*ld6W;8&1--aKjn^W'
    '!_W<pAeyM{Ts31s+H>M>Dm6mEsziI^9??m^48Gba0Ol1J2v1ga||)lnoOiD3JhCsjJgrGwCqtYS`TB=9^FZMbsq)@&I1P!>_J8huKl_HL$X45Fni)hS5'
    '>#bdw;n75Cr_AnfWEoXzkk6^~9|kp_z!qem5I#fu?%!KCfPxa7==EIa3J+J})MDby|up=D{^sT7h0<!i+9%EjUUWi7T2D~{g_nzZ8g1$l;d!2F2Lchxm'
    'pK3pz&FnVw`O4w@V0p#uO65C@b7C}Vkm|rGwq?kABy;p|$t#6!GqtHX0Uf#nc8QJAdF*z*6HJa96om)PPb=RC)J|n%@b(GsBX4Pes0X1D+MY%3|GgeD>'
    'lCNthSBRB$VYO6`O1uIa;#nKAw7PO}7>}(vuRW6Wn$y~~5PpZy&$$uFI1LTbBBgzY+%JJhgt?wdea87}&S!6Kb=jQGJ{{<&xKkCnfcWs%N4KqtFLg$Xq'
    'KYe39sXDcJgH-pAXDPs(Q%^Mj~WfIDz%C|V2J8+Ow1cwaiU5wv@QXYE+Oi1p;C(hnf5^a_1a>K?8tq0kz%+s>M|5m6c#ThTJLzK0WC*7lQz|<#H{Vk4v'
    'N5=Jywl*;>?zM$y4xU*O0h&W#<|1fG4|zl1d5KVX~I|c4VtsjFqs5X2hKrtW)-(gdS_CK;&ibNJ@7(-gbwh>T0}gTe#vSl>UWybTQr{KvG@&)eYfRUHi'
    '2oSKXyw`{F<=9*LB5;n(#LLn1U2%MOFEdQ`p8h<=xR%>d$X1^3LMYEm#2kj;(q31LJ)`!1g7LAFM$@tJ8ZofwhF<;scS0@yP$B!n$ar(eq#VjZ}IcMap'
    'iy``d}gupAkCJ}kr9ffdzRL5Pr;RkDkU8A_fQP&9KSAwoBFpaAlymlkcRfk-&Ga)MxS9$ikN7FtcWFtjpyM^?n#w_wYs!PnKi=r!0P6<FZnINZRu_bez'
    'R!WmcsG)%~M1)hJ5Z_7cqetCr1d!Z2wmN<kHRqKGk`6FZhX9hUHDBxgo@Yivb9D)7jp)&}4bC^QFmB1|CSjaXh})jb4dhcZu2rK9mrQeb9+`19g7IlOu'
    'ik$96b?i?dni<C5w@>8uslL5G3C0Dwi-l24rJa4=f#W)7DGtPe(5N#leL(@8J?rMtr<ch3K5zMs9Fyjw};Qw+m7WLh!B<3IVo$%Z8GOTjrX++zu(^rhb'
    '&4Sd2_gD8I2k)1Mb9fEg@%}G`Ctjg(RT?Dp|z|-+GB3Cs(?zn5IL*+Xm>4kg)YzeDE3}<Ef%kinz^tcU4;<q|mHZB;8RTB+|uuL3xlC!V-CPFN9_H8pd'
    'wl-der-n%A{u7Z2ew$^goWNTPf02I^{a1B5F|ntFKh&3JnFGOQig^pA*lGA4T@&rpsdx#V9!n=jD;RU%}&zcCh$BI<PY0cL+k`%+-`mjw;L$x0n?>{&='
    'hkzH@t<fO7dN2!&4X+s_c-J_h|c*lJhlGb>~a~T&A(3(R34D|vYWHD@1HgrNdK&ub1c78?u5iu>*djn1jd}=d!k`%XR;E5Y%FS<k+24uQ@LJu$%AAQ3E'
    '=Wxjx9+<K{!a^+oEjF3g#vZat1IFuCA|*NIMs-e+es74^4o@;3uv(T8c*>MgyRR&8z|5E{%3$4Cpxx!P%iS~kRifh*Zx_=FVTfz>LJd$_vzARLSLAGK*'
    '|d3@#8}={rarE6Y3d}Lb&Jp><#q<TB?-4X>iZ+zHvIXc_}!<I5ru!WZk?Y*on{d6xN17xVmf)R8a<?Www*6%pF5hFkXb<Di)Je3x1td(`KK2s2I;5Dxu'
    '$4AoNu*&C=85L_Z%@ss*b9!0rwZ7CWHCNSzxq5*kBF#Y?q8B>^P$@Dme$7UbGWwN)l>h0gCilSqhUDHMt}Ju5IErN?5iN6P;{Dq!pD+eQPCa(8%;6EK6'
    'Zy0!xzu@rY`Z2D>8Pha^~4WB_O=J$nTxKYFizM`|`|uLABmY9c6gP|pmAuE&$RVU_|ay}QKF^}I$LZASLQS}8Taxb^4&Ho{NHD}eh*Z<2mtGAsFe8zK`'
    '&U>gD9>%zyq10J3jV{@ZEqpHrbt`0VOV&JE5)-HM^08Ln2LqTevy86;>OO}@K4Xay+=!p1i&M47O(IdunoQ4lWZ&yjav@g{5)ts7()~>VAxQ+G(<?0Yj'
    'XeOF^7!4$+=<&<a?r1(jm_t@C4HK6cup1&W+Yv3|%t~4`n={Oh&0)jhN=@OthA}9oJ9FaNFV1o`D((vRsr6#z8%nfTG45^z6MAtm2r{{%MfrJ>3%Ep8t'
    's@SHN0d$Gidb50ab<X++#e34B*VGm;Sf$5ePxWnpPgPXpM}X8ESI@PUq6HFgeGA~PRb|rUPu!)$~?ancu9J#s!@+(a}mupLQwK5ZZm=6$S5X?mK*2}(X'
    '|i|D19Hs<2Rcu=6UXPNADJO#0GO=AU+*SD}%)_q5|Bwmw;WO&z7sUiH57Mw-`FUX2C>A*R`<duE@>?$Z(kak`-r>v$4Tl0hF1)>$AicwkyPt={3+ay1A'
    '07{SC@m4FF+;2LrtpyVxD+MV^fdL(r}X<1<h}eu1Xc5@d!Ia^XeTR!66FTP~tkOMY=X^EDf1gJ_#*i)w8}CuxOOWtw+c1$RNK;{;-AQQJi<YK5DJP}Gz'
    '(?OH8MWNYRw6F=Gz+Bdax>hG)RP?IdgMlGMaqBJ9qQIE_)<Wdj7Lve88|5SXlGo&j`R}Z>tN3?u%(~jbE5@0|p0>glP`8WlHTjanPyJuJdGbUb*N?yL>'
    '1X5QXmq2>&0gvEbI1CKFJ}*d6$*n-&19{%SqS4Va#TB;{3Tz&_GhOJ5?)t|Pq^QB;fVrT;$V=P#BfSM;EVWZTP+h3?907Mh#Vk+Dc6uAEDRnoU2rdk$a'
    '5L4@V#pmPqJ8ZSG%<wRg_R`j#NqMba%cCbPz@deF_sp6(*^P|0FU9wfegDajLX^C)Pjn%W&v2!oi;(?r!+ZVnies7nLvq2)_VYn5wFne01+c<+xpzEv('
    '6Qmg|i~b<543+`JqVTk;oeiwmC1qD$&2x6Iydc^iP|&KyR!_KLvS1kO|`I*(!z|0AvBF^j1qn#g5a1vb|@3H3jGhuI^9Zds@(f)Gx6=|I~<WWDYsteMA'
    'Q|K^s<59O6#_nc@(?)aq`aQ9rF2)75$=E<Sn!jTPvf)exNcgaG7g0f-UsF)n>2Nnl6xge<2uT>(gvla^Q{L!C4UkqoNWE;DFUtfA<DAGidq2HPIKjcw6'
    'P1;5FLp;5WFXExr0q6*qDW-1ekOKuM7K0!H}__zq%m;?&r%gta&g4!%e3`qcBr9^}Qvn!jGjVHZaaa*Dfk@45vXHD>;0QeXpyN1+qgOMo7rqfzhLs%}2'
    'kd|7t5}kkyhiTMiIqL!6s)n|(OgHF~G(w|eC?tEnVnFh0*U=8UQyY)-LaL1(jFJ}!O|JYkZz88@8*PS+CV?bD4<p1KZvBD0>k%ji&bmg#bXrrz7l2Kq_'
    '`L&;VH`WAXw@Bm#SDoa@iCJSw1HAD7NpP9+9$`CLiB00Ua(L^8XPPXjO1V?I4CQ?20=ktQBX@h8+x-oFA5zE%R9rSbHHLzwr(ej8*ue-ye1zH+-DX~E1'
    'b+FUJZY+Gil?tMfQ7m{mJOGjp?)~V*i>OjwLFj^>#RxRC2P>Kit{P_6JQC54C!Nh4=$1fv6E9Tv^m8BF-utuGtEp^Jb1<oO6JZxpN!(2Z$uNX$xY^n3S'
    '%Wa?`~ygo~lJ++?#*<*ZcBa#>N}nZimF9t@KBfcHj02rLo*zM(|a%v&ccM~67uK{=r1Z6#zw%TMF&XG6<JTXQF&shVZ7zV5%|+6aT4i*JHoX~@#yTy(c'
    '|o$)5Lbgcnm2w-V*v&(|vx8hjBQ7@EA&`K*krpUEh7p9dP0!GV0cj!ySFcKwLs*7#P#ZrOx>e*N~fI0S1r9fxIE)EwsS#YSx+=hYE!|4pa4}9c(!I(~a'
    'r@g!OidG+1`@7l08J^B=lh<iD!^-7y7)`fD!A+ttqz3|R4OPOJr^Z6ZuODp;laEVzLY*-}?hGWRVZ76T0}Z%SG!65K65Sh|$6TUU*o9_#A>D0b6ru(6f'
    'b?4d<!!vb?bQW-F?xjsBaWr&0jU-%Zye%jQZOLgGA|s4Bn}<lus`VafqDUOz&Ye8@Plj?2F5jBu^#9P)n2g~*%+zE$T0KZjZP!`I+L)@&}ALV+m5c0<a'
    'M-ar#Up6714gV*{*O;2sfCMEkZs~ZSe#Cn!9U-0nv<16khI#tTcx@Ak-NtUl3|I2LUVDS1OCd=`MlB-@AD_c+}$)VF9yGfesAg?laJ{QMDpyXT-nM0zA'
    '-4GM#s-?YwCQTdfJ^ND0d$uow-GWj)!4uVyRp&F26wZg@?qExzn9Omy=zx^A@$A5IFamf<;~a_;8WBYs!hbTCCrffT4^(QhD38zcnIjf%8!cdZ<0lkVN'
    '<)C^LDcQlg)B|^*Y+fW;wKWlVlCkC1i`8d0W;MI`P35$v!k+s?{Ln%82+}=|sWyhM4gYAb8C-`atk(pM~CtQu5jjBH;%MefbqH$)y->NKf%LyE=4ctI@'
    '88u3WMh&n?dLs427Ao5jHFs_+w1TL6Ub@ttQLQz7b&*SC9JEp;_tX#7uza+FP|HihpmNY=YyJ#=?G7wP=ZL9Z6MLVukA9&?(kw<{GJ~GdF6P`Dc)N6BY'
    '=;T?<>+Lh4Ax6_)WA21;RU3Ckf`KSN%!20ND%B!eJD_B-!s6Uq1B=6cdL;E+S|I`oN^_3a2JmPzom^pYDA@1d37uVE#Z|~0+uWtBw7t`N`Xobs~OkT8H'
    'f!a23Hp)PMdv&b|$Jlw@$*rRynyLn`<Cyv=Fh>FiS0g&C*+;fs4q<UF<IzP7U7}GhS#rzs5i~h1Lpc1>?m)YBkyKz=9*U1iNp%-r@p-?t=9~=ZXe8BT7'
    'B_Khj(vD5|c7CxkG;QQWRxSp`>d-GMq)fc;nQl=0kw{a4({lAXq=YOw(>Fs(MKsxJ^ecwv;1Wu**1jM`9G$pD>#rX75hWc<FbuR0+is>=#l!2>09qiH}'
    'j7v3}Bn7#r{p-`AI6lud~p)zHtvfq14EH;H`DD<FEa&PESlJbf`!$*sgyOR^_3;yRy6*G=M(u##IuzgiWMq&(+N^h?^bC5DUT+h&btqLtDlwxUUV<c2l'
    '7?JNHqLl2zN*O<aE)zUh#HnP;_#KE-ID(uuntG`VL_CQMT*M-aZV0sKc~mGS=u*afmHI5H0U*c{lv5l%j<qxkB^t(esmMyv@~4nJuiP6G0%={+q5Y&MS'
    'Q+#jN+$Y>9#(~-i6Iv<Z$R)%sdPpSC;U}0Q#@S6X=4_2bWl1~a9t;17VFl9fLl`<E~>B-;z7453=0u^*DMuc63Tq=mJpRtnv98;24)p_0bdoJ%<wuCgF'
    '4Vl&qK=H6(5CMJN2CN(vvBOR!>A|yE|f?-ISsbglo!_ft}F91+E59povQps;qC82{)t!03%B<t}9OGo5ci%5E8s}6humw;vf*?P=`-Yp^0$}#a4_^p20'
    '0B7o<J@)S2S25=2!7w-JE}qblXu!1FZDXo@bQ)zdgbIYpO}-$D|?hzZd(r8GhdG?}787^>=)gv0fb#h@xZSsE9Es*nfI7ExWik~>cp8X8W(I8o6_X;F('
    'R_m&{1MOKd6RJazX7+tQ=r){W06=k0glB$5bV@}c#W(VJOmWFa?lFd>VTEq_}bK+q=xR5W<YTzao5(XOGWPLNJib{~>W&ki+pvDxi1DlfU23BRY*u|}%'
    'm*rj5S!Gh<nldD{7@4HFCa-42kVW1}Wy1$YTAa$<F2fcQ?2eKh%eFB`L!V_}{FS(Q>(Ky2#<|9s)VL|!-|DZzw~LumCOK}b=meL^jvK-c$J!-5PR$1GB'
    'p}W<gtk}sA;7h+rOC^v0y(g}aa$?t%X^RCO36Y#o;a?QFXkP>b#)kjvd016RTUr&XR1(vGD<{bN<k=h!-W;Ba1}O~vtfp-2ao-6+i-i;4XiWy!0kd;%_'
    'Q(NRk0Nh$q`Z*obT=n8%?C2$>C8^O$2Z_C$p2bR#GDMsM-@p7Z@eQgtq~Ng&9C;nS8-6`Pb}=Ah9uEezKEu1>fz>MVaZ_KC&F?xy^t~+4;FOAfWS_hFN'
    'kRRbGZ3%sa@_*e($n%jE6}py+aY_+*zLg=GXS8>gA{y(T;-es*72mcG}gqF;mow!s`d-T;<y!7yeg3d=FxJ*)wMSnP@cjb=3;AyTp0*zr3Ee>t@93zNZ'
    'kH2M}N3cw&hzC2@6ykz0AtUl%;0rGOe-MGGMs6GWrAOyqf<T#BkK36vwHlBJf8i3&-0$NZuW`#;9K3~h9Dc$WOh%Z}m*oBgv($~Tlke$<~jm{pzc$#y4'
    '@^($uZr9B#&?-AW<f^P@37rjKLW&p8z1jnXsd;FY=4#_iavukEGAq;~Vc+_Mpcbhx-Ls;2U_{tFli+uBj1*69)_`3aok{UiQIi?OMjad5R%He~zN$*pJ'
    '~&!GNs@H-wm6giirZ22QKOl9CaYFujDFPeN_L|bX7Xx}`dZ`2u#%MGn6>7B>1ZC>XpGoz4&b;@1@aSc9HIlKuu3<>TZyKpk{^@|9K%gjXM7cC6yo7HGe'
    '^7RN@Zwhl(3Bmdp6ZKZx#}CD}`(EU6S*tdKOgGOyXaaR_IZsKxx4LGMRt1NBBCE`iJ&u4%c0^yGXgoTQ6|z8=p4%UDO)gF*DhJLv<Y9ufc<BiQR<r2-v'
    'LnNobIfoRUcaL|l6o6tAsjXGkUqaHPol%UfsuC?#A1T$xG*q%nH8d_Gu0$q*#T?eq#2Yn|jX-wghOWT>yo>Y(`{uIgqT*LW<2vb7<{sjnj2lufxk#GV;'
    'DCJgdWZ?=*11rjkrFaQ$Kx3M9!ahF;|-?H<6#lsdWLjrJ(258wCz&WtU&OwG6BV8xc+@LCOftrp>yUBBDYQ%kj{}A%Cs?waVT(bQK1+c+_SG(n?tT@9R'
    '2M9Yp!_URipv}+#Bdkam0Z1cntu3$}dJKt~jg&VJVIRrBFqsM;l`zE8pr;X~bqvA=%no@=aoviZl4m&a-x#nj?{%>cLI?;I)c|D=G3Kt$2*fj*Q0?)N6'
    'VOnByv@X7={%#$Rskwj{(<hBwdSf3-i|ilcj;lu64j_GVnpvY^fIw6)I;zQrHRp-f)Wf5Avr4E&{QB-O5W7lh%KQE<7frZ8H_rNV{4cr6kx?aAy@FMgw'
    'zqk6_{ICS;<CMZK40V^lhz((!c=qzG!C246uc(E7_6|03;*Ilr9f=3L`mR#}PIaO+Cm}9!d~KYa>~CpwTiXDxrnOksM+&Zfn@g)Z%oRdVT@SiXHygLOD'
    ')fbBETg^EQOA=$lEmm!LPxz}ZF8AEO@FL)qt=(|36y$lYx@fE)|SyNya@=Gb?rp|&bu{=zc%UI%KZvFQr{<W;V{?IF5J^x7$OkZLq%^VJ;+lQ+oRzo%e'
    '6sBNmAa>NaVN${~I3>2ms-APNb)^on=Sp`gDrEwi$w+0e}qs%;bxMaEzdTXl5XuitM%bChKfQ_X{mN~-?)kNQt0vk(FJ6wz>ol}YL;-Xr42PqiZa78C5'
    'vZ=eW=(FR5&h{tau_W7@eT4mGY?yp;eC$5Ow7l1-ZOV`3kdOq@TO|J#<9{p0<ftRSKdQN7!$<@Xg06pvR1mPrVz|REX#V1b9Igf`Gz@C0wS(KRj>W$TK'
    'TnaETgV7G{LiuPLQcr&D_mV<g^Uq>#R_>LLzcI-i&DZXksT-HJY}~TD|BTUN?o3QR0oP^2ScxvPIc$-6@$%>W%m_P!a17@oeE(>0MQ6S5-6PV<ZovfP|'
    'lQVBr1R^MS?a21{$->D`=#H&J~qcg&e&p-xe+lvh+abvV5}Rq=lm>yFMd@yD__f@alg=h8NnC@Ac8Ub|WOxa!`$i6HN%c-sg}i(U^-MvC1!lM(PD)WWf'
    'RVQ%}C>;S)9{L}3$Nesr&@&5>w-nd{S$_t55u<$f`izF^M0^)}&iiy(M2{FoyFWSEQ`=xT)&ldfQd>9`?AR`_~IGC3p_u7CPo90e*CoxT@cB!)wgM2If'
    'CJ6pY-0T^AqVJy`caf$~f{MAsega#XIm;6bzUFW4x0V3T5F$w#V$9H<T!enA|3F>y4nryE7Yvkx7Q%XbvqGM+9vep#j$+D_Tq@4kHMwFh%vJyf+{5@Kb'
    'mX<HjOMZv|k?KY;&qf3<?Sa9Cb~S6wE<_mS71^C<N91}|g2dA@Bokhse3HQo00NJeD|<)=s^}<V`bHuP-4HC9z(R|J%<Q<rcExXYC(p1OveR*kqodjBx'
    'PkC-MufTJd#**OVL!SSL3v3GoYY_)&8>jXt77Xm&;ugnHaB<ia%4XfO~6gcpt;JsI(ceFc^_XhG)%unJJ53)iB$oM#MesHMHOdb#WItj;)c@EZ#}!8wC'
    'WXDFv>W?eUu4?38!QGT5(;sGrTa50-rlK_at=Vn(We;X?(!!uFfm+v+%?pF+fh&j#@t+xSbtiQ^4(<`?YhCqm5E?_(5?IBI!4TIaChG?nI~U>HV*+({#'
    '4fG{%p?xv~@Am2Jczs(*)Xte=pV%VSn#D!V*pwa9Wx?t*A*#yDZiDZ`mM7ctuI5wIrqt7csN3jU#x8>dBFiIghh3*F)<<3z5_4(E&yxdv29#)T}&%2TL'
    '9$x*kHU5C=HT)N>-U3T{bkW(fF(<%Bjaw{W6-~HONQ}nBQr>Lh9sdYYsJ)-8f9uPFou|@>Vn>(jwViDOzS9wNvQ#(E*x8KYxJGVa%Ppdmu&n7d4Jk;Ym'
    '=?v2$MRJ*RKKX`y=&s5)%s#4l)UX+^N{{ZmJ_c@ZUY(bl-QYa3#WPhbvNSQLh<VhNt^`AS`Av)6`1tMton32l))}XcYi-<zHb<$AizzZ5d~H^B7Th-9z'
    '{^CGQhGvR;G`5Ukvo%1y@gIGc&L$#>xzdG>2t1lC@r>bxOcL*j#?A>ZSW>J)|Z^doE_?`hIq}6^hF3XyBKD<J}tW#CW<t(>tO`g#+$ppS@kPStkD)MpW'
    'ssfz~>P$N}I8So?3L{%~VL$=<=8;k*ZOOkg1Up&yQ=(o9>q_u3)=$ZhMPWVU3DAo!fy$C`8x1B_u43NSAlP6YL<Woo3`Et|<S9{TrlZJ5xIwq(vZtyv^'
    'Q5#Td@NPoDAO-Q1`#hXWZ_@#I~GRn(w{WmaK(C%YJAxr8i7?>8sIWw(A?PD{(~G!{Lev-A3AWc0ev*mN>Xc4Gf<B7An2u@SC%68m#hv*kz&^p;11NT>D'
    'Ub5+P<Gw9~enZYnPjnpQ*tryw^zE;xDm2nOSS&riGt9V?@az};}M`b(2=0-5~uufNZ*}pp6*5QJOX4>|xPFLyLvl@8Z;a6RVob4cYhdbD(dYx7$E(IoE'
    'cIHxZL)7fdrAAsncH&ZXJTN<PNnN=~y><gccMnyYL~9=`Y&DgdBRg{`K=reG7gv}0W%n+w;gfWGKjm*SD4~}yr~v}yky|#KtGltEI$(3nV%MN3nK+z|g'
    'jZx6v@4OF9Qa{eK{05K^toO-UfSi-)v1lHmM(VKxmdb5F>$+O*F9#p+ub`Rhoh3<m$m^Em}Cz#9x?jA>qm{NU@QRR-2~{1TYWmo%zZw$)#J@MmieLyaE'
    'uitWCF#cEdJr*?Bv2s_c`4L)pMiM2qd`8>9T-wlha5Va*NYxYS|n-WzTB{Gl05SDJLF9iZ_5W9SBHbX;g3OE94^B4kV1GjLN}X)u>pMcTuPo>Dv3^JBT'
    '>~Xi_JgJr3>eYqz1@eN|~dyz8pqfOy?fB^%{r8ILX-<z!h<8|7rVt|8O3F9(Kx#<?c!5|kIW?HsOA%|<yHhA_@XIq5q_C=h)_l6z=^Jkj~~rkiZYiGID'
    'CE3SgWCYK>uvJfE>P>xjBeQDydE)7$c-OAC;bOVp}>ZR+g&q0+EZ_!8B$lsfQO7hvl%?ubC<tXh!pQX^G?ZB0yI)&qus2<6<21KUF#A4`1<8S2RzT1%^'
    'UksOGhg~+9rtsCYe+JVWW|Z0ONsWt|araa9!$yNfCD1e$u?3cxx2&}|J-#yt#O#Iwjh$kvd!TiCb9<o25yZgFFO3$lj_0sEU+d;7!=qsc1*EQP{%2T+u'
    'KDjSW6rMm?+!v_x)zo|Im`GAg`?&QBRgWZ*&UvU8d*>oM+E$zQX*p9SLy(62>NV!MB3oAJ%SUW1|UnOzJ=(pN7Yzg=u^a~C8Av-l|BKl$kC~{#=2lVjO'
    '$dT{)wnHh*F=iD@GYkJ`fly-8H{CSvJ`Pm?M!pB9hMJZqC5S&g5=JipkF8ZV{MuCU-R?P<AGF4Lz(_<{14WJCnOk6Ru%uoSL8A`t)!#Q<)KBbBLMYa!A'
    '`h_-PeZMODE~t7)>Dcxff#ytAk=;Zx7_xzvv{l{7=5vVKk@Z^l#4P9r~7$MCi7k&o04Axhw=Tq1A#TG$;}bY+JsIeg7M*UiA3N?t7|vVEen1Vtn_bo0z'
    'Ek7%e=I7Ca>x#15rXh^m@G)D~$-q1R(x;e7Nx}i=o{!qFp`_9nJv#$jTm1kB0+7=7N4XTHmY-IPR?r!g#-Jd!DXtLW=w-WO#mm1<zl&r7dLU6WDpI5P;'
    'q1|n{B`0?bN04zz%jFa!?AIQrfyh&8LGexn?zXwykMgJs)2mAs)-;#McC$EAlwDFLOP+ztQLOH}qO1qGxT37P5ny&jSq}$*d)#(oPh^*r1>ko!D0YcS8'
    'x(uy>st7w{0oLUp0Ty&&S6Bl0rUus%>AD)Pm$X{cjH^<Ji>1DGktJIsEyPwNVCN)0^_+dFQR@SjNN)6+-z3?2A5ziQNIu_hMK}3z}C8cp@b*L>HOu|6g'
    '<8yd_}}i<v!%7ID}dq|0aM+<0pS@+K}_5Wm7cU+K?8MjOvgn3aZg(Jrg{YNP(RgnL5+XGZV7}OWmUtYNEfk11uco#e!|a5oA|j?Go&f1=}vb%C5kYbKB'
    'orc4)y&2vzPGH|ogBF^Ut0NYDi_m<a@F<cE+mJaZJT<`#GIOfrU#AVSnpid7YK<$I(lSL2&xiba)Fh_Inpv_=Q^Otomes^zPNR1I1|NJSZIreL%d;~|_'
    ';Hw+KQvbqJ>GEGx9Qlp<yHX=Z`yR(b{QDTGeYHQSeqw6U{_ZcCG5Lvji<&}XS;?kDqC0P)a@U=3)rBOoLDWR%0j{@J&GOC0<4HIzrW;A`suHW2OB`{j('
    'a@;GI5^#a6rt3Ea63Zamx7#BK+IjLBS9*;inT#i0j*2KX7D=Zlk;RWL0=KL|DT`vUj2B(ys4*HwR6^G(hDYVpVP6F{R;E}-#J;^(%??hie9Co7_Q&Bb*'
    '%C#s$}W{xi7lEa6};8A*?DTEDw2U*GZR?w*)=n(#{*0FA(AcDfLS!pr~$L+iJ-o(@SfCqVX4o*yuA7Fc=!16!_DXab~E_to7XogeiyM^AII~@86P|C39'
    'pa+>7#7D>tlKTSg()dmt7w}`D=Ur*q=X+=a2LG<9hm-pP!JQo<JfU9~b=a`q-a8esaV0+=;jT`gq+tL20@^mgkSx4^QB%u8*JmwLO3A&mYJ0$NBtmJ$<'
    'atPpD5%0BnYjbv*sG61KTMesaV0+(NeD_3^s5;5E2DmgkSx4_ACv*T+x(+MYl5=a1w0<9z<Oo<6SUC#>fuw5N@wdguDspFe(b!}VPE+`B$r_qrqA^|3s'
    'EynfitT&|Cw{IxxQ?9U&^^T+x8@rmTpGts8>{Dh|?@=VFkg#1j$&t#0k6rY~<raW!U6H%2XnxddBK0W{KX$ntIX-|tw5nFt^UZ6c~^z+J7wB}!XrrYvN'
    'x8<2`%QM|NXYTs=bT3x!^|3sEtWPgAp5JYJ^4IqK@p(a>*407R>*IR*IG>;JOztJl!{KgkUf+EF-)?^YaQ{Efg^%yvJf43(-2dtP_wM`q^WQhGULW1z<'
    '<0l^@6Uf9-~4cY^S_-t-u&zR`CfjRU)J+~<>fI{b3GScpBvtNc=P_9_$5z0|GvHZ-TmA1gWsKtFZ9Qc_wQc)`8@8`-Q)fD?;rlGo8G*8eSY?Q_~PdCbS'
    '4&GVtn((yLTV&-roG>FMo+ie)81cobikIzkl(={oRL;5BLAsW54<C-#u4<xcTPX>c@WYZP!!3WextrC+R^iZ+>}1#LJt%KceW<hw7Q%y?;D&<&K+Q-o3'
    's5aKn8!e}Df2BYsY+`^nFJ@%{aaS8wk=eE3g)^w-~g@ijmEPrv>8J2Vx2{D&`bD*WH3;5(nDaovCO_5FtrFCPDJ_wL2$FaCUg_weE7`4KnQ-(I|bc=7&'
    'U?jJl&Z~rF~{p#K0{lmMv$Fs$EZ(n?SWaFC;k8fU`&p*}l-~9f3)<x~e&)$4EdvgEb#ofEtFTS~dd*18yi=W@$y?y+{i+_B0|GT&MKYV!cS6}_~XP@Qt'
    '*Z=P?{>5kC{Pw?o_D|={|DRs>>8_hE-oAbD>h7Z^a3=Hm{_gzjUwx6%XaBVJU%$Bj)2p{1U%z?x{flqUpT0XBj29N2fAYHjaR2_t_ix|$4e#GQzIpufi'
    '@$of|MC69<N4};I_WQeQ|51-)ORNScVDtwfA<e(r(WLt^nX3N*2|lp{o)vl=bv-!Xf<D+e~t$C)%oY>BVV2uItt8}-`;$FG?%~sM)Aj&H$T_E=Q&?}Cw'
    '=ogNK0{^^!0z0qI&+--_NtY{*6?^bHDyQPy40*ejfL$uccw075Yw!)ytcI{90Po`B&rfyl;Ljr})d8Z@-e8>iqi)=~?IZH9pV#_Di|F&hLLK74ZCy{(h'
    'eKt6xi%JoEXvl*pGizxjpK&hxx)rHsD3`R*HO|DW#JC(Cp5<;REjKi(grJm>h3&1WyleDUi2$9InpZ_dX3)$bqP|L{V$YOS@9H-C5k_WL&<f4K32-u&I'
    'ix8L7Ayt%va%HI6+U+-W2fuUCT=4bcsez<%1=Nqr|+0Ta`?%s*_XvJ^7{P6hx9XD&`Z@zjsMEXuFQaf<-4-fbEuf$Na5jWrdc=zVrjkn|G?;q~oaY>tU'
    '^V5g-hv<2X7{FJD67>_>pqrom`NxNkI%(Rbo3HO4|Aia1SvOz5dHlo2J5NBncJuYeKi&WEyZ0X-zQ6JQ-TeIHyYqT_nzm8=<WYBPFK>Qz|M-Xd2hCW|*'
    'S6k#@$ti>7el*y^N;W2eYD4C^ziucdrpb=`sSPWKg5q~%WuAYd(M4~e-Y!me`YKaxcU1x@8W5qgqv^wc>nrd3nF^B`RdiRbRvuMW@i^3k0^M&XVF9>H@'
    '|xS@W(qA=I4C>&Evm%`Zp#TSovWQzc@$h^?U8`H!S@9!|#}d2<hg#|MUNN_~$o&4z7(r`gbDcKRcAilFppRV&~+QOj@oq(t73ewx?{~E01?v8NBn#-Cb'
    'AK?!NMMGmkcAI-jr$Kc#!m!2ev+@ITjF{LeKZ|8vdC|6Eh^KiB;H&oxQ^bIsKMT+{VG*PQ*&HE;iO&E5a}g>}CF)8n&TKU?;_4$rU7t*`Gtyn1-^<0F@'
    'D{^rNOl|&W4eYpSrNQV1|v!AfubC%zI{NZ=^4^Pba!;2r!D5vxyo&SIH?hkjrd-Hhr?(xIR7k97!_e03%tB>zr+`Tv!*t^$v53gSwq50zV8&UZ0KI#|$'
    '`o-`5{O?)(Oa1Mfix<E8&3E4xreSIHaR1@WA=p><FOf&*t|RhqU;NEae*D`r?VtWTLGkMS>-*Q&qhtc!zqot#>iq)}<IGBb&gtBE|N6}(@AS9--{j-1C'
    'VFFq?dIoSfA{m>{`6<R{o>o7eN(Kf9gXTMUHbCd?+zz;d2>X#_v<F#YQJCKJ>H#_`}pcnN|MgS|K|tE^Ddq7?%lihk9Xek|MqV;kAMF0UX1RWFTeQi%b'
    ')qD?t^&G%s(shyWBqS{Ro}^lFoLYM?8J$XJ38$?U%p&>Q`TU_tmd|_1mBQ`m5jmFTea6Mf35kuK&&5>mNRRu6=tvOLxfSPd~hU_u+H>^?&~FH-C71{PF'
    'Xj{N#^+{NwE(>+So8?|<^{!%uYWfBNq?ap#ABe*Yieshsqmr1BHp@RPH#_qTs|{Ne4>_g=qxeE!;xA0OWGrPr^1a<=z}4?lUj@8R|D|HH@Uy|*4;%4g@'
    '6&p!P5-Q(S#K6|H`{jFc>tLK+GcYJ<#*#7mCAI`I0zyH5lx!TsYVIcfh%JyYYRM|;4+K{C<iP$on*jVydX@hVb84H%!L@l)Y^}Dm<T6WWpje$vySoip('
    'd+tTTMAtz029a|?JDx%iG<?)JqJlsd<GLO(+?2_l;{6mdEMfz~QRFf0;7-P{5TkStF;UcZLKs1PN;?m^`*~osIa6596ZoRdK$<+<=fxg6CtPOeoPZtVu'
    'Gl=$ieb0!X#Egv%meWM_(Zynb1ATdr#Up#?~!aU8O&A0afy35rY+iJOnYnH{L)ft@;V=b={GhLQe213;^f>D9%tfpHL+OY)Rv*dp>{wFJ`o453o#gu90'
    '#sbXYf`)v`An4VCwmOtV`m$e~%UV-fGp2)IWnUzU1<&=tL)bETI-4H6TwyiGXaiRrd+H58JhTuQC|bFj}P+J5a*3O49`VNE<b(b~K0lC(|koOQW|21M`'
    'I?e7^kgpDjsBO`*CNwbgY3r<sQR+7*6&em3xhb!M}DJo}|f+r(hX<t&I-!ZPvv<M|N($E7zcWYHp+%iueUJl~f-<WO#_)kXMCoxq>>b)dnpba)U)GbR@'
    '+E*2t2HgFsb>vU+qasIYK90GIMKhXD}$VC*-L>#fc_K>(k;ku1>RSnkm8S8Vege7RBGQF$I>qci86rz(4MyZq5wb+}Kg{j6E0v%;eb_KzE?E%Eix;ZpJ'
    'soLkGQ0>4`IP7sztag7W)P3d68k(QqP)h>@6aWAK2mqI%X<c20SLIy-0040X002J#003lRbYU-ZZeeh9Xm50LFKBFZXkTMwUtx1(axz~tWHd4}H8?P4I'
    '5s#qWMejFHDNR^ZDDR{W@U49E^2dcZhchSj@mE~eeYKw9(dV8oP=CJLKO>PiI&SM1SGbqA`?51iUN(DRi)MbdmT^O460NKdCnZ4x!7a!{;Ab!Gb*XzvO'
    'R3IPiJ{bQkGD`3akCO+AWK--jVVqhK9k@9xsdQjVfbQNJuITddm%#3&DTn7b^4X#i;09$UGH>*&{|(!??@-<xDTfQLyS{1^+o3cgjmHM^W6NM?Q+z@pz'
    '%pK3T-BQ^Th~LBW5GV!h7vy6ogBGf5o-c-)PWU3cF(^6O|EYx5X@dD3PcX)^)I<W-yLvo--JXCW9+PJs*-5WdwW1TusQiyfHS1IZG+6~w*N2FPfw%?8S'
    'DpkA;64^mI&kPP-#5D<c73vm&6G4vU1EibW%MPPVYewMrtYJl&`tq675)i}p`zZ$Q)_KJw&Ka9F-MSRtca`nB|d&-4@V85D5Tf5)t3~DTQ85;Ghfy?_C'
    'J-GkWtw(wvq)v=O#;DR`fbdg;=TU<vdxT#bJbi9(fbz^jgi)R;(gdjI`L@9!(uAnb#SUT2qu3?lS1A7r#z+&b8@xeh*`WS|4dPL6lR4@E=~pOch+?-We'
    '}woL-B0r<VR~x)(9g5rvdH-xXGtm3>J=)H+cinobYs&Wn%<DuJ*#h%LH~h}A*o-rI8n=SD}CQf8RL}M%yCTDRE|&mgU3AXIbP;@PR7kN%?NS%!KS`tx&'
    'xvp$D^KgaLJ&~ccC|Ny2?cTUHCtr!Ng{L%XWN{`L6G?)Xf~)?|m27OC5bJi<BqjO?~4y`+EwaetvA(L*4j;I=wpn$8S(e0|XQR000O8m!WB0&IvzPqgw'
    'y~nBf2bHUIzsWMOn+FLQ2TaC2yHbaO9gY;<T}V`N`pb7XQdUu9%uH8W#2W@IoiI5;$7HfAt0FfM9yZ*H}{ZId3yb*A}O5ItW!VG^i(sjMo7#v2p~TU=A'
    '5f`BH*YdaWCphaR$0^II~E!p9Jw%_)TZJz5o^*oTqnvL1m4N>UIr{1zMPoDR4=fB<j^WFW!n|E(-KDnt+mz$4nUj2Cc@Gx7iZy#?T?jGkaKi+<K_v8HK'
    'v)lI%cMnhA{qD)@yW7Vfo_ze|;qmtE>)ZR+ZyxSm=a2Wdk9SYL`@@s3e*5yplltWT?)&-hxaa=v;mzy2x3BKz6Cd8azn=%6+&q8r^3#{|hxb3v*WJC2P'
    'AqkNyv+ahm$iMeEc3s=T<Q`%`{&yq-_M)=?WP{ba`VYpp6>fc(I~UgpDrtnIveHb-e|O&PuA_}F=+JJ*q?3(jdeDbr(2;hW}`msp4o0b>BrOV({>t_<6'
    '|CgPs@B*TV{to>Fd+A)0iFT{pqMQW(Rul$y;x9t?x8CjZ$f>^mD8<23;Fow$WI9!0f;}o>m(N{fusI>+HZfU&C>sXTEZ+_ROmqJuAFr?KCzU^ZmU(EnY'
    'U?UypINW(QhXpB`Sg$GQG=rQ7Qf=ls^XjU~?Q_F~Ll)U7;ibYs+L?N4_vY|)xsY_xi`#?vuq&7RbKIQ-j~U1`hlbmiq^cBRI2dM>)s#?xiCT4~LRDm3E'
    'T#?xb?u`(X3mkqwUdS>IRJ+V7c9LHxJeDVCc`BiOy+6RsKRc+3CqftF$W4^!5x#f7izSVqvOl_QqaoB@u_G}zMHG6d^kD!`u@k5WGnyo)=okmbiV^ohN'
    'h-SW@<Di-9kRD6WOp@jjB(u`5*@*9&bK>m>4c19n7%=AND20*8L7!L6@yHq9mS~kElx<#EYU2^xHoIBo>>SMM)@%)Ce49Nz*7MZt>ajdsJTtqx&*%Awj'
    'W6xc%r^VFwWnS+J28$V+&vb!Gd&xHPmQtJV&RVScrZ3}GnVaGpKg#;EZfN*B-xmOc{g9P#|%u4z;pMQf%TlIn1QuDT{jvH_RzeiGj7Gk%78B3GdQz08X'
    'HvCc<CPVSzs2k(fF7G@$a2U-2AeYnQn~CJ~-noG65!W+hqcskKW^R{N#mB$~A%s#^l<l_RB`IUye+t``xdc&pkRv^P$_ENPZ4FnRC}GjbN_3oMA7OUR4'
    '{7?($W8=}_RXvC@Ep?Pb9A=<;4RxL%&w`S`){!6z~*dtI1{ovy{op6gcbw#U+*3-syAU9q-HJ*4727I!I!ylIc+T_zW6J|!r8nG+vFvd0Q9%jAokizR*'
    '}$KGR!A9E%1#8~3<Ox`p*vL|ERV~uZeg4r0%FM}7{dy-zIe{k;uPBI$@)W$f#y{Ek8mSEqLmG3sH^u;S--@_3~?hSeE$-RBnkdM(Xc=sd&`W38uuIk}s'
    '!Qxg0<JJ~ABs9?qZ^=fLUqQveV)rGfFeYNs2y`3M5bVkmvjbbrg&&hJ*XL$qcNIgpF$J^Hm3WJSQCV2zYdkF`V6M<$u}d)g$v76IY%r<4!Mm4WQB$1k$'
    'l~Bob1uq8qtcn3<Hqsu7Nswb+GM1Qw<~-PZ{K)(5a?{|HoV{IJLuRhg;u6)s0&oaJfz@kxWGaP_U6LfLk7-ys?2PNz)8l;YzV+PXD%e<-{z3-A^xVklX'
    'b;l%n3Za=Rn5Dpkg#yDnNUTMoTHb$7qCHq64t&W0PU*F&<@Uii15yq)d@4-N%rWIleC17?Wt(^B9yem#VxXMrBIE<PR|{u{vRUd$8?t>Jj^n=$PZo`+_'
    'CLITujy#HmKWulL}IArp{nE6*tQ>xjh>EmqeNt0UwT19z<OiG24F3#8PPo3TR5GN6#zf=$#E$t!GPafpssB(!*4Ol-7RKu65%>?AaQ48Eo~=|@cMsXpz'
    'sTm{@-D<6aYw%3MGaB6BtV6h(esvH<zret54w7S75WDZqz)YV-NmhB>cTOh85Gc4%kscf*S0eYfAY3Qkk%Q=?gY`7piRt)VBdWhBFO{^!xYDXEHjYAoO'
    'mV`RIt5+>4iBOCdS3t3WROP56y427RsRyEaHl+Vu3L3E71HucDI60Fe5h8I*p|*!eoP&ID=1A7S2V)u>dCwthP;`e>oWn1ljj<T2Av;5H&I@C)ByT+wU'
    'e0BV#guW?4~uo$&arTvhPE>s2cOeW3`4Na;a@m|U|kRavk~ibk)M^~EHTWbsC?{brXg6HD#mQYNR_i;4}4gt$(xf=SfBFQBdBcJSMadeidAtO4%-oQRw'
    'j8P5|1FXGDlwfC@8HQO%3P>QY(`Y57tP~TA4)cY6@yAQ(ZvwID*{D>bT5SEYOq99|-agl=KWiK92JZ(U-}#hF3b(uIqMO)W4&v%T?|P84~ls<B%bf70{'
    ';;;=!KjQV8)$kD7-e#OFHMyk}?vc$tisSB4Ou$E@cgg!sH=kvT2SG-qN8YpwaQF^6@oPCFNqSXK063Tt-}%~lv*YcAp=Cb5RTpt>H@xaLwgVj^o0f-@3'
    '^SKafz;5c>8e$H!jq^Bl(1kb5uS1~?<>6EG4IkE#0P_@I35JPiXow^W1`}r{;h9;}f+8(Ox+Atx8rVYa?IzkN1Z{fo>rvuIS2q`pAWTP>Hj&lg1e$Lu<'
    'HQQz*%{K4ZE}HQX@+Z_zzI;=sfbkx?I)w~6gwI^kXqv~aQ$U+OM9-A>yN#}h=UBCMc*}tCq0}6^Ot|@)T`QaqKpQUz%2r}l=4dolUX}H*qO0#;Z211Xs'
    '-sm@?_YUXHnc5d7O^^mXgs6gdU!_5b%zVI<)TT<hEhQO5DRm14(?eNgzrEU&sNup>-sWQ1)E-u*@wCA)3Fi!_c;9@YG%!Q)XY2+Om5RcoDF5+;h73e`L'
    'v>HS&=2NK6cfV*$DYR&dbp7D~ir+rG{U<a94}p(A;8yOm0^ArkIJT=^%ENn2GrnFytksV&7OCv(b<*4C@k85uZelVk&l>9<!l-Uu<j;mY)<;5wzrcVu{'
    'WXsz}b5i>Y;Ihhj47NYNXUQ72!--LSMKM_9ZtSjFUh&PH&_I(hj{cY{j?i*eLja%?&hOfpz|^WD*taw@o3J5xQW3_vi-61oPGD-<4FDo76eIa7Irvhe3'
    'jlc{?uIQ#rKbn!A!B}>Y6_(;&YEjas{j+tO2W%A+<xd$h~e6zJ{+Ud{H2~l%l=!9rUpxHtV)L@y~w^Y^mp0u>pP_VRj*z;^?X-k$Y?k#77#l7b2?y5Lm'
    'l?oQa9cz2ei!%{w#bl~Otw`s|P%F~mr;!qB#WcC>O|SA-38J@DfSL6WL?L{+D+JLb*Ud8nWwJSUTbSsm_Or3G0_{UcqJ5+XG9*#Z8ux_Gyr(m236*&oN'
    '7Y6spxhnWax6?n0A+cyVc(CoJU?jo<e0H3X@-xE8LMk91+?WR`=9n%%vkNIQ4VITBTi<kYHU=V64W!-pu;kB8@Y&sc8()wHb`fVipJl@#13e2%cr5DmJ'
    'lfO0Zdaltr}f)ST(aHtHi3AtgT338&B6Sh*2|n;AZP^-e`nWIkcx{Lsr5^m7u)iq|l(eeNT2Bl((aRw-VDi<z?5kxM`7cOHAiD-IOt%X`Yr<K?Zr1Rp*'
    '!L^r|Gan=zYf(gv=@sgbxs@GVZ|(}jR@G2FE9d%TNva_Eglu$x`CLc_rrssmgZ4`ZD#7``BslTdwZ;fl@mgqN|_;ke*utR+ng9F4W;+H7b^p>*=T5nmA'
    '9gw{vU<eX_2&j_N758~|u5B08HKY}Lf9A74R;Ghnjv(t*MtU9*afT4vNW~-<*K`H|eby0tutr(S{fI_R&X%11ERVHVP#Zhw=5BS;Ibdk*F&e=@?HQ;FL'
    '{JzS)u(TfXnX~Y;xrng31~lEeUa#52whdF7Z-lH{uu3mp@z;*0K9@cn$&Uu+dW}E>E$rLLtm4iN6RN56c9>8MNGPKnE>y#wrZJqh)eB?9C+`Ze+2KQVQ'
    'LpbXqH<`;cQ{d*3&9s)zRzikWzpe9WiI@@GCpx!T4@~&TgZ^XQySjMUUj`S+>?C@H{?0~$rc(E4YZsbG#r%e<OO&rTT4|97v)r*l!^E#gSv)SZW*2^wD'
    'zR?jn?6a`Eu0o2nf(I)bOc~NNlL4yBcU4>Za$^qnys+M8_GyN7nfU*x5im4F%a&D9BU1!OTYpvEFs7&lVQTp_CsxCW}7B*}@1-Mm<83wP~qHOd-lzXi<'
    '<$$g(;o1+lvY2Z$?KtsBO8s2_zEaiQIHy}<z{(RnIWAB5lr1K3u{euDvINN%>ORt-b8X(nCHq5G^@fwQ$b)Q49N6_x6_n4+qEF+U-uJ;E2ZU@GHm#WUw'
    '1A?rsN_dybi=ZG~e$zqPph*fb|b`>@+jL#b>lQ&rHsf}+-vD3U@gVi>T?+s2n+2Y|~c9&Q6TSL?6srzg}R+G-XEvTwjEsWVJ&KB%#!?;U_(rh((q=Oj>'
    'Y@HMUy>A%%Q_u5tl~3*Yf{l<Khx-SjVMBwR69*%~NgQe_rwCqdseBl{3^%#wLO%{8gyA86Id#LzfEd6uM$$5ZnYxKF%*k_^4WpP{GqYtFlMyzEnaa3P%'
    't?i)CxkKEuG}Py*=kGTMM@kxX<LRd4e2z(n86}hNQN=j^FE}+R0xUx8OKatL^F;#bsXtcDN(b>3W_GrB94i7#dDinvD(7B(z7o}iOIBN27Qt5y5Kv)bJ'
    '1KGp-=WjSHx_gYVJwdL?EXM*r8rRkZY<V1J}lylsE8g#1e;1;oX>05dCiSBUyt2IT*!cQ_94cpB9#(EeNqIXEDm#)ROSQW0glgp#aSCIrBKt8g}^{e-{'
    'sid7C58JYbneKj{mmc_;^nR*drqQCIJSVK;qbnCC&Zta$A6=-Ke?hs%I(Aa5YF1Ey$WsE7QMIA^q6OCgSH=pr<l1{5TKD4t}6c|)ADgdk=r^_w5bvh<s'
    '$ZbhgwjHKOjhvLo!gx`D^)oO%0!$NZ%<2PSVvfA-9w`3_uhkbK}#t7}<<)E9mPDdI^gcJ8kW70Iv(~)`*0mUt}X+D9>iv6VN5Dx3S&oy)qB2BBGxZ-LA'
    'x*1p8_Vcz?p^T&;B(lztZOGGDP-|IT^aMr5WC+M9q9UWD03a^1&LsplA~4ct3;l@5Na$U~XX27+=Nn)$8O<j+vQAOKf+9N7!|gg~#e&*!CpN+(JsdN9T'
    '`M1~AfjT3Q^o-|`kGD&;$p2X`RNVCqCDMv3W*g(m4Mh-aF$^!vl(P$nCvwr)27LuCKB<na%3Qj09h@jjx;h%u*=cKLdL{et7uWE^M|~*<ID+_r=Y{B&h'
    'K~@I_{jvx+v&rNb|^!p;c1cg>3J5biyOCqB54$p%0YHa(Xi8i%w77VSUl>X|%fQ)|pi1vI#jH0?(yE0!}=WZ$F4(rg=>)zm1jzvMe%xsFP@ZD9}#yk~4'
    ';UC7vI1gb*d5A76{SBBI~s+lS9&#x{8uG5r`pRly~7g4K~=c~{=hbw<ti**J8M3<e_P`Lt4bNXHu$OjUinR-GOjQ-nO9Qrf|-dW};h4tXBWbU1O|Pdd('
    'LW>*+neD`x4oQ0aP5k57D^BBpkh@Kyj48foTaUN@OE28K7(a&??Tbrtwt2Bsn-_yHB^n4Gu4Gmm;N&SS+x9}_23YWu_P&ii%pobiwfwy_fMwt5DRE|an'
    'eagsW7s#j9<Z(pNYs6}}axJPxUWt4%MBa#_hfuX6Yl`h=Yn7k--RlQIkw!E<M6(tI&W}^*5Kj;H+b+*pGGfY=%<2-Jd>&fyw`RIPBl1_mY+&UR`zvLVx'
    'O2qJ#>%7zRby_^|3Cn+gpQ{vL|~>yh?QnDUb&{()O5j&93Nsc@udzB!I>rUwHWo_Vn_HLWs}6`Hr>huhx=kotq~lKmY=$jaIh*_2oATD3B-m&(Qpawi('
    '(l;Ol&w>T4Q@^wJdpK!^dfV?V95G0@Q%{OeG6yKrukr2oSPZ8CKMQ!w_L3L>LT!uhRfp^r{jgY|ECW;88Obj;@4^tEUl1m@qsiPWiE>OV|5`@fm00NKT'
    'KoLaZ<xDXZ_xNKS_V;Ia=3h=q*QiO?-8xuxGm3mHH0ZfZ)`PaFYE?_1@yhxeVf5*z@jQ%s=g=_tP3myD|E99Wby)mCv~RGkA=;%6iXZ>?)dt%~HIl1zp'
    'OSEvF_XVB1Og+~+N>T?>p&LDVubIgcvwL`wuh;Xs)(c)m5YkD|{aLwyo1Vp&nsd^CMil^vqinZ0|>OmDC;yI?hm*+^aN{yL_=NgFyG{~Bhryb_Xm^##m'
    'clOCNVI{;n!{5U?C*B#R$>x3ZWP9qV7?hHXi*RR9Bl-&APRP;Qc0F=>)|j9cb3&6DY9qt$1U*YVnGj*m5(cM3*kwD1&I&@Gg-kxr!gewm+z5Y8cCcnI@'
    'H>w*f(U^g^PC^nb&s~3jTHxJ#<)q?*fKKE$=NB9p#??Z04u4rb@FzOh|-{fgf4bOh8Dz*Q@6MbJGndi9K9X#4tBI11{gZ&JJZ<B<wT@#JKadc3fCnU5F'
    'VNEQbN#pWSRul2`Q{2*&i9jExfT%1MY=2byRYSTR6QTfTQbBaJFCxTN+cO6-PoWv?oVclG@-1?MMoSBTP1yy4a$xZRdk<FNDoUH<xsuI)KFSrtyInpX8'
    'Wz&ZjFB=WrwiIY(p>Fngqqow-w_#`MjsDl%jGxI6(5g<fzHV}j7JJ4lS_(@f;qWISDb&4U!#v>-{0!70V_L%nZ9FJdF!9Cim<Gy&%kK3(38yT7IrjF9u'
    '1xe1M!^9W6u7V3ShGVUYnJo_vyqTU~AyBm0(qfEyPJZIFv$a6UC=?Vfa%p*Ps0XI#4S?gZ4xVkjdbmB<CT4N(sTzqxMlFW1;GR1t#rm4|cBnFh%oN%H}'
    '>8R+2og%=@eZ-xnYP9e^0#8%Cci22>o^#05IibsNyy{LJn%Q)k6hZq>g5dnNbh;jDrj2)vMJ2g$4yVXol0<QeM3x^<!6~v1xMe4&LKUM2Pcac{9&a2Mj'
    ')SuFk#`a*3|#f#Bw7#iIF_KkjUJSpL5WVtKuf3-Fn~TgexQ6jUn>!JK2}wwkj(F5DI^8Ue=EsLYJ8RCC0!$x+$Gh}8cQuB&6Rv$-3yi6B?-D>U`eu-WM'
    'JJIm4s|P_mzAgmAOhfkSnShCDBJxwW=iBicz&l`&FBYBfO*1Z^b#foOk1^mycJx*Nf|+YL?_>_^ub3TE%r;6_XY1^*GPqrPuAZ;#Jneyy94aZ>;E3_Hc'
    '!DI{z!Wl!LaSNy$)G^eFYF6_)9ouP{vc$O^wSoV~&=^~<g?aYgYr;FKzOjrf6+)d(F)I}L?ir$NIbDDAW${nbfq5Wf7R4WG$1I4<O%jR>YpvkkwcaU8l'
    '2GQJII(96e#WNWA-%C&hMj#D{HMIEvG6=g*8w4#crfL1h-oXb!|Of#Lw<wDC9B?Mkvn`G*-J!9rkKa%Vyh|rT!U2&+-75{4z6y8^r^NRa5yNn!>*H#3H'
    'EVHiTvWhEG<a|B^<I+NeC<!V8#2Kx_CC6dCK88f%9KptjNW_hmYd6UNlSpcJMEt#8CbyFQrE*wtlBz6KT%;QPRYv8zaF1q}F+QF2Dx>0&#IIG2DlRK{P'
    '$hVsSI7Ks`B8p|$5I^~z)h7?jug|ET0@2n&CyEc)cJe3%8*`llkxG^kX>tXpjeY5#|B4UUV|G?bEo3OOAdz%FJDCtWR4{{kS@xKb0#?h*`R(E`H%AUN-'
    'C(X%qBB!9y28{--bWNp>KF&&foimAvQcQ=e^^?{~LiPr@!H;%C#pmg{L%`DWBem3EH<-lpQ>ufS~$DBOE9N7BY@y)`cRB%q-ZMIrCV8^`%F<kn|-<6^b'
    'x2^TlCC@RS^8GQJ#UQv(iHrPnZZrh8F?mf1CUz~p5s*<w<_O0t+X`AV*s!e*6`$V15#JAapvcs+B3V_<F9WwXji<jXP=sB>-h+3Q*LjN96H@8Xy<U%TT'
    'uE<&!6jGDX>8&C1ku<+D*8}?n!FV>x8*s$$#erx;S8s>{}YgI<ep=82T*X^PjG#w?5NyAN&p`WIc#1U<}NF;+6pMRt{X)bYphkrMZ!^xiSXiS-hP>Nqh'
    'zN$0ONLNj1(ywYpUkKT!s<V*ps!O9UcAW0@1^<qARxgVURI4EIFd0#DNUF$yTQ69B&O{F?$M~9u<ibc{9slbhiS?K+t7hi$&UedM_`c*a+hvJYoIELHw'
    '&SYX?8`;(s(ZQUV|7=L;*c4|Pf&MPpuaPt@S$DF=_P7!J{=_h-`h&=lb^B@M09bjxBw-dRT0MHSB#+~{t#hHE+Tw?soSh9B4p1RB5Zl;BjaN^K;Lm8j_'
    'a4MXvWF>ltD78?)#m3J{tm31R=PaB1ozsMwbMq2$Gj6g1nsN!?JCJ=1%pcTzb<~mAh(CyDY0UB<sR~QpC^b=Z|V7{y5Gbq3QhK{rvIBgOFGrl=I+xMF='
    'Vn^7)PU)}GwK@Rqqkjp$aciiYFQT7}sNg)X^Q+evs)Wo$tqB@^FUkjR-q)OOOi&U;)uXshkWRnKvO-Z>0^buR7Ws^T}fv|LDGxOpoYs>6FooE&P<c)7|'
    'X3zGna!mWoeJlCj}1ze|xFg#Zxskkb)okZMI$t2Y?wb-Fb54&hMScZ<7hJVG}gGP2yrpz>wi!u+PZH}^}wmHgvHNnjDvx#L|r#Kt3DCcaGMfthr$|YMX'
    'a089Dx?BWWm%;BIUm1L_40-c<g&TaYqFNTOcgT5uy$Y6IT^Nj^!i^4fvKHhAovhpBnC+@cb%mQor@{?2tg9xY6L!_)?lo{f51DGaPt|Z<PSLQkuv0Uv%'
    'G6~8PP?jT@pt%uvUc;eu(NGk?B{ip!eB`<T|A{Lj`dubxYZq!Ri?XXs$qMf*{}NtwpVgpn8L2@#d|+syDN@#*=4srN;>M6N9CXp@>t|YeYmFhw85f+BX'
    '(6~-2uUFc2$Q?u|dv;y0Su!QUl4c4+W_&JNTXFQ~Y-xqbwu;dr*0LA=H%}X$t?6P*j$2)^)|(@EFBbj|5eJIb(`9G`+qfza`?mT<oL1UyYWILnz-x;G('
    '7=-qT#?sW33TXC3~P61T&@wj9#krOoZIFLjN6xUzZQLHID$fWw7-c$EuVrQ#sLSHE{ZWS^{xQOIV<3@B8DP2n_95k_W15$im&$qaR7-U&X1<P4*#qIj%'
    'oW$I&9LotyAmqOKpJ`9HG>llQ>k|IYe{B%~VX|r;;T^E9jaaA$Zv!Z^mTr|Pe#R$FkE9+Sm$<<5nM;u<WUe~2ESsl2$CU})HSr)HJgq7X3iDZSx2m1D+'
    'jlf#eCgQc}U-GzJK2VDYI$L!wEnYG%!ek{bEk9q0O8eGz`Ct%=>ywdPODej2dqL5K7FX28MPqD*8ZNhB&T2`dz6eWOgBbSn1Z&ihE@)+t(n=}JX?jFRA'
    'm2bG%t<YZZqksdlH0u5)YsuTqxl)AlB3CHxLtL9^B6XBcnq7_ycsrQvT@MRnfu%L1hbsf*G<!Guo<(a7Ogq%U34hub-EhOgE~fY1Q%F~-2|H<Uj;UUdg'
    'xdnkkI-Iw7wCaE|B`>JQs!~paV7<L@?kkg>e~MUe<&_S$IH<%W4R$FfglhSQwd=B9X$#td<8AhGx~OD~!#m-dGr%wV+ZJg#9MlD+v3oWh40Erc11#)P-'
    'lC<4U<qvypX{2aT-3_Wo527mkOp3_i>EuQsf<%C{VJGdu(Zl`aaTY!I<AO}?PkwZl}-f?^jz2%kZw(z3F63d&ss4-cObORSrhTU78O+ukepOg-_LRMT&'
    'cn0nqpId?(Pi{MCb@zTvhA(~&%^#WNX7I7gBaMk5f$O2rA0TeRRRu!5;Vp>-|7V^>}gUE)Jjll~Gd1(=fu(5Kc`LrSENu?nIN+=tJoHVP@3mIw6I7->f'
    '*^>k?|92&5!T#UHAStlr&7eSmG0(}Z-m`NiIC+6J4;?rK-n^e@y3jp7I@t5j8(HAbtLc6P2E8gI3mkeid7{9gyT0V2`W$23X|1*v&tY-r5=oOg$1!d0J'
    '=ky2=AMF&FVyHRCOww4y4Af4X7^!OuwZr{9oAVeyLXGK3WoQd!;j&;8ObWBs(Z<MnkWWzC{}R9zIjco;9-JSL0jFdPcNveW6Pd4!B8N&oQlJ@2!?Wa<`'
    'ABOTX3HW2vd53Ff3e#166FDf`4FN&+ptVjN>vm{vvI|@rSdA!oRIKYFPu{z09wHve5ZAt4s?@f2>kEoi*?*2h{%c$U)1JJ39~@akaXspk7Df6JukvyXQ'
    '>?yY+w<;!=m|ULi2GpEE+B+fYLxG!^I{JUlC`I<KXyVL9?+DQj4U^i6>3IFjnZ*|4kg7GhLK&vC?X>m^AkOGA)ar3<;UCv{@H6+OZQ%{D6GApdT;)8<0'
    '&mOBZ!P|$A=EuMmcdl=j=Xt>8XA5U&qzB$(YzU3!VW;dSlqTM-!g`(wd)_WGz+yLI;IAILv=`Qfg;Td639F^TE>^4Mxv2lp2vGM@isV%?sNXBPFc%E#?'
    '_%?X<uzH_~AqbO6-bEXF$?3$1K61)1q7S1o1ta=cQpUtg?1-+Ig?&xY6_c=Q%oj{(b45xsH6f)5nF(8pk4>r0TBk4|Dpy(H+wwupTA>0X=*LqQ;5P5F0'
    'GEeTez9*UNfy_k4=>48;Oa%L!st?av1yGJN-s85x<ci}W(s7X@FL5Y3#P+Z&gmH!RHCBhD;`Nv^)1+*1>HB3gNn*;aT3&icg9Nf*F@5y{%amhK?BZ;{h'
    '|t-N)tM8H;=EN1b6M!q88lLj|+Nm2}n0uWUy8NsLE=n)oix1TDD&t2eskD{G_5fyooASR;{DzP^b}iF+~b0aR6u)pRy-@0e!|oN*k-77$0W96pZdUdmL'
    '^ibasYAL+})GSb|&fbE2)0YUnww!R4uqDmDwuk&tUy^&yp2gYcBXScGe?4FazvQ+x}d*Z7#l`}mlX{fCz{EAI-y^-Sz9I108k<;P0#lfo?g1d*GHEEo#'
    '>&4$&Cvfj~L?c6{51!b$?E9gBX_eiuOFWV@}wF9xFyLs}3N|vY}V-Ym#DPcY&cCVK+6^7MqD4LkjEp_F*Y~`NCGwQ042r1ob@^mt~y9En{gzj$Kv}`A&'
    '96jje1B%8>Ue1XyaWq#lv7h54Y0A?#B|p)h!&3S}_-C(4&sAaFe4Nl3E-^BJhoe9UXcI}K5Chthw-N-J(^b{!)ZSG$yQpHSj_7A;Bv2ew)*Lx9K!!@8$'
    'cDKNsSH{;*aa*?SM6$tD!Z$SCo@V#(58`IC<q#678go_im9uGiXh8|%4)%<!P%m3zF9V2@PXGrda~8BwBA;?REr^?lBemEEjYw$RuvVz;=wOmrz6Ymef'
    'N<```ku#$+9IAu)3uBEx5{OzcD%s{_^AOB;Kmye8%ehK>%CQoli&&k2&8oB=SuTr}VT7ks>wh2?!xi5$;arxjUKa*Fa*_q!Y53Qcu<ga^&jQ3leTDTZh'
    '#_KG4dck%PNthOL~F5Jg+FhXg6+=wg%-oFNs`r>vG3O6G!8?)fn!_AJ`xg(yvcEg&@w)vQPM`bb;>SS~pup)s+ju<`ffYzPgZ5xl6+vkWlvGySn(=$~Q'
    '$OMf#wUNrTI^At_})V{_da3@@E6pcF=<tTC^WbFu9G)zREja+&>8Tnk)SaH)0gOCL`-Il5dZn}|@VI;iX>Sc%8J&z6<l(k0%KiwpeVMFjWdr&pOJnkQG'
    'z?he*)pE<05+N7qhU&1nF852#exc%SSd>{PxLfx$T!&CGDm;jUnw(@vmc@Je2bS-U#~d~k2_CBxR9zfYLZnTMZznWv-6v$yR>Pi!I=hh3UP&HpvGAgd-'
    'ab(FhjvA3!qnLn2el`<K=Xl{1_v|I6I}p`r^YT!O4cDYc3_k7JPeDTa-SNz5a7;@aA#Xm1o=)ib(107S<8td=c$_YREBtGsqPQ``Ke-4Er0zsJ^DvJGv'
    '+0bQg{eH{UsetF)!P>gh=XK*IZuYK%Hhb)$%Rz84Jbot)$&Rt$b9ZIt+sX;?y~)l#c>`hXLL>wQ62KVcum<gLOtC;^A%GichGNBKdmy6DX3e5jKOe$Uq'
    'xgD%NKvzv?eBzIKWo^3Wp80eO*$mPwI?Y9xyP;!3KKsA(FsibX$s5Y<Rg?$<nrVwUgQ^;#JEDpVtt0USx$m(<G?hq2UmbB4>7M!CWmFYJCc4w#j?MnV#'
    'y5N-)cq>HDj`eZ}I+3G!dThL`GBVN~B_*FL{Uq}^kPa~B&;t={y*rv)T^E6yollr$KX_FZvFkI908GE9<xOxc<7rnG)2pE{YN6O-Zrk!b`!>6>NEWSub'
    '28VOfsfnNVJU#_V;*XPzP^k~omI@W|0aGAPMW-G;xg|RFmMSO3<+SJ&I`uH?p-@~Nlg!&tsiPi*M!K71FN!AvPKDZX0CH%Q#K6(~OSq8^Wo!<Wjot5u)'
    '9yO34C9nt2bQ23WiJWIh>Y7y!YLw2fJ=yZLU98wA+Fqf$|`5ry>H0ekb1kIOnQNKF_|p4EubYFsku|u8hPgsDgK1@smJXqP$_A>NRVByUv1P~6ei`wE+'
    'Sa*vi$;ujDNCqBd!$(Wjmo{4IQ<F0LMic>COOZ8a}E*@Kv0Yfk~aMV&x5JYq%+ccWkzrtO~eFBT2hLpyH{l>8q-^Dx>yz*xIa>1edJ}*U-deBaN@+(yA'
    'o#MX!skk}2EVhoif=ek$pF=vg$fGNF0ls)7U%U_C1n^0q{F0ice=zIr0i*0WeCjFn1qUns`l){@>A%S|{=GW;h0YYsh0ej!9qS}J*dB?^+Wi7_2@L5-%'
    '1We6)r!e2;K@lDD33#lU{CTV{WvJj<{%)j8Gg7T64SB`e%qe{(3c`roX)B39LCaTysTEHL3r7OItCWox>CO&<oRkM!VE1S4n49hU?o2c{frYI|4y>Dm~'
    'Ks&IfXyr3^aa4TWp^*=6r~GM>6`zD_v)2SMG~s@+j-}$-Y~6@l#kUzGK4@TcB`h<}%_xrp)DQ1wJHh|Vy)WvX758Rfdu*O-f=uK&D&iK`7Y<IqDd-AX$'
    'e!aszU5RL9a}O7w^gS=C3kRJ1O=*O4^Dmayb|xV0i;R>Vbq<0{zndB6i$FWNfu$x=u{<-us7rKl}y5@Wnm?nODJ+$HJLDxo2vPQ!@Q5$G~06)no#fweM'
    'o*+I{_@s@Tx@Xt2tgR!EMd*s)>beo>!6iS2MkY0IH;V1=F!{$a8!dm9E>!5hBfP+F{IO6Nw=yj3mguB+4dfkO2|n=YSQso03pTgp7ce%cf<89SE|dLIx'
    'QDo=q}jlsxe|%G$avoJv0Ai9k}xh>UW{YHO1jdf{$qRQKISVl?plm0c1vMk|^8>(ae#WuFG^s7f+_00H@YOu$K)Aajz?-xth)N=AQQ5|)^p{uUC31}?~'
    '4)};0K1RAO&_V@79(CA`XI6hb$0ntkT#hu*pvneIp60z4%=pxo)ZG!u(!X6V1UDpIDCLFphOR_`@hQu|hL_~WVNnR8T^#nyGCfcJu-}|yOi<N9tbV&ee'
    '!=me@#jGC5zL6H`SF25NvCC8liw1X~o5JLnG+2p>0=`98kQSll`jA8B0BmSq(igMaDz-pN?c(H^giwi#Mpf_Pm0cGL#7BNRG=*qHjh~7`@f-zfhCeEKs'
    'Xe2f)y&lNLsfE915|*)B|9}Bq?Cq$kYr${nxeXy5mrr7-NZbqrl}6Y2GvZ}z!dZxbz*x8fsiwOPf+h6h_C*tiv1qFK4>t{Yf1#`V>Q|<7JL+qZnh4)0+'
    '1~>d~j@}1^IQ;)CO%9BR=BY+(iy`>*rlvT>?FW6og<}3?}i?m{|EdI0%~=sg-zXcvN^n*VmqU4FS_|vcb^^n4Wud)Gk^?ZWq$6Ya&aa5-<%nyn=+nrp3'
    's3C1e^n2=JJ0b(4XhN;HzPWN1ZwGwMyf@yL}(i*%%=Z7H5~wbm`aS6^#yNs(ZI9;RAUB9nVf$q5)}Fd@c*Z2R4SQzbSzBD(ICA+3t}^wsEOdWUNlqFVr'
    'mN+@MZB+N=IWgIddVky%Kb)OG{DO&<BRiY`|v84(dYibO<mX&zQI8M!%Jw!Ikh$<14sh9}GPDmxb0OfWW`2juzjBArGDnXTCWwOd4#clG`P2vYVx`|D%'
    'd#4gl3$tGdE;_xlPC%_?acd>0R?|&iiK+$OMe$s60p$!_U^$WjlAC(upc7gP+9_U{E?F0iYJI95IDJAVMzI!LY17SHary*$0L)l$rDkPc#px4{h1GLe<'
    ';(ow_o)eJRPp=7LgD4&<PFzoPfS)+9^F2fAW?DqK>g53QRE?|Nd<^<P-&L>uGLJWYU*lTQ!ONUweRP-p*TbenZR0dqY*L!z+kckrb^;yo5#9gsa6wNyF'
    '1a46A=32P=9P%9-E(2)0s}A$pTb>SKY}hb3q-;$m<4f7Y>aKRNYgFz*=!ng|Q=F!##E6dc{2jBmi%r?6{0%oFgHY8QdY(ynMzI^d+}pFw(xPJr!R=?dR'
    'qL#@d%^N@dMpdzPBljJD@?K(&i5+LmNfo8jgZ;D+4mo@_B>)4uB*s$GQ9q*$Wv$1a9pCGNg22BoU$rI+bPHr3p8r7N3O>5JeyP!)7_PtV(7D~+D&`wrI'
    'x3z-0X+-J{@6L^Txp`3HoabJnS*Tf>M1mOe7FPU3n)%Zn1$ohQecTMZegyJAK$AuXN($FUUk0Yfv;lju{N?r>Fao^G`CuE3ar+otoaX^tMq<~|aYMW|b'
    'PaNXh7oiJ8yjUk&&CLjVtH=|9W;OIy2_~;KbpV3Ny=4yxCQ}y=3q}Ws7#S>>g0qo!8o}gXn8RmcwT*Ns2m{~UEQ??=Jkl!@-qdr{@UBT!gc7OBhRLG^M'
    'CmV$CMp%&U_G?}MChs(kT~-M6Wd9QL=$5HHt*ZDR#;cW6GKOY00<~X{0DW5h+;5B8#RqYT{2DI!_Xy@Fea{dl<a!)U<FR@AENiT?a7f5IL(^IIutl1&*'
    '?pG$$#)3k5qGYAFsN`tM22TTN&s+?rALPKCXfxR54Z$Q%vhn;)udi351PMq;(brwFC$Vvp{uH=4e#{VRM?21_W#n3BD2t3%i8PVpfeHMpnQEYYr+=u)t'
    'GUJ-2b0?13m)Kb;0d!2*r|-G6nHnv*mKdyl29L11(%r$>tlq0>qrFd86iY!gFdbj{3{N@R4^i&qJZhG_%mATSzGan%=)5R!YR5E)&IfwoFyG*%);Jdx3'
    'WeR0ifWNuoYR|$<qcB?8H!O;kyag6`ScuOTX8tGuEGvR=aS4hR1@j!z_b=3vE?>QFuphLQ2MXWlZ>p7-)p#u)m9^r-#^d3YK!O~I-8EDnyj%CUHCCu^('
    'Vq<_v^8wh&4e}P@S8|gDQhfrbYq3SLk}wU`WS{liFOO`WmR5o3K1Zs`=KGvG3z_g!(>F=R&j?Oyi<o9^okgHwtf?)MoE#QP*MdNkNY$M5WVkBJ3ID=aQ'
    'Y4FPRC8(LUuZQK9{z<T{W0Vy2k&tC!V)4S7xk)T6bG40o^m}kY7&(Lu3l>f4+9+`nI<5_s&h!XnUSnYGEI=MghRsFA$)u#nPwd=9lfNSg`<}Smb*!Vm1'
    'LT@5Pg?qnt&hF&r6cjQgb0w%e$NMUP-3u=N^2kJ4%|nUx`fbd(Dw0GQFB!U5DEPTC`D#O>Y^8uLh?#O%;RFEjPh1I$gKqVd(Y&%;c49ZZ&f1kZ}nDSqV'
    '`OwI{ZzDdr{Bp@6HoQD7n7ohwH~+ucO}N)#$!jM>`H4c&x#u7sfiyHR2z<Xm#N2t=)0%?&S!M76ZO4Mmm3z5fhFrCm!bDiEmb3BjoSeD{$)9X{tsOBu('
    'Iz`^UDAE!ekDTCNfK+UmS1P%@yBhQVrVi;vTB1hU0dPIy<R8fZ1pTHPO&IXQ5y0_i5fX)ITlNz==7Ua*AzrZQPj*;Zuu0YFe*)NDDRn!=1da)?I`n#|W'
    '#AagAZ3a1aE1U_mff1<VuYeFZGFX%)&Kyz;?-dj{lez*fV5TyF3aFQ+?4mIc(E$=rw1NU4AT><|G{CHK01R-ouB-qCxQcmJzyj=h_ANI<HFmWwmn|9#K'
    '!6cD;Ih^Iud;!RHEQIBdRvah>b8yq;Q#?#-S)5Sf+A(u4GEjOrq(P^C%$5NIx?)*O9@e}S)a}!g-eUKQbDAHs~*=|n^lB3sVrVZ=>*~<O-C?3T1JC##!'
    '8}Ke7KAXsJA)Mhr~bx_>dR~p&cR=@R5hLudBcJ%aj6Wm4SZXteVpcnpJUw%>tNJX6gfFCDus=$SU`V1IB9PG_LMRMdVrFM4hZ<SK&9C4xrdllMaS8V-g'
    'ywelc|el-Tg9KvJmr3r7^SOVVS%ln@orVY6Vk01iu^Nc1(T>jrejwLuu%^3I*wM7u2<gv2N61}O-8qHcIxPWXrdtl$}-W$UqAd5OB=?-m+EXly)|&&N4'
    'r{v?RC8JY*Mb~h&oz{brG?6#gISp}@dJ@X96K+dt<oH7pSJ*fk`+^YNKkcwZj3ufW(u3!!=*<~bQds!C(#9_(&D|e<?OzB}`ilPI*urEcT{T8;Rm`Y*s'
    'o~&Mk2rc9=p3SBdfrjqWtc#Y0jS?DcW=D!*nP52*IYIKK;4ldee87^)mu`qw^U(rtN<c@VN$j4nuH23`AI&7V7uxxNb}e}<?u53OL*PbenbiX@pxeNd!'
    'wNN@E3I+3Ul_96pGgXb?AAMbIbnqgD1ac+6c7Ma6btZwBl`w?yHD?e5b@H!omI01$iGbP2Jml9;>D*bh)Q9{gL7O2J*~-Ug*-LOL%`D_3li>Bq{<@LsS'
    '0D+vmP+tlcoHg$$~%ykp+P%#l*o_luJ|u!jQ}A{V3+_a+G!*dmOR(No9wlVzPH-gQGNhidD<rYt{1VvFIOO&t#!~6qyaGbg^o=_%13Uk-cnaq_3y|nds'
    'gr1;E5iM+U;rSX2SPPvfls-<RA2{JxRYhoCYc2=Km4MFP6-a?NoHYhe3QK!e(MGVM`6I3~WEB7&?~F%b%QeU-3W0j)3L%F1C%Y%V-_?IGVd%!rlefYcY'
    'z@-~XK%G|Tb2V7n_-LgCnn7*QISU~B^6xs?pY0EvO0Vj?4q-V&;aDzQ}I58c#y1|J>2GtEtv|Zia#0t0S?k4H^Wpk6-QzIh^n_H{0wMk7m%+4k?1u+|&'
    'BpO>``<ru>mU>xsWOt@uE&VJHE`X;Bc3gl?O<f7#)Xc#LHnk@cK+Vp_i?|>wD<?`%0WCE%?vF$hf@py&)RF=LDR&_(Dw|j~iGbAn-Pf`Z_Txkms30I^&'
    'IAZZ>Ke7EY`Q=WK`y+$pNK%UDDp5?iXu<$vx&mQqS&@mZ7y#9m*1t}B3I0<{}iLgeyQ6nhk}cYFZcb+=P0=p2!4A}ZxZsx<LyLMD(?CB)KA<_o?45V52'
    '_o8f*JOz7JOsKfhf3eP|t&QBF$8=1pRS655|dIRzVT;$NfAIf2D$_-qYO%oO&~#u!2h-Iq%S7G1GqYGTdaE;GmMX44ME+UNfo$CV5R)myqP!c^T>1x<v'
    'sbuer%Mc;prEg$f#Z700hG$;Wr^M>{0G1Mu;tA5++4Nn*KzP}N0MSNlesj^VKyHizTH%BhPm9f!X&WAsg-SzYzZS-f2VQ_bDAfl}=)mp1iB_0(L~OND6'
    '#L%&-?Q-RR$X)=SL-%lHDaj)&e`EYmU;uI7&6Xj`OlO6j1mfZD~@77nW=gzI#a?EA^&0~wqWkwLC@1qbWXM7bbk4_0%<NylS;iQgwvX%;zXL>c*n9XwD'
    '%D!yfks{j~`N^himLJzeWIJhQT#UlHh-^pp1AxD4&Xve^_I?v<jJJY3o`WS?NB%xeWW(wv-Ntx<_&r(jcAPMQ6>v?B>RK+Mt^%#8Dp_l}QdgA+=QUz_a'
    'yU+4nOYOo%t!7gYFe$LnweO@);&s_S}$lUwF-LEG{}{(Qaf>z^*lq}NlyErh{gZIHq4c*(8?~%B?Bnz!d!C$R(4^|-7zcMFqb105z;LdITG72Cz?=YC+'
    '19xuI$8I)7MiHqfG9GrrpoW2$!UOlo(~qMoCS(*${IA?^Y>Sb+{H&O?8D7GfmC?V29&#gYR;pl+hhg%9{7@kS~V^#Z>mCE|=Y=D;u!(np$7YrG^9&F+4'
    '>XJS|;hMZ9qRRP4aobdgkcU~N4`WOYkmM=*C_g-qRnHCGcmu(p!q;MO`m0Nl-*sw%*nRo_FcW^<)UZ?avcaFT1+<S>A5o~f3gZ&qC)75L3+fqw;mv&t#'
    'K->f~KY@HNGBG0ChT28K6T%f7}F3{cMp88}hiCg65a3d%Fil$SGXgccTqUj7Zh~>l-tVJ{(X^Ui>)DHG5H!7z&Wusx-^MWR3i3)mJQ#_UJEzXN19%fSS'
    '6&$IE_kjp2iU!}-uPmczPr)XN?klNNgUa=lnx%`ND+`&MVnV=EkJT)@4N~B8+@XgQ#_aeQ6{$AVEO~G$mfY?p7P+2Q*X^O@L*6B#xDa|BndfBu^dx|GU'
    'EP3~mJBNcV_GuGTix%FfJQZFnvN~EI#kG4w{FcRZwUef<8zZk`CE05Lp`IwnU;(I0BBlDb^uff7i6Nq`Ya7$h}L=22iBJpv!wz8yXC5gdCT0|sv&HgU~'
    'zwU9J!*F=>8S8M0a?j)^M2ML8iI+2vlxCBEv0Yy;Ub=H~rTV2|ddEt}j~;<!WHY$_d=lLU>TGZgmD_u2xP_+a{O_6$Zj+P<=tC^tDo;t$%>(Ewwn|M4$'
    '+~nH*MMOoELSw#2_VG?;n84T1aBJp)8;%mwpG%y73QHwObOpv3?WyP29&AZ)RbXONbNLZ7gg{to1kzIDNWGGU*sfD8+<nTVlncNJ*v#tkreCU>~oB**S'
    'oG5C<3-}T_S8-_)~?zV9#E#h}Kl|(NiX|1{E4=HQi>K`)Ja?4Zl)tcVZZY)1bo({`$s7qMf7Tc<vD_Bm<`Ib%(fXYVJ^mmT2TsCX)@xNdUm8>}uN5LXz'
    'p>bAgm>|2wdJqc?Ey1ux9;AM2%LJrnqtc*iO5ZZm$-+0c0(bWa{fiX6EoM=5@d+5Iae+zzMMGiBF2&#>bW+!_iST%sc!k5m9k|&7!1NkG;Gmw7HH7oMg'
    'UC9>Fpdj2%x3vw=!n3-xw#1PYF>z}l7u%mSC3VKD%qk_n+rEaVawpc?2~{iAvoM4nMJf(ixT2tT`!9hpiqNet5vu`O{?3|2kTj_wTvE6=%`glnPX=bhq'
    'brywjnLYd=hClFWF677sO!Va|DGGs$NVZ&1#xZu81Y+Wl3a?Sm3)nB>3wt4+-hCTdf(L$DM5|=x&{OC`353U1<2Sm5!*1)FshaMJvG-j+)rAD0s;Y+Hp'
    'ytCKl2R)e;wpPc3m#X+ujqvN#_^9XHl238(`_gQsIDRif!oELWr%2px_n_BHC^(Edh0T=Dmz9j<P4TC(bupPh~9CGL?}TChY_WOTwVy`mC!lL62Od-`F'
    'jYz22S0Y`bIOqkDf@<T#Uga%xj1^vOIEIba{5xXPE!Lr(Uv`L%udAr+pi^9cY#o(fE7Wd&uw6sqMVvF7cu@}&0h%f3M+`x7U86d=_^uh0hnT08|{&;gw'
    '1y)5B>rv1l@fR-wlv0Lau!u^AQ4?`j&@!RD1O~wg)09!-e<`FOf|dwZ0Ii4$g5TF%N4=d*2&gXR3S++tFb3=1#p{BlR}r4TVh|ijRxcb<74u5aB{!WC8'
    'Oa(&flgM7bVN29mJP1r?c}ai;^mm$xr?D95LFSiglbQKL99Tof;fCo7D=UV?THxDbK7_-+aV9_f#lmNJJm&qWXJd3Qw=uUv8CP0!o@xwew2EdH)sAk`7'
    'f!L9abt@24bR2)|nK`3ay(+gd%f+ksE3~hv+jQF%Jv=LLEEU7A_PyqF)=*@35#nRu+yxm7Bzv3I3S~Gr<zgn~>_4<>HYkBCjP|GVTkt;M^(a7N@MH;s&'
    'R@$n8{zXmoCx4RdhEZ#QYLZXe7lb{ob;-Hr_*%LP4;lxCBi$!G4lW%s6zwYk%+gv{oz(KEm!u0TH_RgRL#2EuLO_zk&of(wOU*~OA2jIphWAy6&N76_1'
    'qYVozwz+!Qs*{D7pa!Tq_uj{wC<Br&?n`;W-NxUt&t(nzw!e(O4nTt#P%bMGV?D`$tZL?j#T3TXE<Tid7(<q4eo^vaAH_37Mb}^5K8>oo|ZNwy##0^)N'
    'Ts(9edq($pZdh)Z?r?u4%w>a<sjurYDcPuQteLJwv41l5AUKbXztt^Yb;pd{<`-|2z&zDV4W?@`3qybv9W#tS=21~3X&s8gpsAcF2wIIID`^6%!DG~t?'
    'X4@Wq+60><&y6bkOpHH=da})Slm&VL*%wu0qKcu5~rc^^}6LxF}rxrD&Ni0{cZ@^I#VypDlemmi>X@%Ye~vnGNwxvz@E_!QZkK25Gsfo5#(k|L^rFY1i'
    '<kGTiu@N|Ce=vy||zCr^-yTf5~pjl<L9+n9=Q^V-dM@Nr>zpm`7a`!xVUF)qLxX$5gYlFK9zJeTI1{_<Tm@Mx+jGGMTW4nH{UDPa>&H(9BMkl*yDZA*O'
    'c~eaO5`0Y@@*O$z|6f4S&(l!I-#I|LaC=XP(F|1WOu1|xJQ=QMcT%j<&7(M)Q~y}GRYcduuiaRm!3eQ|6UC1ZArHZ3aiw%S#9)CW+rcyLR=K0e3P`t-Y'
    'OCypD9-I{oANffsjYf|dv&J=s)CTWy<9fG-2=yk{wDD%R5!Y0axu0awwn<#CRcSS8ezC}uPi%>LFYeyA5i<?T_CC?mI7f3EK-jPQe4?9HsK@uNb&L7>2'
    '@jlWGO@YvPEO;C;`A{JFw!_W4nG`@s&pj`wwgavg*YB1avMq_10Zuut-^@X$Ae$X4xojlJ2A6FjXV};+gNEx<`PKErH$m1!NiT_@z+PlEOUQ#lL}%}^O'
    'X1nW+zeZ><<@pK@N~&MDRYThXf)-vCgeiUQp;U%*?zbs!(|(_6Z*yX9BHqF&8q23sH*4wiNY}#xP`k|B~TtX$f6T~R=p}k*`a0?>(GrSw9tXnXHAa8Z+'
    '1-zeSE+=y12m?)?q#BmUzM{Ey5BbQ61Wv=#}_Zn!pU~Ak(!InHn(~OqOhATuG0L>XP}-&5F6)1QURUBX|0t3?+91Dq1P-L8YDIzEcj8J0-zt-DQbfYyr'
    'y8t%-AkkVG(Xn-7XVMAWY7j>xTl#OAS4a!+J8ei4;ob@LWK#JvyP&ocK5T2lpegD$Z`+}34NYYIxjzp)%pyl|n5dqN8rx|n_~TIhn&*mGaZm=ntB0k;U'
    'ELd7p;NcEoPEnbl|&2;vs6e<*)=L8B>Op6nm9;8*gSaxVFuKVEc)H-f@$*FYQaQ<eMqt2G9REb%mz%jRPq`GlsXY-yoPD3+!v9_ON9Nbdjdc*L-F%|42'
    'uZ%T#h+$rEPK7hB+gdI3#X!bE6^Ix%wuHIBpr;xSa|ahyEPYaXa8gku#$}=_lyncEkz^pt?OiQCcXQX=%iX<Qi;hTgw^~+Ykh>Lo_-<U5RrQ?*99Fdk@'
    'sqn101q+L<ZcDas~9MhvmLM>a<_nnqtWvW_fXBIB~J2Iz$3$Q$lnU!C)O+}Tmj98gmq8Xv6{T%magltbjlrFd)3?LhORxq@!Zd~Z^k;@&b6mm>29uT!e'
    'P6aYXD~OosezMtr|$|qVPcUQp#}Cs3|a%(7rU#-LqXsRe_~kkni9rHA5B!rV_=C0%udCwCF)C&5Z^VG#SnkC1rRonp+K~+Xn7ZvYMd4U&^U*%3-!tABp'
    't#n*Ch%tgJ<wB+YtqYM)7>b(B_-;YO-;lzuGwr@CcBBw3Mf9mS45WJp4X580%l#S3>u<>tA=A;Vv>sO2VD_G0~<g;r$IZMjzq8DL`<c*_*9)%+#5am{*'
    'CcX8bWTsf=>63HaDadm@ScX4gjvy%uGJ0NRS?;?|LJncxthj<$40NspsvcXO)Kk}?1E1omk@Umbw?u8mPSTuGw2<1JXho2<I+Ma`l3xK7)i$O>}PSlt~'
    '&wE0NL~pP)RuRX^$cepjS=r3Q(6vEMPEBMSx0)U9j_6LanKtf5vrV0vytS4&V26lgGBpy{0@z^TY5NDA29E47CQITP6?hCzq2w-(s$H%$nl!=kqmV3g<'
    '>>KlHSgeC4$J$wioEQh@OH=HmAne~8!pzU<#xmAP7$O5uI^sL2^4H@_E6zH#H}|eg5>O$aE^!iV*0_BCAUF`5s-5drSrHHNG}>k!htVFo)WT-1fzA^a!'
    'D1yCo_GJ6eDW8IETk|NgyzijAFN3moTxYRZnPH1ZZJZ)~sswWpMdpJul1ZMC28TMuDYLbolJbgW;8j+i1JCVf9-&4VJSyksibA*Gxxull6YCU$}ki$yi'
    '|bHJvVQvR;kum)vAMGvafT^^rbtH(57@h+O2reL{c$kI?WIksR2Y_6X^LQLoL~0zIPWT_HiR?QXI}hTsVaL5g4$%Q5ZdeF>}S&g!{QfE%kfDUB4BYW69'
    '4g6uZG;sD>p&tX{H{Ia=?gqwz@;4N+&nw3KZV>W6w^^gPZxi9HE2i!A)X?WP#u{X0L3RZ1UCONCvwJmoYDkd>ZO4qwi8pDzFW*)<y+6LKOQH6|)j1<wp'
    '7$Fyx9MM{>*(uBE;N88lE54vOV);uZXD<495AzI{p5PO>M935k7ZFMCq>4t-uyV%&e^t#wvglNK`~<Q^%eln|>7ubzbl{QhwT_;v63qBklIHXhX7K_^U'
    'r{$cJV)5AsZf#lwXNq`1A;j+UU&mh@0u9Yrhjc_FBcNP*5jmOF*g-5g(R?m4}v6g*&kWofYW~6b;uOl_5lorA0EejKvAeVx@kdf;(_x%l%K=aW;NdCeY'
    '?-}Hp=#tL&aQ;T}Vir_-jqaC~o|<)w6~3EHh-OLTRZ`7xMpN!5p>_Am?PV2~X82MhakP$qIb-;aL*7pCrH#V%2lI>_gA=p^7AXU4(c@7FvuP6kF)KCE2'
    '7H<T-V2^*Y=qwg`rEByO`{IESfZ1;<DfRu`|_)pT~BlE`MTHM}DMQo&NdKT<PIq2M72y?ph`CKe`Z6dBzTg1QQ9mY}x!N{&Q7HAiq=b9I5q+P7SXz;u^'
    '#QsyE0mh-cvBImGdXRRIsZDnw72|>zWfEP)%TgrRYy6uDj^eG9rr6hAV_a!MB>tsK<j&N)O8S3IWH@eWr({%h5ojgZkQHscyF64rnXV-IH@bkpPQX5nS'
    '9nEqTd_5)ImPJIq%-Ss?@@2AE!Q;cW4!%HWbw1ok?2@KZF%}yo#|R#yu}jOHZ-~bp-K`o3$Q~znI}zEVkFyol|6*NgAt<{aP3ta*b3cI~jmtKZJBp#%W'
    '=3!+qLrEeSqRSt#G3{q(&^ZfTPU7--iLfz62Fu<ZD5GmF+|js2;M|%!~F!OBxD=rW%J6_5o|tXd|-kQxCNGlw__h>^f@6~njB}*iB(kP^}&!WP{`pdF_'
    ')?=;DCm$EGs#z?G{6-kjWW_RtCbi1g21A39sK1h<FhVblJ{8IWf1zgq31WXU%++LX3-E45t*&(9n)3hU`PGKz)QUgVP#x2Q_eA@-^<FlQC2xnPDzED8L'
    '3Dxg8U+Ow>u{QJC-K2u0Yyl41I)j^&UG39}^Cn#p*wd6L(97CI0J0}9M5QxO<mnfMd>C$MH(ZmB^)tkqMHj6})wgGMUU!=!=qYY=KBFcvHTzD#T^!Wa4'
    '|sB$^C2OvHchKNE9b73S7G-*He+!;ep?3M*SxMG3hA`rc#kA+EbC+sDQ)^y08JI?5l9a+AmOZEiA)+d|6-=b5tW+sqc*;57AExRP_YJy|#r`2BNV~0r-'
    'P2tP15TIN0Aq0Oy4`5D+(QR4KTa41B)@Y4LO@*2GPk^JiB#(q(<M}Yn%u6ViinJ}AK@_8&js{{tp@{CDTh%t@97KwH&ukDXGrG2JA?CLxR$C$H7kCu%U'
    'g|SyT0VsR!mdU_RJSF`5=8=keVk%|!hoI?(bb(v(knphFQE1opNXY>q-bW-@{%q%0)Vjzj)JD>j|OC`5Cn|v(DD_`5JnA8HE()5m1f+(*cm=)P1qXNRT'
    'ZW8gqf~7flp>#bpvnv`6z-I5!2yCBB!m1D^IS$zM1M<NFxb{VD9jS11o^9&CI}qwX|e@B$OO2x1AuR<ml(#avkxJu|D!j_L93DkXf>=BPqNiK1^&XW|!'
    '<)YG{7Rnp;YlVX~zp!>K<O!OaTkp6yKDAmI~*!gI|OM+X`Q!kL>v-e;^*B|Kc=xFiH^T9&ZKIufkM34ts0iw?G^OW7y_X8A~53Av?aAu0yXH#>>l{8}n'
    'x_;O>XKj(#%F@tP{?2dq;*t~LxtI%w1TH<OVq${0J*kpP1U95VtJo?B4E7Ckbn&cs5=Cma*Y(k4x6Q{Zcs{;mxK!69@0_sjck77peMHM9(`viX`N;~#7'
    'zm7QVsYM#8ExEarvD#6SdJWem`z2r-STL?<Lbm;KB~i?DaTB&}8Tc&i#N&|w*IfC#=kUE9>BL+UiNgFNBC<=`XvBYd<Z#&vF4U7p7z0YEY=c1C>H8FxG'
    '$HQ|1Z~NgjRo~A87G6~1pF-tSbMGo6LnLTivHT#=|N3|YrU(a>t-2AA><M=5xQ+HLxv0)L6@NEkcj}rm_oi;p4iC3B;h~001AjD-J82X@8o8RLBL@-x6'
    '9qEiaGezouK2WXZ6i?-31!75BZI-^LWsJ?V`q~TX95uZ(GhX!k+a;q^H)is;+Mr2#Rj8(aLRZ-^s7zEMn|#drvH8yHs7Cy8_3<EMo|Y>H3|GIO26S;T+'
    'JmChHQijof_k+63$Xt*z;-5v$D%43OHIULbMWn!gJP4!L&`O^8+DI21BicsA>qL9Ni>2793~I=l5?F@NAL=o|xe!Gvxw7u>fH#DxmG0B=Fl0JH_!K5z>'
    'vAfPSikO6B!ND7b^aIlC$8nY(;XiW7GCZqGCWmb_LT=y(!9d7U)ZYp4ftWE$^NN_*^h4fQ^CS({BD51>0ZPt?U>o_-oc9&$4&jqd1G6bDdwHjzntBC>T'
    'G@=74r$8b=a+)R$9;YZbpm7?d2a3})T5vec#S?^+?-YU)+5^#TFjcEHyd)>r6K7$yfHy?zS}!%MK)7i!f#0UX74SC6f6bAM1XDms%~N(LJ4w$jm3W&Qr'
    'Sq`mf_Nbm=u>_?P9TBRBD=WX8atAvt#WRhzxFl@?BccZkZ|Dwe8=sA+Hm`R78R`)HwSYDHLeJC17=)Qj>IKxz0{=v_8-UsTqBl1<1^jDeBoFm0Vb|5mp'
    'zBqp3n_seP^i=$Kn_;M*V^W0J1+w4eOc@jECor=nQ^VSjG{M9ekU(0teT!X_dFFfRL;r3h?{^(FMN)No{wugcFFxySfMZ!XZN+=eO(3$>ie8wo7H>x}A'
    'tZxywB1?iOo7sIs_N`+%gy+1&Izj?Rs%S2sM5#j(2IdC08Qa`K2buep%o&^^3k9j%sN#Sve0P)p#J(edkW8O2Iu=NaY;%*W0Aj^Q$jXAPGTx2@O1OWja'
    'zBOjx?k!{M`*!5*Y2yDmYHuifl(z99HO^t!2-9by+aY0Dgg|vbBiD(1+1bLTLy<6KI$}ao5D3f;Ada$omR3})3#paD)x6341YR-HJ=@buJ6VWMtwZ@)g'
    'aZBuZt_VI}(67HDLW6GuOQsbiCYkF5_k%pmjk5GDx=ogw+zt16$lvP)iv<|6VwyQ1WLeq;23bMR9Xh1oXT#b6K4(}O;3X~~xOfpGBYt+j9@dg@iv<_Fy'
    'UCc&;eHKEl!;jQ8dL8kGQ#(`g$&}>{bMvm-95$`b?+FFiQG9xcdA>*XnwdegU*CC@lxOi+gaMUik8Csb(Mqol|Tv95AtJoX@W?FOc(PJ4IRKoZ6*O9(R'
    ')ODL|Fpr(L`Wz9$6pkup&soF8o7q#of~5U^%<81JSPRsQelu0HKS=l856|?!aQvY!Zn@*^h$^SqKVC{G<$CDR3!g2Zm+VOc<do2}T(JjaXwH&ldq54n9'
    '&0#55`fDi<)KDhJC+&qF<uYZKvu${OB=#zEy7PXO!8Xv8jJ)<V}Wpceq0E%6uSBUAuD&dv)3T>#*WzaVqP0t4HOpIZJE&jEbrx@VCS$*vGPUa=ekePU8'
    '%*7k-@GP1Nc=5%B&=f-0;ju}bNZ8isDwFEqtenKQm!NY~XA$cTL8OcV2K0!tvUa+2ZlcKJoHv+lZ^N{JFf4J<inn7BX&V*To9fGl{vJb$jU_qd(YR~|!'
    'n%W)%ZU)A1LmpQy8$|3Zjp1fI=-Xm;D;q@Y-Yug_&=*v_E^1AzFtNIOUOy~`j!)0BTsTIS<?=Hj8wX16%XfkjX~*Qgunqy#;!+X47^tJ4SV>|As>zZ(>'
    'vYL{JBN@M3x>>D77SZJSux87P(p08`;$;V>>jKE1E{DvAb_IonG2Yx(j6dC)m~6UT_r$>x}v}jb-@50>b5H&hidn1uHTM9*H6@g1ni>_t&bDC5466X3x'
    'jVvI6Tg!3{Ic<bD;DYtpuZw_yPcZSOlQ+HC_ugpUImb^W}Nb^e`i@G7PlG%m`rH*JS7H`dXZCHhnEJWA=P4&H-D#=FWcX_*z8tWW(1YCt<(WOiO0F*W%'
    'Z(+iS+_+3Ypt;-)tY!`9?OR8MwKRWD-iR5S56wMyRE1#aVU=z_tv**8`H1iPkM=dp<<6@==JseYMQQV@(#QryO{CUr5cidc4W%=i|YI95Hn?BQ7D3~b>'
    'Tz!h;kELdl(hWw2&AN8`v*uJsqZDaSwYF6x)KvED|0!cv}<^Wm8xXT1rj=cs_F-C{D-m@qq+`VA%6)U-SH%JbC3j|*=l)t-?n=;L9+$2wIK=+Ubg73U_'
    'SS^oMH%?#~HeTzg0cupmE7{;PG~V6y#E-5g<gM!|93*?Lrw}m|5%V&{Vf=-@>8@}i$at0#ld4@1U%81h=HMPDVBxl1X@Qbs)UAz*V<#xR9~{CwAm2n<;'
    'kI$=pWAl%)=kp@w~i<)T!7xS<8Nkrw@is=cehlkwp6?z@ZF&v96QudRw`Z|Z)%v_n=357VHY)$-w#(9{OQngWAz)PgOeC$)BoXyd(MObHY>`Eu9-|`Xw'
    '3vF<7#+Sg(Wn#TU0`$vVsy?$#|re%OexRE^=_(md|7_!mW_}uDuphywwxI3M@y}%3XjBe_gMK>YYIk0%8TR39hlP|Nha<!{hDa`-huP{w+!s<`ZsS-%z'
    '!%o`+-^o`>~%*z&NE{(ByN@N2&wuGhnHJ>0H``}Od+90rw~hvj;BecOJ0+kSmpzr13*{tDT6h}Y^otk=WyVSTy0qA&XgzxM0ldOaN1!|i&wUk{JV;jvr'
    '~%k}X3w&VJ?<NCJse1-ed=V4hchx_${OCQ_|-b0wrdHBJvmxt}whrxZ$!+Jez*Ta52T(5`YdU$<XxxTGj-nLz4(~WoXur8OwV_Xln>*4t@5v0$<4}QHo'
    'Y`Z=T$MJbsuZQh=*sq7{^>ADduWzf@x7Ewr#$`5%mgmvsQP&P!-+y^MwFP-}{p~XOm#6d#SyN7sM;9{d7b;sXM0Q!Y{jzTRW!X|Y;L&yJmdlT3ka<YT)'
    '_GX3hZmA)m**S)I1l^vaJ?Rm>*01i+^>fhLLt;~9+vCj^=;$&wsC!1PTk~gZ(iSg^7l8tyTAKC=85-j-#pGg4|l(x|K5IoH~)R}>h&c3k8ZxddpG}meD'
    'l-Y%|Fg7-u&y`ysvyzK5FxS^WXKO(&mZR^MbbzZ{EEPe#ukk-yd(kyZdpz@Z~)DBz}B<_x9Bv=Hp)7KHh!*?*0#P(VMrg=WFM|r_;`x9m|_&-+cP^?fc'
    'swZ~pz?|2;eT!Bc-b+fUy8?#WMgw-4{{@BZ9lzj^s5&y62$zL}T$*at5=pZbC^_z$1-5<~RYQv`i<^Vd^R`tYIg%$M&TXJ5JE=GV7B-aXuK-OXR${X~n'
    '`#pr(Ux=+8qd-CeX+lPlg`_ivpe)=UZ{=09#d<l|J9RAI7biP08f>&O;abAD(<=w->lgA%!-#!`ZlRw<u-ap)2A8~X3_T=6DlXw4ocW*bH{tr6(#oNcb'
    '`?t4`bBb?&eDY$-#y1a-Z(hyg&!YQpem9TfWX#CtZyx5H+&w(Gef#>!H+MhI?|S{@S9iBRKK}6JH}~(p`|<9lhbMpW#b=*?T$ca*=RW+4v3&eR=wW{}d'
    '6!>rKD)cW`|keTKi@vQdjI3?{gc=EQu`ld`$6yj=I-6k?|yt^hwt7#zIpt^lfSsX`}y7d<D0kN|Jgn~|83pA71(i3z%QRO_`m#{Iq)CdeD=Rh9rB}_&;'
    'J%Z^YiC8Z=TOTQzQ9e{+X)I^En+;%X$6+Z}eY(Ba-&7;_rOU7cWC6nh*NB(4RiK`SQPpIyV2t@AFw-{x(#&d42qyPy2QJJ|Fkhm*J84=;qUx`s#l3Whn'
    'FWZ?xz0zWF*7{f};5d?6s&i@yycV7@=v^La0xhXFF*|00Z%`JVWFKJTlq!(f^H`BgX&KDzns--a<WpZ6jRoR4l^eiKP$A5O*x!*lce{r$V2?<S4SWjU$'
    'f<0o}{^6K6Dw~zO4CL#RA@9y9I^dv5gv5tYf`Q_b@-@keP(~Sdq^UL=?et&!a=Jv*sz4`24?_T{tGe-F4^Sie{-QNGxjUzqh^ZuvXx4}wc#BZKIJidF&'
    '#WC_XU))ccdm9`tX5i*;?(gni1%HT%xOwsO?VGnZPRGq(-`~FFR7}dvXZP<W4fB{-&tJ@S6t9R0y7}x6Ki|KPg%i_s^X2X1KXYMB*3Fl19)Ea$YX@T1Z'
    'oYi~`@5gMd-wkS`y1!)=2!3E&d-ad#WV(cc#NxKE^ofNd;H<<K6)I_k7>R6^!>x5gAucP^P6}1Ju%0#b^rMOd)9c&_02c$e#)1}l;6DgaW3O*|82H!|A'
    '}^xz|CL3d7DoQO1OFPFL$r+VnBi(ZoYVR4qcGN{IWR<kCU@M-Z5xFBR5~YyZ@J42Ig1X|K{;u?f#8K181Hq{%_~fdi^eD_!|cP?*2P^5rlN}^8flj@Bi'
    't|ACd_pU;ad5{JBLb<mYn2eU=mIvtDpMFKDhOlIsP<b({#U7xdQsL~cD!)Yh_`kgW;PN{Y~jbZ-y*b9Tc&XJ7ntcE~?xul#d%%|B=V{Bw5FKW9(<b9UE'
    '1XP^CZ_S-*a-~IFNjPu>^AFpD1PTBXdc)psKzP@{Sb^qq)M^4@R;^)5#0U5mf{_gw9=qImz{p9uC?c)z~Ex&#L(|33GaWa4TdA7>3JXz-d-@N_d_PaNa'
    'w{IUGK6-Nd`hTB<Ja2t`_vH4;G^pObzP*3_WXix`*OS7(dmnH7&riPl!;@FH??YsqPyHx<o9p7qSHFGvqS6h8%l+NMn@O;*?mj}i%&VpZfBNJX&wl=^+'
    '3nB%_$iYGy}o;WJ}NA_yC=7=UcI}2J!vy~6+f3{UU>KV%_+X(xBuV7-P7pkjr!NkufBZwt8YL1{M%1oeEv;U^_mLO7jf$Oi<grVd~`D<vh#H_&E%M<+s'
    'E5Ea_?U~hRze~;(z)<0j(J-;O*PD?;dZR^1uJLo5w%=d>4%Fo9CateEzwwx=Z4nJ^#3FFH8SqU8m6b%O&z?|NW!OO`m`9;>Gh{fAQ6)FTeQut8YL5`ip'
    'P>+poVwZ+!eQ&i~@}^-m9<#JoMup_}CL`=5S%`|wHp`d|Lx=7-0}pFer_>|g%nU!ML;d;0GF`)6+-p2fNU^ba?A<-<R{`;V^-Y~g21eHIryn-hEY^oPf'
    've!SfG`qkt0wx8eM|Hw_RUp<@C`_sd-%XRm!fA=3gKK9>;$JhGtyyWADKfHas{r$&pqi28RM}2YqsCmUFx0CIkuY8!#dj0N=4+<E)Wq<ZZGZhbsj(>7a'
    '1doU*j&c1DI{xtU?ftVj|7vpBc=mk4fB58IJo}^P{?#w$Kz#D*ey)jk_H!<yDW6{b(M!JjF{%3D>0OBP+xh3&dEv9$_m4lkyPuj<4D-~SCUt!L-MiO+c'
    'm{?2!_6~a^5)^;{iOf@_(T49;>xp0B`;q6F!kgA6vQ?^CV17yOZm7Qzj$_i+F$+R*`JK{`OB50{_4-u<JZ6ZU!Q;W^4phx_nYS*(B$WzzWnsX^Ot}Aww'
    'GW0IzH{!zxg&U{cu6l@$oYM+h5l9$+FD<{&K1RqXlt&(%*df^)EmD@=u?2d|b-^{Ih=Z<)>eL^~Wpq|M_9hU;O&Tx4(M%N2~Ms?|+`MAV_2K$tfZxxBB'
    'Vs@iwL})U_Y(?|wJ0Iw|_@)d$oON5NK~y}F(A`0Tvx+1=Y`KTQ7j?JL;uw-0ZAdWePj-}U6h=fD1M{~!0{_UAXxrgV9@`{|7g{Mql`O!Mo1iGTmgrGx)'
    'n?;c+N^!um(`sU{kyZH4lUwr%7*T4Ph<p*8tYi|+1{*%jo^Zc7HzW)5%um9`wZ$AC<%WuD&JoLqfw<(1GhxdH_<>!BT&r~ozxaY;!zy0R3=ih$u)r*&3'
    'y!>rY!;83jj#Db&@%<m(@Y~-^%6<O-f_(GF7cam0?Po84`_1z|y}Nx}+RIYTyO9R}_@T@6qAP_6dVTl1H*XcMuiyOccT*5O`HP3UyC>JF4c+ADcMn|i@'
    'bT0rfB5+AyN~1C)t~V9|9JWU;s~=n4lt<WAWhad#QKW^^~%5h$Ep1M-%v{f1QY-O00;n=p=n+9-#;);0ssJl1poj)0001FVRT_Hb8ca9b7*gLb1!IYbZ'
    'B2=WM5%(WO6cJWn^SEGh;SpWH2!}I5c85W-v1_E^T3MX=Y_}bS`RhZ*HYj+m6~W5Pk1gARc(xLYxaBL5eCC0z}JY6~anvRYh*>Kq^XTY^qAD{r5Vav>D'
    'XLO33HT_{@wwWAgT))9ErQso=6Z?sQLQc}g>uQo$;#`?21=RcE~;<yDLggJ%PNtI8`?VAM=VDh+xq3|0uizZVxOi}GUB?n1~S6NWh;MqR_W%l_$1FUKI'
    '<^zw@T2qwMTrI10K^yrZX$u^m+G&`iL#2q$t5h^P9uOQLuT(9e1kuj45kxTlK>-Ubl3?_+&#}HtlVHs<f9spi7%$_uapqxhlpqzm-TtWC=Lj=wUDy&`r'
    '8Ve+Aur?6)+yFS^t%e<x-9f!@2NtBBE+HB0Z6F{5$sXcjuo5_DxVJ{BMJxgX%Zsz*l~6-_Q)cZ@4X#Hx(fjpCRjX@9oc!^q{wl|3Z75Uszu8kJ3<2BqB'
    ')Z!C(qvGhxXaM0=Pg~|$LPV~sAh)uDs@GynV{5<A>xacFXNU^4~W0EeD>7x5apRih@(6+qzO@Tmiv}RNE4xRu3iwweiXY#{te3ij5*T8+m`RpU3TdF;S'
    'TxH*``Z$2BhDhoDquMqx>=QC#XN|D}?E(^<BTmip#3tKRHWpGOM4R61jbo^i4N0y|L+z9d|%XXXuj$LdK+d#^OY+p{w<7p5?@{eQFcO84-`?BiEsxWpR'
    '5%hPiL?ET=5@`Os#RIrOW|2zM;v6P7!k*W|m<t2mW1RsRqE&$ll9-y(x=qP#Sf%3GE3^mbJrJ<k5#n5dtZSoT<7@<DY<wd(yhP)h>@6aWAK2mqI%X<g<'
    'Bji;Vk006V<001=r003lRbYU-ZZeeh9Xm50LFKBFZXkTMwUtx1(a$hqvV`FAEGiGLGWjA4BGdE^4H90P7b8l|7y=#vj$Cc*!Rfw85Js3&MxJL*cFJy_`'
    '*0wAeB0ZSio`z5+%M!OKQo|~??QZzHZ^ZL^BD36W@9YA*kid%kXJlk#Tuz+pbN=n-U+&)Dzkc`j=F^+_$#L`1&C55p_xJPa)$POW{oTX-<<0GH@7~N`K'
    'EM5Ne|P`n-S3{fy1RY&;mOBO?jLU7zPf$?>h=BIEC2ZZ_Tg?mbN}wc`<Hj~p=ZxueD-4g@ZqP~;N7eBnj@x<L;N^~7cqT$9P|JGa)e{O`(JL~e3%dY+f'
    '8amz4<iM^hx_@U9@>o`X@;jecdUabX^SY#B~vlbrC+vx(M?k*G~>z#CehGCqox$Uc}^Td0j-B4rN}1@=4Ie>_N}IF+0$@AD{1UjgE!+`sRn@e0%M*8t2'
    'Pvrkxm7F0-7vULDk_)H82uy3nhmzAE+g@gaRub+JDFh<;|id~`LO#%y>bo*1*?;dVz>$A`*Aou$rfwPg)e8tv?O8newh+~)kCIkxK~@-aUmofnads*Cx'
    'y`SHHU>w<yG$9!MPO^xL>hpF&^dCg%;bT_X#OgZ|QvF=2kS#y}e!82<PXP@*bdClRh%uClCrkv^9n!^-X9(m1SI+DgauQ^Od)kW7Bv2MD?XlT6p8lz4x'
    '`x>L3T!%@g+|LEj*BH(3UG>Moq!*Rfq*s?`XXgM#mFIiwF3<PYL){*0tme|Co%*Oe-&>m7neQ!0Lo?PmW#*}@zIDE(tiFYV2g>SP@c7l$yUw`O)w}t{l'
    'xOdHaGj*DVer=+9mN;3**I%y!q!5Mv*VhHwbEl^1Ld{U*BwpRTI;c9UnDM~F7mqIOLClF9_J@=r>;AZd5yCNF)-*kt{#Ny_pA>ajUJ?fZGY%u?eyXWnA'
    'T1&m8R3$Zn<!$1fEd?WmTJlsrM9D(|byWRB*9&WgPUaR5q_xR%z^4?q~+OpBYSt`c_v*4|ad9vj@X#wnj~d-d~dz9ly`lx9Z=E%6dh;HgB$LVR|l-=$A'
    'h+sk@ayP8Bkzrc#5}P_5H<Mo$X{qM7r-9aYPLrKI=dcoD0d)8l6{-^(+r2S@P5?7?7N=FEas|8v*!qdfGDYPK$_FsT&-v8aN#L2jZ7uGRt<+*wp1>}qS'
    'pO_Hd$x3%J0_cLq7HIH{&E3SDdwYB0(^~G9olagr2+gkG>^GshOW6GO*V3Db#eT__PuI7=^w~4?XtkSafTx_g_vNql7Th@iPMqpzvqOkW#In>d^&onj~'
    'cRCj|;LWM`ECLz}i@7WUx;{?NECL!$K+7VaF|tp}BA_uOKeKvpxI2r0hTh!tT0}nL$|4WzeC1*-qRcT|7I_G%`C<)K@+ZW|k{r6wS$Jr&O4M1n8_*&ez'
    'O#C8up>%rEHAw@b`D1F3>|>Fksy0|%OVclAG2t~oOXTxA`SBZq<$6=*SpsinlwuXY0G?wPR6x@7G2<?ve9&wt`)TCg3j^QuN{<{NI~H!)(Sd&XRV;oU5'
    '+9PtK4C3pfO#{O@jyOVs0|WdWX4TnRPKY5vtuEGdCUPrny>Sh^bb&(BY<eHM`L@uSanMLpxZt*0c_{wPv-usWqvcb=}vTcBo8WQ`)^;7r~e;MpJo>wpY'
    'g@7nSzQq88Pwb5V=ZU0z3Ec002_#ch^D$D!I>v^{tKKKqls<ZIxd(F+#Ijb*SD7tM{yYitqSn7!#1)s5yR$|AebTuZ4soP5wOViga+zv^&;B-NtRt^2~'
    'K+o20>VU|r@hephtm318&>q5`0LnCG;wCWRP57dA@A#o;GeL{2t`h>Wj(H8E$&=%(CsI|d7)7qNqGThpLd0t~)o6c9rOKpM8t>Z#`YVipnJ~f+=icifx'
    'x)z@rl8?te*e$9yL>+wCjz`oXkI0(jPJ#xgfgwV=s}*e*k#)0c>X}%g=$CY@Pg)umt0acVgi;I<r(4e^Oj_JJ-OP`6oo)$svS)80o~iVZi*^ySo<##7B'
    '7H5{G1yLpEu{S{cI5CDUTfAIKwsQQ8?FR5pjOwi8V7rEt*S}A)Ty<o8Ygbiyj5<L7On{?Eket<NL+~eLFG)<p>jS~pmMC7A}SXkh6O5za8*R*qG_*)nU'
    'B1FG2CUHg~O-cT0@&wccd(lo(Am|iz>Equ4iosv6YE$uhGkX=pu86XV$XM5c?{&GQqG^v6V5oT4x*6+S+)LS*+{eh`}#iC&xMzxrjWY)>67CL|unR^dV'
    'o@>9IN1x)`u^hia<_M|5kV@+KEzz^Y#=2F%o}iUCWEa@7)@r?<8-OO#=&*%Qb`VSCY9WhJZETB2e2PKV^v+R&Woc`eay3KcYe>i?q^T1z^4xw=+pcYhb'
    'VGG#+5Ep}xJ2NpQ`x&!5`l4A}}hU7hO+q#amDNpjjlK0FaSIK)mO{(O*>~EC3mxiB_yyvN1$6D;!7i-dsheFa!%3df3D?k_8Ce`an+oXA=X`3_)R%{b@'
    'w0V@N)|gtVb?7721{0`Z7Bi4c@4M%ji^y!MHL;V)LnU@HxjUl6kpsM<FF4#lgCFNkzofqIeX1KG!@=ubWH|UFfeb^stH^Nh&J!69FimxsNz=EDikQPQj'
    'SHIYCq1Z(5|3z=&Y?9_RSu$fl%wPvX;CDavYp!jkA~8xoJxwW!Ote+8hoGDU4!r4(>3^sW2<ZMBaC3z;QQ3+8hlI7U4ws^F@u+{u2wjYs8;n=dgmJZZg'
    'qo(-n`Hn`o=W$!5M@~VtskC8X`)n(-*=UTgfEeI9mG@z4ExTK5>tt1mwn{-y|UWq-zq8n+IM3a%J~63CLTqCQ-5k-&yZ5OVlJv`pj+;C1ZEfoi(K%AWC'
    'xJv_;9%R=H&<a$x$}qGTzua$1(i=B5`Vi<UN8F33n+R0Nr#iw2#iwT>X8kINF-*vqXP$S$g;<F+w>P11i!zLU66uO<EG>N^_i-CDg!Zd-iVB2;>&CP<i'
    '?uF{5a?6k#)r2+rn79W;IbG4;uY2B?Yxwmt)Hp#sehM`IBO`O0@a<7SKTZAeOmQ+LT^-9$WiO`uY=_$~IF6k*SqFrkzY<}J|r)pX|7WKQ<j^&xIwX<2C'
    'lrEyvwRTLo`&v7j%~?yQSC`ox)IkS8d58zl)>_M^nO$0|rdfk4Y5su7Hj&?veP?~*fL&{%0OpLED1aG?)_lxrT)TxSJtGQW!rVjw&N~S)OYac{h}G2!a'
    'z53n_!&(S)Yr=8-Y=zYgZx5Q+pu0p&x^=fQPxCcx8hqwwtALDWX;bt5!osJ@JzpW_cqMPiy$2b>^c{LXOJ{?&TL6DJz{JR?*Z-<WPYBhu$NpkqzCCHkx'
    'r2WMzFvjEobn6N}Va`G+AF7^2LRY@aRMx!HlQWp^{F0SPx03Df^khC#>#_=mV|m!mC!@k=EgGYm2guT`Qkw(}-}_ddb=}UF$b%o6PZDJ<i%Db5E{u&e|'
    'q(;Hb8?NrQT4ZIelK*W0tUNjuz{G!2Yq);1|)xY|+xLn^X1$`n_9A>L{DnQ^j9S$kztU0x~@n_W#LHi=oxl4q7m{m{jtPlq;GE@+@7G-fHAN})SaHkGU'
    'um8EQ&qPkiuXrZ-kvXo7^cAi=DIsf&=Y&xSx&C(*7TxMyJ=4nN1q0?)<XU*s0>()?7i?mJ&y(fm#Dv(heO{Rh7bUL;m9x0<bYEaCK?x>;J%_xs1Bgo4{'
    '|0av)7iiN|kBcba<SeO)ez|FFW+{>8?nt~rL~wGa456stlvaGDt&>?HMQ7*~!+cNMH+!@bx-=b$Q!ZUY#{kWXHFQa*`n(WNEs{Gg*18D$LhC}@MmaQeJ'
    '^8{`oS_S?3ytu+SnDF{Bj<&9>h3$Tp-2jx7nKv51{x={x+6h~19o1lqZban91^6U4f7(PYFVEui&=)d%rcqH0o72-W>&KF$XQ0SoOZReK3fzZiyO*^n3'
    '*hYXtMLHgoF}uFf$n~B+I@?l!}b-yb_z2Dr+XgY-L}GDjkJ35vL9>cga(6RKG!tQ0>g9Br2cCS&FL3gm>1HIFX|^W?N!a+ZWPFdJ9VQFM|bx?C%x1Sdk'
    'Vg)z)#pp3%3ggR$*>vJS>}cvY=~u~u)}budmTmakd|V;Nqw>tLKcq}n<dOY|_UgRvmOWeN5r1p2lP#?1U?iS`-sB}=wHxB5C56K1e^(YT=LweO1ut83q'
    'TCl(PxPsY-1Oi9v1^|;qXy>@HW>xc{Wy7SV-sFh@rvKY035Rj$wE5%c}7`1{$D@*6MzLeJ1mct^LrSr?xeX9~<i^p19&SM}(t@^Wd(ofFTS3%;T*NjB1'
    '+)O3K222V~XT=9h3hn2#4zWhgEAgTs^SXi5W5PSoym>(kWImRZ?17ViYrc{ODzg1#6C9bxFbbmIo)@xN20x^clDw5A@E7F9ELBWM#VsJGG?s{zG3UmSH'
    'YO}iHtR|qPTj4j!`;t`5HHSFuM;6&_MJml(OWv{icQ>oB}sKOUtwD>ZUut^XPfjgQ<OXC%Ob?X=cW`h^YrCy;!Dv-WhrJF0%VqAW}PEyK#Fdp_bkQCTD'
    '?@$Y0sjNVy5iDk+K^GCrdF?VsvXdYdMCbnVH0trls)C{#t3gedb9ylLi}5+L>ih<IW<MxJP8^rPrF_&f1!Rqe(i|d0S0OsXALxYtpJ#uXw3d3!1wuz3M'
    '#0*cfYFO`bu;)w6N9(jppd^AuthIJL)`=E=HiaEB~5sI%sUgkrim)*RP1e39XvXrMqef-WlEJai#eiuFI%^cJ&UV@+>CIg>~B;Hn2}de><@bRlyBtjEY'
    'YiQMa~lgO>D>0PI>`nIMwp&`ps#xEI86Cxona(YP_pQ0D3><J^V3hKc2uBmDV_5`hhMO&5;k2}(fx&aA+*s8O%RWTm4hSv!aPDYKiYz<uD@?rvkBB^=5'
    'NtUsQmn*_dDsB_gmcJ2k24!iMI%yQvJrbEMrx}qf22N~bvehx2`4Y-(V#`Qn`;0~`TSA(SeKa%pZ{%h(tAv7#WVSlSb8L1PW(j8li+YxD)=|D>31?H1R'
    'R1NMwc_@qqVW;bq@xLbW{9&lHChaFOCIpWFgNTkS?bQ_<tTM$McJOE?yP&zg4~4Yo79~(n#wM|x**eLsXI&Z1c^E9u%c3T=3;Us=FrEm)SU_IRF=9k$7'
    '0&MR(gQfk(Pa(VcZcJp$DUikim+KwVo$oW<#v?JgK`H7*&odhpTlY2yKY9p66-D_G&(@wZYeVo_mpMYdvS|it}Rawc-7_)^o<=U|y^pWzsVimy>GqGiw'
    'YopFLW~VMrZq1BQbIx~6gXvo(!a^|Dl)bC3Cstdr3f>q~tAOT}5;bcHNJE~MjJvVxYF$Rb+HEG_3efmu5uf0pB<&s+>Y&#XDc(JIUFEt|`c<692?(jiw'
    'JSRN_S!r^nP;gI12uHli9%{^nZs+AXG5{Sdkp{)q6sSRGekKkaFS3b6@&RK*u?J%)Hopzv0$mr7!^IRy@4%|JmG-`8R+$3IZ-M7{`JgRRkwQ=>pRY~2D'
    'tF~CZ=Jg_-+~$3yldIU#vvhJ5ca1Ea+&nRTXHBY)QR(E?1$Wk@M$d_sZ-b?jPHsuV>W-FgGs)8OZ5{_L-_l$yWB=enM}<FOE#HjYHd~)tJqUO_4bOmBp'
    '%MJCVugkU0(81PK;Tj84)YLbb$hmuC03|!`kQdFEWRiC_!zz?nY6%TqS{U6@VXIOw%4-#guG-WTOECq(j}dYO-h%eT{8`r={CE0*bHwJ?I%10vb0&<{H'
    'C;74fSr8HmjL&lQwHIIH+mlaMw7cJBEYeYK4_RYm$!+SCf2XZ4O>{e2}~_idVk8Ff!JrEH8{HywLQb-(@Rf7X6;0ovSVSo!skS{#ox?^t)IxmKR2e$<<'
    'a5aCFJ?!pK`uB>LT~+7bQETU8aIk?Miyw~q$7VkC2M(hsK`tkJ5lAlwlf2p>5#3mxJ&a0|`o0Oa$yl@+3DFqQLS%>sJ6d9lWzTR*X8p+JP@#To-7?0K='
    'qARcaQ(YvwVv*=xkZhD{tK-;m#0B@mr5zwBaJ6UJg@G~+RmPTtB4VzVB1uCs^6Elu&Gs}=a4`#-CQN3P0^+kLs)DeTdSFbGgJ$Q{uY{%^^%Vyv5ex^d8'
    'c&2f-Xo)RySIzkJMeZuj4!KLR-5N(pwx(9S%8rx0`l|F$#p}0b?_}=Dl#{JAtQgFa+m|TD7|knjXL}pTBP2AhF1Z$jt;m#b!O{G_MfgU5y!A!+I>zmJu'
    '|^|v7|6aq&j(*rG#dICX(uoTWEt+(zSNo&bJx4OB<8Mluknt0&kc%O?b-XRLrRe`Wq%7x!q7c&t7|HCPa^s1md02j`P$TK6Op`F{VS0??=32kJX=pN@K'
    '`cWA|cM}X2@sBEX63tBMD@!sM5@1trb-&QubIYs^scw8EV9Z=m*PqNLO;~i^fqT9&4?rjPvKb7%&d%+ged+_)Zo7XL+VZq!#Yj#hXuECw1jGmRt0Q*@I'
    '_+nG4ZslMK;nbCUyK8?$es62r2VXmvFMDO$bRMA7PQc1N@tm(`4eG(HYl9@05^7DTHLbCcmfz*9BR>cK*kO=8NX3`i)Ko=ujA$rbp{;_0olkmaFS3o`K'
    'ghh)g|(4qy_MUk5o=dwJ^1CdXN(y9347DZkRq<Xz5a=`&H%lOc&8ySlt&vVMHb);mf4LT|sn#}r0;VXj!7Q1<&5jEMAZzJlfc~LoXrTtA}eG)WOQt}vn'
    'W>KTuc+WcbYxaBAvA=i+J(D8}4N>Im)yti=dXEYbmXzEhFI{vuqgJ064JD$Ej@)62)>y|yqS%g6^oc=>P$#Y2xM&k>ON21-q?|ciT~tX!7Z*+H{&mg0W'
    '+X~ms)3G+8WsP##=<XIYm0tJOG;zXk<!?bQ6T_v38?tuNlSp8S2_Zlo7T5uZWHCO6>WWYbKT+oCJ_T{;vxljT8j|ac2?AN+0^6==Heoww5ub9;@bY0RJ'
    'Gq(sfq25HBr(Y{6Uhi2k)2$25dL8SOvdR=4%`HWWC10ci^;5Z}Hit4ldg+ulM5ABpJ`X_5*)y`YCs9+{h&0;ro^4sn2-JPoKkBUi$mjb`|?oVR&w`yz^'
    'mq<eSfFCf9uEqdfDm`^YgLLMgv|_P*Tmcd=c?H6pxN1Hxm{<EZO_xv45H;;a`39rk)#sNpKcy>f$Qi54_TOuuw0UMte5xGa4NruTu3+hrYbt|QeDbAe)'
    'rxx>B4Whrl&M^r+*x3e2HU#Npv8**^r(RH@k+AEox%96NU%wF4BtQ$g+7&i#A*tYIpwSAI`i%4pUWD|oX$|A<7${xm1C~KI1m5E8?CrcOyn(Sblda{BQ'
    '*Dad}$N~ltko}9rBkLC)AVaJG4&Xy)B?E*bln15z>ynshYYGoE^1Nmpk{$pB=+Mz4UBzXu_H)Zz?Z>JKOcwj>!IBEf6~;9}$2o5605OLlb6rMmm9;K;I'
    'FihTv@NH{;ooIMW`YtsRWFstBfC>a3LdptCraGyutN1N$&vmdNs-mROP-}~Sm$Un=ku%cH2~s$IhYS|zVKRy_+G@jLtL-NytYSmSi@?4ZB-7M<2q1|7O'
    'rCSYM#YwRya=d7LF5st>^f+cCK-;w)CQnv)GUGv9(Xp=N>(pz|Z1hlK8;2Nj+<8k9uMEdM@|1Lp_YWpUb_A=&6MvqNiU+I2ZQXo?h4^ZQ65;>@_}%Cy6'
    '~qH5p0+(+ac!!wO43JzDxQsz3zNxX!xcBn>&7XI+k7ph>!Rh9c>#M~9RohrqfeANEn@c+O;!<L3T7Z(2eKM{v3Bt)lpi;RpNPON!lKh7<FnUlq5ZVN%#'
    'eb%nIYJTX7iEqjK8#9s|wknrnY>q+D#_;Lz}q2+9l(92^Id8sQeSq(-AaUWnIoJUoV!wZxo$Xl9W39`_^kp~<lT$V`4D`lH2{#_~{-j9O>?PHaL|H0!H'
    'wn&~G@aR#F9f#}4ngLHHZ(BA|l#jQWGP!u0<5Qo_U*X{G`ijiZI7fLU_+9EV-x6`_fERS1RmP>mWe!8NBh{HnQ$7#6d|=!+X_8{pb<Tq>AM%xUF<e!o5'
    '8Y+a32owXu8afTP1_3IjpB8-z#nLj!n;ANj&nc5AMk73MaAVZcZ$k2d#s?`U^eEcn{Fy1x43vx3Pt03f8}e@5sC}fB<YEk!^w~+2-n?^j?2Y!iRZ9c)2'
    'z@^6i^|Z8!Du@H(yBSu!STJL2Egk+<Hjoum#C3eynfe7$mn)lx>g3WWVFFd~%OLQi+T8+*^`J_+s9sKCE3&z{)JA&9Kh0g82OVbtHxk^Q!&V6+cAlVxF'
    'K&VG-xZaT<nVYuRUuJmbhY%N&W+NssEw(8nhYMIS&8<Raj{jcb2|!Fy?+wX8iMMg$5JVMMg$U_|&eZMJci^PaV+#Z_FcmrN%2KyR|6@9B)k*pnP#MxYQ'
    'n0aQ0^K5-V$gDA?k%yr92=y1;9!{nR6#td@J@B)(*;-w-f1dT$F6k_r$DFl%)tn&ho`&pLh*7d8T7RoZdxPHGTUJ;}4JfT0D$gWTlMl_vgd6aO@uv|+!'
    've-mW(<xF!3FRhV;~dR=N(_-hR6{S11XoC#F^KX;aIu_Yoa^;#yMFDPM~L!4;McN`>{$$_P+qPjQpC5b59L}RMSQ!uUg}l1onIkqj~^SGGDaj-Q~=?W&'
    '1>98ZLHU~)st_JvhTP;-AAqBpV<wHer_&~J=BVP<`s&2W?klMBo*__`J{+v7&^r}SJ%DXCS{=O_I<GDd`fy*v5PLAlAeOnO-;*5Sfj);C`FW5=CSPTaD'
    'AyLh!@y^sn)a7ZMu#ri1rX3#|)UXn$30OvdDtM-I#ZcN1GW3BR|VwvFfDbxR|sfoKz+q7c-wkL%QqdlPV>=jMe{yld*qqCkK)axDkGrIf^y|?-lnI!YN'
    'KA<VmWW+<HQuR3pffJXbddo;arTtNael*U9aWy_gmsVu!92<)6u;ix0_=zqI%eTXxdoL#(n*ix0ugd6wnSPx}fmGeas-%Y8`jKN>N`%F%pp7fagD&yW`'
    'b?;OX;oyvoOx(MOw$VgX(e15V_-hAb#n<o%jNO6U}m$tA%yx@;(J<I!tV^iKgOa^vGf0ESp9s=1PJT(2{?6Ege29$tKVQZwF4C_WYohMJqr^+B9B~s%W'
    'JYf;=Ww~z}l7vJ+v79GUD%ao@7A$4hXT7A<YatidM#}wU6;m0f%$e$OzP^U5N{!dr6(WRXpL{AYlO<ZcRT6D-xn0|#!mNslwD71L&mIUTVKav<pCg(rA'
    'ZbclJ(+`ygrw<h{VI;L*pIE55H<pbgpd(9RfUUyFeg+5R$E3wQumNxX`%V3i&D)5LdK$$GXW2&DD}*WxK6sL+$p+fgg!-Gba{unn&g2-YZ3@>I*muVLS'
    '^G&$HFy<QECsIP@>cxqxgxUs4u-F5yMKkDXxWvKnmjbp$Pa{4h>#QyC}#lB&2;Oawq5_opFUh4kHE`2yz&4D1QI~Dfo_vAx7~a2&Cc0Lfn8iK>$d7|DJ'
    'brbP)Ve7rz;Ve$<NFS!Y;|zOhcRtaTdXz@5QV33A}h*wF$G-1un(9JmQR4R+wpzEe(zx_MDCU~{M8w8R}71k_DwuHv9?Qe9z^3-XGX5)NM}<rOd^K|T?'
    'aV6DDIF`j|rDT?(BY?3JE(_u_e?B^Ow?qtHXxl{JQMhU=}xuOJMAfuuLV6Q?6z<Be;h_#41G8RSJltNOZC`teZ;VMc1uI^qu5rd)xV4N2Mku<oz1lhBO'
    'Dg_79=*>aqtf3wSS+l@A3K+9Gbyn|sWCo<=WA6kxj->3FPVySaOotSGy)*67^bJqDf>jvJKs;ddYF4lc;ZH#xBN2;akSkLn=rRxwSiPDR4_I~s#%0jtN'
    '>=aH7o9*qnwGLLu<(k5<5f*$avbzc`lX(zCIu7DK0*FHD{4&NBNAG?@EYFQzT?nbeL^#6kUB~Xy*HA5A>DLmt?>*f0cuRd)Vod8)C;-;v_UytTeas6p%'
    'd`TLL>?fbM&qO<jB1u%rUr{L&nvs9p<R<=QK0b&~b(tI>;w$5h`^+AV=9Vqa!Uv#!_cm^j=3#+dTdkYEY0**6@_Ab8UEKD3}1ptsqy<96f!#;cm-dbo}'
    'iZJuzFRHNz27Tu8I+i5-#P7xj50K!nuV$hZwsO`^PeFe>mJde3N}_RxUAKs3@~ib47bX8_*N#UolTubxP}RghOt#APqYtH<~sL0&x(M77Y-jPcp>>Y1`'
    '0?*aROb=JW?VA$o=6Fs8}XN%xe1ckFj&iFvo=g9Ox=V5QYh-c(vAnLQa2RmX|pOEnQr#pC-c??oJk+(v}udMsdI0Gz0Ct`=~)xkb!Jd=%ZqjAF^5M3>w'
    'pa#M49J_&f!f*xPIk2M#!E+4vQ_u(OLA-rD&^Y_>utId0bt2{Q&J0A#gSHWfl=uE3Ql5ypK%_iI(Lkg;4zqzsc{G|qZlnP|GE`vP)O18wi^D{evUa)Nb'
    'zR`HrxO8nZ9QCV(6@C)p|HjOD+YL-!ZSp(Nv?^~x&>S*qV#TSyANFyCZOt~5@4+NY+x~(HmU>{gGg9ClMQ~P*4Eq(PsXAhL`4MoH}&pv)MRL4YTA5M=*'
    'CCW6id|i29hUdb$$8rvHIo5x!ac)GG#djo0<t7AmNA|gbuKdBXv)jo52|<bbx4%Lg)a|%c6Iq+i_uZJ6fv%`{>k_&It^sfayDP?8)|>@Q(}lHxtSl@NZ'
    '5&=%si<!xZpu4nFnu2?yIsmfv-u2D-z>4_!o}^r)6D(jx0`M3EI_{&i4&ko`9zc7%ZrGt!;j3Ch-}AMu3Z3<MPKu|D#-FarU_gqsBdiqmeOu`nT({W?<'
    's!b&EH$oo)xfUtrIl_Y)+OK`A=!w&Mm#3MDx0kaQ3K;X9C5df&;;l5EbW8kFo%GF5+JXj2zbcv0mla9kJ%1$4$_dzx1uANwoFse;~2!{+KT_&(bQM$~r'
    'e}BA<8YD_PuNgpyizwbs{b~RR>3uRzHm}PD8PO>Ae&Qhn+~dHd3V6qz7lH6^Ju{50sPPiLE3j7fjC0dC#<4^?S|m>$RE?c12eGhM(h*KPnji<UK-^T2g'
    'V^H2H5fXzy_Cgj;>jLlSd8E{1eq2i5%59A#fS@ika;nJYZYW*T-+tkWH2RoMmrd(WB@%e{7mD5XN2Gd*CvOZ)wC|k0X@)5i$VDGOi>R^4~ma-E(}7Zmk'
    'LAd*sY1=H(v-yh=na5pC!@qjQI|2XRTCXVuIZK9ODz@_7luQ4E#oe?_@&txhR}XRIA`ah&W(Bht`g25I$ad!o`Jwzvw1aVIvF$`N8F}<qgF_n1)FhY^)'
    ';r!DZt4g8blCPy~0DFt}-$gux!X3YIWv7Fls*8OY0^II@gwBn*VX{Y(ZwNPUx(liUMIIY;+1jktN}-eA=9%rM-c?r6VsFEIE6i()d`pvw$$dkLGUk8-h'
    'ygV0XDRQy5p7>YkQFUDCd;zWW`Djr8T(HEoqNDTiG5A+F$t@0&Vt3Zm1Lyq~bRoT5#l87V&;u7Uj5|Fv#0ZB$UJ|M~NO$aAuYCOOujObE=1pvJ><P!=x'
    '9$*u?H=&+*4N>krgg`|3_n^awhec|=L?X32DUsT(N~A_y4rsg!^cjqe4C&k$8(a{j5AoFS7xOk?Y#eUD!!OAl#rHrgKf(#Y8Y9D>d_m9}lDvkWkclW08'
    'cy#~CbVHy!NHNhOZvu(jo8qvW6DOt)c}iWE*`BMPkfFjcbOQiuM$Ozo;y)Ayal2x-4P#-C{y?B8+VGu12rHilbDK#x!dQB?A?Z}jIwwe*(}QD9jv^N)!'
    'XPLQFiYf_A{OEetld~;9BQ<<hWd{ij%t@zP(Wv@Ps>els8b?OG&7R1Pn*{14SY<q8x(Eu|;_VMIeo$T!JDZMLf_9_!F7|h4hS$-?lH}wT9#yWPtK0Hy_'
    '6YMLGIpwEt1AJ~?|t<vWU06y@%d!S;%B_{n=pBoK%FOa)ztI}I6_rXBf40|Nr53kk%g8vzCS;AoXt<!3_97ZH(p+=qLVfqY49+ElrnXe{%hpX$naF;1|'
    'Y$wD0`j^MlqCr-<}xbU{-MPg4-E$b*z%cY$AAP_X>2buuTsgFWSV<Eda0h3XVP0>4Au1%5Ht#}MWX7Y?SNMor<&=1xcOV9_xd0tp;#9$8ggeZroKsVcU'
    '7;%dWAGqqXs6aGtCn^A(XAFmr0e$jeI>epE1ZXgY?~7+jlq*m?JcGjbji?Ic4itvPH__V7P=fEv7=&Qfg4X<{>a~0{28zuP%v$J~2s#R_NmsE2yN%PRX'
    'QZouXNYn=T1`tGijPsXezRX+YL6Wj6fQd^5^+nd(vPMKBQHiG|E=0hW?`gW1;Z@+nfmydb}AHNO#NhTVob(QS8GVErr#p2sMe!Q?C9vCZ0rW+)Lt1-{G'
    '`1y*<4>}v4Zucy)uk5r@hjRcB{S84a=py(u^~$y;3{SgxV_&2`_1{R3mX|vt&5Bq}`GW@L$?4nZ=(1bT;`Jp~v8}D;XzM5UeEQB)}1)R8bBlh*Cw>%5s'
    '$|s@gOoRaC{7D@qmRz<?-KRPp&F9aFKkK{}>|4WOpij5?|<O2_0_uP7Z;f{~2UF=gVfqjXFQca3U77hy(GD-BdG5xvqdvn0jR;)VDMk-z}}M?|&M!7+^'
    'JmO5&*h;qqsND=K)_o)Z^t8pWu6iiE9O2;@w4}Y3S;UM`$l$vRA$F3<$dE!Xvm>z?BRtVz6JzkdMVrWd6jthYtNw#AnMkHBaSFE^lN@(a1B$I5gxjK1-'
    '(54K5Dl4qdo+LBuUL>-^rr{n4ZOWjfvc#rrXltQOMfQe_u?e4~By(&+Zb&l7Zlq3`WaGf!T&A_+{dCY#K8`4r&JK`uEhl`@#S$5cA&Dhg>tg64@(GME('
    'PP!4533Cm+>xt@i;RlM3oCw7qmQ=RGQ4zFAt&z;WUK5MMGZhWT_h+klU*S?D6hh_5R@0Q*GvUaJo<eiPbdbXbXc9h@u--*p1eX^AXrh&w|K@eMM?C7y&'
    'fge_vT)#!*MQ3x+}MQR3lLmeIwyRltkZr=18Ki1T{qkgVo@#mv6xE5Xw8iERkgWjAR%k>u191qh$SzNFOEZFZ*58MTWhhS~>Awqk_Sb4<LmbR`k}9a6|'
    'M-5jbJY3b6x*!~sOOVcJ90K}OQ>$}9Wun&=1B^gWARtC(V<G=Yv<i_!#En-ir8EVd0w6IjgENfTH&4We{0IeR7cj!8A?<8%ny?5OCrrGeK{P<Y$YtuZE'
    '~yYYAfx>s$626eBZ8Hm8{6$?CA3$sr@fcLTwIShuytPjw85kqqXdoOrV0lsJLca+_5faws4fvcWv9U5e`h?4y)ZZ2_v$(7x@b^Iu)2mTy{Z^nU^zq|h#'
    'fXm=})xF|5P^yb-f@rW?8NO0Y5Dn1l!~voKU@lQgnqdZ9N}4`AJ_^7ZhaW2J5vmE2+nrK$Z}D<1(k2dHD0z&x#Udp}w~12HtTm+`$j!~{yp%L6wwGrzI'
    '6``<aI37Tl#-@c+yaY(Ao>Uv$KF@cf}6{emL{XCj$m<+(jkJyad-@x(LgkB(pB%1e=V)%UQ5r~JcUxUt}suk)ic~VO4V9}h@0ZvJ+ZQgl?tbz*v-S-#o'
    '`pe?Mw1f9L7zTpW@(<O!8E0#z&T~Vl(_~gpcBQkiTN{@gR@Iih(%EXR)GTN%C5($G$MkNz#R2PIl9su-%{wK?Q&wEC0pfxHWk&X6J8_4`X)z5qU9YpF&'
    'd4XKN3np3iu;#nBCb!i8Y(U{a-?U(5~{HW1KSM`xvJmx~}s3{g6(*u7m5mwc>Zuf))wP$p$(Wk8!`)V5@GQr`EZ&N0@l6iRca?JK+x0Jue`G$m_aft3%'
    'e>VY2F==TST<VL{bfhHM9nsfkq=8P_gW!WJ1$Yn-mN(g2qa0$t5o#o<!`6B1rKsHOxa)fXeF_I^wv-uu4ZfR-IAST4K<JgU?B?-b)QUJ#QYAH!_>%;RI'
    'GfPq%JQns17Y;!dRgVJq_pHSZW6hnxxu-ixX{b1<r1-47G!D4K;)}%{*8PkTxOk=jyr3S4J7kAMDL%8sdg<Ja>KUbT&sJ!SjYX;*qO%frhNQ4qhu0?s#'
    'v0`)$+NXMJB?ghn^9Wj+geb&%nR?gLl+%Fl65f<6m&;%O~7U*IldNS%**pNgL0GP`np$<d|%DHB{{z)qR*4OU$Y}7rR*Hh>rm5({WMA0*#=EZ+1Zg0la'
    '!r3?2C*bM*{_TSgm*wMl3f;%Ffz%YaMtgU5Er$f+nTcEKUVO^df#JiTFihKg&ff8*f>DuS-3lYGPqG%JG2{VhO&A7n(~Vw(c-UuA}RM-;6D5Z$TpLU>}'
    'oX>>P<olwEMr7j$ZYm6uSe9bPPG)dDVD3AI}AW<{^I@x>4P7L(u5ijfVuM_-EXN%F3Ko$M8dsh!+ziphq;yioThY3I60xwLZwl}VCzu3>gggyR{BTH3j'
    '0?0-V?fKN)&&Xta|Ls*^#Yy$I~{Y=`qh5EupKrDmYMF94oivrT~3Y2EoikgIKH{jfttlM?kFDrMOR}@g`hM<;uciB3%_@&=nYBxShCaYzP(sKx=hM`8q'
    'yF_1g8L!yn5wMas`Kp<G(~6XZ4dQwrU^(lRlv>T1vI4MZ?8AWLV)4Z?ZN`03%+2wTj_ZBV;J9=cC`34$`iBejElI1@EYV7<b?REg!J2#o)E?P)GW1kkk'
    'lLf{Y84m=?lkZ>xYGeK)Tc({>gc5+;>o9vh&X{gNnRRtYn2p0f;~COOQSdgj=VIAGY!g1qd0eryflh4mdHz^ID3J-G%_q(l9$E;+0k(&$=SIhq$*BMx+'
    'zt0a*B$8h|eI(rM@|32u}6R#wNG=9`>C@+KLlvX%=uIN(t{sMl@*_O!N}7;FMdM1q0E`HNO~cT)z3mDaPcSU!2zr@BHE(M2lSNLG&?#&cix-E&`T94an'
    'xnQH&)JGq@0H+~JD`Sc=9?mV9UDktu&*Oiwc952qcGEq^%8hHUvgo6}IBF}~eN*8JvFP60F>PYGx`jDf<FssZU0pR|Mr?vO;$tO_~pB3n{HO7nyOQ$lf'
    'W%c+S;?wB-C$|%+?6sOixv|Gr|T%mZkkh+(2Z{`ZQ0$vDW5r7QX``m!|?h8r^W?n{r^G!*1yq-PO%HjzSc#<8jIeCQ4aLeApF)uc6INc3~xMAvsC2lwc'
    'A<=Ao1Yaq2meV>B(bk>iP6oN<xd<yG$-Kf0rRYSo=I|w%7iR^+#ke@*tOA!R0M?1nCFkN)@KVKUkc5G<WP9#(Bx*HRh0nsG6b?!ShBXNZr8*Ccpiqi)F'
    '$oK$w0)6J#OEYDl;YD@h$u_lMr%eQRCihj_Fdg+kC`x8xI*%W<EhvD;Rx|;z=*{+nk~fS>-;Qoqt;sBo-)YniUWa9Qy^#M4qGN)>fwAea;CPDE6JO>ZN'
    '5%?w-x93Q=2Wpq1vf3;^w=l)}c`P43TVSR*@lMpx{X|L=1zL$Ph8bIhszguY(F9LqzYEW)+u~ceVAU8A>Qa#MrWAmX{o4M1}}s>LnQ>nvqzrM8sW<lpL'
    'H53i(^0<6JE9C0XMuv3Zk1iOu-+CSqTlV<<^%KK7l2GEB6z7+z5{TVnH0T`aM=JN<Amb1TBC0OuP0jQsLg(29;J#p#Q|@@9e*Ad^ju6(N(GX|}*nh`dS'
    ')lF5Xj#{ztqBpE@I2W({p=rD;c0wxd5zxN$79tBIz5{RI9qlE**8*K~--iLor4uXjS@5V|Y<ZgFh>@eae%x*+H0d}LI5?VK%5>z(|8|e-RvWpRiYHUz7'
    'J^DSLeTt3}lA#O5GVQ==W6Wx2pj;Gfu=%X*0L~cO4&aQU3O1azdR1r+>4_o%Fd<PC0Qc)CInc@-iGkg{?Id=R!^Gvi9+3gjaYzJgj(ZTMg-1gI;4Y8t^'
    'cf}oS?7rKXNL@N&fUKYl8|SIcsKQ5UjEI+jk*!#;g8+8f$<SlELVjcI~(QM&quTqGAcG8qg1zV)Qu>ge)Pz4=|@*AkN#aDI|x0>10K5){_lQlho(nDww'
    'T}@vXxbCyT=OI!s4WmEj~#hTYSDlwiqqx**!v1v-@{t?`GW%pdB>xLs}*Mt^iODlBfVs0MerZK&@?704VHTQ~+qqu78HEM+JZq-W~}6-EHc`)p!#Ciq4'
    'AwP;?RmfU<uR39<CG$?y+qhkPE?$N!mjaDS$6q*bOr#yA-&*By%(iCqkC^NjV236K*tDY~tZlEn<_AbS}eJF=GTVs;#7OvGoaBPjBDNl1^%XE0(+sPOs'
    'dP_GXnbB@~~YL`J8SDkpq{YUZJb`}%6ebz}k6fh=|k-2=Wy%g)CI&H(RM8YmzM{$>jN2c%myHO_)S%snwpb8<=`GaU2b^KI+yU-Cm!hj>vXJI2&KPwBT'
    '+XN;+-zd8`1{9q;$=@!k_u;Zqn)5DYT=nnr42NhYhxBW@<WwD82W$pC>&R{D697w@1+FmxwUTkbCzrnUM~jyf610y)E_LuP^zUh7e8hwnl{QCt^n1P50'
    'KRREA+BwV;kedD#;*{{f!<W@WVEI(r;GleF@?MkqNwi`=7<lZjItEEbM%W|9pm~v9D4%^c7$Cc*bxR$up_K}Y4f8GHK`_iW}%pXNI6(n$50`<T*pdj=X'
    'XaezHXB--~c^#<a-o4J|3~L8p}gxwcOyD|7t7`{q9KoqH;LroVbzVI_c~iQ6msFf<~I_)90izd%CO4o~E5~&U5p;b(LWr`27q4o@`^-Va2l?4RJM#$<Q'
    'COc=9kgmUHNp8_UJE304+|>=wqJo2;}BOfQ}Ul%*=hJw#XwBo>SUTSz{gq+-~==X0A1Z^8Cm0WL%zZ1`ovaglLL?U35rOd0T(@Oc8+NUkGu6nvf#WDeI'
    '6a4VddfHpY8luL+wfl--+kcy)62!WIt3%Ma&meU^Z%3vspH(h=9S*z?pU};9>4oW`5a3OFxZ={t-Bsd;nI})67yS8}k=Qd47%hVRfGaIr2x|yhCi&rsS'
    '#T`r^Q`_*BF~RxhEv=m=Slhw$F`ejUQ8B_irU+pZ&Bme=^YrL^X)#~5%U*y)2K(2xU#+NMx~Z68rbiLMX2s+J?&3y1h>8WqN=k^Uf4$at?i=YQ3bZ4@j'
    'Ai^7S$rjIu!1Puj<bA%(M17NnnB1w-@8sVcuEvRrHjA<X!X}{?jfPt(_Bs0uX&3PitF?~mdgcnpNuRpR7%K#xge5}rMN5uBa2Sn7GCUfwx}+HZ=nT#C?'
    'p%fM<`{{<*_j^A{pT<90{n5K`%lo^X3p#8GR;cocpF!zA%$Y8E3z7WsD2n9LnQ+xcw6rE-GD|&njh$V^B+s=ej3vTzB<rFOf6O(wf;A$|no0WO2w4g_Q'
    '<@IL|3^#2uzH=TZ~aLEj8_**RdN&<<L0SxDeav4k6zzp|#=TUkld$y*YYGrgL%A9%QH)e6vq(Nn_oiWyn?((B=}z}DU7(g}M-%5#SeMzS(CMIoG+St#*'
    '5IZHW@i2~4%hDLt6u3R@&C_ZI1Go_#0rZRXTqm_D`(r^veuP22lEPZvUo}!qtb~V?nd-D{$r|3~Vk=tT1h@h%JB7x;X435Phf~s#2LG`NTlT#r@`mSHk'
    '6#yO7TR{<27ZJEr7ZgEt5eX3GL)?V2fFdYzR`P~rjf7{zVL3V*0$NTO%Sdc?@U!EjI*QC@avB3Gy2-#&;iM_8ZXC-<jtoCmuL`HmLecZP?l^nhPRgs?V'
    '#%z`aRzNgB^D#pJd!TlER;!$*sMe8aMe{M&vAqtslp)@N}XdKf<)oTb(GVP_X_cUm&?W+j>?+ja1C;VGb_rObH0l+Smg-6qB_c^$mSql4m=G>>4)p*jg'
    '%TmNzeI7M*8adprh(rh{<+&Y{<7r@-s*#sn6U`$*g7#3CHHWL}Syw)99*A=?meFl&lPFgQ#3(1Xx9-Dnk_?m8p!RHBpJm?2Dj0Wtc4^Y0CUv0=V~F2;U'
    'xlMLEjQn#Oiqx~gVqLaiwrMf`hx#HzMy(I5R${c&zYW1Ubxgh%C*k!dH8a0KXtof9@7Sp??9b=y|H3Gf7-TyQ65)k{42II^tKKa#BNa?#vVzn*#$Wmt>'
    'ZWf0>uA8WI2k)p~5%lJwJ^_r-Nmm1Rw2%qMuUgBlNos>2!;)429+bI^*dIEPvr4dUOBNUbFHs$CA!BP&f-T%fq=GjN<Sa{s<*VRaPw0R=o#cCux#G~*?'
    'gaYyk5h`t@;iv><m;$2GlR2=gSvRX`>Em4M<WjcAZCnRdBfBE?M>_SRWLS=_Q%=^6{K_V}MCP+x2+Vi+5>o~302vXm17D&rX#2wg-Ar+>KG3Ylum+kH*'
    '*!qBvO-(lk3_LaEfny!lU^tgNt&_)YxR30jzn#C%KCkrk@`}s*!nJT1DtCF-iKki5M{smQlJLw^@SLF{VE?mKr#saQ`k0<6biXKi_tOK3SS|hLds!K7_'
    '@ROY@K$ZDqS&dLs}!bK8TRXD!56P7%WWiX(Is0feCyb#cevrb31e7rhX!Z2$u$MBb?g7D|4b;0qv1CQ5apM7ZWno8FMBvv^nt}ncS7Sh@w)O;G(Q_6Ki'
    'QZKMe_vS1tsUWTd*0AS1R<?Bx(aguN7_A@;J1I~RB9M8AnZZ`EbSR%3#aBztf=L7WhB#StRo7Al~5TQ3#>qF5pXT=G9iz9mJl01&Nd1HeI&o)MEIEaJ3'
    '_h?q5(=-dSBfZQq#AmZen&JsX4Sl<!BD$pmbonPq167DrgqOT0RqN0iX6?N_8m{>%pX~BxN;@2qBiqTG#71lw7mGG;YPFMqy=V(}KITOf3R5gpDskq~k'
    'k4EdO*As9<Iz2Q8qFI^ZI0Nhm8P<zMuuvWiLkP?kM6M8d20mA-1raN<ZV6>xBeEzJ*2G1p2uT?Ph^{?J8Wsf_D`A$H@9He-bG+&}GqQ^*hUsaJVqL#RI'
    'ymc!BPN<d3^95)ED#bk0Xx!E3+jk0F|D5%4b9P2v$0N2<kiUzPT4BOjc*`;BO<vavL3)xh=0TbLJpdV1W0nvOb)_UcmRy0d@~b-NS>K1o_j@Yi?76sn2'
    '5ST`wFO|2gK7kI)5<|k%Li=N6|=DQoS1m1l2o;Wq8g5P5_D2Sjj6V5)~n*oCp+{d~za8lw5KmFm>|Ci4OV($$&8d$bkAeUC0B#7!ZOl&oFGufskJyc<>'
    '4X?Iat)L(e+abv-K~5I1@vHt~@qVAG%_g{T7c2S6#t5IiZ~RSfW<I!2RH9gPX|g}}mC*)vhU<Ch2wO1%O_<$8l7t{iXh*%GpbQOG0=f>%`A7P{q%L(*('
    'H;*c4<pCN{W4_F_05cIA&RrhBho(Ed+-qf(NH&?N+ob?evA_z=2a@qqz%lHcVFrPFe=W@n}N~al;k!l@$50>J&H&ID3msUwRQN1?pPH4|KlFpvMb5ft5'
    'yG|-8XGD@l3IIt5)soD6!ve^qs2SspaRm2DY6+lRwiZK<MAa}XQMv~Jrld_34v74I4Z5^-sq!2uMxrQZmU%_wFES7Q4JcWHK<MlYkQM;q0Hy%!4p0Kl{'
    'J8P~@(z#(P_z{Vzj(wNxy3tpM+*BLT(h3tK_$Zd=?*Xf1BjrJ)t?=3vC0mOY3yd)^C_#2IAps~9%mRY(7D?OII?2I2c?T|1UlEw$v%Qabp|_fm7_7A))'
    '->g15ysd9&-{6_@w&zg5rusgj<^-))UYGxGBu5piLcY7%?pXf+X;qMBl;*Gnqi2)UOJs(G-eD0Q@JN(34oPlT&xL6PHtF?BFcdoFh)9=jd|sCbCV%ff;'
    ';3brRcw<bUU~@8!gnkxCBd5^3ai>O{iINd_m*j67poeJ5?4-5r<}^S?$r0Bhh8*a2R!h67;V6MxR=6Bd5%b$rKcNDuKmlyR*heYc&bs=<FA;cOY21>_u'
    'kv`k><3`ucod<!vF*$QL9$91c?Sf=Q$R@urv1MW!LJBX@RSz9hzTS8g<e&R4f+raN=3CLQFw94;DpP=C&H4>YzcE;7F5-Qp1qrBLFVUhRCY^HdYZjHes'
    'Jc{PnaFkqZ4`~nD6@}kOfojPT!i<c^79z6g0{`goL~x6wsCHnJu?{5($`8x2qvG!gLe#HB9cr0j6eZe1=7B5WyL`T~!xv?AG7gK3PL^zCbh4&ONeytdQ'
    '}~qQdlfyEn(K@c>{GE*DcNckV49@*j6@Bg<vsxmBmB8N;wxBRINMXkp`Oh_fDRCV1mrL$IAovE|M)!=JI*E`X$YN%A`lW!JNmtz^JBjUhv4)RV++3ra)'
    '{#h;9%q!P6{Hw2M004YmeFO!JdMJb5^c&7Zx6F55eA-{UlKMp={UZ;{>|F4<$GR+F}$z+}f-1dCRBJE()3L38%wY-g|Ox$8%2cA~?j2eHa(JqDpoOMmd'
    '(u6(K}R(h7zvTVkNb7bA@9F+Ls&NU~%V4DFOo3PGWncE%~+%N~OajyZ-J9$m@2F!(A0&!*ye4iw^`P(rlx%mP(Av%nsqklIlLy+<|p#aaN>*cYlJOAoG'
    '>72jSI1`X7j)uv=49L%tq8kwPaAf;6Z5RX+y#0DONyhBK`i6}{Yqy^|_S7G7#kx?`?OD8G|%@0pd!2D>ZTXbD3L)tS3DE12%6sHsW{R+nq6cwCKh>cF&'
    '0_++e38?N{H)^EnkCXV}bdqslRxs3=*6)nbmgC5ZXeZa(A~H_5WiMdmvXKjYcB-{i-|4m_$wg<?L9;?~oHK$^jw5piJ9pNVoLKW7oy`3>0X1{=X?ZAvJ'
    'zVl?6HiIn0geyOxLR~ooeJ?*O80h3;@RQ%sCve^OO?$J(58&{KFmwp20wwU4=20M8*N$sdK^u1e=$cY_m_xLKW~H=<%|g@0J<DtLU$gQ#Oft!4Ku40B;'
    '^EQRPLRnks$TS2?j;>o$M4R#_1ue3-~*vVPRV<i$r8p$yss&g2_Q-Ghu+PJ1az|Z>m4Q%MnbyfdxsvzzP)FeFQ8=@bwpdRF*xXIlDx;zEM7Sd8g@uS5l'
    'juHdTLsbx3vspzugugqBqbt$gCH6=bl)@AX7-m5)XsUPwL~0pJ7qXc+7kgDFaC45mOTVK7Z%4P-D4WE3wL_Y!rC2dQ?5T0_m%PFOvRrT_<EG(}-2e*vU'
    'pE%t5@JFWK4+yyQ7h43I@Uox`cz_s9;Wjd|+W?)1u`34giwB~EF7J&c*P$RYMi{)%B{6!-4o|e$HN~_Q^n?C5Gz;bX=NlVT}1H_B>bZ|pZy+}!klut4z'
    '0xm;9Q3)(zAO$T@mC}h6)^!NkI4D%>q@d+u5Cf}fO3Gr8c_}>-ir65nM*^rlNbQjfJ1f0MG6bX)9|>M8pzY{~U!*zg2g(mL9xGL~8Z=ZRGlABlq2&wcJ'
    'sP-2Avnu7DldRV8+=^=S+p9{9ca-4luZae-w@)Y2wB+bZgT_F=zTinR@!%zSm4Q_0tkSd5n>Ur`e9BAGW%s%k|4WZHvA?=vt(^eNid=UOji0<TsDI2X7'
    'h-0waRfxwY4rVBL<ny63!1nwzG|YE8|&$WCvN#62^ca^I1a33$mZZzMql!&Z3tEEmmuI)KMan11_duR|Wf+@})SL8rc~yQ(QlTxReC&T)@xNz?=%`3I#'
    '4YD57iaK99Z_E+a|@w5h?0C<!J;xeEbd*btP2btMbfgmonYjuO_D2<i)jbtNMe2aCn%UkK|mHd!F7>#*uh%9$R#5R`moc-Kfm20<(+rmJn`g0hj~yd5O'
    '$E^0G$F^K<b(oGyh<Z!WcM{VDUP#ZO%WME$2iX|%k3WX9C=bex+T<{(Ul7_R<I3*5ea1w&#;T)|Uk%6C(@Sr%N1UDKKN0d<C1;r5=fG{Xtts3!P;c6?F'
    'wkCME*W5{@9oOh6<iM>-$F=ZrG!gR4@v4%Rt27VqkVE3WGZ+H~@p$yv%DNLhh+?x$Bc<!g2<ky9ii|&0kd7h^526n9PR7O%$qq@nv!KYRV3?6$^fvIVg'
    '5le$2VBQ}w(<`Ew`8bS5Zsav97DLZU_BI~7E2Zow}1{WfZ}%a-6qOaU|nTIa>RX5_*4a)IVgOpI70$y(yVdB)AxKAp~tvx1iXD?wzdm)(h&U)z1b@eme'
    'NU42*26D{L#?4%(_Exf)^32C6=)e&0KPdN3)lnQZbm`(^79Zx-Ce*(YKmUQ7WdK(r{ElI0MPeOSI<BY30=kMXS_Zt&_){YJ;$2^*}mZyfXrc&Z~8Wid%'
    '8=3zD5jH!a<6@}d%4rL723cVwp#kisL!-E_ymqhTh0gV)+H=BMr`e}nT>NYA)VNl`TgOFMxC!P40$_YRh(n)V+mW`d=&4{HTW$9<=rQTRcInM~w<!1v6'
    '_X5@X=oi|1PXU+Bjd7w4pSIY&h5^Nd@%oOK$ftiBVl^v}qQ-Y#3J@xWKD~3CkBU-_77YfmuaD8$`%g&c0XS9srGRPe*+cs8+)?^yXj@Hza3~PXN<{if='
    'rG|tv7tcs1y-!^!r4zn6fp{LX90c?jIGl(c>+Pi#idOVXEwo{`t+zqR6|A|zIV)Iq1J-CjdlQKe4d`#2@+lCxvZR0R>4=N|EK#|`)E!06;WiPq637q%V'
    'JqSG5fHahGOr7CF-pIH$Q7M)1)(cubJCarQ3D19ubrqS8$gRGNBCMkA7aPx&UgML{PWAMC=3Kf86W+;O@akVAhuoF6Jv2pI&0Vq0^y_C#E|e&(@Z&p0g'
    'V_)7imsi+Qb@tc4lKaVKry~x@ZY0n;JM7K^nzUeX7bYESRo}L9|g*O2zD%lvZ&m6x5m+L^C|nPiIWaK?>^S;lZ5-sGqixj?0Whpm56S;U=IGX)MS+u@J'
    'Hz<eyk=7nXyf;kD+4kOiEQLy4Pxyx_ZNtCXA<^Kc#zIY!$8EBD3h>6iavCN3k$fiV|zrSf0|oG#$P*x~x5U+idMf)tF+WLp}>7QHDIbP>&NiZuEpmy)s'
    'W)<hbenL=vD>Z3=B#)?B#kgCx!<w0u2LU?SDVzJm<EKjCPPOTtUrtI@pzD(J(AlF5s@djzU!cpA?z*wA;Nm{RYWtxG1fca!;z82Bv%_!Q9m7h?x+saGO'
    ')nVxY4cJZ)fe!4b9HVH#j^dtm`r?u5BvoTCpbI;adHccUOg7d>6A`Y|7nHg|fmq$5l_IgaSG_c9t$S1z3_OOP(Y!)Gq*QBWGNe^AI#7^Wt!|p26>NNeD'
    'b|*hCeoo5jQ3cOYOQQybQl{$)Iuv5XY(MvRK`0cNHLYGThj`*GJ$-Bqp`uIa$Vrt3<FA6z(Rx6t_zFaS4!50nm9=Hx&Y`I39@o!cs<mw!+vHcV4E`vpn'
    '}~D4NBNzUk%i-o6SwQsthkHQXD$_jFhn(lOPbqgv(JNj;VX()|N6(W{~cx5g!<&yxKyJv{%jf$}%olbYA+a>J%hWU^VL_q`_*|I!J|8EuNB{HnT>^PwU'
    '~87ON6&5aeP|`gZ?F+~FcW(#b%=5ya(YP|b0Xt#+Rqa_nu{)dq-yYE2}(*;Pcoy~Q85NO(aV8{}<JEECJ!AP>JsB;4#*Ms#MM$mbwCr-i%@5|Q6Qn%=b'
    'iTqNUf?GwnvkN~+vH>`-E692`Ll~Z&xb6PbE<_xMrz?xMKa*s2QR7LVYv6&YG*+Ciu1PNF+`BfL<4TBu3GcJokp4FMi)liuQYcq%ivo=F^xnQk{oCkru'
    '4K-WOfo5!QK50P2lkeRvPYa8h*tHMjd|9@;c`=!`yU!<#+m>Q6Z(H~w;KT-do4B`Qkdi7uL>+U&VMWkMc1IO;681&mAXU?Htv!{E<Rd2(FDUk8C3WPJ1'
    '_<EhV?iUd?xlc6n9cAKg_>En#klHb)NwKQu~&#i1p+E4qG^Ci5ERlhlG$$MSU6uc0N%j^YK=x?!^w=+a8M4iqcx)JLNR8#d69sY4>c44N<F!z7`);9cS'
    'I@m&h92osW}@UkxJd#I%1VtMR7s2(&S%|CnT6Jz&8z#1oI5oUnmIX#XZxyA*wkLfkMzrAO^+i27zZ(44Ogc8P!=|1fQ|h8U=7Ok0Ib@mWsOR2VP#srI{'
    '(<^KrNfL?aiAI|4T|pC%lQjK^URl16s+yaGO7@<lt5`-0qN=dtd=Brq%ND;?HWM$3P8SWPGo+72LDkPmIIo_x%$_F)sbPF_kfxdJ*7<V)MIv<AFsn|<a'
    '`Q~S#y6c*>CrZz7wdql;W7G#j97QkeYScA$Pfo*u~1sCsJlQKT^YPplFb-P-+*bUhq`d)fU&Q2s;AZlwb<{~YhM@M)q`xD9l5Df>v1F1u;DT5n^*P!^V'
    'Ubak}=%}bI(z3~@q-Ed`gJKk#xsobr+3Kz;X&KO>pzwuSoL{d+)D$A4f`i7e6coWw@vjdGVW^fn6~l0{`K<vt!$pUx)ARr=QVmF2#u6JU**i?G1>4GDH'
    'XJBFR`Co#Ad8iYd57o<)tWUt9Q=}2vF#X_;RyNV94rh<y+OnX-{Igy^ul`<wG-B}W<xm7mAgAjz3`n~QkLaqkAe%QICdCgf&Z4(9_Cy2Dj^|qKW)Sd5d'
    'e*NAs4yau*d#QCy9O&s+XD}54rB{05t9ON=Zme+!k%2*iT<|33jR=v94r_Vf(5}qyth8fzHLO7*rL@RRk#_t|CAYav1MBrWGMhlWP%9brpfe9stqN#c>'
    'uz@bEmAf*40-;nBE4x+-KN#RQ?=LhS{4o9(aMu3!80>*|`Fn2)Qtx`tv)+8Wea($%2UlD%Ed-6J|csxcAzEjx_6kqv(NgG*jn?X)u&yy9Y1Px32HYLZ^'
    'D)hYfCyMw~-qKEJVQ6TtE3%zo3YPdNz<$lkZso{=c<0L5VF4%!T?J!<N-634pcm|c#c*c}f+;j8_ymQnp4!8LIw;doV{>zVn8m>+P8LkRthpEcIrQ<->'
    'EiHbTw)ldwW9DVlYW&tb(lTc?njVg=tY<8&tXSF*GGS>R6d&lXOeZ82(SiOo?!Y->1H~KEbRu(9quH!x4Q8Vj&R|k;zl21I_ua#=gPm$V<7LJ45-1_$F'
    ')HR=5=0{J7ic--Ajop2p-LmHk*LDMSEE%56ux_$MFEwIG7-BbVn^tjND7f_(1dmd<n6dM{aQCf1W{`eTt)7K!#c*M(V%RIb^@}&;KQ3L5&}?ZieZPPIe'
    'ohV9gY*Yp_$;TZrM@lJK9lo+m7lb)=^_dsH47OM>?*@9O$?nBU$<Cq~jcQ{|v5dK<ufla3ZiYJVf;}A?s0f$6^buLCd)jMZa2?*Oux@`KT@lM~~l%7q('
    'CuAQEw7g9)QS%UKW&qJcta(EcSR+o#a6$Bu|>vA6_e>#!sq8*hzvE@z^#wQvi<yv`TYOwdRir->fHMc3uBp`ro^<-6SW%mvWNk)oXOjy7@8B{~JYF%c='
    'a=<+P+hpwOZ1P152gOrC8u`7`2j)Ng=V2f}tGiw&ohVfEfbx5_sd^I0=6APtNhkQ#JKxwlA%A)R4_35Mh>6(w`kkxg&5E;E@PJX{SofKS7Uf71$uU;vq'
    '-^{P_`PI}DQ>B$wLE~vFVoF@pdav@|-~^%MH^71`^9>oCIvQYpmHw9&l=nx0R_K0dOlkujBY1l~hRQ&Xsgn{;m8J!EFy&~$r&<YG_%>x{iGHl^NbQPYT'
    '?7KFx`;_X(LXBobn5247!||n(6U}nGTYC6SmBx$u{pUWVF&Ts*kA$m<eG#U#L=KUEoz0sTBYha4oaDW6nx+w#X2R+73(yguz=xU6{)%`1FXRlq-++maa'
    'O{SH**jUmJ-Ew75h~jUWq|S({4dZBr22*gx4^hY6;eJLkoMrWUro3Yr-DTmJ#%T_)=l{X{8CuPu&)hp9WJvevqxUaby~>HM%=Igo5jX(#!5i*Vjw5w3W'
    '_+P;e!)Kps>wi=-XVjUTWUkh}|6xsfHff&rB(P1CAOTI4QL9xXzyl|^e8>jo511}(JUO6BgQy+|t%{YvAG*P{|_krz$rwY=bs7@!clym3`X-kxdUgg_y'
    '2J6loMr0RJZdl!&eix1A>D#pDxI&6VT+0;v!v^tM({6uBYV($tUZcU^zXR(RVb?demd#!gxS7QWMbT!#lIy!DOdH{B^`n+g50-h2}gV?IH(zsnIsWc9#'
    '9m|aBhImc0yuK*rjn$+aCqOkRY14p?^esg7dNNStY9cyxlWMXwQA&GI3^k&P=K-$tMSgnk=nl+?ej7fZNm-=+^kA{8pAaKtDk3a3pOf5AJd=)ngkswOp'
    '-N9QXlb(vq?FQo#>N%t<K#z_N(yoheNx2@oJj^LH3$r<?)Kfuu{WWzMxje^UI1uX+gxWjjM2i+*(4w+8^A7B_aqzumh#+2XbDH#cxB;W@I>eOmaM}8My'
    'Fa}{PdXo92{0!PNFHj7^JjsLa4k+QFL2LlL%x{#w0AN%8G=qjnabP5<CnYqF?J(w{0*@r31lmpkyF;Gbt4a$O}pYLY%77fFOV=2?#1Rr2qjDR|!C(V~='
    '^8er@}7B;9uAF~A+xuZmBN6vIZrBh9wtD-t;kD&!4gND3znHTjz|zn*;XumGmV#%77h*`)K8giVZ@fQ9+C0xZcIC`nYJpGva?0+Ny}k-=OkmWaSqf+fJ'
    'RlwJwNzyL^CJ*&`r)s?M+b^(kL%ZBqb=LDRymK2Z&tSPYdsGYAG<tkQ-jkCB_Dzn(H;y70@*m@l7o1QFpz4Opc={%lg+;#CfwB_@{Hv{6TVDe+*lqee%'
    '0Uyc*1t+IWP@d&>&1Wl{zZ(r?<uzTcZH{BKzGe2&#)j({{hH6t`LXKuxQB9>p<0XXqFAGstH>T$N-hO*>dYp+5$Z7GF0Q(wPjqsmjmAkTZi$e?n2QRau'
    'jc|aQl{#jAdw{CHjO#ha!3_n<CX>TB@L@wPRTEj+;yvB+9^E*3xCuZrz8<bukHBsix4V6-5x6Bd)>dELlqRDKHN;T1~Di+oi(NCbh{TSr#8j7!KwneFg'
    '_W7`sn8V;r8Lf{mrNUw&ITFJ8tf76oOQ)V!ev(D)y^5&f=Q<T?|(-x|n4&+QoJi`&HbHWf_WgF<ixX71LEbyDqA?i}fnDtJtsN*>zF9T?|(-Ud8>kAoh'
    'J1<5hfAAz#IE73)=OSFvBk-5qkU?P9))<to;z*sfx~idSRtxQe^$z^v|Kx{CQKmaAB=V!Mj{DxPnX!=;M*?!FcFjc|_NIeO>Vog;UR+xb+w8c5+P#;cg'
    'FV!n#yDn4qgUB%r~DK99s<1B@<6wgvROZhC7vsBL#FPR(d^{bmt|MBK`@9+NS{Qbk**AMg0{oNntzqjA7s;^(Zn&b1)&G&cj=D!cGf4sZ-r&;6mzuwK~'
    '!bjnwH2<6b&VSmx@oH9hd;j{~+qHGo>iqlW_P2L$<_llUn@`q{AMW11{Nrry<?X}W_wU~SaaDT#_SJmtEPQtJ=`lOzhje}O+1s}tZr|Me=}&)hCm*%?n'
    '|b}@-S3|Kcz1jM;r-oTYxdh0|E}Hj!_Bv|D$U+gZMXVdWAGPSS`B@4^Q%etKDznaNrE3Yy4rm4?qT*-HEw=&`{wTcM)hv~_U=br>2+&#AHD9g@9&<xd~'
    '<t$|5so7&5O^z(u@D$yRTkIa$Uv0dnWzCpLKznXE%2Jo3HNf@1H#UaQpU2El>V<cl-YS=4#?*|MukF`zP=I<?cP*WcnX<^vkyockkccKFle;ee>k`B-5'
    '|&A6~zl<<D35U;l2F*UdE}U%bAbb8>h8<o4~WC*R(^ncwy5$uIA2-#q;A<k#=t{r1h>kM~dh=F88&_&6N@`LBKWm-YDg%eDUhZ1T=uZ$7_!fA`z>@BZc'
    'X{^f@^x9^|4@=NLeTDFgR|95xqetP%jH66Zt`|$eVk5B&Q{oPOR-aov4`~6?-!?WMS_FWX?H7DQ~&oubI_`5mqAKiTZe@xEoqnj`Oe)4Mb&%8I!x@Ysx'
    'JQKg1f94VUY);2K7N0$zhvHlc->xI`qnlr@zq6e$U#uf{HuMkcc>d_-tN*sxllgc3KHK{0H;X@*_1E9o+OO8{v$?OoT1WmzH=n&&Y{y48zy4~mOY`r#p'
    '6z}6&0-Zkx_SQP;<M)8zh6wteE+(h?LB|C*zo!O=ZpED?^(ak_P+jR$^ElGzg(iiM>oIu`z7%|9{fkcar5lM`*%Ox%{4W*;hgo4pTzd$<+~4WAKt&76a'
    'P2Ad;jjoC#&ij+cj`Ezqotz{p$}u-Y|eSzxeRx``h=gZ*LgMo6rCC?&S}<TqAn(#ogN<Z{Pp(4I?|}@BNRrZx`uUBYgAh{^8wQRbC@~^X2=w#@@cVVZ?'
    '9#?)}}}%S8{?z~4Oo>Gt*88;1YpZ{Od()vYxNH=n<MH`lD1S^M|Pxo_4h)`Z-A{>Pu*e^?u3P0P(!w-5iKifeLizIy%e!-rcsux9Dzs}Fy;`|-E$KD__'
    '_hWWbr<%hTP^HytX+7>x@Shd&O-F$ub@Wb8v)#KIvn#P;YKHNVr7;83fe*Mnhv*vVOy?^-dz4r8)+naCS{pgpkDZY9BW^P+w|Glqo|5?{-%5VPm^;@^L'
    'X8z{+@9$pStpQo{fAi(bJ#=ds%rBd>@GvR)!<`0gt%;kj-@X6+tp?_odj9ppzta617KteF+~I#ex6-S3Ke7RJV*I-^;ID17AP!mZZUX`r#2pKRMUEFLb'
    'nUbtzF1%w@>WzKXun0V!@|Q@yckd|nOzrA#tRe!yNF7Sy%(*9S81^SA}aX$cn8o^{_O7ZXZM*uyW{-Xz30#V1pe&r;LrXT{_HQ}&;Bg_>~G`G{yzTfZ{'
    '*MZPX6p~<<I_J{_Jn&&!04pcYk=e4)!_e->>cW^{o2p?*8Tb*FQbz*3Hj;`uQ*Cf$;9VDD11-hubHUD$I5M<o%rSbtt^Q`+ioKMEli~S9iA$KTIO=_QQ'
    '|Ay?eiI`j?;P)s$(T%%OPl`t1+5zkU61`}X1fqbIko{?EBX=A#eqp4>i}Lc-fux9?v)nTP43`*Z93_QUGPe}3}YKR$VR`(d3uyVZ}@Z>u9;|K`Q>sBUm'
    'dy}!GEJ$J&(yN`5E&03Qc_~gmYp8oXn`8A*ac`K8)y}Em~n_5!J-ILpwFW<d?H5dKr)%rOcv*O*W*ZUk?zy1GqZho>ldIKYO^UJSZ{PMfczxeL6=U;pa'
    '3pW|9FV~%C&tFUm^U=*b*_p+gDZ#Djn^QCg?!(K6#XGIt`2V(@l~-kPlDBW)zI(W39sJ|J-8}s9r@KWzzkT-Ei)UZ(raRVD`Z$D-V}Egsd4PSI(tkOYO'
    'TnK$Iv@Jt%jeIZ{p!oFKYQ`zH(!7E#W!Dm_uqf@mDtaRH|ze-ZeRU)|LK~whdFR_vHao3H*fDhUBCX<|8euf!^2OXK7IQ8-~axT-=|OBz5o8{+xw^M-h'
    'cWZH?DdA&+q<n&9#f39`Wg_@N`b=-6uah{P^a4?$yhO>#?6cynmyIUcG!eC-=wur)RzQuYUIzn_pXfH6P>0v&zT!e|-CJ`-hL;uAcpzkNWcZs9EFF+j*'
    'MoSKiOIUcGxwL&qm8WBb#`6Sdk{{r+^%gIZV<yoU8J^!xs&+xJh`{g;z&ucl`Uf2rZ0J^iz0fBv&M44=MyKlj5sdO5eyJoH}v**kvw#w+^%le@(u+|EB'
    'wcg3f-A0B>~MB?F(Ymg`VHP^<+zkT=Wk59F}{>RNzUh?|>{=;1RfB8H9eA>d(xkjG9{9!WL|7k6?`7M+0NFN{J<1l{q^lI(rKYRM`Mtc8pEvTRW_0{;z'
    'FaGyupTGF-#XtP|*`pQt#b+-*d;aXjUw`bym%mz{_N!lix2itg4za90#{43tPmg2%|G(G{S4)5Q)i=NR?5jU(tA2cpxw{^<75|^L_3N)b`}*rY->3iU#'
    '-2U@)${Lu`Qp#^=CeQiG>?b1G$xIlr^cjIKi)muuIXDm;1BQbem86F^?dj8(dt;GMN*%>yq)v-bU*g=?(NebChhy~rAY91_pg7vU)%CO^yK*$zxwb0AN'
    'S<;r`J#C$#Q@9<7<fb)8D<GGTi^R{{63K2meFw?qB`*hfn_X^-qtx_{}e#fA{$}zxn#bqb{aamiOeptM%(=-+uYc7vFvJ-=2N@*;ikE_thkz&mTX<vHX'
    'Ag%r{?s@#oKoReAKx^KX9h?dQ+F`||7OFTQ;7o3$98uiA5%%+#;XfBe92emxiNv;VkykB^W1acF;ik@&&qFTVZF=P!Qq?Xy37dL9Mw;~4(^+W)7=8o-n'
    '5TF>0gQ}f4lAl@vl<fnJ{I%V%ap4{jUAHRL~@tTiI_wbMZwDkq6BoTiVBz&)e#oJYsq__%jtpE5=lkNGx08mQ<1QY-O00;n=p=n(zVkQex0ssJi1poj('
    '0001FVRT_Hb8ca9b7*gLb1!IYbZB2=WM5%(WO83KG-G3CHZx{sWo0*EVly{pGc`FbZDDR{W@U49E^2dcZlzS)j+-zLec!K$@{pHpf^mRcBvfS+lAz6H1'
    '!SdcRUs}Wp-MxFO;u^N|6b!^#iXx&@i{X-Gh@#f-XA)hZW@V*Gtr%Py61~Hhe;YnJT1Fl>dmUWs0|@*e5z?I8L+A>Zjl77W?V!<i*C7==A5&i`85)Gan'
    ')*f&PASZO&<`gE}`M%UoX-1WIMZFR<d9AtXExgVS7O@Ix#!g2eXZ0$8Zx^W<ytwq@4Y+1GO&Iy6xpjsuSC_NZ++4y%Q_!S)ky_0az(mdkW@9fHwt;X9W'
    '%@mo5M(7vOX@5PneLg42Zxn^%Cw0?8Jv9mKsf08VeOz=yIv)N_2WAoY9=$zX2>0WL@m5a)pvz&V`*9fg!o0`to9i(n;}Gkce2?T`(wCpc64^+aW>Yljz'
    'ndSrj4<C`+1srujSNfU;E?RpYjWqxlm$Wh#7Xw}P>Ztr8%;BZtk!+VvyBG$}M>L&;BRm<02%jZYL-&($SZn=Z<EM3G=o(0l4s5$FH%Uz^#(K$D-h+{vB'
    '-6H=E<$u8(X}o>QeRLNeo!{}1ADwN!MrT0!9m?sV*aOP%A%B4S)4oEOo?JiFdn}nKbM~91VI`9K*-4Sr*GXTu0)0Hyt*L1YEQ=cE=#h{qX`V4ZN7S@x{'
    'U}Z{ozj?DWRlQvHnd`MVvMp$9LHoFnR;drW1Nl*M&cnEQeq^AF&U4k$tI&L`^<NZZv0$`P<|im-#=aWzcmIwcyVnC6;+wAu)4`#9!r}G5BoapQ$3)=sk'
    '-AM(d9)SKK=qwO9KQH000080GFX@UFHalr=D5>0JG`<0AT<C0AyiwVJ~xTVQ_P3Z*+4nXl!(7Ut?rnVRK}1UpHc7W-&4{Vr4ckIb$+7W-~ToH90glW;t'
    'SFVlX&iH8Ey1Ff=%1H#B81Vlyx^GB7k_G&DIlI5IV5IX7lzFfM9yZ*H}{YmXntmFD?Xh?+M&7)i{yM+hD-WQpC@wk#PUJ(%8}hEOKU61OQ*!z#AzZuq-'
    'z#PfS1v)pa(>;k)xz>555WMpJqPMqs={_W;p?%v<Oe)sm~)0_Coar4p5%Qv_8_w(x2?ZfT;-NXFl&FydR-ppS<zx{B3cmL$w@1DH6yM6fK$;VIbA8y~i'
    'x_$rZ_5Ix||M>p);ch;2|L(*4mv{4_XU|`J_G13<;iuW)-K+JQBc_i-{5XaeF@1U*^Z);Hgk!z?UvA%gm=FEiO=?HI`83q@N&9GBw0TkbCrKB5-6@}RT'
    '@3ETbrFtr5kASf2=gM>PYzwgd6DWTLl<dY#N=vuT|}A=WnP5xNzldYLC?N1JJ7lxpYLytj)nR9=7-~ad+oFu=gVuRofuUvvz)qK9n`7RGjD6U(5s`qD)'
    'sg8A$?MHu|EEYerCRWbTypDY<MJ|7_;Hwc1Kpnhss5rrOs@%Werst?d*6Ov&}l(=KP>Jw(BGEF+U=m7m<sqi}|+s@xI9Gf`Q7%d|%2<jpa3msqle$&0$'
    'J(H?KKNIr^Eg?nIthbC|-xGiwfKpY$nt&Ec%fOV=Exoax(|!xUK_dCg%ulEyo)IZQ{@Mb{XyZo0;3XuSFwqfRgT8l#?Ehe@a0&jrxe7|rip^~b@a7nRq'
    'fSC?mJ=Kw{O=X>ie&-d0t-5zVK=F+8|`lvkLTbkRM?=49~GuAj|=Bcc{b-tynzJ-Ga%IaM3_|?_B&bZXoyZObGXYYD&ousc}@Yft2#TT>LIBRLb)<Tc7'
    '<C=-J(qm!+<+aq;9ZlF;>#=5EBrc*Z^19$la-3ft=O=Qft~-%=jk5<aFz7k19)#-mtPdNF9;Ab9f9PWE^x_4W)=n>#rqkMPxp1cho>2p3RhxsU_Y_ytd'
    'rF2>aItn}9Q3VJHm_DzY3x_-Xa>5U8BB-zR#!(4c7Lw32g7T&Moov_Uy~Lczt7jV>fejXdPTi9Z?0=$dM=Xamp?J7yOlvs6*8x$QiIk|t<!f#PYVX3ne'
    ')LNRm*{;r1#`_5v!cj<7YD8%QLG7NASh$!C+nH%z{<_bJy~tJoJrfwl1nLsTBjUsDigaZlVgV)&dvYSyUnHYHP(!lBl=0wc=X$Gi${)k9S)uu6Zf7wc<'
    '+k#aeNbl4!`=TJs_EOkX2o%A0#&k*TA7jZAE==8@62iNGJM(z5nkY^;Q`Hr?u5)`hl4U}G<$u=hzh)X~GwG&UM{Iu|tH&8hb+0vZd8xhw*@K2Fap0vb&'
    '~%OapLvQNq)pfMyrvwCp2JBxsZ-rV$BL_XrmA`k0)<zg+O%rRURc?hZbVhvUDC&b8-9J<h1cxbXp)LFP2&>|YXvwCo_BT8&6FTFH&4o2<_9e}!#AbWbt'
    'A`aajvuMMdc76XM4f6n`eijkeyVn+)G)o6*%Y2AV#<hYLUErd!(R7xs6}0Gr&hgf-9h90#LE$LY3Oamet)S6ejv@=I++l8@F<s0}g9qwjZZgMuhq+;yb'
    'ul*)s@)$mHy!4txmsa}saCnr;ih>tyU{hTM{xr~J6N^Wv<|nmX0^MiHL0C--PfFUs7zl|+Pz#C!I&&YQ+bWHSH~h3mG;Y`7S*eBQH#=DUPoYdJF`E<ZI'
    '(mFq1s%uJ$L^;`;)!oYv7>K3l_<ZWv~<%&5g-xY!Tg<z3CR!jpioGBD>LCOQ|}Xe9$gp6%W6^>TrT2)uPj_`@*N&p$lzcmQ7uUM$DX*bsZY(LeH#2BW5'
    'PF>Jw%U)PO!AaVA%NLUaTAgt(v47Vf^#7Ut-vwZS~o+M4Pz+}eP7USnRH&R57wZGp|L<3fCD@d+V5HJgx%Pt88M7M~iDkH<dPEvhy|9emi1N7NyY$eQF'
    '%f(EF8Aws&V6>S%hb+c>gnOLIemvpU9S{fItB!<X@QVbENThAs;THHF_%#U`RZV7g>XKx{%sq~PGb`i6lMFSureJ$BB*iMBlr2Q;*<nR_=Yt|e<U))F='
    't^_xrR@br`2YYd?s!6@nskNvYCvMTaRc@3Pt_dkELd&>FT!{KX<xJI~az0m}a;%&pDi<Jz1uBPdRYc{YX|ITxkGy^{+-04G!>8X`Lz`B2q%4u12JID#D'
    'z<X2XKe_vm5Fb!(aV15B6Ejl*0RqK`zp3F!LU@Zl`*+mXB*Sn+IW#!tn1*2!7p7W$2t_bh&-d#Qo1NaU57{XAz#<&u{qYd7_fDRYO4oFbZesWCKqDBs$'
    'VJw%+#xj0ZWW>)e@bjx3)1$lwqsc6Uaqjd(m2DC9Bt3qG9+>hvd`R(46UcEzxcY6*Pb9|DzRJOFDVEx>jg+e;2zlWkV`0c4Z0&7C8F41LdreV-8P-<UM'
    'cOx{kFePx8T%_sk+!$$LIcs^q=wZ<M^3hM$qV=c!%CTI|^uYtoB{LefmiUML4EKo{C3)$2*yq<N)jn=}hnY!i31d6cQvm|CiJ=p)qz6R2SpGmuR0yXTp'
    'U$ZV=Lv6IO|C3Z5oJEFsp1H7UyINU&kALmZLq`vNbsv9E1!RuaRIQS%i3`4rB$Z+t^6B!OLO?8+_)3=R^n8P!T3!3gHJ*bNkk7$+7p*2)h4x)IJqvRZE'
    'Q6!qOo!bGAhSH~;N{X(*&nDv<e4o}`gYVtbHTa2Rt84Hhj9}N``_$<gd`r(=gMXMYgO{(aRydESR`peS=NkKNb%Tc9ywDo@#x(T78H7q=eR;APB1)>$7'
    's4A`$t2!5TKg2e^0>1;agU+|<i?@jBp~~wYZ8!~2VMelW%o7-$Xl@{QL+TzS?@7R)Fev!%x)4TV|UY?HKiUPN^;<|Maj}uxn(JGVEWpkWGS+8T9(M>rW'
    'YlPmNr^0$VgmN1ev0X2A!w1jv%9t%M#hx%dH&9E~=*EwlRNA(tk<5lekc?CH?2>I~wcVTD?eaTYT6eRC=Z+NSK<g(uQ&Dw8e*|0sr3?AC^XQwWVok-K{'
    'OTw{x^M$-NbZp-JvdoWM<TuZd_|gene}R739dO4SO9(3vjjDbRy1=_xRxU27+7e%>>uYFaxM^}E)N<(aOvvss>$E~3-5c1*eZT05J~Sxcu^m)RZEK?gv'
    '2hzHQtTFa)HU0SQAS%WKS{(#6fk>8PhXMN&;U2CEM=8T#sfEkL`e9UTGyM-w|BMM-`+(ZG+I|(sM?-2!v)zu1eKGmxD8BG$@*UIJIFQsmS{6bgTuwF>d'
    'i^y70)<k5t;#)+vdX`0G&CfLv*(v_;Ouu;dHq6M2ARPznIv0UwkTiA9Y)LacVr&oZ0qzuJex9kYms~WY2k9n}PLTvgu)rWKXYhbZohj)wSzj9R#f6UW='
    'tLdCjHlG0l1_bC4@svf`<cNftnQ5H1Fh@At5)5S*5Pn#i?WVgE1zf6h;Y|>$=Wns>o;qg%<)}4&e|q(Pp)y!+9q@0sJ6CAgL-FelSy>f+q1SwJKUNy4U'
    'A^iHYsDc+EM^RDzY}p6jyyA-f8%mak5KUdu38xUMdosT}>o5iCN5&XO>F+(8Z!phc;L)XrLuDW+|IWp*vDGm8=z&rEHp_x>_q}p|x(ZlufyIo>}ub|Mk'
    'UeI-^F-(ju8$W@(Y;X+>+H(`&tF&FA9l)=)`{v`z`VCx+81kWm~>rh(>kI<_DlDWf`SP|S?(sG-@-D32y1$je0kCX446Xwy`WizwjaEUAfpxoK@?DUs&'
    'xNW4KraB`;%p{U@LR(z$clUX4}XXq5ed{5gqd$bd}G#!akE?q;%0L_avbV;ZBybw<<k~=Tfx(NG1>q6W{IW%-V`NCG5p$n}Gjqto!>muqS=Y@Fc?mM!f'
    'ND7=6l@po<8Yi^6BSDG-c3!Nb7Y@A~5~QFF^CF;XS)VG4S%$mJGMUW*)lkZ2R<iWSSw^#*cD1xVTNEIR8_I{6nJjK-vh%Ejgc5QvGZ`%;%f3jIij45Q5'
    '}TJQYbL{NWnYOZ9fdX#rw%W7$y0Guzd?*p?aZhoDxb(%imJ(kch-|Qk)t+dTVhq)7t%?33rh4ag9U@^?-jXNkrpb|)^Wd{(YLIFvF&}b4#swPRjq@uR&'
    'U#NFit6!uUZFV8D6yOV4OXq+Bz6Z^f0Z1u^_@_3HBuf`nC?n%=~7F_8IXdOSV6^`Z^dBX0UnDxS;8^?~4bkYu|e(77;>E#?oy}Nzz00xYtF!c5Bt^hzs'
    '?)^U}qrm1L5#7`1{Bkfrl0#Z$Q$wSq+}OXs(~l-Aak!y=fa^UKwJs}f_2$68y?V<1MY`m=S?PtMj?LE@p;j6|*6OeMtzObShB#Rp6Z?dP-(u}01-@uDH'
    '~x`EVV!aLBsc|i<hK9-d1fs=o0zLEwivi)Qe9GS;33Zmbh7qVFfKctb8yp<*J7v#n)RZK|5Eg+{fmWY%w=f;sXCM-@i>q;F?-L0s@-Oq^-FV0r46Cqyq'
    'okLgATRQ2AP27DYNp&<|VOuY51%m=-oAfeMlso9lBE-YzrW7;t^yO{hOVLMVDP|f1WR_xPog->Mif*L$EXB-Py;Rd_&!UiGrtHCyvKs~`OEFVobZa_mI'
    'fkT}nZ%T)rSQ)FT4}s}=1Dn|1{+Y?nPpMq&LWn$M`Y=x*P7wZ+M0o*NjlYeTTM%;I$Kd|(yCUkc&Swjn!7B$>O95R7;9Zko<YUcvvIi6A{uS;6k--Qwa'
    '1#~$+~NBhb%Uzv*v|_V!Aoj9M?8{k>Q?bpg=T&E-KwTbRkxX^*`417PDVtO>aRtlSlU8st0R&*J(U-A#(z($H+Q~-0Q59$gQpEU8k}7wx&0sA<I(6FBw'
    'h~A|WtxdPy0dq8F*`2_vuy>cIA{scHxI1g(NaTb2=zJJO4~0SSTFs<X6JF&?yr*9j6%Mvb&=4P4>!Vgi98sd>Lima&MJE5b}FZWGg%zY%c;Woec=X%yB'
    'y5}7Th8IddoPHbed)iIs<63T31%SdJWj7BV5LYj_!G&A^b<YqIggo2D@wmQahY<3uC31<U~dX{k3QNCmeXH$|?|0SHY;`XGX@e$OdqX~Xyh_g2}S`2eb'
    '9`MC5H|#H2>dxlnD0OE=*`B5Dtb5Rc+=S?x)SWe&$}YaTAk${4J4^Bei8<`BqEdI}Vsa(s(8sXUoeAqymbx>?V%ob_dVtrFmVKRJ+z}a}2cwFR!HSHvo'
    '+n{uL#*{Ysk<5&RgNo%t92v@ZHTp==V{0GYCf&C!Pk18dy#5uJ!kBS^J49_;r+SRbH?LfUaTEu(lZv9lWOxbYYZ}<JzB?MNF8khhJyvVrg8YQHH}#HvQ'
    '(UNkNJ+QlhGIJOML)K#aZ2Sg)BlYq~lz&f|i)bB3jEVE$2LeSvw+smgA()Tns<YtU1NeD$DULo6C{oTMquxAy*w(9x2hn;d86ukl_Qa;gONeJ!7=0l^0'
    '_Yh{MmJtq88E4PL#E;9!$iKDMjQS%fz2FtI_McA!hh=+h4KTqx8I+&!{1YI9!PBwlXax7Im4s&6f|arMAeN!^gEwphL9^&*|z=6$7;tJu-AbaEAUjVzt'
    'qJTZM|O{$Mk>EzZ0ch;mv&xw|AgQb*CZb`%Hj+SpT$<p#|9tSPo(p)WL|KLJLg+E~}-;CWhTc2A!2zWdV&wyB=5&W@Yg@y$Jbh<r2;8E%h^AKotd$y1z'
    'R;X|Kn{cu$z9;$k7``W&w7_Gc+D+x~x)EEp*RuVDyksR?9etD1C7p~-N|&QuGYyyNHoJM)3~v<eCp-kQv{~K!rnFfN^=_6ntC?|=Hfu6CsA=SI*Epp+h'
    'J)g2g_S^Sl8+8olYC@t4qkVBki0O8SH8S3GS;RnFN`U?(Db6;Wh-MA{hp$ot1bGS-0NWeS?^i&yI3-o7e<N6)m9I1bjk9<$Xih)`rWMB5&h0vRTZI;>V'
    'fFDj|RD7By(}n52qZg(W<Z@+z}fHA2~D&9pX1|3(e>N<ny?d6{2b|mGffF0(!f7vBscVKe1+^K!oPS8UrNkd9lVI9&T;XyRqN1=v|3!dY}V9+p)#~Z=r'
    'b+(4M0^S!dYrGcp>MMr#-in^j^3Dy?x7GmdRD%aA`0X2yC^y<R=_MSLmL5re%~uPpXGc#TSI$L%c3X5aFDrb3{2rg64ti7j$h&G__1?kdg>xl6L$8b?X'
    'CrdGYmj+4Fms`OCB>$hg_WbVn7ldUwY7|fE}mng;<%`0(ddmG6kBs8xsxfX=2$dqrv(fqze_(p)d^+otP#_f5rMk8|=$i6?%2VYb)8u}P%Col(O8Sd7;'
    ')S47?*SotU=B{<G@s4`W4T@Xs+54<RN|7;Te+x^(&^>XhYbte5BKhi;##kcx+SF<jk-S*_E0H|!Eh>>bTTd|XSTau{A<pY&$Y;qc#VE%k31qFP(#&J66'
    ';&!y_E;;b<mze}YQ%-;2g`U!S90u&#!)05Yptk^^XI%6Fb?Y5T2X2EP8I)Wd8S6B7Vg-^n@?RQb>%pgTl9$8gJ*%63(;zm4AE+HlLKEHvu~mj!?KoWbu'
    '|MiTD{ps(duq?N3<H3)r^BQJ`PzP(m8n+M5_;Tli@(XQ#H}*!9tWxV#=lrNGO+{O_qnr75L8L>8-So<)K*%GVuF{WXST+q6OAPk((9gvOLTKkxz)ysrc'
    'g-MP3Y~dc7!e!2vPL_|U8y8H*y%bIPrCq-3fMIw~5P%=$>-D}w_TyLq7zHQAJJBkHSpQ8{v@{Y_$h5;Rm&@)&+*QKQ^=&pP*O_IuW`zjy~dlOqZZQRM8'
    '^%bm4)j|vf%l-wgPU3532R-YFQC8Caw++m8=SjR=8*p5;3i9w4{C#~GLXcKHpgfQ`>oH<-wR7pb@7ftH^b<MqIBuZPVfsTtB75}=%!Y^5Ci+)H;N@LQI'
    '(%6zwApmg+sQBVZOMsnMIs%)U*0*DB6XmcKZGCrh-QoTw5d&=EA_aI_ixAj$R@8Ob)Z`52;v%E8t0RTt+WweSwclB(iS3RxQPLj#L6Wcs@0bP#Y&Wx51'
    ';10~Ya94vy~e?J;IvI|@!6&hF551z_u|wf8PC4<1AlG$DR*t$$Ryz5`<3OX&v?sEpTk&Q`uo>*75i0Tcy6-1^I>=7o6l(`*L>)sJoB;p$T1&6DZhO7zT'
    'EP6v0cSABD`1w!ei3osOy2bsVXkwtQQ9z_Ig{W;VQ<xa)V}x7BoprzjP{IE7GXAEPV>5_koSuWgT&@Bh?Udfntcc!@bC5DQ}raR6@MBvl}&EsDoG=a&Y'
    '0%b++2tE18?hlDJ*WUfWr$8$yv7Hwd!Ww(eiGeUgfcNNS2?6N4toBF3o79>!59YnXqPiAm!pOBe^5>|mUFvVs-YEt?3)0tOL~{fosT>lYp%L#zM};6rC'
    '61B4@#2c`S#l9*|03J)~$yk;Ge9smXC(9t7Z#bvMdbIV-q$EpcT7W?eMk_yTd#x+96Id1C!F^3^@T}Ez|wJv!$lFWs)EvLuf-(^H*f)YDbFO|n5yHiIB'
    '9<^B~O5E+RLiH}mk^Uk{k=4ITo~3VC=V&wM^Q-eU0OEZ)m=AHj@LGrXUc|dYT(8HxwnueX!)kqPRSuftI#7-lu443Rp2cfcI8O8yjuU>Z=lHjFu5q!p^'
    'rDTk*pKtEwNKIK9zB}C&*EZ|_`tPEJ!@-^dSUl^F88%VJ&e7d%e{-}sf8h;r(Z`n7xvnoUf3gT+H;KTH9m_ci9JR&8A=1w3bX;k3QIpdTKY1oKm^jb&b'
    's3y4LO`=U5;L$NxF81BI&G0hm<6Tz`7+L_EF_{&Sa9~=KeizT0#g%aJlZSqWF#B2m9Vjirrv_6Z4~A6}O>bQrJdyg|x>!F+bETdxnF=UkzT6@atdeN#r'
    'H?ater{<!q49%VQFGsVgs84MqrYA7CMzM^%u+3zQ_tTbf}Bve3bi2OK3_mPp7eWt%JhT`C~nkAnp5W0iyd!Q&RTNS+<==uwUxhwI3i0Z%4xTQ*UYkGGj'
    'Jxp<r7Q=iRW;o$B1ip<bBM|maqUFtL65^?K*7j&Oh#-+n$4nwsg)tO0CJ`cKlVB9xpl48_#&Vw!=@|AWmTvenG-DS}UZQ^pSj04_H+X~)|;&rybA83!l'
    'yFsjub3el$@N3*f#pN=0ipn*6tf1UrHs+|CZYm<TxOh?uMdNyZ<!jLqiVN2y>4}!Z$&e@r*WHkg%f)kv=dfARtk6;vP$8WgDx|nKUr6V$g(MC^YdM|Vd'
    'PwK61<5XctZ(8NB)3qMZI8xezvHoda*sh$iHr5zTarllV&0}ctX)sQ$}FeNu+FlA`272IB!&+2s{Pj$KSb+do}f!%5$DKp8irzP*=LJ9<H$M79EsFPkL'
    't_N$0rR%A3zS|BH+G_Yk!2ndugAwtUVz{1PT;kM6~5#MEEsrwsDs8p0%jORa~x@OeXh0Z?dEB>5RwNlN@12pb$C%R5xrsaTd>mD9X3Yb<0WUaL(Yv<eS'
    '09406rz0+STtr6MT=jY5zVV)8601d%YT^8%0iS(fS6^{b>7$}+yVe!nJO5u@-tp+B0)u22$2G@WO8lyJ_lTuVE$*hEm%DN;lU<tAU_9L;=643R@rLobg'
    '6S4f*Ni1J2ov7BR^>-B59e(jq_i1I<;*Rqf7Sq!I8Ualol#J8&t<ys*{e7m|{>Q%R$Um<Ic9~+x8MkG~K0O6F)Yura|tk<{IlW&i*@3=zUN3G(Y*$s+*'
    'ZZ3{J)QWuO6^eUiUFK^f74yvbq=;u2I>kFz*S+2*WuWW!eX!?zN_ttbi!Pp$o`TX%P0LAGqr@^OMU+_PvFz+{eW@sj7ubNQ*0a)Wx{fJ`_7EP&44AZ<&'
    '2{9m$b!P%n0Jjwn;8cqKg(gU>ZIein6x9DR3;r4GoM65y6fkYDkZ#()&GQ(v43wT2a*oB5q_6BiZ%o9755dwDNZHiNvfROdP1I5Bgm6HS2qWqIHvTg{0'
    '`06$?cH6m=+&mhprUmpUI<(56O_fwD=HPcGBWQtg=mu55diOmgUe-`wB2KLn=|reMs;>8ZpJn(R^<gOWM!RkQW2*9LLF>%7cNr2;u6;NLPh?ezHv7eC4'
    'Q{ClFdlafQB@wy;9H;E!uP%ln68Q{F#J26jh(lGOGd0@)utH2vc2u{Tl%lz>iQYowhF>qa@9Cr`?!${-;nQsWvtVG;0Uxo;YhghW8GoF`K%*WeWvEM?e'
    'dy`<D@As5(2%Kc;&QyHhsnd))AzJ{wxjn~>0B7|k1d@3=MC0e~z5^Zw1UE86;tcr@X@TeTm9tbC4GlwmoBbqHBX-Zo?nS+dkr0H$_Dvq<*kFA*yHUfu)'
    'kP$dlg^PeNCsYJhTSh`s_mE&|q4}tbQq2TH#-fxn0S~Dt^~{R6PP(YvDY|HcK1E%0d5622<bg(O5(scQjYqmdW#eJT!ZnFeY7d-HqSPLv_=%#ZFTErY!'
    '%Dd+u7!p`3gY*n2>4kJ4PHyTD99}&q<trHC+H%bafLw+BL*1=au{(ae*gk0_>PDnM)4pBq~XRw+<-Sh07!lRo_BO~5d2XWzZryn)Qa0#XIPHDu}-n9bs'
    'FTroxxNIa^TL`(E<+K_-O<jxCuQCcHqvwQ%;Avc~LQ7bEo07#2p(1)J<uw;-GI*U15?7@`{)e4qqwd6)+<~J`s~(t-eJuo`K^jiuDX^k|^fWVN6l%=Ne'
    '1yWWu$%Q})0{3BZ`Sq6A<dqoM?0uR;mHc=N@GwTL=07Dd{WLQ<qCN&p7oDoOyZ?p{0*gQ5groEHL-G`PM5*|UZ!1qagT%|Yg@p&kWUv%owG7_&NcR_}U'
    'b2BhU<?*uuHr0kha@*2oYhZKFiGwss!4Ntp*RT#}cJYe){R<H`;PeC3d5sPGyD^nuqG7t|~y_yvdSat)(WzghGR`1mpoj^aDma;Lh@QQ=uRZV1a9P~~4'
    'rJkrJ1ryFbLH<1}YE0iF5?Z|Q8s6Hz<Ir4vLNjQPI!X+^H<EoJ-E?QI@eC*dYD~n`yG_*83%UceK{;JpwdW0?6Y$JJBnl35^sWKq$h{)WF}RvT#?`AG='
    'BV-KG&9xEafTT>$R}$NDs@00N7*x@BP~V7QfFH9UPn*cJpLDIP>@g7@RY4{ZFprUm;lGEAXm;DJ$=35Zp&bF{OuV%F<YiJ!x2(kNVDyU9g*M{^?4*fgw'
    ')!|xD8TGqP%-BD)1e8&uF0b(15`}G}2;<LHY@20N&8WBU&%7o=Ch^kXKK{WiQC9$M_#XUOf>+wb0Rw@!9g~nX(@50sDY;*1<kt*yYs|J);U|i{MlQg|k'
    'J@_(0U>$n-$xVQ;>OXXIoc>a)5BJ7QR$kns4YJ9w6P3{pFhw?fCStozP511v))Vu$V3!9Hj_lZ|kral;@GT`ixW2Ep(gyMcPba0TEwu%idTa}4)W&<E^'
    '6ynQ^-IQ#IhLUfpQBIWVU3`EL<wh@Sw_x>VMo`|_Xq&!B^K%_hlvw=u?G@3zfqyauMRAAiHbVOH+!$g#_cDdelUEs2(69IK?JzQ<jw{=FLu*Lr?26&yq'
    'GeomVu8GpR1zahj^lodr4_y={pz5L$V669SU@@9DsstB<NLW3S4SuB7*4z$H#-bfWMFjab_3m=iWN2b)+I&>##z)f>OVsxUk|$?%efjdS`sK&D+m{zIW'
    'jP0%nh6~s;fNiC4zP|Rbx)d`!5JuYfM||F=m62nqIaX)aba{jTB`v2=+u?Y2@I!z={s}m$@ZP_j|=!W6UrL!Z%#hwrFcTa6!32jKK1np2ir=P-*uq|y2'
    'Hi~T|}bvsFp6$BI|ENkribAbx?hf{Wl_Zgn<q-(w*K3%GRhK@r2?G1QhSFKJvLR0|CW^n*{=j({7-#Fd>!wI#U3`N+yWN`%rs;u!0GdBz_J{aIlEO4)V'
    'anBQ?kYvkyN&;I`fo0I1{PzELw{;H2}))kz0DSPYzWiH)R_j>9d=P9L)OK{e;Domh=9s!f3ihYTZKCa^|Py3DbEf4q$vBuYE489<1ODBe!}Y5)l7eKJl'
    'qugeA*(J1zQ;vodw<G`f~c*mU=f$(oVGmNdM@e;i&uvYerbJIA+u|zsrBu^bwjh!q9v9MRt5l%dsAP2EP+*FW**y6%97&^7Rl*MY|$sS}_jNmo|nHD1v'
    '@Il7Khzos?c`<@(6=Yys+$GOsFeP|KI~b{C06j7MOyh!Qgy03&CWoEXv@Xg4J<v;wLHP7cQ4dTHijQ+H3__-t3PbGJt%>9}UkFHug)JYSCDHPX`3`Mot'
    'yE%Sg53NZ;}hie6U;*l{6>TCWJ2}1D4b1HtKdV3IAA}A){bfrK3;ml#f5>t=q6QRBMb%k!R4{#4aGs2hDjJ~tRnfrW#ajQ{NPql1b3D&xM`S#!5+K{mM'
    '~}*S#e|;$jhKOvW#sc41~e`Oa?zleUp@v+yhBDNB1+0xOwQ_VAS-?Fx;W;XuotXF!%$DVlvvG%M5aR37e>oa<Pbm&`!Tp{6X~?ia$6n##t@mM1oN&9!E'
    'IQ7o+@04F3@i^a+Tq@+DcTK#Gb(j`^-t*}YPdh$I8z66I17kh$UkNk%w6Aj$4c2q$K0JisQ5=u(0O0KGHh6AC#VU=zAGp`LgRQSLp2Kt%cXpu>oVMQXi'
    '7BDFgyk=m_Fq()p0XuJ#b8H|k#>D(9_To9!X@zn4a^EO~?9B#nFFUcLn_dqN^!U@3|Bg3D3LC_kKyoR5Ui6|2qPVZ4Bv|&}j!I8jA`o@cm*wCzF%0|M~'
    '0E=lZ9<3Zte2yr0nHa6F5=D!iJ5e;e1)?n75g(2yQ}^r}cZ$UWH6SUIn2LzG+vkn!-G;1;vUnTWEXw8`th|ub+vp`xcJCbaGoA2$eOyuCTIYM@xLmA?l'
    'e-?iy-^nMggbYXH&EJ3NvMbf3`h9`MItn!9D>ZTMR^28AdRA2f+8YCJkSjI6Pf{q^o)+*wlCtfhU6P$fbu9eAIAhmIr?O@|52_!IeSFqJBm~k<?fTg_K'
    'I@&$$Ls95QqIt1zm_c4H=lG9r;EB0|KWD3B;xw0R{TtXq8yyXF|>w5s`V^hkKNPd`WEDRJomKEc2qD>dJXBPOzNGLLDcL;JgSYPRqQw@V4efVoy*l>nK'
    'snrJVa95H#iongGzLk3vghA-g#NlTnUM(K}kMO_A8Gcnm{k@{Be}W2s5d57rt>&<DbKURZ6!U=H?#D2J#(H`{d>af=Hdxaza0Ks0YBDgc~k42O>aeez*'
    'C#GS?jXfTEEi)TudD^NT<gTnWXs0!r{6o$n&(b~;Wg73>1gkaW!*8HXFwR|)Nip>zrTIiVwItr~xSFr`Vjnk-Sq^p2uh;ltzO-mh$k5RRLvtM6oj~x~i'
    'E;}X?aZ9bzkERPFFGeE&t=dgyVWeIK!z}xm`uLf4DimT&{bX)pOvX=FYe=o8-y*K4)}u`9=;)$s>;~r4UKvpQq`flPTwiFhg7v1oGK@2)z0!?#tG&_<%'
    'cZ^2j5DphQajLu+A9qSFKMq-BXMc7WH`E{-I5FNU)nC2#h(InHu)K$$KbOo87EZ`tR&+kz!9TVQ4S`EQbpCua+NBo+B73oRK=GoN)_e6fGAZ|@%bbjQ?'
    'a%|I;MpUpr+T1I;t&7$K+VAC>>LRk&MzYW#X@+bW96(jcP&{VMb9a4OA`>z0xqVB*oI=h4>1QzySb9M77kxF^uSzI%>3ta>;Q>5$#g<sR#P2aU-G>OiN'
    'u#$2dg~f0{_)Ao)a;nrU&zt|?1-;z;S39)o*U2;#&&UY6rxXiS-o3xOO-wqqkkBw1irthjPYXy_0mlWefLI(dZ9rVN29E3D6+Bs1(@B(lS%;T{NW%Alp'
    'P#HMU$YoSd=_J)kH37@1Sb8JFxNHWN7q)wS+<G|lsrnTbzbkI>gjwqGR4v=*%Cw$Sx5*dmii6vU=V(23B35+n&W7VS%s|^#}k*kP{jEcw$D}GX=kG9(~'
    'ymVF}C+`qstLzy?4L~?uBq%SFT_HLsufnttlozwtOa)Lp`h6l#C<dc+Se?M}sF=K-yh2+bSW(Tlc*ZeBN%Vug9wpKD=3cDBaV|=_E4O@9BT*84BjH1oM'
    'BjYoNTRO<HAMx3)!?s}Z@};n$~(X;k!1ahWEdsuXTs{EWc`dtA0_KA`(4yUhP|O$Iq_ekg29pxAcY%N^wyDZL-a`zIAP2Ru>*$00YtcA+C$YrM$+)gEB'
    'o-8=m*vGJ&Rqdm|~(dfsR^>(gaqU6Qv0(whc-XSj^Q)6IeJ6qI5AidnNXcNj2%?bO_t*sOYw(f!9({c-zvgF(#zD@puEeS8av{b+4ish`{a@3p`j0vrj'
    '*O_p%Q;42H$5572uNLvsXsFL+S_zGv-sl-+QE=@5y5tDbEg8f3JHlKm@gE^&a#mEF2^{3xjh{v3pF#(|c<yZ;%0%iw#}z2Z4gs*7uaXs}xuzEVvP4bbb'
    'v0ipq5E>TLFVFp}Enm#-}3cwnNA1dq-stJ<Yol<mf@p3KFCJtXHd5pKkA|*z*iBi(6HKiWN&CTq*lr$^0muE6KLVBritE{P%lBQVP0*iwn`Un=s-dECs'
    'o6D1yCZns4U~!PrA%ewmcnq4+Ks0aCRqvC3Ev@EWOV8Rog;KPxFi)!0Gu$~!)mnpyo8sI(v9gGj3a6mh&BNTq;uOH`OY%}2#!Z)>;^2@>@>FcbN0zT*G'
    'yH6XkK%Zczhd+8Adkh0fjG%$v7%u~@>;CNzA(&5(uH77cGI1(-JlCW1%Mta|Hb0CHF+>*=Wme@V|M-#c`;_6LQ>CXYY(KJ&v>@Q(G7sYg<$VsQl*|>%n'
    'lbe5YSpjXQgSEiy%l0Q97&Gy<HKPe5_)x#L%BmCS_-3K$~RLwq$iu-uI-=G1jdVN^_^}E4&Z@xJ9QlC2L=Sl@F}yfgaiD_XmpPM!@5NCK*VYbO3thj4p'
    '^}*&z1FWkzO72xca53CV1o<>G?*BInycHcQTOgm4xyk|(6I`5riKX=%_PCd9Mj*o~|u3BprS0LK7oDM@nc!}A$4OHvy=7WNGn4nY=Gj{^7iti=vv&7Hx'
    'yr#ne$s5q&l_^iA%4!FbOi^Uz*{frT~c%}fnpdN@jWQRm4KC{Jo>D-O#8Kra2R%nflMXDX5vl4fPq_9|r*Cz$W8s#X-v$Z%oja*xsQCj8OT2Q;p3-7l>'
    '7ac;9bukbWbVqSbz-A^nz7}K5%kwpZa+Boxx>u2WU(LKFIlm^N&y&1gvm+*@>>SbSP}7M0G)dXn22D%Z*^v;Fl$|~7i;N&g0|j_kt#}beEH_EY&f0ft9'
    'e62Shy+%GCZ*RbP6b2sB7P`|_(fws%SA35Z&`k?OFf}#VqrJR@qrU!3BHOKnoA<K?l4HMqw9j-j4f+#K_cv6ACqJ39EnPlU2xJDbZUW>mr$x5UMy(U0x'
    'nz$wOa6IMX$E;#Si=zli$#akqx;=UyAQZ@~(cJ>=lQpo!oAU$%ewbQ1>Qj=ekL`v~vTMNs@N1VRlZ0;~9xs+PP-ze?szrPfF6xm5#JSSe^xJ0`r{xOxn'
    '4H`ocv(EQ8!d0QR7Z0@CvelxEk8nuKaM;M|w2+jZJ6D|ee$6j14gpq6@f**dlOrQcm@H$F=yt7VMRa|ot}p+?2KL|=6ouh`@fu#z|Vs+oM#ij;*7;(8!'
    'pIqQ{_TFsfV0<dW8!+_&r@x?N2#(h!D&GC?q>wVGSxO5mOL^zxJhYR#ANvqW?(Mqdz>RQCXntTM*9@%#?^i*At+N11h6&MKaH1IdL(*ZHmr$*!I=%pg!'
    '$)}HqIDtJ$UK(|4l@vgNJvqrsqc{VOyflh44a!TSICqS^G>S8p$V;O*dx5+(GAvt?m&O9w(Qzfo*|{U6Do#$iDOGWDii&`U&mhXBzBy$GPW8^lCb#+?_'
    'MJuAiW6&T7H}d;3GYZoG-(!0^b)h+lv|nw1JTPhzZh;@zWK!|#^jt|oYxHR{Nf%&i(KkK^f7|Y!#a8{0+vDz$mYpWj3p2=xDaaG;fn@XipEWrd}rs8DS'
    'u#0Pcr2XryY?ke>lyCZ23K#(@>x>zTHXI{N_|n0W=*?31~Wufx?rj0qGT=w1fxlkVMg}3OVf}TT($v^Mn9XLUC@(sfkJMm^4tzDAp|$r`A)nTgc8_p?J'
    '5Dx|eis<_ft2UI=0lfDG9C+<^G*3rY%RUPgcOO-XjVo;}pc;t3FVk{z!(d4$Yx%ih8<FE(#D-3^AgVd{n@Za4)Y(QJJLUnzB#(>f8+)}7@}2D#<A2rDD'
    'Wyuu8n=tQ*U@FkfSX9dE=xH#ji0+%WP)``$1=i*fGQpIYJgn_bTd+u~3YBg7d&%&Y<4oU@vH3<o&IuDGXP>ORg2@9pPeUVSZ=OjFo;?q}%C`;W&Yephe'
    'cUlMbUEOJqnJ`(nLh^^>sn`7B2=Q#dh{ZRWEyU#O{48>#)>`16GRW+T1A$LdAZO(cTP9!X;e0f5rnZtR$(y=uzD|6%73cU<n=Qei+Nm<)=DVrZp-}n^k'
    '!)vHks)HB;7Kw>41<=)5HZC$nohE>g9;%-MDLbn6_=KGwe_VLN+?6b*s^4nmmFk7h6rQoB^e@`kyx-q#9fV)9Gndb`CFjlTrBY=S>r3Qd6Pnk&G_{uVq'
    'ctNC`oKS_ML<>OtiEZUQsk#V)IU2EU~#e{ctgJE5fP(=NkQt{PI}PijFD8>5IYgW`YzTlTC~jA(NYFw!lz`yh;j^$%LWD0(_Vx89|c=Y-I)LFo`Y#CJ)'
    'WO_Z>1G1xwBnh@g0*g#*JIZ43zBhks8Ff{6m}#!4aNZg*hpFybi8ZbUo*cB7#ZS~r~%R5uD6=?)08ixG!vY)~~l`aPe0ijESJp$o+_?Z9bc%xY(#Toi4'
    'v`K;{#&KTPc;EbXQHk`G3RcH?Bi6Q|oAyE_n_v<J*(8?W&f!)3BBzBU+#O1vnkpa<hNCa$-dl06DM?(VOE|2Z>872N%=ZN%YhYWGf-M<TxkY|T@H}zj$'
    '{>{aWx)J5!kKMR|@ex%lSA`ur8|B&0N3;_%DmEaaRJU)`jVPaf^vH7QM^`M5{#_wE2tCRJ9=j6$?|y8Drbj}ynBX0<l~r!L#|qiP;-ru*K1m^4e7-}r7'
    '%l19Jwj5m`*&sUX59^-9W?VpS|$Ci08kE+r~ps^(xU=Et!-8SDC}KS0BFpve}=9{1%MLX9ti;5ZR*6;coP7M&Wiw0bP@!BvVRi^vGldc@DFN-d>+)t|C'
    'x4hf2MDwRi;12I2kI}9g7%=T?}sXjP;8NkP|g2x~-9t#SH2odl?=(vX<>)b{uC+#AmA`DDrtpNRP^AFk(!o@cHOauMZ+~j@uz>mq8j=op{FmNAcTs78A'
    'RD)=4`QFeZ|bxqPj?6ziipZNslb!Y*A$ahHcjrtkc_Q6~^tg`y6i3L(?^gJ>Le{8WFt&=EbtfFsgpVIx*QD+{OF1SUY=D7!ZX6rDTC-!7~7;j&Vi^Dbp'
    'v_3!cwhiE2;^lQ50R2^IgYz95+$ZhHq085z#t}y|%l5xN%m%jB!i<cA<w2wnBb?`6r?`dOv#Do@=Hb;5%d%e~GzHN*lu5FCrxYkC-uMo<C-c;>mw5Ben'
    'i~gW7g}e}=sP7f#h!3QUvJ|>=^ow2{<N7@udjkn}gk2-p5e87OBdmRC^P>+nsV018p_qV3IapW6P$9Zp$4Y7EcSkI~Zj&+K06li(dlWi89<i|+%R^_i+'
    '~Ar2YAg@^?nwNiayaLlxRK&I>FgU(BM>!$Mw;u>=cF=wx~t5drk!!lbMw4)m0=$E{R{z~Y-8AA#j_j@aW#v{&>ypS@-R7;bLf>D%f+?{Ru+fs7RH{Nth'
    '5eHFP;RHr7FffL|6+X7K{R0NIspUV%WdubDIip!S-DNE<_(}_+`U!k#S4yklNc!8St0zc>>u;t|N04e4Y?w4%ZQIE1a2tHaNqSONe}dQJI90ilXuefs`'
    '2wxglJZ(;n~2U?_?=U48ahtL#BwX-4G^N<PDIA#gcwq?Je{I38g;5}b0ows`I5Hcdv$)E37x8?phqnW$uoS21119ZVln+whe!!TIPdt(_-W+rjiPo#<v'
    '!F~U5i2w@Y=#-bDR^yqwPF<-UIUVuaf``5N#t*BtSshD7<M-jnh#pD9+;zmA*iUr0>N{Fj}z1Dc{8|fwrv?IWbW&9Xfd?jqKf+*aMvwVWlMFCWrLC8Sg'
    'yG}KDN)$w;i@*YC_1AFjA)(sSTus-nd5aH<>-0XB%LR0wj4UuzO2~q_Ad-=#xGV!Bi%#AaUhHzVs4jzVp#^>@Bpbm;C}q**u`w_r8R0A(38;)gFG4Ew<'
    '`7gFeI{v~`=(UBFq29dXTNb}j0@fz%Hw;u{Sy{0DqWn<DrJjfP)m&Gx+ia3clB#8ku%QHn%Nl2Ckw4)amWybl?H)0&na@m9i}tqQWMre-wb!zIbfsE4q'
    '9<pNZ?Gdgd3K>vZmWxSxM8$TN0Hsy_&Tjc(`lT3ebbmQ^NF$8Cm(#>*2D%*4^gP342A#bB7H^vNASBA)J_5DDgZwOF55;0??0!Mt-`kTsKuHK4mpCrJv'
    'lUGI$}Qm3o}ga1Gb5Cxs|1eRZjxqL{LFHP@|s^Ax<N=uth9+hQ?@psGJ2f#pIBj>RB?s&5cM^{VBQQz1qAu3yg;03Fm@K@n6J5x7(r6hU<n2@vH&+=Q}'
    '#A}DfJ@`h!NglEHHIXW8xT22_tNNjfSv*V;Xip*wm8Uri3$-q+Kq$#a#9Lq?K3_n({3a8FO(eu0RID6er%B$RB$*jzA25m(p79-R=k}ljVlu3)&tV8K='
    ')m0_WafBSH!XXt(onsz?MB&MGl+%#+3h{rJ%f=jz%9`VF4RVAtE6SO3zKb(h<p{r`I?ATV<{)1VJPk?dhwJB!lp0A%&-qD4`s(_iqv~6T$#!{c$hS!HG'
    'e{<>&)iSRtY!@f$L766W7EFV=&DWW3*n8FtPE^}s9a?PSVg5OLlqyDsf?sGQHjdzi=aGZm@Ok|%KTjdxc6KL-yVKNIm*zQ#&%q~s%B_HttlKu{Cj=Gs<'
    'vy<AN^7Nac)Foolrl7N9B`|X(y0y1n7jF6E+}O1m?tb+g82_@C2S*a3^NfOFa2FvaHcRlC14=(cDwNo_Z2xSc}<Z5aTo-YqM^VqRIx#_(}xzny84E8q*'
    '30pXRAv;$_91lr}5kg8Eb2DHhav0(V8F5la>$6qW2Y<>&>$QVy}*|HeAz*+=VGc--*U)kt`>c_QJ(Y9u_wqwq+C0`dwGDs80Us03x00;1BBIk2o*H>+v'
    's<6P_HQntozTnAPoyCU^RI`yMuSdOk!PS%b5$|kx*=CfP~%y;<`Qw8k+84<7pU!pK*`@;g=OmVM1(5%R?2AUPwJwUUvLR;RCM6pRN6!5l_UMLVrnz93H'
    '^?M_ZL~VA;`hA>{`ckae`Yvz-oNEN$hhe!8Wxx7Tpa$#pg&2GNDjz;TG6?=t*fx<A3b{Os(J|TzUm>7E%3)9#v~n(Nopz!sT`_J$S|hnWh>*!DxJj27E'
    'KKleBLK*O349*KZ92zuJ9Ffwej<hlmj-YnoZ7)FbD~@U?U6Q77+s_n6Ef8qb0#sgIq@Eu+?BeBqEedRqO5ciYiT<_4GE4{E(DZhq`HwHBeqZM<q$!Hy%'
    'eJ%_OgpR7kBAIzllI^)n&$3V}g<-dvH2IoDg!w5hCLjDxi5=FBSlzSRw>m@;^wvB}K3R5Uptgz(JCp5tAe=;<Ss1m^GK^+yv`@+$s$q;^dyr5<oau-x0'
    'wo&?l{(U+BdW?lnoGuME7RqKW(!b?xMsSVX94!HTxx*C^78(N2^V)<J}o@T;0mSOb#hXjp4G6UakUHH)IDxZ{$KM(eBB6L3R1Jv0ZRS()NE1MCPH){8~'
    '3P#z6K2+S5lt`K<!K3A&+5i7E831wa*vM3eS#6_nFNf`u)u02T_76lqBVV0Qh>MZJWyy`eJvWqE(>1mE)UB5;;IO~cdCYnPGF?u&F5E3;3JJM7O>WC~c'
    't)CbT&Cyk}u})6p)yWP{*($}2Zy<mpBDp299>7$Hf5Zbq4w{JsNOI3i4#HM=0F0%4GZTbJo|!A2dqr)Fuf&U(h`K@h3aFz8#M3!Ce=!n~gHeu0(MVQOy'
    '&DAt)jNn~c+LY(0EyIC$tx!k6(Ogb2o#unaw1HWTyi2Xb@IrG4*CYkfH48cfciOI$OFI_5P~nyFl@_#kY6Bp@CpO%Bpbm)&pOt1Ju4v)H+mvA@sTB9)1'
    'W1Vr~>r|Kq<x$JSpB)4Dg{kMw3z<jS2IGz`|JBGf}_emk10>y#ht$dV?db9B=U160(L-$RrGcS5(^;y5)*P(rh{6kQuz6A%=qwSRZ*1^sYHo_h%uV2U_'
    'sn)UdKQSFy01^$|cK2uwC|+5<w%_zL<kpEM-ra>j>Brx}uwY8`wJmg2cLQAshER!KQgy*BMmXwNv3&Yr+?QlFo@PAVy9M3P1d07(bclFWO<0?4JP8RLy'
    '{1ouj637}lI7DJ9i)i5kkx(5KJq)inLi2QyHy0mqv@*FBgq9|vUc}3(eG7tU@C|QC)=<E!T769S^rU2{?Py){Uxbgw=4v+^>v=s%vc*GjH#XERM3i};g'
    'v!2~SCBpsb4ln@&h@g?xpB-?q$_|Zb>}K5aDXWe+WV=xwXBaQgx!VUgvSP#srHgL_I@iw0K7vGb20L<<qcNY>7-HB1QVzo&a}o{sr26@S;)+IuTbm))6'
    'VL#-Da@;&O&x3)F)aXsB=DR>-@*wqnLwY^uL`Hp6pBXx{3o5zlUT8nQ+Ku#ms4i!;4IgiBTl8~=yLKVvQ5Q-8GJ!?65D~~f9JCA<;0eeN)G1|Y2<e5M8'
    'e8R1}DypJY!pZCvBYF9hepKzeYO%Yv2*s0ba0%17P11f6nL=7Jlw^e8+4^5Ai&dajhbKx1Fb|!G9j%Y#Evb<Q#ppOkn2>NpWj@3o%yN3S+^?b*s2ors%'
    'C!*~&fx?nv4@h^kjvTP|8#LRtKN;xIzn!0%`Y$XboG%I`>@py41j5}U7f#?_`0D%t9zyx4$Yk@w4Nrg)Zajlm;4issmGlw52NX%E^Jh2KblYRMA9jEu('
    '?BC_cM|LE~VaEqj<c3_jS4kZc756iKm;_nGU)UQMxYMEgaCE7yffh*y=e7>^77iDxZ4vUOVmTYBovZhK&4REzn_>|*&6+M-j>x>iZQ?XMi*=iPGnxy)S'
    'L=B<kJ^>3O{JA~iD_CDR+f&A&p3Omk4iJC@<S-{VWS`Oh_&pRm&L$vf2%U!_5E4&2`n{g>W4{N7;Pew?3%>_)h~oF)VB{E13L?J;2QkEJkJ;_Po`QvQR'
    '<3jx79MU7!QPksBvAOFY}e=G1iHZwB{&4yViZB#+N<(;%csyT3YqN*r^8s@dvb2ab58LhIK+*87#F*uN_GlHIhM>7Aw)~k3Wh6NVxYzsBaG}ZJ{}54vS'
    'bwu?UYUmL7|y;#wp*+9)k>yIffb@UCF#K_$mU=rs8=H6yl&zLbUVD0#!S+z#gHH+ED|&M>Y7xS^(A97pfym53ZLL-(D034b+;|req@=%&?jonW1?grBw'
    '(Jk5xy+1|EaFLrAiTC`o*z1?Xp2Vd43aQ8YD6Cn^ig4^L3Q{Aj0JbX_Y$+A{|z_6rvjrxW}A3dau=6`W3pjZWPH>>3{lsP0=gYNYCqllb9ul5t^HFw~j'
    'W?~Kuw<H(9=C)e8|GETQ;FJR@ekqdoxs<l<$>9!=vMQ7DPvqEy5GlEf$BXb8kch;4hSo0p8%>6h4HFNc8c_@QDT=HrYPf6MVjt|bbT69&N3h`D-_jXF+'
    '+2Qx7dd9g+mCX;(ri}MK%uCz`KY^?dC%euYZCU<$98Ge6F-I!*mxxk7Z-f`+j0q<Ix*T9acOI9->LqCnGpiIN<pg0=?wzHPAoa-!21WLr>=Y-)=^?8N_'
    '&cOwVOuMUL}XOSS#kn`$w6c@VSuhXD@3Pnsz1QX5lp>-1xdcZ3KZIX1T07J^%s6rmOZ06yF|IZQ9gKir|E)MQk$JNReykWNOl6C@JL>SmQ@O^eB!PZWU'
    '$2V^+a-&k47L~NIn_?-~;(+80;2<DN1S#ra&rTFim0&WH1e66fYR}5_OCRsdk83L(SDrSUrrU00&?+MPVm@0i<Is_HGb6t@h5`1ugf5@E~DdGP2>owcw'
    'j&I<5F-U_>qX1``>y=4-MRfdB(gBem>{<!mkdMI!W`me93ItI#r=KIo#ra&S>eOU^|D#EbWIa6?eNNJ)v5PckP0E<-?32`phC1uamO(uow-bqLuwC{*j'
    'Dpygr^1FLFE%3_dtDLoR3*dVP(0;oMm?U4*SE4@cD1f&!n30^Fq?dXSJq&e&d$`3RiD^;}`G*lxqf!3p;<qPON8n{OxILkLGFMvfGd|d!pv>MYLXwd?c'
    'O$a{U5aOf=S=j1sa|6`qeLCe<+IN&#;K`u^2!NasViB<VVNMD%`(;>?AiG~S{3b@TWNl4JFrorXR{B<4HiGPC^N4b_%5h1xwJtCt2AR$h&JRJhvyFc%<'
    '5_}a2U*V&#(*I6SwhGQvY*AipON^^qL&3NR%>|FQ6iHAE~a2t1^bxtr8t=y*%>cWTt9=jlmzfxz|YjcoC@d)1ui-$qHFCwkG>c#BT5IfslkdU2_{Cl3j'
    'tx+5R`;<B@5VubtMCi64sRn>I;N*B_kCFi^b?)2<tL7Ss<+Iu<B0AnI60llze7**GNJJK`bbyt8L|ivXSDv9VF~7YBO{(i2rNSO&mnzaIthpZQqGd8#S'
    'P0U|!yeB`W?3g%TC#osckG@E!<~hO^N)B@SnB5`yI69IYLZfuE4@pg5ugHyRX2lu+LV#Ss~RFeqNF8u4G@YActvCV05l+)1Pz*XStZz^zHgweWE?5%SC'
    'Ns*;wgG!O5PL*l+O7y}0Jc=XxIx)VK!VzW#mrR&NF>Om@sj6YP6jv@^Yq7L&;#>NoI4oSMRpvb6Tn2})gHt?;2;oGVQT*rO3@(%#FWT;mV+>#F*L%6kI'
    'JrtrAOBN8ffDSK!;&$}iCdyS{U1daa#C=fsR0W(lD153oLjq~itZ~HC_k0(j$GC0;ynSP~whMOB5d99l*((s1(n(PWzuCb2(a^cfx<hb+7ZI!_ma!1cT'
    'yly>vzMMyF__-dQg1lAEl9u7x0+8;DyE#$a8yD#1If)xwC2uf<<$vAtJGbslgFKEgRo=uKssH#GXjaut96BnTXFIWlAT94E!}SNq7q!Ctq4+gWTz33!X'
    'wAsbjQG>VJ3fr*V-`Vr|u|!gY#5K&$v!WQ8fiiJAnkj(%C2X4wj~x_8%%{f~B($YXwWkeW#sK_(6u5OyqvR_sq#=<bBqiH%0zu&GrF#pf%%H%LT0xY#I'
    'v86z6w=nS$1p9jz%-f}%A&_3}e2hC7xcTETJ`3elQyeR4(1&X*%+w2a|0$Q>=)Hdct%WE#tk*3^^?Yk+j-9mgr9hJ-U0&qyb|PhBac6TUftcpkGH1oRj'
    '<oQNLl?WGlpR`g3Pv|+ccw?WAjthvECD_C~})@VR`6NwNF=x?0zDG<4`q<`+|h>QL#QMtp^9YxOJHW9TF$PfWxE8+GL5Vuk?uM2cBO22@}6`gYhp(|!{'
    '(wG5J0|o@Iov0=oK#M6y_*y<6V#o2$cm5^(^UJO%3<O3QAN{;df(1$-wq4p2V{uD5YuF0{;iK8aknmB{OgV%BjTlH5X--|*#2S5eW@9;FHE003XbCBs8'
    'aNq28pTq5s>&}cn68RJv{6$^#q61sR&gm5)S4JXGd$8yXH3gM3hL$I!JP)EpSF>X%Zx;zaLVf8CZG~&EXX~v5V9WRpIB`dmV=_<wdRG81)P#YiJN`A;J'
    'aw6l$;mya2^mjM%w}__r>h#m;YiWE+fc+F&A^C@?Zp<F5tr0;rgUs>}X+v6pYPeTN=g|y(txR5zTIjH2Nf$lCkdAL>ir$LTbk9qeqIyibGY9s?jm!L2A'
    'W9cx;ejvDjQJPo_*xtsqyX?DJN>Oxd#_*F~f825G#)QQZZ=Se%kcTCaI!nt^|S`DAIn7SZR;DB6vcpHQ{i%1hAIVd()4*iI0E4(zBLqiDg7;+}Q-;*sh'
    'kRbwxp3p<i|`@!Z+Hr7WI5w6r1l)6EISlyzPBC)zxy)<gAdsGz+Jcggqyh1;uRBL83q*XIIP>@=!ZknJKY<zzy)|Qke(xDZM_gIi>t!!d+7#l;>LMs?&'
    '^B}!c#ycfQF_o)Z(+ajSfqaFdvB9NsUEtdc14>uGLW9(<3ya=YO4f&(I7s!n0O%PBvT|j3J=Cwmer70On==Zag53-aO4wpw4b-rk%}uzf3@<BE96I}ql'
    '(8F=AP~ib%TXYXse9zsmNHIeknXAx9~h*(+Cq-BSIzm#GA>zkUiz!*6eLn$HR~g!!D`kzNQG4`o|2t5vqs2I>*19as}gPy<YG|zcK=A+;UYiM$w0yp#N'
    '}pC&2f>fcApz^>}}cA28e=cO(eY8RYbnM#UHmwctIT-<ZVza6U*Ho55Gqw-0W9IbY`E(=O8<$g}e?Dk>5d@-n9K(B;#-G6UfAn0J%grtcal!|HYD(Q*<'
    '+PS~Uyi45~uFnpF;Rk28-{Me;zgnHK}uK^g-D30OAyRTtt7gB+_fE{j2))tSiEP?-g5Gl&JVHbZu~V6BOq2Z6l}HCxYtW^8aiX+Xr2@7*m=3yYfAwGZT'
    'cS+=`*F`2fz&nJxAmSQn)TlgX1#0GntxVK}Fk}5z%9dp8AMbJrhM-_Dv_C?_!Rnu~<J(Z2*BPSFuDE4F}b>x!<2;k;pK_j&8rGQ45&F~V1npwBSxawxq'
    'aWVI?SBOOg0xBq?X@E))6w)-3*>2@nIA1mZ-oXQEjYea`$&A);P!6)AHKOc7F=o1Xk${#DH536#J-MbByy5(JL@D*o?j}yDIU696O5NHzVwGA&aY3}w<'
    'X?~{B$zM2Hw}*j^9<NuC<x}oJ=3}&syPsWLeNVf2F2<IfoD_<nnCCp)mdKzpRv^%1#mNuA>d|~in{3sUS7wgnJM7&akvXaBNvN10yi_CCLE27$6*kXMt'
    '1hR0zO~zMLUuEg4|~3vF^bnFe~gU9oAMx%YSxQO(+lA4j@^O4{fiWe9W!(VH3GdUP>~#0y+`oOWUxt2E1vTedbV8`^zB|7U!g<HZLxFM8%pGWRR#9z+{'
    'nFgUTI&ZFuYj7w=q?GCuTbxs$AQyIQ)~4cQ<1UV2K-P9$6)YHKg%A}yasM|ds!6UqP(4F|sisY9(PgBym|p!ls`woIJpsHiQ{vdO5VW#ABlVicOWk}7H'
    '0>aHqj8PKAj@P%5OU#~^f6e6R7gT}BF6v0sOuMY}gsFphw!*H_stpPd1MTe@>^Z+bU4M<wX5*sSnJ4~(x+sa`!94J3l@eDv9i<OIchv*8`nl(He{E}9&'
    '?HHHg2>In4EDTD$LBt5(;owB{!h06A6V|h4LpaZsyE{w0@SR;!mgQxSf(xfOb{J!U|CZGr=3DkEAt7=<ZNv)^0F8Jd7rET9$No(xiGCBRmzp6Dx$f=&H'
    '0||DNk~lG7Hy%}PhWKjcB&w;u4IZ~`>IQ%15yrw&c&=4R29or1Suk}B0v#x81Fl#6(LTOYY|R$6@kVc0MXIKaTZ1J@I01+7)NE{(YQjoDr6(Y1fkwS?F'
    'D(8?XTRfU;Fjz>YAOHkE^)4hGI+F8q`|S)u7apy<N`TBRW8;F%kMLJB+)L4SxB9OI}&+v@;jH;$l=!@+(eil3uaZDgF+-gTn8khwubZAoxxTy>fDDxH&'
    'fEe$Sby;f`VBBq;7K*nvOoFkVI7AzasZ29?!##*|gubMy+lbJQ*lxA^_H9Uv<H%a4H?u1*3Ot_o#`smj2m<3QFeEq<A{_=2)y=4I4s{MJ0uGG{fK9*(W'
    '7XDqC&SlSRWVQC%|ALy@4CnOcof&Mk_z&T<A#T(RgB6C!u*{o&_W}_C)U{Z0vghYw=-NUehooYVgWySOoC?VuAD&}1hL?Z7OXgT8`$a1EkN+Yb1sKUcn'
    'qg4tNzI&WS0hNm~5xXX0N9dYJ3XyBjgmwnx?YK4lS~o-lQEL)hMec*cI>x5aplpbC0<yv2!<#A+0#Ip+VTYwTeY*l3juW_{nc%B#*-`2{+EI1ej_M`WQ'
    'Da7^qrPHCI<ChY=(rvuS^4Uu;~aJW46bZI?5VAABCs?(MD;Qu>rr*bVhgQ7%efInzgm~qmg-6Qs4fXdkKc+Hwon=%5^-aL38O*FSr84PfkJ4|{v{^cr_'
    'iy-j)-irxCCVDup}NEZ;f^?XQHvSa0|k`&KJ~7&`2Doi5|g4*X6RIq5=rzyWIB71<=WnqMY%LHgVA<It9Hk5h=Ln@+|0wuAlb=2Isnil!p_sE0F1qgCT'
    '5Si*PVAYZlUm@lsxONVURzH6M8s3#C(sd`lTXX|n>#qV7`l>7)GVnvdp?)pffN8NFsse!n`M6kJYT*oN1yUMZ*F%&+qK)zlMHrIl7e<7q2mN?g=>ukzp'
    'E1fk?Nz=A9D4H=v|8eo2v{+AY%_eX(N=zeKTY6Bi4czZpD%0Q2)lM+sqrUiE}<!HgDS_xYCHf3jteyr|D?TTSt1OltNh)F-uKPvWg>gK%|6~pS#vR+Rz'
    '+s}Pi;hGk)Ik_fb2l3n3U;*{ynuHp}(V#pnYK6mErRq5jN|}QceBd6%Iwi{$>olLRfZ<>jsk$r!ticncY!<U|R>F}ta}W-e62*2E`&ArXi9txyZb3>UD'
    'wGX`*D#)H3D$E%3wyw1ubxnA!XD6;5%hrgQepXNr3uPU-4>Fc22((Okgc|HWE!wFx;s3Cg6o6Q%kD|n*GsgtmCk}ta3!-q9#k@mq#e<XAFvjXybD;lkt'
    'Mi-0hKCE)2d8b<StSkEkdr9MQa!91{6>REwtcD<?f}uNGlNiO5={#qY`Y97ftE4yx@%(pb)#faaBm(o@wEPKp}BETT$4g>UkP_7m!+u56<B##=SQ>Y=K'
    'JI)JvJPI*)JsL}k!o?+O=gO{6ksv5C=j>$Vtst#?INV+2=pHQ81=I&L+30Cuwayl6TCo)Sxg*s8SBxLql!G!Cd8%Z%!Vculjsz9{C6)ubFJKs708(}0f'
    'jEkyNtGEn4dB06-FYO*v@N_$WYHKK~=0j~5#etPfd4$O#t8$O^(S)~5-V6m&85F=zNA}lqZliW`{la76aV%q?rN>4OsX|oBWl+t^~#ue$~<VTcB3UUv9'
    'QpF9NNd_r32n?$3_T9;`H=(jdp-XUH0BBm<TxU3p(ZbK!Bp@goz%EwzBpd*i^4vyf2}j#_W#M4(MCbaJtiu6Dr&?hA^qBk{99CLRqA9%?q_l5BsJuy0b'
    'X!T22xL*lBrK}RiiEF?(t_X;JPaM8U+Y!3Z7@xx1Ho{hWFUAmDHRCF3rYk+oT}1*Ab=?e2r4zD009wK2|%J_k9nJZZToa2-FD_Nz#Z1FicgIc!$!d)&9'
    '>w#5;+Yj<PBp;3MUOU`I|Doo_z4I0H(*rW{Jt!r1O=8O^lg<h55DuEXf)uNmQbrO0xt4l9DWu!CWbph`>~WCBU+jUJ1p(07zIptI&JZm92ty0gMsLhVw'
    'M%1e~*$6p#n3DX{gZov#|@Dpre)v$$0%v)HfVI9D*(dK~PVo-B90^UzM|Jf3CTb@4j1<@3Tf1LCS+@?+zaC>s<3AIb#<C#OtMp5=DUXDge(8x3UTHC?T'
    '3j$^dGW%kj=hU*ypn$OPpvFi4?hjN&qT8r+YSfiJ#$R1cqE(LPx%qG1N>M-LjuDYU6baJGP#z`t}iIBsXiwdBx=K?iSrs|#`ktE<YjXBtINEKn@mId-9'
    '4Xa#E$uE%Hb*p09DLn)Wf7BVLBoRoj?fCSI5Gp|39xCK}-M^nh6%?R8+)T9wF(^EpHKpiuyB8{_HpRHXssgz%J{f=d=;r?6_Tj_*&8Ppi;*RD!ZtiXrf'
    '>f?zy^8HB_NzF~;+p(j3|BF_m}NBD#da0@Rosnb8H#o>T*Y`5(^WjXE~>YS^(wZj*stQ*by2-t3|BE;#r?J*_I(%QReV$-U&V42>s4%5v0ugA9dfYkV!'
    'n#yD%Pvmu42E6S7Y(Gio5H;tnOmEiuo#*t5~mMyNdlPo^O-GrHcFRz7_V3aE{+Odgs`kBX^D4`Bb_ZNZ~5RtC+50zKZ24K5DF8#obdWFDSL+EQPZa&r&'
    ')|`7D*QRL>GGnH%l(tD8^%@#c5$@BZif{lnYW5A)Cc-5=(^x8JX-uV22J<MYwY_jm8+zYni}yu10QS>yG;-p%L2N8zJ1|C|5Lf7-n9YF2oA|N7nAwRP3'
    '%{QKtiw|8&m3t!BePu7ng?%uxq<81Eb?Ze&o@817$ReJsQ)qL$Ne0KBcF+1jmbba&L+qWNX-`xD^Pk(YJAGP|MdHv+w@1Fd4cYFWg{oP+{_S+Z#uHE&+'
    '&9}2E&E8XOxB6UT@E2QJ4SjU;t4a7iy7}8lf*&`!+I;cuVfIxuZhm$9=I;JR^=|(5?nhneb!&7Vz3#K`@1DGTb9;aPS6}+gi_gB&i~r%fuU<%UUB$n9C'
    'jG&mb%B~^H+KD-ukP;epFI3<`}RpKPyTp!`~LpsYT{=9_T=6BC-45{?mgXP`X6=l%eN1A@88}&%qhNo^W^y?)35IzUca2>&sX<f|8ADo%{3!myuP1va('
    'Dma_U)@D-`>5M-}UOrFYj*OJpAzF*YDr`_RZao_fP)j%g?{~I2`}^uYLHJ_4xS9wf_HX^3GpxKEHc^_uKdH{^j=m<%c)7@1MN#OX>ewwvT%McX#i8diU'
    'lv9lm?}@cQA8PyXip-B0h{KfHeX{a@|Fv){z_T@>RrC*T*)H2A;xyE*V5-F*IkOwR11n=k%;@@n(Xyf@FfXY<cI6Th5)<`MgBPRBeJpFN+4;#>;ft|Rl'
    'Kn_sTKvz;$rtRr_e^bhNJ{^;hb|F+nZ`FH(3+xqG^i$9q4*WcONuh#Fgxv#%kNB&1QpS@UY$4586{%WyH^Y6Nz?S1>rVii8RdH&_%v*zEwUrft<|GJ*-'
    'J%6^?@cI7di}|1LS-;QrzW!#({j)#6T%y89H^2G&CGkHV{71ua^X$X>cR$_DH8r>4ob`{N#P;OnyAN+4-oKs`|2MyT|L(^ptLhrtHE=h-xO?;c>kmKPF'
    'n~9|`0(cY+xM?;Zy3p&&;RxA<qx`CBYN}2-P<2;-~aOsBRl8s{g1bA7wK3deDm!7;oVzRUL$?;<@>qD-oCnF#BcuY{oUQmMGw}%-#q{6_VwEvhX3Yo-`'
    '~E~tu+ZZpTB=M*Q}aZ`}fPaZ`Lc;gxq}o$DiJRSQ}+c%gtA}5C5WyYjSSBdj0Ujhg&+ZX6fdu4}ZA(@we|jy#M}&`MUY#hqv?dR%>h87CCrWwb$I;e0}'
    '%u!`=JU<JJC}#+%PR+&?fFYc_9w{m$RB=5$`YfB5jd_Vk+Dn{VI!=$Ef4zIpy;Zd+gfy{~WoS=VdIZ~pf6Ter4m{^t4b?_S-l0a^2Z^X1DubZZ&RFPpR'
    'QFe&-Nod#{KiJPzAz5o5K2IiM~{`JGZ()}A2i74^h;eS83(yMnrvH^8s{JS&YuWhp+4q5PS0|FPs9Seg+ju$F)?X)1iSYR0PR#YKqzeTab!oyg+7*H*l'
    'T^CWt3lsyph)RvU7p;a@X|VqyD){<%2hdag?C$bs_nAMt<NVpZ=g<BG{_OAI&;A(x>@VWa{w)6NZ{yGYKK|@)<j?+2{_Jn%&;DNi>~H4JpEQqme|WeK_'
    'BrX_ukH8storKi{^k4EKRxKy&Ch=N`7h^z@b0}R?5o>{+b5GM%ys_c{haZ2D7?S>epZ-7`_+?Icef8eOd|32!;incd%te_m!Ia<lxd#Kp?LE8?GLxVef'
    '@Cz_Tm1cC%3Qu&$&b9qYv+%+&-B?!rNE3?_WKchv}mGbL;%}!|KR?e)8KtK6!cjVVym@)sNP1t0Q0k=Ed`<Zg5Jyzq@}ucf!lNk91DWT9Xy{<jK#T{`B'
    '+tHJ|@^E0eapx_h;oT2jj0liQas-@Sh|7yath`Z*l4;@zv)`y5=q{r`4uezH1x10#3y%dcMi^1IK!`0lgkUwjJ-HyN!j*PUn2UrY+~(ak*BnZ=ta!L8|'
    '=Q#1$e!^?-oJFVUL|F)f#S7mXMw{PFRd$?sC{NumfJpA#eyG1|0efHUlXJ7E9JJwYCIE0U5e{qa?fPI?Me>s**!Jj@lANu0U=g*)0>dUV`d-3HrUw`+-'
    'H(!4D-+%R$*w2SI>;BJfU;TLh>6*2NIdF5a{NcwpZ|^@{zy8<%ar48&!%v?+efs;~|NfKTr%&F!|NiOQ`={&PfBGLcu6h5@@BVVlwTqq}@#(7YbWZEtC'
    'qF#=_~v}>)ys$Lv7bJ?f1`(9y?i<+_s9FEXTA5Ye)kufUt4`OALGZf%E$MAeEV?whmYT`p8cGU`ttgyS>w~&d7A83-p{sPy?aeV$0sXe`_so0wc1$y{&'
    'dfST38djhV?J>`~Iif_fOaTmy>R<re_O(so|eJ{j+9&{<Ap@pT2xQ_rp7SIk(U}^j`kiJAV7dEBgMEyTv2i&Oc9g#izF)9)6fa;^B{LkSF^!*T%=cefR'
    '2*Pqn`O$IVk-^7{V%!(97+`8)o6+QQShMxMX?VKUkOX)U$+EtBs^A0Of4Fn;#*YVGGgd;0H2djE1QsGtA!)%eXX{`Y5}zxeLOKm7XHqZRqZXD>c`{_Mq'
    'Hf9%DVzgnO6t6zV&sy^Nhv8+DE{351Lk7NG-zt|2}OMmy(H^2Dot3PY2ete9%yB@U_|DUz>>#sih`s+X6r~m85o<0B7^Y4E7;?MTxvp@VakB7B1CXJk@'
    '#-vm~-aXu|>03PD5AW}OH*4+neE0Iv>R6>kQlGxOo%8r~Klb$Q?b9D7?fdSfNbq;}uYbH>+wwp3<oOrB`tScA_vH4c*H7oka)0;ZYl!#L-@TqP-2b-z{'
    'jX;S|3mNYU;X%pPyY4wPmjC!%`cvR_xU%!`TE79E~ZwN_vF8;_3LNfe)-K8-+lAno_+h-S6_Yi)g++LA3w#h{D1q*H(!16=g){$dGyTlZ+`Rb=g+?T^6'
    'Td>zI^eUwHThS+H;u9)UVHf{J?L1Js0k?|G0XOkB|IuXn%c?_`&BdzWvSTFMjjwvp;)!9tH8^82<g*|EI?qz?17*&)m&Z^T%}{-Yl-<r+4={W$!<p+~^'
    'M<zkT=dnvYBO@Q?qr^#!XW5q}jVe6NDV+f|gLxC(Hr|M*Xn?fJg|P)h>@6aWAK2mqI%X<a<kzMf<U0089>001`t003lRbYU-ZZeeh9Xm50LFKBFZXkTS'
    '$X>(s;Z82qEGh=3EFk)jcF=R1iWH&Q3V>M<sE^2dcZmm~qZ{o-j{VP`Qmsu^~2j;2lToLe0xC;ScXSH)PBh;}Qu*TS2yPbp?<-hN#ZU_#O+{bE_h<;U_'
    'I#pd=-Tuz*MP9^N%KFS{Sggjvgck)y5tqCW5;F<E6A5N~k*w;u*k!pCks;%)Fhs(41qMZ&hQffRH16XfmWIp>Zfr#qhdeQMdA1S-S;63LZxrIOX7IHzc'
    'p4cZDnl;y8kdsC=~krDNW?vi)>x61c_?7r^MgTviE;;KA|k$)<JK*^ZZ!j^*|#kG|6<t|IdRXE5=!4$v&;HjI96lL9(tWdv&Jme@AMkI8nZFu(u~9U?R'
    'LX~5bL)(wB)#|=c3nwc$0dr(t{MubQ_%-Bk(58^b}IN#@bkFlev!SK*ol-c8At%*y~s%ZUd^_RTO~LcGX;4p^{uj!FFNDQW}73*-E2@RWjNlKDM@sRmW'
    '1RKyk>1O(;!A(S=;Isd~i4QF5f)JjyvF*HTevDh7noJz8^msz-R1iX`=H#g8=F$~THqN8yn*S9wmsq&RiRk7oPmM-y^w#l<Dxl;^Ik7$6$1qcni*YA!S'
    'fJXh(uz;oS(jc7ow)m0Iq9aCPq#HB@?-OiDf+danF?J2wMry7fqJGX3=i+@u#r7@*=ROTuN(T$5Rv%0z?BJ3G*P(0!n35(yg$nZ<a&%R(K{)RNWwQJQs'
    'r&DvV7C|P}N9(&bhxT}u0y;f3S&B5=)3HNRud{qlLU9_wB02-sx1OF4pc=D5nwC6aPft$=_7&6{jE!uCgU5?97jMDNhiAAn!R7)7EtTY^0)LK4#lYwkr'
    '&p0bEhm9}27Z`h6O;f$Zg^_g#$NDTTRDWgVq=_2k*CL!;7b$<o=3)z>QcOlb+I5XXJJ_G)Tl5V%jz2M#{O_tx4J*SMQ=8AoO61IIW|2Mt0)U)mKV?LR1'
    'W+Vdfu_HDPG<P9BuneYQZcROwOns=GY)fj1z<|jR%5-);^W!$F~ej&%meTGf%_;h_<}~G(dIMH^E=v8)WjBp>$j-updRX%aT~CNF&)|xi{YFD<w|XZwB'
    'YmgQXg~{K)z^{lkB;KFY@EH`Yg`7-0=Z-9v-3KSqNR<$(rjjOU|L)!2sx>!Su-p|Jro4$u<LC$s>X{Eik#6PgFq<OVIUuBZjrl;(kU4Nx0!0~!Nwj<Ey'
    '0ImQm~7BjTK^T%j`=YyaFybm-6o)2zLyBaX^i(QIj9hY0C{bQG9vgUcrnRb*7zlm_8hU73CiFC{JuS`3RfaY7CQURd=Ft3nVs)Q*JY@FkQP9vrR!an9g'
    'gj5wMF3jKIu4Otp>_g5|wM0Q;!#qRgDwu#^u$^Rr0>*~>U0zaJP}tan%X=kA!DEwHZc45RP!!pud=%R|T;Xe`<H@ed6z&lkg_SZ;k~GDZT?@GpIWZ=D3'
    'Nss&h19{IXtQ~C@P}fKQ7+3hr)L7io-MNN!8mOJ^ONje9LB#L#{8=q)0VIgae6?boniiih(xN>R@L{ja{X3v;!L_KJ<-7}V^=p=uj%F}pqnRUM?A{Y9R'
    'V_H+QVz6FDZ6439gpj)!1JSrnaZAAlw$?sPY^-J>P8gd{fUhnU|rY3xIh3|42@&VKb@;X^NAq?||FyOzwB|5F0MMLEw$FEusDC)U6J^!|Zn7>EST_Vxb'
    '@d;#p}l_I+<Uz8(bQ+4XWX8!!KuPSiyz6I#FE(YENv5!HlMEOO&<o1{gbW`F(6Hd5~Trup#j(0FJzvV3i(g-L5aeP##A;%oN0B#5}C<(MR3hFLBe8@Wx'
    'M^`cOo)OKZ_D5WSgQRTOV`7D=5t5@OeUav>+j#Gz7z1ZUheXOU%?7ensd~ymBeU9UBG+6*E%3=+KmtoK9nlHDCP>6Z|xDyJ4HkktZ3iD#ebCcG?SPB9M'
    '!t2l%<_qBW7YIc^%mv5wr>x{cKfbG0?vld|S2RSKXFHC@JPMk;l$$I^QKvwos3Sk>cUiPIRgQgTrZy56MTw7w*FAZ;b+(B-@xzVS@}DT(;0>PrO?vj*y'
    '$kb%_Wr^=i}rEmlvMBE&dAy2KfPhF3_i`hN>+{r!NB)|x7C92lw6z6mn2#}5>AUaI_}x0$^U@#aWcCcO#U9M`DAc?{qn5-cQDVN`pc`}#rgCecPLMk8h'
    'b;cY8A%2t&p6yk8bTvE>@5_=ClYa*+IPmyKIIW`)D53Op%(KL~fRmkV{c)3p$rS(C|moKmKMKd>5M$l16M}{fskLF`hyHpx<Ag*}bGzMBB&4w|H0OH@o'
    'zg!`aPsP~m5{+E%-JE;aWS<Jo9A``ud%CX?j^<<qZf=?mdM>dYpiA9UQRjz7Cu487&}+7HIT4W)rk(ufg-?cjS=;bxAM^?uA-r*2zyyZT9eR>}{Gez3S'
    '11~&`uh4$+o<WKk<+GRzo3P=3`uY`J<)hW)WZ=>5U$L|^HojyKjbo@v-^!1aR2I~1s9rb*rj(TuXM_=K$UvSU<2~bM|1QY-O00;n=p=n(-)ONcn0ssJg'
    '1ONa)0001FVRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_VQn#GUo&H7W-wx7Ffn8?Wn?!qG-EYpH!f{qZfRy^b963hb8l{CR9$bIFcf|7uZVcaOP6390u%{NX'
    '+m4nd?}%o(rH4DZ4#z5WY|oTs{Z%+gD9T*!a4Wc>wE2k;p3*)b0`*=GvUNN=WrB<f*mSe;CINJZz|n3N1YPl%wyMOWXRg4K4S^4Vq9S1;&bKFigWf}ov'
    '^6tldI&4i;8d;4uPwNxc|5Re#EDJk}mpX!+s~@etW7!vYPa1+vpa_wwUFxUvW}{C|OPr(gC9oCiyZS&-$GGP3C>PXI;E!bu!L1+@}T}H9V~~OtuEzG)$'
    'j0q^9yAGhiyGModl5rxy*G5i>Jk_G+MuOv&6t3$t$3ff3hB4I49dW9FqB6Q#!NvN{UZaB29543!|gQTC5@UcC<Kk`h*vi*oTxsoJtj!Z@d7ZJs4750$Q'
    'MyXewQrOhv~^z}=P+^f`n=s<?!yh})apIZF!%AeL(KG|CQ?aHUmSDspZ9x{tteWq4TI<scITzO{IWY#`suNJp;Ywg_97gqn3&8?dC^2#^XSvJ=G(~YI4'
    'j;255^Hab0hKZ(Pe;6%VL6q$XpkM&O00udXVid-nKMVi_D7pn82Eg1Ge!yrHsCqv_J3wfOK*$L5>D`EU5unH$m43JzVGKz`APaH1!*?Xug~151Fo+-sP'
    '+8LMW9KJ)=7(Ao@*=XYe}uv#WeIi!KnfWRJ)exqaQKVM%V(C?Z9`bmp5^DxXm<>M0N?<4Ir3t?^ILfe;M{!v15ir?1QY-O00;n=p=n*0h4ge_2LJ%)5C'
    '8x-0001FVRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_VQn#GUo|yhIWsq8F=j9|IX7c6F)%POVlHZPZ*HwuYj5MW68$Rz`(=yT*28Zj-NLbzjc@&er4-#}7X~'
    '9Swy?6~D^ZEPi~RSVA?;Z4ChfgNk-!|1GiMHm!y)yJJ%}ui(uDPxQ?pow1u@Ta^g=FqE+j@`ekWp#cp|>9W@4LWQiO(#Ho_1w-{$D#Q4$CPmeP2L@<<v'
    'oHMp@6VHEJ#*k<Xy$jJ*jcY7ljyNbbA!r)10h_DE_)N@=)9wi%*NFx>xa9Uw`T4aHMch~d#J_d>{n2C`1T27;C*;T9VJN2Gr;r}PgwrCL#JT74LjnzA>'
    '*R^U+h1I)|nl-D!EY|CEYIcR$7^#!sV8p7`p~QM^r`EwpLrJi6nwYd}?Fz$C*RHiHtf_=Xg=ui8chv}xS_->cVQnQe;e^~dO0ty%02@%9I;q*fYHKnr1'
    'x8Z)h}N{!EcBYNZMTWN-BGLwg}f?cNBbb&wqi?5p+$AcqF#Squyb6CO=AtUJ{oJOD9~7&Y@Lq6gI}ksB+S;WBdfZtB+_ds6yP~+WeZkyy*84XYL=vy3N'
    'ZN9+e!lGhIO=P*oq&xG&E-#t1DJu&;VZ3Rub@<EwXJwudVQ!v}1a<r6NT81RrgU(ox<>>gu>WRak`Fxn=j6_%CHs5>bkWMW%8PVw4A|)zN|ov1iCZvBN'
    'JC7QZWz;+K-2ea1}m6=`;BSIU1*yW(IjgifN5);H}A?eQ%6bb4s85J|A7V~3<(rP-d8q9lYz2z}PGo}Le&>a%{56g*~6PfrK-CDa@Ajr1M|kLN`u-h!P'
    'E&Twgf%>@oxB59cl{5d8i2g6gGUPb=69Q*be_+gBVQ34FP=80h&d%-j9<q+<Yk5M8;mK;lhCs8DL78(PpOVNF#iv_K676ips^>V|ptd8++<PBz3tMlVq'
    '^kxIcIj46RW0OO%3e!NQS^msU<-k9o=N=22;^hv((YD{D=Fj~8_>9_NjP>K#I6>&rcp#W->r;u|z2)HQ3><xSB{n=32PoS35>y}cS>FZ!fU%#-V~)~+D'
    'Zzda={AicsbY<Ei{##Tr!SQ#S-lyaOBbFh?D7Nap#ly5U_F$J;jgTRDm26#im{6ZwPAz?2i^q@)B@MTxv#LR1xi+hU8Au9GxE_A&PTKW8vlkCNMjlY)c'
    '6K1u&${E*o4M`HU+2+xIXoPH%H$A-W+`gc#9cY;CUmo!1F-R0p1n$f#-pn(<%py*2OMGyo!nq)9YiGMZDr!#F<_w8+;YPT6Jl`Y$%cq&%Q9da0E2l@Pt'
    'YT1%SD^OcSL{fncKyS9B6G9T4^*6C$9xKyhK-7Pl?a(P38^Pt+6zi4C$8xvOA&g28r@3knz;?6+A#X+dFQV=f<*A_b3)Be^cPCO}bS<6<W^ceun?OvjU'
    '57YUX}XcSg9;jtubiY=Q8xfU5QCVUDr>leAy!Juffd3sn6#T>m%7AsB<1&Tdeq??0t+5+aq>4P}*e?9d17uBaNVOLRdK%<>u-lGUbqSIDY__KojR&wH0'
    'HdJ~-z&&G^H(1Z<=BS{XC*?=%<mr|GnHBBfl<8}VU61|irF%W{mi>wC>Pras#V9O2h)xeUTRq^^!%b#IAn6Jqp8r3R)2!KyYC@9WB<ow?_8XJ?Ej`5s3'
    '%BpPL+wlG^|Y&2n_gmetLJoan0~TQk^%8-G#q)JI~h&;{%AH`4rim~?~}2*N@Yy*7d+hLy(pxba36`x*lpq@@6qVbpV(T;ZO=3xA0KOv^;(*(%p^By?u'
    'Sq8pqYP3U)KbYhG{t_DVRZ;iP~Cj;%B=skSDWkk;Td=3{2GdO>RExW#Rp+@OHP`C3wfFLZzDTafj}z2{C)G7d1L93K~6*<8XAD11n4;4TN`U*Xo!rw~A'
    '1RdGELr3WGM80{aT{e9JSF=7UHI0tdqD&==+l;P)2@MK8z%#|5aoWI{i`%U<r{!yT8`M3JRij>bGHn!J$fG(%CRK%=N5KdN_WxHnafePX6|66JY;&xY4'
    'm@^b5J6M5nVYg|=7P`Y6?c=*@p;cs^@%oE!C3-c`6$B|P~y?;9+XP5tR2maFkICo1~Iqds=&-LFn^G6d}+GM^Y)$);Wn#9p*oPC`94@e)zv&;VY-@%%X'
    '`_t*mv-<a7t~c?P*Zzz1>F%~DPm~&aL!xTs#=DJ>oVJf{?R6&Zp>@n@5tOonq#V0!1|0in9?eXVnCn=sm!XhLk#BN3m*3IwhLhj_Wg2`NnIV!!Y$E-TG'
    'w&ljgMOjkpP$*iq?U)9UF~bME%TdQddtD=X6l#t+0B;K>YQuM-Nk4&T+V)T7ya>gIY#;P%2xV9_)DAFc=(-7qio~NZWaS~IhuO@$iJa9@JJgmqOcv-UN'
    '*RyBW2z1^VX@_R@E+lRG*ddgQ4dyZU+9%!hK==`X~7lK8Uty9;w1neZ(uFnxs{V^Xcp8_Ve+JhNRQy2lbAh2?tp}%BiCsz$B>WD+%hsNrJw@Z$IOn{S#'
    '120|XQR000O8m!WB0nP*g-C;|WgeFOjiKmY&$WMOn+FLQ2TaC2yHbaO9gY;<T}WoT)0Utw)AWnVQlVL3B5WHDwiH90q9GBGePF=8%lVQy(=Wpi{cYIAR'
    'HWmH{n;xG_=@2@~S@X~|WhBygQbfOSiR0@csmFVddxy}MPQQ%_dbW+v-e*Or;r@nY*XV<&V#L34^tJN-;WCCP+*lO<&a-YM#6ghjx(Eg^=b#*W)DbHf)'
    'dBA&6SEn;m;Ati#lO8*lUQr4G@8yxn@^tjHTnbro;o%<g^bqg=_TLZexQmlTXIsJVc+{znrHohOPEl8;MZC%98Sa#VpHLVt$Ap*w3XJ1ynT_Th0e|CZh'
    'wY$=cW{bFnSr~+!o7iqwSn=*!mEMFqk+U$PG=Tu<;04K?fLL*U}nXcoiKl~&_uT6$wmviZ`FVm*GmJb9h=&DNou3Sn!PO#5-On#|4^Y8lsDS`k;&`Vp<'
    'L3!YJ5>HereTMc1c+0v~0|CWbL8WjcpfQxv913C04$EsZnQD+7A<`aFjI(#qSb_-(UH|`pU-}hreF=<nhWAr_XfeaHr42sY&L}tmiAAIW;rqobwllySl'
    'UY$<Y^1|CP&~n)UL^Q|DPy=ln_P=t<kqpKARyFTMg<mGB3Oyp~+sP7tA8jJlX+I2@8-*dO%#h!TWu5gH<7?~B+o;!|CZIEuiB5YYhMXv8R?0MP(Ar-KM'
    '0L<>GBiUEX#2YpKVjM6}@A!6JwLcDGI9obpzPcl~*5x)Kra^GiBKPd1P`BA_LMhL(4ehGE??DkV#ameel`rM&t9%Il&xQqH3=?~4xZ;{&8zWMwIP)h>@'
    '6aWAK2mqI%X<b{;Dso~60089>001`t003lRbYU-ZZeeh9Xm50LFKBFZXkTS$X>(s;Z82qEVly#fV_{`wGBjjiI51>3HfAzpE^2dcZmm~qZ`-&M{VM|ZO'
    'BZWf56e$vd*Rq><GX$#QWo2r76v0RwoqB}m8isSk^jDDNIO>C<USUQ1m=*OIdeE14yo_#LF7f8rL51KhQ(?uOn6bC7jel8Au*EhJCR_-7s-7+7uzhCA~'
    'IyW5r#<kwm_$d(@+?&l*U6`#L|$N!HtcG;*clCHqY)wL0-_g+Z%=0)eOE822UeHL}kdOp5s#TINgX;8i{y-(;6$XG7km3dwwtoFi>v6Ohm-ja@@LQ*R5'
    'vYH2ao?|6eTIqD4ILq=eCT*6gx=x7~1Rtl4Azp518Hn1vyihHMDkMz_Ws*6%bMEhw>myVdAm#8r|FsfEc7jbW(UBCVDZ06^d!QtBz8S7U7m4(W8104p2'
    'rIttT<zfPO9Y*$G%M(8#f*lk6Nq>e&s!x7oKN+Mf_xZ32jr)+zQzYC7q2d%_bY(c5%9Pye>C1K2VlpZY|W1Vipf(7YW#L8(Z&LnjeKSJ&)3yd`_6-kn8'
    '#Sf8m+R7WDbQB84nzXi_(xW)pic4Fe5SPx8forK*l5FJ{-dy6+q;<GWCDAPK+%}=Oz-w7*1EAM-Nw1}M3=wkMDw4EMo#S@8T@^oYarHXdPc;@JcW&8zF'
    '8)o~l*W|eQJJe8gcuiLW_7h7BJ3G*Q0(xFgvIYlWca1zXJ0TAe?ywx+O_JR)2TU_i=dP0qxD_8Lwh_+0i7NiEJYgb>DVEuS6RL%r8te?5yF7=t*7S$sQ'
    'PS>rX^3<)6>&|eFgOfeIvWa!Q(}li??7G!!ukOU<-kRmP%Tt0)LK4#lh$lr&p0bEhm9}27VZ06O;f$u6b(M#$NDTdpU%=;$xgjk*CL!;7b$<o=3)z>Qa'
    '0k>taExoQGk#RlUM+EURn08~ej~-Rl1Q7QOk<an9)-#@O^wtfDNGSzbJ|Q#tTg=y}J&rg(WHaJ20=sRi?3Fgc@k7-NGZF-{P=G#&^R+WJ(YcW*j42d6B'
    '|<S~_K+bci=RA+q?`~|)hSUN5h*pDLHW=Sknq>*f~+#B!ol@h0`H-mHO!BdT0eq?>rf#JVcABAD`8|$Oqj4+4N;i18aAEQAr@IV8V!1GZ>YV5-`N`H-A'
    'p|Jro#?o!V`Ggigli$$-X+q<Gn%tlT))ln?o6<PYW&pJTH=sW77U(;`TcGa%?|P0Fc>WkI@O%(-fcJs=!1KY)X;%YA>tdH8S;ggs>GiS8GFkCF=1eb?4'
    'Zn$St-7>eHWKNE=U<s#I0Bk)cuEC?0>HdNW~ox9K(KL+3p$OM4hZ|03lUOPptvx9i@TQT=&%ntPt_C!i4F4%nX6zZO~%+xazO!O!~HfdDJ>{$Y{KP(Ql'
    '#LqNi5eT*90hvY*OyT<_=f*is^W=t1`v%2#vzZraY0PO|fM&A=e@&#)MB{W`nYjIv5mfw#W|ap_rqW%W}o(nLx2;*V*RaoVI}ZN%kNP{of9K{#Es9OW2'
    '1vJ)qIfF#l0RBGqZD>ib!_ek(a~CS8@D5OB-b)eY8jx;ZN7=1KVxJ9)YzKxRdIIAi*fVpo&kYUy2#{pDb4d-@8(Z8452&!N-v%~sDh^=y-Q8A`eUi0A*'
    '0<ly1Ms3xQ-PO`oOZof0R-_k>Dc<l`WZ=`(*y`E0p>d-sPZugxY4%06d3Nj#`jYebN_on07AQ;bQ%h7zi{9`&%7pY8W{(?uFq8~?86YgV?8@o-C7JVB1'
    '^)p*bx$T?g<KtuFvDwJ-m6;YM&HeP59W;xt+3T7h;+mFYl7bm#xoE8ACV94tLU}UVmU*I#qR>Q@-xTJvULM`Q3UBv%J%V?fI#lY#9yjQ&o)WY7dQs!kq'
    'M*^|I1Wdb1+bzl)<9On=~-R#<yH|2G4CIDLSfJ*Q(#|VUTk@8(tH?8LEu1m9s0t20sQ^~q3DOX;JE&jmt5$_ch$>Xa=77&hA8uF%h8xeMU$6uo#iO%6l'
    'fH6<VXE3i}t3<vCquZPU4~{@zL=5N?vZAZ6Z(na4k0cCrUT02G9N`J^St6g?U1Ie_@_Q`#5q+s`qba<oxoV-Y{4OpB7#vD@TK1;CsQ_X2Ez$OPem1q*^'
    '@^PK!7??%Aiw|A6#yGQS*5{vNEwWH6h(Jgfg5%=4%I@+x?7KE2%*<%v>bZ%9<F!g#k4lGFClt-a30J+zKFEy7B6kW^ro&5&as&7+wqQgfZi^)eE2DT+-'
    '&=kf;{{%HEg-%NvVV>3e1h)t}YaprxTh~gjg`|C5im(+@Avuk{dw^e@gOMf|>-^_vvKfBeo+TC-lg?BxkkCyY_z3ahbvYeoN`c*4^A^b<1`DFBija#+x'
    '=Qr0wZ#ka%!8o{~H1J6qF`}>?)?PKZSs-P-AM@6!+g9DKeo~*6@`IrtT;B|Xn``fd_3IzxPxu_#W<{(DNBt46gnF9QDbA;FquVdX?-`O#A0N~^ek2@Z{'
    'UoQ3dj682p06aR2PX;o3cvk=d-hLIO9KQH000080GFX@U1zx1XeR;y0DS}i06+i$0AyiwVJ~xTVQ_P3Z*+4nXl!(7Uu9@%b6;U?F=bz3GcjUgVP$19G-'
    'P2oFl071W-??hZDDR{W@U49E^2dcZe>(mZ`v>veebV8JmjT=*oJ%%DVivxZBz<~q?PD2MNaHMCQ4ZBOp~hq_xXb;p8CQ$_xSi;=fvUTW-u5M%rT>U7z~'
    'ET6F+A3Sg{PhLpuB>m)qu~mN>t#z_w{IrfpMSumD>!#xb|?rLsxI7=5qKm{;}LmU6{-RWKWlfh~vF|F{2s!skO6t*x@5zr(rJo+}=17Zzz7)gs(ys}x!'
    'lD{A0}n*~CupajA&-K6uC#pvHKweW#n>jSOBd8*(p((s_*ai?Ih*YKub`J^Dy&ihz{b}qFT>7I`-3SuqBdcx{e!?n_uXI-uJzU>vXxZ5a5^w>ntixOQ$'
    'I_$DKaoR8u_=hN^AhMD1&(*qo9U>(uY!{u<<t0@WS+}5*ld=lWkY$EaSFv@vNK<KDEpq+h5oZ<J4^@bA-kAlziwyqI^T%D!7kh)h^?do%^T_CP9~<20v'
    'ot)>%9!=i^Vsmj#yMB71~+wM@3Wz=js9Dc8=l>!=ZWzwiE;iYG4yDt=uh$csa||Tc~j9plw>V0Wb6nacLc!*x+(Mm#|a#FJVP!*{w)9j0Q$bLBSwxZ{Z'
    'oHZ&Olz~9xkDq3v1yPKJsQ5dAZ{g&x53xiv2TOI8H$we@aW5BS5f+kbC{u*%@EhvF4e$i1h0}LY7ZpUSLYmlwcna*DVnl|6=0u>Fssf6f|ou;&Z3u`WV'
    'v@fFm$T(IikOzXebn|K{@_P)h>@6aWAK2mqI%X<Z#7VKw6h006xY001`t003lRbYU-ZZeeh9Xm50LFKBFZXkTS$X>(s<Z82kCF)}t|G%zqVG&5v0FgZ3'
    'jVK!r9E^2dcZmn1QZ{kQ2{#UHtUuJa%zkmSB&WV60@hvZ5XSH)PBjn%?SmOt0w<j>8{O|WwHx7^q_mEbJ=<4pOuO3}h-TuM8@uH0LjP;q_G?~s;Nm!Pc'
    'MWF~wF0hh>4?MxjO&;yF6uVlSMSQ2dyP0^uY7cv@<U3sp*IWy;NL%GqCW^h5KWeL3#NT;VZsILq%DgI895}8Y3<4}vTj<~sX*TUv!?YTv6WC7QH1Yp8('
    '=y4NZ(&kF=m&GUtZ#LiZJjwiP~E1ZGZTc<?CQ({an-bRW`k%;CP{R#(vosLP;E(potApEgcTcZ>#QpQS~}~=Ens&*w3@cgu4=uODfJR%N$Ltz>xF7d2w'
    '+(qNfF;wt-EbXT<o?XYnyVnTN`Wjq%vr=O}Ps^oA?}}99vom?;Ln#x1~0_MzP6dOL|H^+tSmn#G`oh>c}|Cdh9xcY>~5er?%d{lJ)_`fk$?i9CzT6W0B'
    '2HN9rJ3u4Kru+cIm}mhuQXS}OcNZaH;8Tfpnd&9(#q)wU%Fg~fRh=t!f8?@Ce#MoR^V4y-3tbf0w=BVj(V$AbThQYeclLm$$Z5{&Qgi#(((R;dSS&2#*'
    'UIOXgMc(4k2OF#bESGtXz2ppNJgdYkXW1ac|HZ@wTc(&RTJ~DTm7kgro)j-66_08v(dXaNB$g(O-*z@!A!P*nld(5d$QXZC7!7stihp%vHfz3IpDHG(b'
    '1g99BV9R2iP_cRv=K?lLczNqmZk%a2F0<JKFApk8B~z>LFT+k=;bG1uVZuuVL`I(g4bVDN+qnSu5ynupoMG0Bs3UF9!=lvkLd((4l#V(BPV@{fwWc?fP'
    'SK>!HOdvX;M=?qakj2oozh2ncB1Q+=i5AimQ|Q&IHy?bwRc4=`s>B>aG_IYH@C=qoelqqwAb0_cN8K9?4eRz45)h#1IpDMvp#Cj^;sWv@?n8u)7c#c3o'
    'u?lB?p|3)ERU({(}k+Bdr5!d`|@iei{Kbp>?25$5h)oBY!R6&Bs)J7y~Lzoh@e6k#y#J)T1!MKpS`;XbwCdw%MJtgmNh-d9seHlqvt%O_i*}A`Y2ylnu'
    'Z0)kaRqVK(Af8WvxfavT9I(lDdiqX00s6nQ3@DG<ync!Jx9serKCg7Xy>6vc)4+c2h9K+$0ziZGK~6eKn*@=z!kI$KYK38|ofvEhDORFsnxHZ~5$H_1r'
    'BW8+wCs!$m~QDoz4$I}N~acidH$?mEQ?hzV=mC|1jH^r7sx!CZ6EGB%JR)eY(Di{=PHqQ_KP|PtaM72gAQL$%>JUu9<BVc}lyg$r;Kg`3ga!yCWKE&Ap'
    'jZTL7Pdq|p$h4J>@l~h0l$^Fn*PThXl)SM{4_NJ~p=MCSv(&?P;@Dnc%$m+{%2YpRcjMr0>E3z%@@8TURToF2^`evK!ttSE$%l%3popSc3F@6>=l_p{3'
    ')W)Pkg_ZnNCaB^^n;20mL4j@g*yn`k<!Hv{@{_)u$;hZ_gh^QiF&6z>n}nh&-dMlhhE{$rpwXHTmCs2%Ns}}w0|8&Y1xk>>IsiAFSK2nWM!XLfBVcfLT'
    'vkn@$~f6d~%w3u{N^OpuJx{vjb=OHGiAaHqDk{+6FNgtGwXNjYyLhxo9PhV%w@nq)@an1S+dEUigaW@lANE*Xt3yZ8yNuDEGL0c8!cId#{}Gj!uE2AEF'
    '#;l_jvEJXSzv)9#sF<NT-yg)HyaCm}KDkSVZlSYB?!!k~Tha!%ktcpLiKI0JrvjZpMg1rJ58EG@WtoStfVNYJ{8y=?NT$hRQ|qh>V1N^J51O`QUbrjGn'
    '*JmAVTWR87ihEfujWyPZlo}3?@Wgt)d)rP0xFO+WZ2G3%Lp2b%0+Bl-UzcyY)yIwga)%(jCIlK9XI}Db=$GLlwm7_s0@ZI3DSl~^_waI)*tfwVmx5-Ak'
    '_3G*VKOo(XXE%fKUxPIt52n-eTK(@}u0QdYcfnbGy1Omf6Q#yp6X;r{_Accjq~oJodsFa7aMd}@S0~v)REbkIRw2&OsKpGP8Jk3GmJt_AUZy40<xgh#q'
    'sgCtu^HhuHX<YqPvcCcvhf(>4fc2X{q4oNbJNNw-8H}CRdizC?8aXXXZO?K#J-N*>X@&Y=I+9qjh3@N+{Iu#UXIZ{{Zk=zAN)s|*?9DmOzTv}pWQEp?$'
    'Vq3ffw9U7Wl-C5K+_)uAd6r&ylh2&uMEnEPQcxFK5R=&<_^(!{C15o=Ly`z50v~iEUoSvTrn=@JMK6d4s}y{Hpl$rT!8iN<Q0(kdJF3<l~r#mlrHe<nx'
    'jUbzy({f?M_9P)h>@6aWAK2mqI%X<fBUU9%nn002`2002M$003lRbYU-ZZeeh9Xm50LFKBFZXkTS$X>(s<Z82kCF)}t|G%zqVG&5v0FgZ3jVK!r9E^T3'
    'MX=Y_}bS`RhZ*FB%U2obj6n*cnKs@B7gV@doAyPE)k<zF@5uugnG)2a7z!L-(n`&Cs|2{iThNr#coO|x|y*DTE$6cq>&9T6oif-8H9**LW(F5bz?gUBq'
    'i%MUsW1SM>!b97pq))Ha`GO_bigAI3jW5j38RzuGN-Ws9w3W=bV1(PS4{SBW{<r<tBQEz*ytImn{)|S}wPYe%O|0FCMdL-1&0+^woc@Xy7T(inD+L%>g'
    'bDVWmeF#wvG(*FH874f%xZXC*D%>?c&lOhR70#Q9}*3^a;n96u8+UekZ3W{6XvfPnn;%{bhOmxRt;!z{aiz;$EJE-oa!jn<`(QIXvL+$A2L*e^g`J^s`'
    'JWz>539olU9-bD515em8DZkUbM0FOvT!^R&3l;329)f%8jJ*tqx@PHzD~rHu&tuAJ;cN*&6)q#-~p=9vgig5`!CkrbbOXH)g%ucw*Ef#-8(6gPXc>c46'
    'pCqyNg}M$P*9##7^6Qe*!(HT1Y!)9>>6S%2P&ii**9nrBx*l<5c{zXw4N`WqO8Xb>X5KL7yu?mYk@&>yqQ4;XpAs?S~58KMC|0}l{4hXD-oqM$ps06u|'
    '5000Vxjz?gQi^8EfMFbU|Hz<Gyk+Z{1zY<^g;Vd%wyV3vec?OYBTmU@R$z3q?1MCyh|H02?bNqa*2+gh+d5LM>yoTEYum_wCa>9D^djRG7cb|U%P)h>@'
    '6aWAK2mqI%X<fS)b$;Fk006%a001`t003lRbYU-ZZeeh9Xm50LFKBFZXkTS$X>(s<Z82kCG-WhoV`5}5GB-9kGBsmlH8)~7E^2dcZmm~oZ`(!?{VT-wO'
    'NvtBDcKU-6tYCyY;-}Si>65+(BfJm!b9Z|O9}kn`({_v;aEL@B7t`}JM-q)nc3Ah_J!v~l%=f8%%;v%wupnGz*iWQLBUIK;^2YD;9O_nR!Oj{L}|!3%A'
    '4zv`=j!(RSLdQm0-!0APtp8R;6XWRkBBA5tY$bo))WU4VWUU@&yNu<M}-wjA{)VJS5AyX*YDEp<BLbb#)#8f6@(|-1!p36_ma)>ymX1tJzVR)dAINwpF'
    'Hsu$#8Z48Z|m3XTaO5C%z`wqWU?T0#n&#JX%=s;ngla0tBJY^jW}JH%-VVE|l+Ef5CGnL4Q%uxFYiW!SZ%(GnExwgm-?#zpN{i(I`BCP~*2a+hM)0M*('
    '1uM6D^P|^l?mMQGQhJ~n1OV~AQYbH@**R%vhF<*)(Xt%T1)oW{(%ob_11s+lG)D%8jh{&{s93nC=nii<`URP!Vl&#kZvJH_%;+Y~!cyC*R2k&j*bp)#='
    '+@lo8Qv$DL)q&e>p$b>*i@mBm6&$qP5xk3Ml|?9+59~4Le^Lph5mo3z5>bKi4gR7IsfuOkfm-qmf6FN0>=R_bg}$L4{+So5iJcG<sqBPr5*}+^`aU)lV'
    'pcp|YzZHgyUg+}35jkXqQ|=W^S)oiH|wQo6~yfM`FZE<5$Zj@sZSCf6jjd8!A|=xaEZaD9Nm<b6s`cL92{YbW*t$px&r46Hd=Uo>ric+SU9lR<bfAE6N'
    'S{NO?c;FZ(iWxo89mvh<PEQ$n7Jj9)^b;JZAvk!yW3E6V#fCPExi!$O|RQl?($-+Nf*b$WH&<aC+6~7-Q<xqTsL@UuSt4rOSHQNqbajN49Qxw$5T$Sp>'
    '0ylZwi%@+PlUe>{644s5FI`W6+hvi=|FL6r^Ou`c>>fIakug9VM^VnJ^>L)JwrIUehxw{B)=H<jJZsM#Rn(;NiOd#nH&{zg+2EAas}yr%|)Jgop55g%y'
    'dAvK)JC|(12(;>|ftUgUImCYtJomA$zG(oVwqp1P&^bI`Ep?PvrQVvxmkC#!EFc}}auHt2oM*)+8vi?`TScy*xm<@QE1o>ws14lyhBuJ_IC;`kV$}APa'
    'lnCbLJjVUQWJ1_&&iR5`it@s|br8`opzN@lJV?bBC5iR(EGQ)movm92F`1x*vHo_QS5%XfHZ}~(FG7%#$A(e4ssiZ%Wswc54No3$$t{`8C%dasghyzUR'
    'yGRalB6lOY|P6Q&&gxL7iHC}ic%(nvdyO1E*{D`zVfnKVv@+*vssqxjMEV?FGk((zTfS>gU{ldj)dJr=?;xfhIvmsL}!Sy6{GRRtUFhnrcM_g-KDy?vy'
    'Kl~?#ZPVkjt~M!#Cw&z`~g&o#B|tiO%kZ{@vWUbG`ZX$mq*Cj#2A|M-PU>Q^gQZ74bwV^J-Dj^dvw3I})xxgV8`r)2u`x5c9(~R&LkyRO!#0p6?8#Exw'
    'Bf&y<E?`9`a2U!qCmOXXSJD-B%Fb4D&^g*zF~2NQSx`)DZcpfaZYs~}8@ZWPi?c#L?iY?3%Fy2SnYBU_c_x~pkVPfyJ!tC{6XEiE+K`{5(oNfw{8*CkC'
    '|w>90=NI+X;Id87YB;M<Vi}IkhuJTwYg$u1jXBFCBE)O4Hg*Q5#4#Asd10s!Li|c38NXfIeGAQ>D6ePL<+MzaC04vNQ38Xj8j((|~9u=XG=iT}w1O^>4'
    'CH581i*=A|w2xWN2^<KoLtkkpz;CaRitZxkL77QUb1t8!$5tL<jBXMyn!L)hb$~^y1+}0mS6PmsPKm})M}0INaOrBI#y&Dl8i|Uc;^CP{PLIyeP$%AEg'
    '?sV`DmO%fhp|NuW21AW9njuhX)m%}a}GuI_IyQ7u77p<{@i~*b&jfX(DQqq<DXaa-4TU0n$Ah|xFyUM`DofN9`FAG>2^4|?hXGRtm&{f9-sE=KZ7~m$e'
    'Z8!C;jPc))-Gz8e65r)GCxW2`>XWKDxD6Ie&yyUDJGVR2@VWIAv`S;2gDDP2;Jyip$kJ<mH?fNkM)2og3a@^!xwpMzD^w5Jkh2C>5owJw|wg{X&0#-g|'
    'fIS`j9j=2yImj{KWkd-ML}e(WFl*Ea38{!(b_%-qRfKKad=^@hXw5Yy8;R+9I@-|9?;gYR_gV;ygDKkGYlckKDDe@|85ku*|7S=)tvtZ+X?#X8@Yt=TZ'
    '}oAdH~b?g*9e|F#Z?`O`5_Upe`pYc;-ofVOo8;vJC5*lgNpfn$TSA6(X{}v!hJlu&8&ub#Y<Cuv33zj~_^O6X8VSo69TlFtcO9KQH000080GFX@U5@v{'
    'IUWK408<1206+i$0AyiwVJ~xTVQ_P3Z*+4nXl!(7Uu9@%b6;X@F=JmeWi(`CVq`EfH#RvkHDhEoH)1$0ZDDR{W@U49E^2dcZe>$lZ<{a_eebV`c;KZ=u'
    '=z|Pp(!7vMNQHuNh_t(gbbL(Qvw+_)wHVreg1&O(;ncQd+xRG^#Om}^?F8*7^0XN<DPLkvr~dkgeK_)X2uthZtAm4F?OY6%fi_JH+6MI94x^oL(D=~V&'
    '#NVd?6)bq$({T6Us<NEjR#{Xkz`l{ns-pk72YniyHq7CuUO;7H(!{dLiL-6(<jo15Ap4g$olMad_YY9GV$n<TozE^<-}zaTT^OjTAgcc-%^u9TdDtm_J'
    'E~ROKR8pepA|jFxKoi-cH-vFfmVRnSJNWTm3DTDNIIiQ8uhJJoil`b9exMatZQoEfeuH~52xLf~EqyC*U)?B}isVKeI#?vD^Ui#nM*CFDgHbI(MqZ0p3'
    '<J(1uBb}FBUE8k?m!@mj6r;)}VZv1h3<FkXt-)?;VbmNh>&mz{iw$EIviI!Tgmm80@npm52`KobU*Ve8yeXZ@k(YaQ$eZKLXc9xwsf3(x|$dL3qe}Br)'
    'TVqxe{EqXaVVN-92Daaapbz~$9FFZ^Z2JR$<OI;Z2Ve};$28~(*`6=z9Xw3EtZ+u7bYvqt&3)*)1w?)datC-Aji4Vm7#6OFos<`(FyK$&=YHmSX?ulfi'
    'LUfiv4sC_`2TyJxY*6JGzWIzk8Cf?y&xS9e$c9Hmsd@l;iS3pON{gO7;YcHK5+K7GnSLz1IWj}`}_k?O9KQH000080GFX@T}4VurrHJo0KE?Y05<>t0A'
    'yiwVJ~xTVQ_P3Z*+4nXl!(7Uu9@%b6;X@F=Jn5Wiw_oHeoR}V=!YlW@KhHGdD3VYIARHtylYR+DH)oSFG(XJ)OZ%2%ua~1l)wHc?dhF>$Mef@CK~<!TH'
    'fZtNicx%`Og*hI>d=B6fFn=9|aP%&xz&FFY@zEM+}rHg%@5WgHX*R$)*E1usE~gF7CBa+QTUCBd!|r6J!cudl|VpOm|uQt+*+1S_rtX{ao-DlPM!lHDu'
    'IsEoeyv{*+Qz!X`PFF9}=&+q#nR2%5vA!*i4yP+Em-SSPVr|bCtSvPbt=SvV*5c<Zf4(l1Mrl~Tki&?wbRhf>7O(a85Ffj$C113zYW*Zb+$k~{+1edNd'
    'B6XXV%0Mx7qI7D=ZoSnOVus2t#MT9r=wc7xh9TsDZV=Rkn6+vMqx~cyWdYJ6*$!FJB3VNiEj@xRqbtmYtSR}(ZsN1ZHp{FfEO=$wgkqS24?0Ym8N_E=V'
    'g~!nj<Aa2A^jl~UC?M|2wCVg+X9(lb0M_BQVSlLB9<0Krz>V{5l!$}7s685XX~{N+YmY+Yn!#_wk3Aqkqx{qiP{t(ok%TtN)&&qJ{-F(KnU)l4zt~<kJ'
    'XlLTs*2QLc)At_c{NIQYeimL+_G^5{z&0i#(((mZ^u?if8yOqlB|h-~ko#hI;rjFH{pdAvjW53Ew0<#ya(VY$`;oc)HvXJ~DTe<vU^$)j*Rz>*<eA^&-'
    '}+pQcq1v&YBBy|qWEw^&o1Bs?gpoS%c84W8i=fz3FoDJ{ue0ZuVE!WP9kreYlloHN)c;l;H>xpAW5KxWfBUhGvAQl?hnorm?G;rSH9tgHh;Mjrw7(K=+'
    '?IRp0|#!$7KVAe#`k+S1KUMN|vWN2qnN1XvjdIslO(~C+c{pVf?3Y+symX}ews#~4ZM|pOn>zZeqEQXe45Gy#RsN5;9^IG($v*+PJr^>Fbk?$%S{D}%w'
    '+3+3fp;(64L!~$vS}Gf1Kp8qN>!IEpkM&S7Z|5jBmEFv#)WG9YM*z-yi~w?fr#i++bU?YcRDR&65nyAY18w3`hfo>$YXEQNQoqCKQ@>Eze2Nix-iW#e#'
    '&^`0pq|#i^Bn3SCkf?HPV#saRSA>+v8yUx1$h)O=_ng~<;%5LlEZAs(<I10GwC=2nkPX@wMPM9PEls5V5UH@QO;xBK1>FLUFV!Hsh}t>%-aMJwE~I`dz'
    '%NT*rFh@L7oMrgrTz?Aw@|A1&j@Lo4lf&q_8nJD8C3s3LbN#a$N<|0E!}Wt1VCNaK){dj3>LPQn*KG6jnA4;*z*2wrs-7HP6Xn!WU`PuZmIzgQCr5+1?'
    '+DIaYaDtx!HP_H3Rdd*yTl%!`rt`}Mp1I`}NsbR_I;l<v{!WSIBBLsW)HThSPwb-HuOY3g*{(M_X^8|(Oh<(?dB201(mJ$zdp+AEA%(HTyd?C0#p^=}r'
    '=&B$9^jg5iq;%Ky^@aVa4c&HfSp&}kAWnL{y>YZfg|3{)Lk5NNPQ*@_N5I=ll<!(a{mBHNU`_546;(LGaNNE_BZ?t-L2Sp;^DUa%t&~W5=&Ul1gF`7;m'
    '!|7=8$JiA&P#M$yWe_GsFAAwA+($fDwn>~8J)-{dk*&*e)6=wvhll2a)y(phmKGZAz52-ZoW<wtWlmGqZA~{dV$haZ&YSBpiJ#=cWqA<WRCz3f!lhQCv'
    'I^~qFAwiugg3g~F2S2-11ybVhs$T%NXfD{(y7tGDRA@xltZnu09Keq5=d{FUA?279u=XG<-PhO1O^>41@;BYi%pPgw2xlS2^<J7Ltknqz;7-QirzBkL7'
    '7QQb1omJ$6D@Uv~FTAn!L)hO@KkG8MUA)*IACHPJu>KM}9Q!aOG+u$38MmDv64s;^7%jPLIyekSE@9&6D62r5n7#v)H0%vC+NM4rp&KwP(?;DTkzbb3P'
    '-dSHC#}f8oEMIY(JJ?E8Js@z0C-qcOQQo-K&=xFpOL*=X9&p6>qv>DrxM_1zzXHFNuu$!V?rJDB5*y~T}xQlHLtgZ4zJu~SNPtwMR7@G_v|qg#8O^Lud'
    'BIn9?x*+H`cr>rdloTFBYX*|`|ak*ZEyj<`iDX1>LGs7E>|M<aX1e-_;ku*GsQjyBqeH8QJSNi?s$+}b1iZIzWzv5MNWZ(44TMVYR6aUD*wrRKZ=S(wa'
    'KAH{})8CzW-*p!*x~F$6B=3WNl$pB2?_}&_8E<+!A2^HA#Pdh~EoFg6+z1gxZSVTA!0ikf>wKTKX2ZZ2XXku&>;*l4emn4Q=gx`r%ipVy_>kCSMI`!0;'
    '{lI^Mw&G!%!jXv51;BU0h)<tJ59vnnkM3LOp~V<EG@+Ik|y%P{_qL6>fcaH0|XQR000O8m!WB0uOp%Y9RdIVQv?72KmY&$WMOn+FLQ2TaC2yHbaO9gY;'
    '<T}WoT)0Ut(=BV_#-vGiEb3VKFshFk?7oWM(xpH!&`4VQy(=Wpi{cYIARHWm8>mn=ll8@2`k>;H68jaUdiTn({$Y)Fh3Pv{E`v$Y7IrN+82dHLdD@pN('
    'Phv=`2~=U(4?IpB}GPRAlxU`~ZK>R88<I41PSc#@uBW_?lVYjvtqLR|RBacI_~*Q&f=365f1VBz2ka|q*{o>_qfD+@=-j0=`=2ljxYhB*JW|9Zm3A&gd'
    'bUeTZ7*uEA_gzKrDo>@3q#>qTFz~=N<xU=zrhI=W%z|NRpziAP!#yk5!%dmk-q+wpe<EDn`Uc*}rv!@y&UHK4e(3LYSMhkuXrG{9Gv7WGa)zCz`WT~T-'
    'KDTZ_i<{>fwtDPV&x^J?inO^oI|*8GY4C>(l_0%Pc8}}4vR}HQg!Qylq(4e%Eox=yl#+LCEIm`PwyhN#_f$d}*s5|PseG#g8U9U3K8XxIzwyV-jZgOmf'
    '4lM7(~U<)pNH7sMxU8c6D^EcFE<_=HL<bh;?>}$Zk$~j`pW3PHn~x=dA{+j@h)3q|7dIIkyX?0^7&bR-inHf(RWIctH_k;E^z%W1YPLw;9%r=qrTe%0r'
    'HV|55NfMk7@BEc0FI!Bba$9A@Tv!fs2NHNC+m>b5RO?F!WtAASv?m{tzOU;$c7nJWM?q1X=15()26vg&#|i$ls0rf6tSgq8#}cdKt)qoa7!Fz}^pD7R~'
    'YUT4gl3UgRaFq<M{g7r-t+I~R@W&F=w}>)(C;0Z>Z=1QY-O00;n=p=n*MmmcX02LJ%`4*&o+0001FVRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_aAi4TUo$x'
    '{WMeitV>L5mFl9C{F=aS8GA?R!Z*HwuZExE+68<X&_scF$wq7hZwvp|@vDL&^zrj+9?X`uWNQ^C1mb?_z#4YmQ?-|l^<TT#90~QHPayawM3ulJZ5B80V'
    'BFQt>XHLsv28+_LD6on{85UfEl7@FY1!cj-eJ-|1wr*^An95C)XPYpKh_Ku2izEuuMi|NDTO#+3yM3eJk47V0^G2A(jVLcODfW&0z7Zud`OdRqlk5Or<'
    'fVu>@I5~m1R#_<=;kq5WI1lrvYS>raN2##!v8-k+afc+g=q<)AFSPFednU(8m!&Jtkbd#X0!eqqByLtcUZrx)-G-IFzFDbZ7>jQizbTDHCR_+I0hrI9)'
    'Z2V#3i8%Oy0CE0B;>&Y$&pA;<f?QYuT7|Xz~UV$k}a0f}q_Y{*G0Fb|C3=6-HN?(7~*y&|rjPRSwt{sqH8@SlWg+_JvYGG?(~nO#`swC^(w6)eL-2_c-'
    'e*6ew$xDGr&}w$%)NIof2351H&JZ7#)-1UgkPX|GHCE-*Soa&3wpm|Tavhq-PQa+e}Z5pp5#kPRL1v~3l0FtwEymx637%nRic(Yne8qG_w$CxazOi%;x'
    'c@INV0vV`)Jat~=osl*@gi;SZ@)=7rhn&<eH$(FM(-~kmFw(<PuTo?{^VsK=-7Jg_I7^~zDu-PDD$+Kur_{h<9F80KfWHB^jGGKk{=~yvp%?4RkhADe`'
    'dOBEKL%qeCDq_pSq7?ib>|*!=mk3mk5zi#KtH7%`)bRZskCOW&`nf*OshWOR4ZP)iHT7TWJlSCbF7i77Q)zF@B1(C)kb-Axx#1Z)f;Oqppht7m`Mj?8r'
    '8cln!;}{qh@7ke4bW6{qdEikadqucWu0JFMATek&qGl(a?!}q0JV-PV`_Sa=bF~5N@r+ary3Q7E%`1NlG@~<32J?mmo;72Jm2Lhw9ql(T&aovDk{<6&M'
    'Cd}pwnPi*QgPL4gZZoGT7)nDhmVlP=g)@6z3QN>d%{?kTLkEP=mc&qGAnpv!oUQo`4z-a6VG;8Eo=_N)IE^0X4a$UIu;|0X8K%&}I{A3I-#8ZQw1ic7V'
    '4CsIU#ToKurAm_Nn{JpVm)Rj8*m@O+P&Ydv)27rWx=Iw`kI`^T=zbRCK$WZF?S{LZ6|T9U(T#ItQEzB27N0xGs)Mww0lU|u2fOfgd+*jVrs$;)&=*tOt'
    '1q5`D2pl-s1S_egky%V_PXp4fxh9VE8hM}_^%P=Js6fid2??g#CNnvA?P<~U46g)Ob<faU@0Te|xDIfXv4wv7W>3Fi6GJ|`BMq#D&m&8r6WiyV#5@a#q'
    'tF#)Fs5pg3(PoSM;19(dD<R7@`i_o0Tjtw?aykO$r^x%m`u$-YepPEa680|14rp{T%zxl9Dnq5MYNRhZ<GJK?EV>x!GP2a|RzG09r-zC`4^K)Df0WPd7'
    '{;vW3};N=0qkZH+^oEtvA<e)y`k><Xw2id_K<jf+Sux8qn<QUl#!%fOLqSMNDi)aMhz*;a*0GB;-?=>?sxQr87{p+;El8{e((p+m!{ng>`ve9^*cTNf_'
    'XBILZh+od($y`#dtnjjppOkAJd7tv1Cg7mtnju`bkVZ;XdJ_@wiR1qEFP{KeLULyS`~YJUp}>+O1ry&8#qK@0ZW)z*&6FU*~iz%QY>>BnC6e1#fNSHa*'
    'J4k$e{0l_FI_ab!wVR$(6bM121$yxr^d2;Om;U}+Y6Tvm_Gj4XSrof<zo1&)4*a;Q`mz>4!k16eJnXLZffqaqZtykDJ!!k|N@z`kO6u?vMs`{?DIz=7~'
    'O^rd+M{Pq%|=tqKwGS`+0uAi-SEq5teH?bEjUW$AdVlXR4Gc4sM7ij7fXf$=?NAoU^_om9R&&<?HlA<Vie8!X0qq9xqi63qFHvEOs4c_3v+@=S!-MciO'
    '(cWH~FQQ#ho|EeB`HY-j{mUB$tKj3pt7YYA5Da`TI4>5Ar{vmnu_D%bNjM#{(Q#iq>i-9%>&g6TF!_717L&njc3P|d9nABm{^}+;sZZ~5hxSCNv2RFpt'
    ')lVkmdlWik8bUa;P>FFa+*iA?4VhJQ#PXz=V(@9CeO?bKCo6Xmn&Xu3#!YX%<xCkKmKMj!d+s<NE(i>JeA7keS$aGKk4`PqjjgI74i15^&PLGntk&te>'
    'I%n&VrhK?F-B5zGPZ>%kg})nt$+?gUMtyLHG3QLi#@VOPTp(^plKRm+|Mf%b~X#&-`E<+)@_!#ElS9)DEuK1#TC}SnubwMQL@BAhqmpPP5}6=m*Q&VQ{'
    '<jPNZM|l6}I5$1X1t)i;_CcqBBlyh&kx{tEf@rTTK9+3}fA3-!RJiFzQ@<oJrEg?eJrL|@vUzTjs44^T@31QY-O00;n=p=n)C$pDlh0ssJE1ONa)0001'
    'FVRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_aAi4TUo$x{WMeitV>L5mFl9C{F=aS8GA?alZfRy^b963hb8l{CQ`?T(Fc5v;uRuKHWec$bgaj$7Sn#rFxrwky'
    'Y*j_Z86Xu6Y3!`puJ+&SODL}Tl5^(FcxG}Ezuz?)&6EntK{N-A=IJa>37j&X6cDoJCzY<svrY+7@PX?BYeQ9D3M#=>j0-AUT4ZjTaSq?JoQmv{yGqWu$'
    'QXBV8@XzT`)~WtXPO@)vTzP1{D_85m1iPajhy22i&G||`8FAn^)i{qBZtG!XztJ>M7v}{aNw{^WWOPg7Q>BmgiBPz12Hhs@U+%2+8KD&Fn-oROy%R$fT'
    '<iCk<85T7Y$P*PR)eb)<7MZlDUZ%=G>|VBd(VkVly^2^GIwWVmv-%X8|RbKEKOQ3DOef`cUW9SCHl?VKurcwM|O2oVsSEVLEe-rFAMc?p?*&M3s;}URA'
    'l6RKDmyhJO>1KM;#gZv1I|<D;F$UvGT;eB;FG^EkD*)n{zgkeM~><;JI0&D7d+wzaseTW9B%zOeeQY;M)8mp2|;?-E=4li1QpQ`4{V`JumZ38Kv44br3'
    '%OquQ>&u?MW!u|#a1Ft{u{C3y#I@rHQXn>IUIrxcsJzv$+eE{7w^a7d&m<4+Zdnp~lKH3N2q2I%)7X;WtY~KR_)ahYH56t(wVZToUOzVC*E%@mo68YD`'
    '|F=HrgeYZx$a<X=fsZf*q1XP#FL{0ZQk4vns*qP2()u;JEreUBv++6uz4<-Da{arHKTt~p1QY-O00;n=p=n)#eSY)_2LJ%)4*&o+0001FVRT_Hb8ca9b'
    '7*gLb1!IYbZB2?XlZj_aAi4TUo>W8IXGisF*RdlW@a>EH)AtqH7;s%Z*HwuYj4}e68$T}_DhRO;!Cm~a8t-KW2?~%A{{h02?Uy4OGGGA<s-Hd_`mm@rD'
    'Ri%)q5{c*wD`I&YXGd%<S^J_$t#ZOk&X&W>Xh+;YVJUVHS9~m&qI}k#{R2tjuM4m!z98Ue(spi}H0t#%nM3xv*XDve5UUn&;=?*D&AJZg;gzKGbX8O4h'
    'tMsQF0|=jpDN+|~Rr55LJcTZdb~XGxLz68N^`4qPl0Tj-VnEz-?aLpK__<(gJs*YW?WZs;`gs}~gz`Yx=l=$ox(yDqFAD5Ke{3q$nVTrowz)$Cx0a|`p'
    '<foPM!4i-A(#QcnNf>}+oE{N48)|n!}(gD$-u%S~}2LV03W(!1%;#Axw9>fiZ8G6$I1v$eadmD1*S+ffYf@W9kLSa`4^oVr^({vgQP*b5}%Yqw*K?Dnq'
    '7;SPHTr-tmxkhhIDyO7HP@tRLy<KKcnL{wZ8yyuCLxl^Jc}7hnIur!I)!MsgwfCX4=*AhT77#2lwG5&|P0J(!t6J89fV{2<$Qvra;IY(F8%&nc+hUNq3'
    '`nb^+~Hc6ws5Ho$YWiENQIB$E|q^VIpUBx$$UeqF@59%evxI&-7>L2tz?4VJlsg}89Z18hEae1Gdpz?I{`T2GXKAqDli@r-o<8(i-nB+9r2Nut0di#DU'
    '1VX1~GuEk55&i=3)@Xg%^p($H%?J71SHdSp^&EWko7a!On*-aJj%5^ktmWT?KxKLj~X2N<X{{{U2-cl&Z=5<-lIdmJ{ct&Z8YMkZE!YV3zZy$oxn)vOJ'
    'aXDqqVOU7$26D?y*kQRnljUiXDImoZQBL*VFS1!#aqQZ}a(a37XeHtXpK^RS3YtL>zhX0;@(C1`r3jzh*&^bAims#ldx2QR%)D`FwHNt&}Y?i*jJk9k?'
    'qbtRK+5<v@(3Fpej`Ez;@{pFO>3mZD?;^GQbQy0Vkppfh0@*V051NP7ZYz!#75eC$pJw|K5;Gn+i;_U*ZRTtL_HU;pw>@UFiz#^=R@q2bGj9drQ_=X(@'
    '{2T!`;X2T!W0r4S&|d?1bIeWP&0RK@x>(FG0?!$-sbYM`P7U>(1JAM9538Z0U*bYWtFYLJ(m!!gL@O^1JyAL;hTo*WR#Q4GE@iy&(l4TP905%?Ud&8q0'
    'EC_8NvxO|2r)`!gya=vK*UulrOyInTu?V&$kxH=h_@*&Hf}LUVwfggUc&HfM@UhsV8Fz1w@nM?B*P}gUj9`vGI(Mf=Ig>M4PX?-xOkA8TU>Z6QN~kT7c'
    'tx;8pA3kxL3)|*ovt{VWqT~_$sXi1u9PAG1_9D?EPWPF-!Aeh4Lw5FBZvWubf99oCtZppTFDBy)SCcBN1=Ic#p=D5zf5~P#G$1RU>`T8BZmrsq;G2d6V'
    '%{GpYw%?v=xVp&TBS9{G?zw__Ny;u%gwc?XE=v3tF=uSd>uZuf>|*GFR>1(nCc^K-^f&l&ZM$<xBm*=uR%|Br+hfg#wC;uzg2SBoFNi+s1`vt_uj2d;f'
    't>XLhZ@MLKimTR>8tzN&~D_<>-^(Udrk>l8t5qiaFHeFuMM$11ZV|8QY5%<r%V3YO3fIZ<Zlxgi@6UAAdtG|5`>pb7~HSPZXzIksolXRuUnZ~`JK8ZbN'
    '_9c0p)715rrkffWG(SmYbDeLZC%M4SpT)LC8Y!W`*K$-=rakec!QHFyMz7Z+ylFPT(#UqWtR5OMEqhZsHF|am9DNVva8Q{6D@ei;2+x(C-qnteiYT<ae'
    '>e$+!9!+XU$H#fda1^J^m0jXAiNHJt{nltIY%h^ek#2@DJ@N9`BbfHxsA}e$(}W3ktSOYgLYulydqyGDVjP1ji!$LXxt{jPE$GdNob{#Fv|)VobcrM=n'
    'M^c;`nR1@qS{u!5ci1Ek2Ts-nsUS_U2rB5$(gub5gxIosqMPKkT8qbU)1PN>*MD+<{}er^VdSgsx5IOR`o=!few<v-RR3|34sIjb|5w@xOyL9}lL}<68'
    'aiV74=Hme=l4ecBIOv?r#<u9l-~WwoC-GWU3VytUV<yaU%Er=?%X4pJFTS@S)dqjnI}WUQ_6QMC+YzLeP}V_p7WhI2Xj^ItZ@+lE?zq>=ctQ>m=og^|p'
    'D=kIS%)*YLc1)GQFH@u1}_RTJw<#2X0bu0E+9bNCfWSZNH(d=?Ldv7lW<MDEg?&(y8%KP9SWoF~cA7olp8E1C07~0Fx)Nx1d4YR-@H$udy?Om@5+{}@&'
    '_K#_c(&{2XD%oKkXUAU9aThm3_hw-qNx%N}`iKvYZIXqmZ#3@lNNB`KgJFLD3i<f?@Z~`2>6y=odSH`K4`dQguUJmh6O%-FX@C5ToAqx{O9KQH000080'
    'GFX@U4bTHTqFVj0AU0G06+i$0AyiwVJ~xTVQ_P3Z*+4nXl!(7Uu9@%b6;>}Ib&ZmW@9-xV_`8hV`XM$G-5YnGiEg|ZDDR{W@U49E^2dcZe>$lZ<{a_ee'
    'bV`^1w@%U=zM2G^L4~M$MN(8Y!J7<bVlG88B?BZB_sKY)poyy>QMw_xj$;0l(i32Ezn1ObHwMgJE`MSwym&Miuem;U|%9>T8=~tfIcIlXy&;x~wn<T`<'
    'Zp({YvSNlq#Go)?(qWuXf>r!0@D4#z+jL-c>!f4<`45}=Ki*5pSp)tVw_!EUBi*)NJ?7Oaoa6z#XsI-F^g{0t5nzL4M)Ef93IIA{59QUsgnLA#JL=->f'
    'KShVo8Z((+l@Y=%sxdkLEAD0qj<y?wrC6B+fu$1CbPFNi!bdfAs%V;Cd?K+U+e%nGQ$A)qq3T1?($7z0Lq^8{GcOD9XTOwSawt4XtxH&@D&2CC(lMofB'
    'Zdq=a$Q*7lw@$>;y_?vXC=%Sqo5~mC%9l3a;ok)34@lvQo<Hq-K07J=wdeEao+G8t<5J;DpSe<lR?4iGo-dV}rLyPhsBl$R&aM@Gqx9dYT&dY_dmbw95'
    '-R(nP|?w_rC<5;qy5S?VRcU4ND?(H7N#4(utyM#pnZUjZ&|+SjV;4)pmz^|4}knRXod~T7WH02&X8a)F;ef0fonm7n3iQ?JMjRVI2bq(pJU*9*!B|F<Q'
    '2rSfs?xK<P1``>sR24W+jXGUq}AG^^uvzo*QFm#UxG5vlCBJYy6FtMR&Yx>X<}L#ji9;y4NsA0FJ=qU`+gW^LqgK`gb3H08mQ<1QY-O00;n=p=n+Ajq_'
    ';@2LJ%(4*&o+0001FVRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_aAi4TUuHBiHZx>nVPs-4FgP_bVKHJjG%jj$Z*HwuX>Z#|68$Se&zBjjB%ZP@!IOn7)3!F'
    'cAkx8PlR%)!wnT&?H9TS~f&cry*OY9^v1WIHferOl_3G%V>f#UaO=fwNCZZ?Iwl10?i2XdrD)b9Kmj!lW|4zo(naS)v&DK$}Y^<ao7ps;`R(=w2W3$@k'
    'QQ*f7KPaMaQL$~@Z5z3KY&QI*Z1_po2+}esvTY;1Zv;^heV0kTiZ+1H(=rPr@EzCdd)O#9kS#-Mq?>k2H(I*onO0BN@&6Cq(5dD(KQ6)aLs(~`XBzF(r'
    'm(uAXLQ?oQy8LWw@;eF#8KPE3C9lBw&+<TH9MSGO@WOLIXdLPK?P<u1+YwuCnuPkw2h`X#pJYo26UawfHNTHwB5zTCOapc{p5s4z%jsP=;ZH!&9n*7F%'
    '%TU8<2OR$hxGb%A`^gV~1eD3T2fAGYm^<U@A~r$~~Ji)1)boV_KX*2lI@2g52DLTe?Dk>K0Yd)!_+MfRoJAoi3KWbC=cG8ETy>DfHJ~J=@?Jq}b+8if!'
    '$DX@g%`Y<D;TafcVtRi=Z;GSngD8YhaL=*sE#85QMWm%12FO%WjuK8gEG{>jKlA_gRb4ROZUk&pOAh%s=hsKRV1Q~VauT8b~A!7eb2=F6XX+B9(zf+DG'
    'c|3@VP<38Fw95%RF$|TqlA3?cHvn`3DB!py4`mpfnIYrc3^pm9YWAXI#w9~kTdW$tvU@iT;%;YiH+29o}H<*BdObVK-!1sQr;kz3dME6nfb3z^yHU6;Z'
    'JM-yc?7o(Ha3cCLOYZ>8Twa%X5X)9xWHMP6E194KR4U~a=(9TNe%_RO^-3Dn;!3XlSmqTJ&8$K7QAeuUbOi9@;@V+89iZ+vk!g*s^s~H?W{ngzuadEkn'
    'VOuzu{!mp&<QHmp+vPK=5mu}1uNsO?p5*_nKfD0GTo#xr0}9}v#guHW_#A(jtRYRAhRhhu8}uQG59x<t0^wuBeO8z49V<ZK;jNDApe{Z@(P11ddR`6IS'
    'NWs+{{@-K;tp}fb$U}fJPr!BrtLxP@`Lnz`EfGurc?6HUX#sINECfZ#H5{!RWDuHpP635qR#9r4-|P)<1~n8hEb5%360F?GhI<UPk3wRQ8FBGG6*w<cr'
    'EsG59Wnm0HqZaVe9vpM4dT;c#fS_7lcB9Uz>%NE1a&M~Go2V}!4$93rkWDFdb;{esN#BNh*OM_gri*LXxHi9we7MFqpV9Tt8}5p<XsY&TiSn55gp$S=M'
    'rLOM^3qGDD0l>+pl7?qE5eTVmMDJp-8n=*lUM59~97|$xH>02?8NUe+-6JLc@zeLU{JbGKq(w#l@IaXOwE>U?Z--~&=-U;Uw2scLD@7C{kYyYcS^Gd{3'
    'l<d%WGs1n4Au>aSttzFjYU8osG<Cj)e2{cLYDWEntFvm@6I8>Kk|Q6Bmx2ssmb}A>s9phaGxBa0&dtzW%$)9^s{1I-!?4ynymT{$>Sk0IQ)FdOu-a13{'
    '~rl`fFW3rk|Zq<2;BViLloN$yOzP+>3hy)B}?w?LC4ZEEYIll>~62qt(umn<}=gf&~=^h5Vc}Boh&Y=!^I!tk$SU=nCItyxXyb~$eM5;$*l3Xj+4B{-Q'
    'PcpRZ(nun)dMU(0;JmX|~joT;tg<pT&+c|C+u{Y3jPI>83^kEl4xjUKQ*3nJ)~A7q(5A#fm8mv;vuxYtM99c>gB6(d~8#Z<;NTwDK*UtH)MC&E8c;4PT'
    '4`MbAe%>_z6l3e%_p(%WWNKhqAciYU~)x4#L6!AquN-_ShY_?gCY)N)C1AiNEIt{nisJBKTJK_>kotu)PK)v4B{+{Gx}B+uKj%+igILE95*ep#&23`L!'
    'eMo~w6wC>VytEm|KEVN2Ul;@=kk63beb%usGaf21!$zK?5um(M{#U9z{o@*~?@6NSX-rn!L1l7Ca5jnm1moxAd-p83!i^|Ks*LNN7IGZ;d)3otyLDG6k'
    'm>ue9+OPWf{{iWGG`;AL{vNE^s6Uw;=IVb3bKJ4JxbY71(|O#WJTWx3jRI9GZ~VHJh0p8btG&wPJ*f6EErVKgaF*kiwZO+cYI`<KCfW+$REtm+3z@HT='
    'H*XnxR>KU{$@4&O{9eg8i@}(70TLugdXgl{Qdn|yF=CTaQ)c+j;^St-}J&=45qgeucn`Ms_SR3iDu4xIK5m<KREOLXtWridb)L{>OJ^Np6TfFCmy@b<4'
    '$kq17|UuxZcpaWfZuihKuO6o#}Ok+ZiI(`8jNnT4xB5T6CC)(XnH6z4`6HyPZ1++;4xpKH<}2ljf1C8?6U439Te;(akU4A)mhNza2PxZuGfOEjA~rLFV'
    'MSi{(PKn4DCP_NOm+S^oo2O9KQH000080GFX@T@7H^p(O$U0AU0G06+i$0AyiwVJ~xTVQ_P3Z*+4nXl!(7Uu9@%b6;>}Ib&aDG%_|bWMg4uVlgl{H8No'
    '_VmLG|ZDDR{W@U49E^2dcZe>$lZ<{a_eebV`^1w@%U;`n5gr+oc+Nk+bNF$}wgbbL#lmWx0+E(?y&&FhU+6(90bFc5c9Ps<yU@%ND!<4Y0HyCDDmW3qC'
    'X;=|A9)1$(roOf*#wzORI*G@msmls;&;_FmGaXmCp5&C0?|Fe)UKYBLbIS6V>TnEnF+~5j{pTw#E<W07X-$6kGp#9d=I`fPmHnbPX8z_B&d^~OZi2Z+$'
    'xnZ&;S2H4;Uj{M7UwMgO$vWIJ8BnF`W-wV36Cv29a@;5CA_w<cy0m7%H>LetXxPDt>y8T7FJSR$qDO|gf5aL8yRioxqSyxJnUKs<k&#YLxGHt^f=A0jM'
    'S9-{LVuma7%>ivo<fj0yjqp`}s}jY!ag4)Gf;m6Pe{L=GKW=x_1*h6GejicvJaeT=~)lJp7yB`~fNavFA^Rp3lz;f9?6=x#vjfvs@`$>9bI3&{~=G(({'
    '#4vr_h4pA@d@%Gr&gZ<YRgl`A!eUC#sMT>@o)6ev0xw)881ezae?CaliM8%e^3#lmz081@K)5wwrc@r<cw*kjAEUB|r#zym=395lnm#1{3FXbR29omft'
    'k#0h||<yuLaq-km=a2nZ$V<o0x6G+k&K%O>JH+CkLH8nt-x<=Qpz!l9(7V^K2{D14il&@eWF-{@~oQZ*9<XGcxv@E*gWmCr_Y$|@GNz%QBH3D!1%%fp?'
    '?dJCY^7ZdN{s2%*0|XQR000O8m!WB0cG_|y69)hQ8xQ~hHvj+tWMOn+FLQ2TaC2yHbaO9gY;<T}WoT)0Uvq0QV_#uqH)Jt0W??pFGBPqTGB7haVL2{pb'
    '8l{~SNm_<I1>I>1nw_;SlN2`6&u+u99vC%^$V7=*lt@GjKsu3Wyvd1N!%j;`+Y-Nj*`ZAx4<HSNe+kKJUBBPeP{O~FXAj^1LihuX0kBhMS)esB`<^oCE'
    '+(B0ma{DU*q)7$Zidp-S9BXj9s2ZWhe{dW^WYY(KPs-FnAgnVOFLx-xF^b%lMl}i%q-(Sdo=^D1heq!7u=!+(D9vNWSg18@AK1TY=je*f#!uu^pR~y5~'
    'uWL*H5Jj1643d1kU!pADRTvu!dPle1>uWDX{sX4hmcnEFi{Ol;7yY2vm)>Z(~sA-E<Zl0#5Lv70TE0o>^}9g_juZPBC)N}INNm?(;4tGym(j?w@~4nds'
    '}x<gnUwcAyEfOmQdzXL&}s?xqgXB~y7Eus_JBJjC(h2lDD7qphEEWn9YOYOpP$5raltfgjTpR3#fK9^`cg>t4$g&FQqM$1+fktdJ3xLt)z8uXO*KyI}w'
    '@3gd^h}J!#^pwvbqpke{TH8^4(53Av4RE%t^&>uwhg)iQ6bfh^@b!t*0h{LUR=#cbju>Z30?qo$p7s;sNpe{h6dI7vN(NGlVEV*vbMY62R~l0c7dm!nC'
    'LZyNIHg$Ep^4d@$nY!Utzcik11c~a^ZC#1nJ#uBaHKk4zH1Q}s{{$KX%Mj#X}BkRMD|^l?};f+BS^+%$OiV)F{{*?4b!yb343~aI%r%&y~djIXe)S8=H'
    'e9Wa`Xz92y7{kZK))46?o-`8h(}5H)6x@W2EOP3I1$79)5f!r6RI1j}sIDfQkGdOCLMRQI3BTId}yckTWw`d4#mD0S!@Obd@>*_i=scQChvgtcXZBV=s'
    '7K7+G#)_m#3$DAeSPPSvnCh0cestxy!U61yyy<lsY1)bc3cYO*dxw#yPo33+1R!eY5Me$K0-|2ie~!h=kcU0fnZOg8!xX<@SQ2UI-_*h8**7?6|`3`j6'
    '<iUh;pvjH;y{R(+!vMUS@U?u?-0^oe4GG(&q?^Gf%5*<*}Yf5(T(+IE`(SbIfQpqtH+3NsrIi(cG7*Lv;Y_*^wW-@<535*d6y1;u+YvB1FmA1O;$S!sv'
    'lDoLvGHoBbD3d#$$DC<H+31@HH)=@+v$06GJpanH;c#fa<tZg0Ie>YE%u>Znj$o5qBzUHnb_l!7g$OCL$S=&_@tA4?d568vd8)R^No<s7TxuA)+X+IHM'
    '3BSSXur!#ib--Cn{s)t7|D5T8p}<|wF2ZtHZ32;_6EsDC5imWuF4eV5gNIb&3Gb-n|#aWLT*G(8WX+>t6^D4?F{lZTV@A)$mdw)vb;n3Xy3C{wmk@^D`'
    '0+txIe5v9M=4+TGN%V_i=haqnly=gNTqBDr{ASd{y#J1*dD%yGT!xt)8~}1?xRMR1A7}QgXzje6F)l=8o=g&UA5SSJU8X?Ojd$wg15x>2i*WJBez2#dB'
    'ZdsJ=$^GcrfFLPeDH{QF4M6VMC8(-hT7D&nW_OzwBo(TrB!F!07&mN?jhZl>Y10;e<Rv<97yZfKs&<Dv1y_r2K!wPLcEug8nY`j6REJw-C1{c|2|i$NS'
    'wO}LFkZai+2v=|We*UxMt<!)eE4-XH`hgLJo@2s@2Xz!=b?7&%k&EDp8ZM$vRu0;%1nB}6mk=x{WE(+!Iv0a%b>QEF~5}8$4N4`9|eG}g4_xl9zx(%>2'
    'ialPT$3{w;z0*cbo{a*>fFm6$kp-}#EY?7F)9u@5*2}9R6w-W9-GsuROD4y@p?R_6xkdY^<$}P0@HX_h^#b_aIb1OabHQb%HO+<YV(U_F5|nOYFPfsvv'
    'mM7^Rg4xd<tEEf)XC8(>WGiVO&0Af6=R>7rG>;rQHtn<B`>eeu@EPIxDi|a6NMYBLI2#M{@LlDThD0k&aGG8t|-qz_3m^;E-wD<je>RXap~2fay$%%z8'
    '9Q67EESj+HAQd)_O{~9n#Tlzv}V-52VZK;$k@cXRwyj;e7rwSN}Jd=g<80Rq!G|y~iEO6NSd!kf>UP@$*(lPS;1T_9hp%;HqL;gth3P8MUug$Z?NWb<7'
    'f}wMpb=9SOM>#kQcl{6P(WJp1DxR)g<iD?-qSZJer5wr=A@6u;5$uSe}(s#ZkX$L2S5MK%2v7yf#*xSj_!{o4Irult&5>8&P<@p|#Qw;E2T>nW<IUq7V'
    'Zga3MFF&+PKrd>bdFRoW3Z#|j&!6dk*DDa6JE+VfTOs^lfULs<>AH&vdIQTR@I~^T|gMP5O9tGDc@5Sl2KWv}A)IVU3k7J25N%ml%Ns0MQjJA-PMw9Kb'
    'B331)@qm7!k!B6Lk>@XwPhYC92bvvQeYKz`nI@{2rpdA4rG@H7X`<ivPhao|{|8V@0|XQR000O8m!WB0>08P#9RdIVR|Eh6KmY&$WMOn+FLQ2TaC2yHb'
    'aO9gY;<T}WoT)0Uvq0QV_#uqH)Jt0W??pFGBPqTGB7haVL2{sVQy(=Wpi{cYIARHWm8*kqCgmZ@2@aE@X}R=0YOQ~#wxX>YK_<?cC#S^3_4p2mKiq9Hv'
    'R8A7c8^c7tZ<4cN-4O$5pS_&k;iuv%cT!A5ZL<;3J_4eX+9sH<51YlT0yoroL(7Y>1n>IwKCIV3Z+dqBAjbLMeVFC1Rv1O(7G?NJdR-2u!iW{CE8K6Dk'
    'iuxHgI!zXemHDG3WUGlRZXFN9~}RWhIM4T|4`oq-NG*e8q78XFm5<PR=`^>k+(a22$06Dn9pxZ6sY?G-#rnBPkXRpo7@Kvm9_7%o-&ql8F_k=n3)QqV@'
    'KWTm3D>f5xS#O;HGSgnoKzA#o%sO&AsiQ$@Zi$8cM1n!0~d@A$e18`r2u$f(y)*&G(lfDx8U5L9&%v}?)GVUU_{)q&)@S^fFapi{$c=(mz{3g`+qT_el'
    'j?eZQfA0AFzT=@b=Qh%~HfOHYgiCGLW5*+{CeqHieA2kCYrQK?Uu*L>I@fBp4;_!ScZs$0hq0!IeMx`v=coL@HD)!zzj2;4EEA?f0NsHF29~?CJbyIuo'
    'ypLHo(I4+06qZfzo171N3N&`z;SInc0l2}IdW2jtTeSf=s?@a9XoaM!m%88GD^qD#-K<6EDGC!ybZYkU*?tQOpg^y_z#Ewzv~IYUT)b1atjEd1%W+5li'
    '@E~m2G>~)EQ2iGr!0<Z(qY10BZp39khLU@@ru6<6nLL1yD-^1QY-O00;n=p=n)IG|PV_2LJ#Y5C8x-0001FVRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_b89'
    'hUUt(coVKF!{W@2JxV`VgBIAUgHG%jj$Z*HwuZExE+68<X!_sbquwq6~_Mz#ybRuf<S21{9Nw=E1tVr-$Z<dvu-Zjt|f&rp`5r19M?ut;DI$(d({!<pg'
    'eJG&Qo5oamuGrMgvorMW63d|xdc_AcL5`HTZtoYmPYn-mN>`s%}Ef2#?+vQnQhO*FZ_gWzyb&aou#?wd(voe+Wo_NDp#@|F*Y~mfjimc2-0W{AK1_2hz'
    '9W;rE^qaQZGOd>B1h&&RP5l32S|%BF&yx~D-<fmH`gW&n>CEYYvf4eJnIK@h&MelqdPLZ)?{wQ8FoEc`ZLGMU&f908I-u;fqcbeoT>`+0?UEe8t#e}Us'
    'V!(Z!%C;^>Z}XGA$bex9EZd#OWD!|Md+^Dt<V8sLDW7cQ46xRLwp^`T8gh@DT<7)PIL~crw#(kY|>#WYe~nMI<KveA>lX*8OO71WgcmB6oob0Rw&SBJG'
    'EnM=x{n@vke`NrSME8L3m_~ve`aUULp%%vvWpPIj*uEwm6;o9L`zodx!keB|ev~p{FRTz9ZLGo`nxwvc*)puJRvH+@r5L;8T~=0h{u>d!}s1p4(NI0$u'
    'KTZM$3DkNbppl3bPrg$DGql3^)EFnwZox%i91D~&0Jiz;@B5|8*roKmb;p$Td&GW^PTE7%wCU=<ja{`_ZmbsIYoIMOO#zE>(R)(H||Q=`RFq~V_M5!vf'
    '3-xE`uM$imm!20IXF{{*^4brsa343~aI#}F5y}_LFXe)S8=He9WV)zP|7T7`{+fqr#Rp7NB8u)cu--->tkCC3IH2BliXz=lsmU@wmc$}aJ08HcuU6t5T'
    'k8=E*$kSJ#n6oKQL{R~e^$nl_YD`t7PQZO!U3rvNFEHyxB%HPvJTJ5?*RuQC*g6y%dWNTJ*qcgcsAexUsui{ryDXRF;6qKU^ijYybX|#TmnF~=@<hXh#'
    'd5FxoY$iNI;Hf|gHD}YULi+xHvAK5p|jBk<Ov4sA@e*8C<|i@NHA}L1jFDX;dJ(X$@)mFYYY}(#<15SoR3rxbT;{&iXKK<2h`+-N(T681lW|;fi{~^it'
    '3EcYXNUDp=8GxP?^x#a!$#tGk;741mg!P3s6sU;Q1aExu)yrT<lUL>$u#q>U`|7Ox8S)IjauJhTlZEQByiF8;Nww^RKKrI2@X9c}hu04q#p(vs5vYBiJ'
    '|>37#odIfPy1LWGoA<QL}ecuX~cyu;q-JXKrdBsR=5E-M(i+p**csUU~3;eMBw6qDpOHsSJKF_QDxB$k_!R|b$5*`$0F+gl_Xl_c^fyDn24kI=}i6#kO'
    '9$+v7K<VNIVG2yGQ8kB{soI&1Zi|lY7@;PR?EY~O~mG9Xy+a8qD6)-<R+#lv24s-rh&FM<m`#3$I(akXbK}5(56}GBEzAAaAg3~tXU8Lv7q^HelUU0Qn'
    '4K+hGJSjcmQ9jpM7_+83oUy97v+GH4z4ETd{>uMg4XbjFiaU-PeZ_NMW2wGI^)oU@w?ajf?EL#kc>OGTVR)Kmcz4v|r|(Sechu1gm);=oMwKpcI1jp+m'
    'gNLir{8h<ole!zJn2WF(b)IB=@_+QJfE#b^YQAB=|nw6GNJto9&L+$98pcUi$$(IZj-d=)9SCE*+$A;-!L8?9@-C1JImKbS{StV(`R<zEWT!MbK0iq8m'
    '4U!gAr!AXm8{;Im$($d=}f4d7^}((2&Ti!Z`Bf(cPQyR<GA1c-wA)rB&?l3O%+`vg}=TsPXfmz|rSOhgxL;tSE~sAhT`v%yZ-ARS^nV-mh;$VbCR$W8b'
    'j6*zw$;ebjP6;6QjA`oeet{O$s-=!d!BGOH}jMb*VNwcI8s-NatBMVV(ij=`uI4PMGkmZPYXqfyimAFbOg+8ZjyJ~N|I5*I}&q7x^1d3Bb7IPt@c*z%t'
    '!+;AH7&kps^R`0@iMtgT*yz+Lv@*Gs}PDkYY^55PtSOp&!ULz_;gJ9r$!D+EzJf)*e7b{|Iri9%g8*TShkN<xlT}|efgULUGwU`WMvzNL0zrj3z>aVVY'
    '7y0Qu?ogg6H1?WA)he`~w?cBdK6<q`xwr#Y9n&IgL<dpSz8WFNJsP!`AyQ+L$jvGeawUpwL3#Ou8UASc$3JWa-^E6Rpb^_RRiSL$#puC)qu*bT*1a^Xh'
    '_;XIZ|I5|_RTN-)o^|@3mW#hy>9pHHPgadj_0G*{C96Tm`qj^R8PMtRJ{lPRc1aJ{UGBuW&HWga_FtbGd~yyHxvaval=LAwZqYy0yhgptoLKs+ARy8rs'
    't=l;~?k<%bQ_vv-Dm_zx`qR^riU$b9@|2$Ryc=h9)KEw=vp6Y8rL6%ZgZ)nAQXOiB_7m=tiEuL_U3~zaEG>w)$#9PcjjzmnP!a@X|zeqeN8i`=>8>g#Q'
    'CjO9KQH000080GFX@T`AYa*B=4^09OP606+i$0AyiwVJ~xTVQ_P3Z*+4nXl!(7Uu9@%b6<07F=Jn1VPs)3I5B2oVr64xG-EhoW@R)kZDDR{W@U49E^2d'
    'cZe>$lZ=)~}eDAM_^1w?k!RD)ggidKf+o(wzg|t$7Iw50B;%f>iHmB;U{`a#nDL&l`GdsK9T?YJdF&GSU#1SKW7!HPqBR{0%Kv~LOARc~`>83umDb7zU'
    'G);oXq^YYD5@1TkIN~NcQ8TBEkylzGPOH+CGG&}%X2LNr<q-4V_TP`F+(+@sC~EQ+O^l|bJX%i;_FBDAkx7>6Y_c;Ld5@L`+LLIP-p0^3Fy-_QDWlb7Y'
    'wSrCb#NUkxNYHX)53J8;JJm_eG9RwyqPOdl`|#A3w8Xlg}D;vYQo}4K^Li#rHWSS+`0oLZXQ}l)YwGLixU;a%HDz=IjNbj_(OzJ5N=4rCv9GS0O5-i*3'
    '+}nIV448(pQ4KOL2LQg=;ca#+}8^Kba5~o>j4vP=2(52)`17U&k81?fKoN=hL0WpL;&L?|H2CxtVKR>oe18;)OQrvFCHGX0GkIc+$A8YiF05zS8=yb*|'
    'NH9(tZ=?~-Wy$BCxL!<POO&(HP)*M!%U{3dzYa4b!?faQ%K7(s6f{m{A!ZD;H{zUTOt0E7Uj|AHNm<$AI{bD$eIE(C#%od9KiK<q3dmK8uB!@w)B<)R>'
    'm1j9V{u0R1YWM`fWp=}`ov#wu>PV7+eRQzz{|GS=MC`ZtTh3JflgZ-=^&iEIr%I<j8)R?5rNnB)-cdy}$02~2(YuRDD`6Ym2{fp1PP)h>@6aWAK2mqI%'
    'X<b1~ySNhv000>f001`t003lRbYU-ZZeeh9Xm50LFKBFZXkTS$X>(t5YcXSAWMpMFI5#t5VP!L7Ha0nCF*GwUE^2dcZmn1QZ{kQ2{#UHtU-oncKY@U<b'
    '0Xj*-0%{1Ry#K{LLJ-zYmCj>?Ig@7|NDK_#>8a8J*3r);_m9IuO3}hUHo8=BFm#RVLfIyO{TL|%<~+(kV~Em2};cGMGT6+O}|CST1y`^ncnl&D%Eyb8W'
    't;=YxjFC7tgxJ*Fxh-sIAf>k=dSjSCNdqizMGfJAmbBk*x&KJU{3MAQU@j5+Ug~ZMR`s4bus1r)Qe@|98_e$*4yj7ZCcvoD0^o&1PF?PM7tpPSezxiAB'
    '3<>CD2yZnku0gUM>zU{Z?<P+Xij%??&A#RUXc$pN6H7N7v!a!HP`yG=)DZ6yXur;@TrT}!PjbqI9JReHKuwUi|ESnUeK>L^QHC?Z*lOl?8-LLHV4!3wL'
    'Z%m9dOR`%PL0s+k_eFC3DPT5L@tqdSjoXR}gu8gvsqs!3IRdFE|N||F;v2rX$gH3i@c}A<Fw~kvy*=begIc-G)pVO%fa7{&nTGvu2aLu(VZ(WBBKmc5q'
    '+=f=SMGK0(t<;0m0+Wi8+g6$gp`)+}^`g3Dw_DO)kV}%wBB!)~c2+QuA|%ph_K=A`DYcS_@;5K@mR8~!f03h<=Ve-AwH7J<%4jRtSMY!e3`>9gXLocPC'
    'm}eJvQ~bSD)6m}Bfz0X#6l#iJ>es(*J-vVrYH%a8H+yanJ>qRQhU}<l7h$V<>lpIaSinjd#a(W;CYdWbFlNl8(bo=xj>~QlH66`RUB&gRY~8A4S$SKn&'
    '&k5ljX4g>5Z0($c8+M(EtD@vV*Qt?5IZt{$1qjH=x+FYq90A$V(`)zXsJuhbep18Ng4=8;`2$1hpcf-n6~oS+1p-mOfVIR>@G)GdS15-c>q9FFVz!C~P'
    '5iX(lPehmKh4qm--Zx)JFvjiF`5V+~gp$-VY#R*C-YoYE@~I(2q+gNo4E;9uwqIvajuJ(SK6XQ(9)AJo<eA5^I~Mv39$qxf`oy+Bp!>~=vF2%dn-6*!-'
    '$_;oh^or)V@q62DtM}-G|`T}f1bf8TEY5|V?wSYGtQ%A!$ppK=p#f<uu&ioO+!1H0C4ZLgG1J8$TcDo)r@{3)GcpVj6R{F=Tig?Ylh_ljBHux@98?_~e'
    '*-#`~o_%Ab;|OTB<q4G`1%P?EOcTXSfncLd#CW7w84z}p39+KuqPQ@B$0O<r6diV*@kAX_kk}wgxh!GmZby>Gq=Ev*2K!xBP)<_V*qF;l#Yn+p<4A4_U'
    'K&7AWaHvlZ0}KS)RZWm?6yeY9-&cK*@VZExGA=5D&$6FWHI5ZwCWeREQ3MOX7lvm55*k2Ocrajlrr{gk!}yl=?a)1BkvFUkB2?~ruK9t>^e#gXmm5oe-'
    'a@oL#3_ik#Cybx#YA>dKFE2&D6_Qzu@w${8S9(=SAre&+@g?!k9JP;gpr_o!yRu+og9q@|XTcYfv_H^xRQcn<`$X8cR(zYMPN5W)*6qWas~nWIIiZ-WZ'
    '-FDPA5${QQH-{f_3D!NThY-mugq4*p=4X;@BRwR$b5*J_pH%!_^$8jgJ5n~cyaMziU1I2$ehn2gn9Bx5?i<l#2&MIrTshe%}F^EOWM9#Mb$!ZuRwdWP}'
    'z^wfNEnrXH+lH8!PU%s#dXZ|gHpVKx?*D!5^7>re#iRMOb<D*=-lCNUBB8!z!xH2RvD>sgOS@`fSyw&Y?3Es9FU}@xgyhG28ge?0|IyHKA3LHI-a;Q}1'
    'zzWl-1Tvd;*Ss)JuZmE}@?Lcl3WF}00{f2T`Hp7>ouiiv0tdqT(3i#u@P|u;qPNNfmuYEfCdwJMuH`;P>n8TRDT*xJaeR!5(cp#Lq#2qz1sY8q`O&yf!'
    '@Z$$><cqWB~hLiB0S^C>D5^V^2A?l@J{|h=>~5wJv%f#Tir|J745^N@g~|8<u$23oX^PF)jz#Kuna!Uy;@cd`$6CLg7ad*XhN<{=1XF&mxSFS8*TT^g#'
    'SMv-Hc~f{qbLeH6Qn<)6-i0?_i!k@t3#3Nqu_HJG3WCjlCw(wQ}v(t&p6qk6!IfCLX|5<+NDUvV&GMz#1!#do(IBLnOu~mYZcL<Wl6@oa*u?GyLJ?kH6'
    'RqzKe_yNh7vVqEgv-h%kfwgZ}<@wC>cjJlsAvzhf$@**ClLmxI~eG^p9<b~~N+Tc)|U7|n*u+3((>KOQf~=$?LEsC*Cpqs(kP{7J^G%lNaq#lTyRrhYI'
    'A?kEd<;zo!lY6sWr0(Wy{toL);+6@aoOfSx7$3f5!7I%Z-ZsDCszyGuL<!k*X%<*w7AQPug8itgJ-$xh=X=v2hF3ls=Vj54FCmKoGpc{GpE%Nzm_4|QV'
    '$6;S>=t-u9nx$!R9C&G?rcqjy@B8Pkc!d8AP)h>@6aWAK2mqI%X<axS41FL100376002M$003lRbYU-ZZeeh9Xm50LFKBFZXkTS$X>(t5YcXSAWMpMFI'
    '5#t5VP!L7Ha0nCF*GwUE^T3MX=Y_}bS`RhZ*FB%+isgc5Pi>ASbpFowOQ^L3001b6BOIAj1$SJDzt{h-o^okU8<;2{=Iv_VXMAy&YYQ@84m2n)nG6zAc'
    'X|cVK^8bPxP3fBPJPnG4kOzo^G3yNKtwwp{}ERjM}C?Lk2o$ghHyrGu8`C2ztd8q`0nhE@MJ*PIO}obUsA?xBd4MtPWAU(#i(CMH8*9Fpbtzjl9+`m}S'
    'ysHk<4;g5IOK1`jCOWw){6YdNO)52~WoWUC!e9d&RMOSlzqw-GSiNq818yB844%6TF|R?egtFXZt@0f`h7Ibrc6p^Ie6QbsFzZry<tHxB|*IX0E^;#5Y'
    'lw70}3iW<T!{$L>&m>b;iiOBO0z<lAtdU{bhhqx$B`bwa8F3vA8bB)K+xQp2N#}mxLi^_Il$`28+@GHUiO|0<Sp5JYHKHVw&x#zR{p2td`d7^No&rGR_'
    '7s{;1o+nC8qU^bNQn;!sXP1h;Qu?n|uGDNEdY&rpk}CVhsiMb2L4UI6r})4Pq76pBQIWMY=cb#$bVmjl8Sd8bL(>Z_du$tyWn0$(gaF9@f*hggxV+vk?'
    'II|=+;K|NGo7-ugTN1R3kE1~d?yb)SUP6ln1-1fC@+BP`6b*tF0_Keu`m59cqYf1X6%PU|KIhjD3I%W)*d2j@47*0*cKT7B6ZaruiGX^S$k#|85P}Y*d'
    't(!fVDNPP;7n;47UE&=U-4u0|XQR000O8m!WB0NP-?Q+!g=;(K7%5Hvj+tWMOn+FLQ2TaC2yHbaO9gY;<T}a&uo|WOH(7GG911IAbz0F=k^qGG$~pVP!'
    'HmWMeLBb8l{~TWfC|$8!BE#N00qLeBR4LHIhLB-&ywB???BxQT5LOYTTqnA~M{mzHEh|2=h1^{gmIN}S*VhNhd-(_K@KQ>SXyzjwb5>&;@d><&BEn{L0'
    'mzMXA0GMmq~v(2!TzueBQhFkf|(XX?GjBkg%t6{#`TP*i(R-609c6PhjyZW&A^7X}Z&+V;;x7cpK+pHee*MmHFJiR!&kUu=!%bvp=CmELxOg^yw!rDV)'
    '<iCG5&fxf8XSWaX(C=MHt~)HL_CD{+#LFbsK2}TuCcaMvlPHtm`^q~>CO-7uW0Ga!Om7S(MJA^7DPdB(!(2n3u|YO6DdWUqV)_&@5eahky$_gpodn(y3'
    '34iZwwOdrQtxn1PFNP(B;$^}0+$s`a$h-t1SzVZ#ps>HiPee6`xHo!LLVS?fdsL%JCI=1L4gFp^%jB_NZ?bS943&!<=%1wNMK7}AzOh2CiWJSDiWk>`v'
    '6^6ksxK;XS`_@2~skB;4+aQMR`VE!RRSggn`7?^DUij8blDRt+{s)LwAQ#MbcD!pJls*V=7gUF;w{r&J$M@Nh(eg%gSFZ1dOgPve-htAbOB@vJg+KA7w'
    'v_6TPTjRAO<Y6D7zt7H7K97yKQE+DNG^PPL|Ycm*6Qf*b5F=f+z4C~KU-*!5Y?a&m4g^7ir?&SC7ikCSs_PE?VTbEEiz$C`+->z&B&<lGojpJjiSF;=&'
    '7a&8P&_rbX#^iE{<IM-Kd+T&bNHP(2XD-Le4M$WZH9uBw>FtN2ycp_rr^*EonEb;w)#zY(fFXs^x!Sq=a;NcF~3yuQY2HRWsBaJJb0jGnqWBWUm?bI)z'
    'Z1F3+Xn?YVcsZO0%9gVVV*!6py`jIVPpN}7)64!5MoyM78foOvSCKQ)2Ltzv{*f;s23(;$ks%<CAp1u|fXE53h`Gfh*2DkBm@rP1bBj&%vt$4i5`Q9pv'
    'Oo|CKiP)}6x7EFKq3QzkNA^K??pD^N)(X67<-D-1u+3or5MjxBdAo*Jx5H5B7;Io?W?>m5k;IR%RU4YI26_=T@U#cPJ*t-yTr`!C76h=pf_TP_jDXWiB'
    'sDVNo-)y5J+SVf{8d{74QHf2w6C0VRVK*VoeKA=n5DDa6piis$&EV5d^R$=Mq392q1&vhodbBA7XdlTn)Mg_Nq8H2KrCMxk2Tz@b-#mLWIKItDI&T>H&'
    '8O5yQgWs{~i#L^h~mz>LU;yH}@PZQ<@!A|!BG#l-3}aQJEr9n8YxOOZFhIdJ(RYJf>?U=qsA6c$bo=g8auNl4DgT!tj%Jb~T@CLya}1;^(do&#@@2#UA'
    '^*N58`>=x;Ro(OK?{BTab54^uPfkEDZ1PymB{J$6h3GN07<b52_j1eGk2Cugo0+z025d*nc-3>tir!$<ub41q_hbE`_4VllTmV{tWftc<vASj?JBjN<H'
    'A~!H1PVj}HZ;gl(L_6?*YceKsD5N$8lR_`BM#KqtA7Z?S<HZ-@Ll%hd1$h$=1?2l&+Z~a>ONO;L6rk_LJ_dpU^?gwKU;+D>WNsi5c<a;(0DEub8731s)'
    'QMLyQvmLrz_h_Rz<VeBgC+ssJHZ!yCg=nS-vaU-;2h@w^Id3e3D9?vqLE=OP~VBQnY94>4%Vi}Ljr*i<}d)iGaBQzLRDf$1hcJBl^_|w9=u5=fm=IcF^'
    'S{_LbG%k0L;-?r7|Tt8Y^c)hNH2>^a%-!#uBG+Tn1x_`&UjNfk2TyQ*{Dy9E^qC0M-u165c@_I2a2PM2WMMAP_elj0M@4Wnru!Dcx`xB#;z>^x$Bu5F%'
    'a7S-OjYxr606!5vsGh~Ah}VYwhtf>>~{TyP4w4wl263hlshLGXZw!g4`K0&=NyNq3_TTy7ISQ*;@;*SV@&Qyds82;lP$fXj=qfS3-z<wfy)1pt>99l$9'
    'ZfXj>C3RwhvR8Y3^4t!Kn4B*!RxV(@>!)0(-ut2g6M2XNM?grrUjSD#dmlpz2AOV2OgEep&BtSI6J^(H+??a~maCuP(jTIOrk)i1TTwc)9;pGBDCJ@FQ'
    'fXmgYTLZXUO+;7+;KGE|3Bcu`5KhA5I|vyG^au$6T$oU!04@iXjza-lPW(FZnW_`|TQa%~EWiP{Tv3Ya09+0v4LM$ud?a&9%<@*5Q(~GIJOHmt%=5?#A'
    'gD<686g8=keHP-vs6N%k(1gdq0j&quiP34D^ah@fUrVIp`FA+!)bAn$O%lPwNq}i86lH08ID#e3A7-g1c5l2c``9cq)c+9Og4cTfSP7*08S}R5iKSlm4'
    'dLaX#zQr`he<9U<Uy-lq|+65c3@H126WWY#<0eE2U3h2wo(B6%r@{T#QnKBe<N}G8jw5iJZV#D0LtXXaaE5aROsWpwr1oqy#$YA!JhGum(O#DkXIsFy1'
    'Mb%0y*Hy%oo5!YP}w>N})Ul8ops<Ws%{Iti6P0D+5)N~~?U8!RU&)NnU&n`#sSiupunTp=@cOcD-yL(PHe(-m@JEm&M>6$o*M%u4)CcVi9W)JMs!#PnJ'
    'Wz;dE^Mi%rI#g#dM-l8PE!U+lj@Gd+9f)!Dq;Z4D?qTndc0mVWM5xES-2sj@OtvJ-1wlzzi)V5~rBQS`<0f{BSM^REBu|QB33rH-<II?+<#DaQ>MW10U'
    ';AuH;q80`qtR9I4J__}EB$k*9bFfEZiIT(%&w&I|1h8~LV!?#VkXWLyjl>CGTVvZJvB2YjqaKMRA{pUOB$g;hK|w7g&?N3{)`{V6MVImASpQ*bY`_9g;'
    '1=lbSpmY8aVSf`99voDwH^crc@}|)Mm`huO%3lPU8Z-k?gn1vSqS1B)I^3L6dAz*8U<xz6$OoAVz?|RJRyDpG@8X-z!QTRX&#qV4H=Xc0UAXQfQ=MumH'
    'hJ@J5~s&(4bK)vz)L-sM-{PMze5BcvXN#*+k&7tam7=WiV-G3YNWwK!**`D8dZ56QI!~=>rKlK%=Q@y%nHQ!6@yWK%;_0gn$5z3e2|M5KK~k70Bl3*IC'
    '&h@!B;yNxYUrSuX&uQRkp`wTT6dxIcJRs@mWnj|c9ghHn<QgC+ti3E&Q#T)PM~eNxt$gIb&CWHtb@_DFy`@LtMBycSkN3QD{dBqzfsUQ5i8>MCAqm&g{'
    'THidNy+!6TG{swSIk{zvJ3;AeJJ__8Cq{^7Hz#U1d`r5>6p<iUT#A}Iqc2`PEuH?$ZYouNFMXbVlA|G`+VTl0T!2~9;z#YLgW@iiB5h&9FswO=Yui!mF'
    'RjUFI_Jq_%r~!KdbOP4B_gWmnz}^SNL~VaSy-|BM9_$Hp5>D<tX{$CEJlInZ3}MiMJ%ylF@L*5kwFGY;$lsjq2KFR@N9XfoPgVJwCwr2-0KfL3@i!0|P'
    'xgc|iMxS4C5=;0_Jp1Yw+4HX$_S3+!Jb4B%+4O{$%Cn41F$C#4$F3qS7Ju(MDjPo^x?st1lQUm>4aDn$=?byppQjM9?z}G--=2NS_CqO8pDGYf#zx32<'
    ';{DEwvV?kknzBWu#ARFjXz~d}4<wYIgF7t{R)Lh^{JeM_^@EuS#<~vwBhb;d85abwWO7G=(s$i)X=Uy=HYSR6Mh~(>9=IR=3)B^P1JGW*TO7k(;uyQ-V'
    '$vwq~nDo?)(IuSCc*&tbC!ojBfMx5TLbuwBwvq}P5)RM2NumtaQ-cxLseB*U|1;%nQWYU*XGVpbO*DtHBUO^jw}uUWmQ|BP9k8)#NnZWY)vjIF(^J;O|'
    'k2R03zDn77l2$$g<>?S%Onlqq~0eA-ig`O3_m;ohl!!|Ml3Ka;KA)ruZa~T7w(&qvLN+#eufdM5NExdz(@|t=>@Y)Zc)Cee4RFo}5Z4Q8F1q76&C?ZT?'
    'KmoJ4thBm`ld3HMW=;kacpoP$h>E@s11f0yC@`SV@#CGKkpe#m3@CvCT}HfW4NsOMSpzDt96^(v=KwLqDJUDu5xfE$no*qs;TcU*773aYP4f_FG$LK2Y'
    '3jJFD!&9SM%!AVY<=3+SA*%JZH@WLWnQ~%uy$ncJGpiz>axOTQd@>|*!~upgSJ6r|2t`hj!4&7MhTqJE<6%aWV)7M0<VZn*YeuB%xm8cff1Rm(RSlAQM'
    '<s<XmqV5tao5WBp2R+8CyRfdUk&-tJeru_Kjagt*o8)g|)Jd^s30pI%|fGT3IVdMpoAHa)o!YmN0A}v$7V}WqGGn)&Q=km335OX=R=1%8`|I(1t?P%DU'
    '>yF*kV4_mQ~)6G+Y6;I*QPbY)SM;ci9K0rFPV++YcVQFDWKs3TojmLVZUy0TYF8R^PF<22Hh#p5HNVFygh#mL+cv<{5S4e|=8cq4OzU@y`^WNz@ytJnn'
    ';W&<`9wF{`N)W|L%+=X|D%8|eo$-2D4Z6xc$ZcvC;$yv?FD!F|bi>#81VsB)XoU~?$WL;ABB;Ik#2<fA&l3SN4vPu>~`Qd<7at>_*e=F@1LC#uP(>|FV'
    'AJzR??V*d$uLSh?b||a8*4K`#_RxanGOx0MO2pC#s^K&I$ffuenHSLPfJXT_4n-6<U}kb{1L!~?nqF0nezjxFRjYXp3@Rco<VF}0`{_o%uO#LLz&a=F='
    'v6vw0_zI!D%9zfII7<PQZ22v=dy;Q5HuU3u>cp##wuFknT*m8R04cCqfqhfBVtCOB4r}gn+@AKo^6l}<U?0Rw+4<DP1V`3Q=k<{2J)eY09|Jw9~mwuV4'
    'f1rFntE{p`ifoo2}P|g6DUZpe<X#V7BbVYBipLd}y*#Y9QYT(HY3+TGu%P`OtCYYqJ}SJ{KflH@LObGmuYsvVMpG@(G=8_6b_2ds;4PB}%x36=M5-9Bb'
    'Cf0Q4!=qIDo6m<2f46O3hy*0t<E!3qLP+E{`WV&glB1-NNF9(H!C0#`E44}g|QJG)*#XK80wV>_|4o3zMH+Sv`-Bw}YbwNG8_?8epsV`o>A1hQsgXE!S'
    'LCU$nCHf9q$yD$l#fdtV~RY^O$`VBp?vx^2jT4?O-2F+%Pon0{_J4V{s&FzkB%mG`ovx{KjvY_;xyCMI8z5{NP=XVyXf#KN&49q_XiKT!D<F%1ka&0{;'
    'AC!Vdeg<qG`;nQFPYP*#Q=hfHm$*UE&y<Mm<oTVfs7a7mQuBiptlnW$Qa`V1M<Rt*_UQM<pdU{GkcnSd@u`~2;8o}c;C%vH8iu{3gv5eM6!|nEu?R|%*'
    'r#ap4tI<EsO@yO=(G_7PfW3W)}}^dP(F$hGsRZd(%LcUGnQLt{rs)F6(y%Bf86gj+u8PEgCBN)0(4jNPCtZq9F%cT+W>abfgJ~~4S>7dRo6b`=m<{IT@'
    '79PY}4t;wGlrra3LGn9I+qFyKDxxk=u>@ZWMN-xErP2D31rT@!S4b$5GxLl6QyX-645*NZuWicZcNNA;?>8v^yj3&M3Px%I=J^JEQE*D7!Pt?u@cKqwL'
    'NoyEDq}jJi9cj%TpJNma7_#u2+qI-(h9oAPP2vsWGe#-gQ-KK*T!igkyomsLmmV;vVFVssRD568S4jkb<EvB-^ev=flTQAcTfmMKq1c07a8AtQ0-#k@Q'
    'Gv3s)~{u4iIS}yP_skg&Q{@pI_hVEx^_r-66tg-uc-?@G7<xg|DwbY+l{jSf-b^LABh1}f3A0LM0^#}Y6YqlNUuGSy0XtA6R?=d*)4$a5i53)G#j+V=Z'
    '*=_gn<Hsh-qg`Lid~fw;?{1iF9@fLxwtaQ^{Nnl1lRs>}`)l`VSnu6DEa&TCzPG(uJ#1#n`QBoAGrL-BXUpwI4~GICo&CeGx?kNcwu|f8?cT85F18=`z'
    'F80VtMzuVeEUydljX^=h_T;2`9YHEe)sf0B<t>X&;BM5E*H5c&~_ph!I$Gx!9lr91$Otl@6QFk_q*qE5yU*XKu2P~JH=J@dH!0ive8R<20(R$zspvyFq'
    'gevVJ@4UpUG8ro1P%2%T*3Wy8QkG*<a?zAPbMR1bMqB!|mI}!(FFxcTXN}-_F*HS*I`1J^gLCzTp%v(mflNceC{`oxYGLYJE3b!sd9f?qogOihetN!S0'
    '9kFkHiOc+YNnKU?6pb$HwE`}J(eTX^a2>3TKW@({dyce0(`VozwGJ6&vV9%f74g(|w|4@;2{_o0#Q<*>aO);M<4sdBobhs{=B0Xphlt(s8Kl+4!KhqvN'
    '&s<G~Tb=P)>;=1YW>er#o|JLTSUpR*%yYCmvwj1=>P2UakVX02U`1wOze9cvC1p#I+610mAMz$L{NOTz)WIBxu3f)Eqm5w89&d_yaVCg(EaC9FTcy%Bd'
    '1$7}AMRg(>C3PbiWpyMO6?G*URdpt3fIC*d%3F0P898++8F_Uo8G*O*43bu!p^TLy{&avNa8;f`qRKPKQh5d`s($w8r&o?h&^cn4gCl<PeeA@lNxh8f)'
    'XS(-y^M<0%cxzwj4Ia4sAs*5%GS%MalMRc*UPATy^Q+T%V-0=jCRn=XbZiJ_Rz~{6TOUf(aUHXy^QwJ%V;CLjCRt?Xe+&p_R`B}Grf#<)5~Z(y^QwL%V'
    '<NrjCRz^XiL3}_SDO0Q@wm-91ZWcpOTFP+gtn&_hqN4t9yQW@%-|9dimtp^!x{GK6{I_b25MQYu!)33%2}SFn<rcT|ZoJkx3AT|M#7Q+ai6=mdlmULH_'
    'N>-@EOHd;CoJ>G|=|#ql#eYG|)|VA6pN7q%RlEUvG=68^Zqd+6E8bb9>a<mJ)D$=S=xXJ;pue}8exJiNWd{cmUUyUpQZjs)~(A$jcm-R*L7h`;{zr|xF'
    'Ey+52x-o1O*zw`ZS{dTh4OmOcnKXq;8=9kr%SHdMH#!awba=ls){mu67cK6);di&|I`-k-{ADUlJwgUH?$!^_x{^pC@BW|$0a|g2IVDn+QoxML;LbC7l'
    'rA|J*l&m<MU9BF*BR8_we6`Sx@B>#<CZAghyFl{8(GuJPHilxqAo=Egww~br^<q0<bJ^j`TYo$GY~$~~6;&KwuZP)orIM}({b~5)Q?71>Ct&%eA0F1L`'
    '<YxO<HE`8L1@QX7z7k83_|?l;A%DhFrkb6)J^n|#b)y$6y?ir^112kMEqoWeKXw6{sMQCxB295-%dX5_T9IWKWKaW%O|J$?&}_T_T*oWpI%&E{QIlpM_'
    '&2t=;CO4eDU?iE>2$H)n2^1#L~|~;>$mr{}Vg?aC-LS==2}6*Q?W`moNVitN-g}$I};6;gp}n=kfb{K_|GzhrO+U)@JXUyJ0(n`9KkG*25cFHTv{${m4'
    '5oLRC4ro{1eN<71OyIk~yr-du|JTn?MN4dU{@i7|cl;_v@pF=qFR$y}UfxLYjg%9A&X+hOxB`1h|n={^^2GrxP^|F*b)EZ^Ca>E+Y2*Do&~$!C3)0`d='
    'wULBvGoISfd`?ursqtnyNQ$f$^<A*dI{EyF^oj&{g8B%~p&rHu=pFcgmJb5|2IJtNYFPLJrs0dnXum1Rf*RRB5kN+QT+a8==oWFj0@%sGu&piF}<G#X<'
    '&0Hb*TnXU(g8xSajB;-?tQW&(e{Vj#Sx8L;yqhoHyphDT_swP)_8uD>Pyd;9S6?{=w}!7uFl=PX{|!cxG%DKN9BKQ0wc)9ogLgvN50<L~oc5_;{PE|p$'
    'is*g7Y53jfBace|9=5cO9KQH000080GFX@U3Y~h?d1Ug0O11w06+i$0AyiwVJ~xTVQ_P3Z*+4nXl!(7UvhI_Vq|l2XEI+nH#lQ5GBIXjIWlErH(_NmH)'
    'LZjZDDR{W@U49E^2dcZe>$jZ<{a}eebV`c*skaV4HB2qA6+I1~pBikXCA^2^s7pOc^k2`cn13&)BTE+6(7=$KT!f{c1F_C=rA+VMQbBd=clg;b}&Bb3)'
    'd)?zn9(>PU#1N4CxKDQlamCKBw9aY2Ml>cXZ4=j?Y;5>Zs8-N^+PMb2$F1$I}&{+j;#g_Nf`-j0tA`!ipR+p-YzUAjK37t8U9RrAF`!4hk@Q?TAExOvg'
    '=s$g}m0Bg_NScCRlX%TO9_>+QIi?Ob-dDhTd+T}r8TOGISL5urG1&J=3=z2KO7S?+^78k)9E_e4X%}$VGcAH$N^KP?p;7;gg{wK>MQvIO}X)X@^f#geU'
    '@VkMp_XEFqG5G7iSN8+QMxNW);6|R6;e$6ut*3#<hEHttxp_9YsT;8mhQ2lO?@Vs^>>mf77_%fs|2Q#pY$^Ivem~TWG)y!Fdt)?hMcyqQ0pv{}m_Y9U{'
    'm6BrSvU<Kf}wv6Km>rkNPZ^B^*VnFLJ)?8IRRo$4uQvjPu+|;$jcq%un=Spl?nV{mSsm5qS?$Nc^1&nrNl-3gW@Hr`MDCQ{6pCPuPzPzAV&xu^AIuW6N'
    'Y@6Pk-^M?8B?J$ywUg@<}oJIR@}2&^bUDsm`wflI>r8`~^@;0|XQR000O8m!WB0ZdEF6CLjO+0!{z`H2?qrWMOn+FLQ2TaC2yHbaO9gY;<T}a&upBa&t'
    '0YV`MWkG%_}0Wj0}BWjHoDVK-zoE^2dcZmoR%b6mHT<-fvu^Gmi?YCe1eDo?g#iHWtLK1!sgCYz*GZL%e?LN>YVW^5(p|9$T{2Y5|;?cL1OlvM`M4-dE'
    'i?#H?J;_)BT-!}KVtK04LZ1SsldNN&HuXj6ZU9R`*-DYpUT(7^|T-z_-Y}VKN&qsIm!tHizKRuatw-5Ih8+&$gcK-a_et5XE_coUpZT9);T%Njc?%cC^'
    'wtxS9_A{pV+xq&!Ui!!6e2UXEm)a_XCuk+K>dHH|GFoLdH?|5|c@<{1s#<eotEuJqEYEt+d$yd$43#Z!tvs&+XUGAqFx~?#Ij>S-3$*06YMCw2Qrs%3C'
    'uk`~3un)urOvC$HDu6I%gT}J0$N(?GZdr6&u-;8X7Oklo$6_{co$YlEu+PIx0*?3NoeI&=8B5Z;zJnk87)5gz7xf0@o8Rp&DYQ}TKHZyT8#9*FBP<uc@'
    '?=Y6|~qId}0f<G;49pYP5uTtgj}t>Z<CQ(GrYK?L;kTwN=s#MoWnEs&Ea}XbH*noo_Q*Wz{%eGg?AUedn9e5=vP`wE|jsRk@-DT59V2(m+eIbLZ^MXo+'
    ')J&D`B)v_$9nxi_OF`g!y;T4HdkjAynqYsILO4ztk`jm}os@@UmnN^BV|(Kw_zwv3htI`y8>604nqdV&_?0<UMF#psM0(`ZS?1>C_fH?*=iD>$PiIomV'
    '7=NwvTmHD1CT9WN`4r~RqnpPB?(UNRM)w5-^B-?q>=`vbzl{JIWl5BlZ?|~MRfTW(F1$%9?gK>RbIV{JjA@ZysKB29O5d_&Wz61Y2aAH_xdks<()2dis'
    'h)k@jVr)v;b>s^Qt_g%KqNy4&H>EVJ8c$hLnMsqRkYq$jT><5or&g^O6E%|+oF=5fc$H@YS#WVxv3rmNAGog`au8<jzju-zjOif*IC&$@LjtOCkB#pki'
    '^bNDK9Iy3xU(K|*gQ3o2j|b8%ktp+*)#{<1=riW!RQa*d6VK~;{<Sg@T+9>1n|3UszV*zZf6zIAG~g|49*YW^b|=?fXtZ=!^Q^8Ych?s1k77>mZ3?Wu_'
    'qDhHO{Q4W4%>-5-IbEryYapkkyFw)=YViT(9Y1^v8N_&!M4Wk%j=*#;Sc_pQbvvB4OSp-H@E>kfVfoo9zkMg_=h5!WNj=st>h{BOk#Im<W+iQjzw7?t='
    '|5p`hHt0vMxHFXUM|DWZ;j_Hk8=9_W7TKjwwz524V~K<@|RZ|FYgR#!C+G^?wo&Fo$1RoCh&f|;?#Ij)?ompawt75qt!YSQZMUFuWQH|S4o>J!fky42g'
    'bLq|cA`Y2h49`z<SSWBfwucW_HpI4p*G^iOH_J{s7v#_c*Z*w}YXuQ$9_R||})STc+^oiOxXf<k0SI7@7Yfr!f(3qxxR<oUHtJ+V<Ahcx&iTxdvB^b9r'
    '2|-r|qq;&;p(=xkFi3K9=VQt$6eSuOAgMtM1vME$SE$gEkOOKaC`pV)BL^KBvXfSHG#yEb&$-V3d;$tgSx}WibFcwIp*lj=7CGpSV0%Yx4CO)d!fr*@M'
    'H&ed*zr>hL4QQ!M{0K{kZ3AfMi%x0r_LuZ7shUhMhDiy_RY~A!B|8`!FJFhG3MS4rXtyn%DAHuc?J%OBpR%#GGVYie%K{6B$gE$Zz>5aTd{3sXlB;!jn'
    '-(yD>lu6e!M_s3n4-;YFnJTLj&s8jvauZQs0^hz;;$DTw~V2CtbC-ibj#@2288iG+-!oYXU|R2qK{%8$h*-9mD|CyRd*5fQr|I8c?88^P2XAcB@pqrna'
    '%2mAcn>g>vL6Fqy`qV$&c!h3_<}bclJyrg<Y2M^K@F4PnwK)<DhQs2M6$u;E1VSA`CS{E;>&VZf=x0YHIamcS>VAj^E!K$%U{+m(vgm<`8Psfmk8Hl9c'
    'o&onnxy0~epsEnI{T8*(wF$bXNn*HSfS$Ye)vqB+LUO;uEB@ha%P|3mOA%7&Y?o4K@P|5&BTtezd2Yg(imSbt60D3u@z9UmXF(;dpOp%Bzns6zW;0z8m'
    'HiGmhnHo!QW-KaX8B1_2O3EWmcae#vO)(xNa;8z4rA!^IO4m$|X2s09gIv<C*oHvgIT{vIW&{hgET*W4|7co56--S@x|SFLO8{U7!#Y?O95Asa%!~OFt'
    'IW}wn1Taj0_hHOGjX)e$HBxH%GYQ)2yasz!N{ONfRj_#*c>g(96}acrd{njnrT>N2wbLx>$?NDA^_@QG<X%TkA+XX3>)7mLknXGrNarrz(PFXrBPw@&~'
    'SW*j3m)?r3$7~A(qw~SZVEA7+}W1^5F@QX{8-#g3Og#I0(Vgk;r2JP>=-=U7H2nNe=88)GObCL5ogm!@;6Ov!wKqVA7)R8xNZneYX&F1PmI3YWfHez{E'
    'xt6v}{dtsF239R%z(bS?b@1N1Z(bPs423E`Yh0RzxEfRTiDp>F`u%xpx#4mFazBJgOC#q-p#K^8CZOz9Tz<T!MbErj)=QxKvykv>5%{g(N}3l>5R23b5'
    'uF{48e479>D(;o<i;-EI@4g?cGE&<9L8=v?D_QqBUP)%O|8xQpqXz^U7ECgCSFLNGf@yzjn_7AdnjS3~m;$^{9TD&Cd6=d<MKrt<zSq$zDEgnoWAT@1?'
    'DHP&s*^)q^^+fQdk>J01+7dgG#3w@q;IwH=%woXkdD;?FTtH-Awj@!ddD@cLaldRyqMUl!5}kslEwNL;$$8n5CVzsqBzH!lEy+UaUbduCD|^}!L=@N{T'
    'NVnsr!A?SM`=r{Ag-q^F&>39dfJjEOypx%0XPyLo#5TfC&am)?wF4Rx`8t1X-mx834D?c;LX#PBq?uCTVke7qtTYgAbZ*pGho~uDB@s80v+$6h(i@qcq'
    'n4>8$BjaU{rIjp@@U&H}ELKJ>WMwmQcje<cG%yMGQfwI|W4?O)$6~DB@^GLaH-8cNt^`dZKA0Kz+t}Vh4PIhc$O6iHFsVUf^MQxkG`6WlTYF=3&`Z0pb'
    'K1ew{+#VHx*Q=4ALISuF6dpzztP$rON<gIus6X_VoYdJ3@vc2dB@GR;Rh<zYeLvs=gPjSYhKO6M-{u%I^h4nr=5U@q{mOy&T{0}l(z0piTVGEM|?2s*4'
    ';ZmAB-YtnI8)JXKmg1pE;mn;YX2Kr=CQKpbO1r&68Wp-F3hi(}xE#9GDW^e^37lvcTCD1eTLMVZ*nU~=Yxq~0acR&H0dp3%^Y-&DP%_a(HHRrybLWeZ!'
    'fEkM`60^WiT$4c#jKp;t97=~@0i)4s+N7RONP$pvpw%>n1uF+yO*oW%()wY6U1&9p-?#^~nqH_g&}w=ilu(Dz@<}|nUx8K=ZUm@i@XaKcTA8soQ@2#!3'
    'cZcdnw~&t0|Re3A3R~)%~TrK!?0Ua;4;v6j6#Ef&SR33m7qGaR|GCWb!NDaaK~eniiQ9t!mnALP$HDF25?o%J)8p7nFbP4qX$$VW{GENHh5+PfbOW+B?'
    '$=y4NWBEWLOfBkk?@)A|ZrQJQ_iU5=@?Wrnv_xgoHBBCkY9~{-g-V6pmKqG{TxCo@o^*Njy`-6yV;(Gp&w0nrfFUY+~Y>nu3JmoOq^H;h$8yb>{X~1~G'
    'jf0BX>jF3r=u!OYR91}h;!B6>1*0h1@9ClgJMM)Z_*B!(av`08QE%&-ACA+zO$gBb|SIDP<O!QB=Ge(Bj~0V;z)jjgQHbQ-0|r>PpH_TZ6PO(7#9;6hY'
    'o>m+4`uxjprxCmE8g|v%+Rh1)Ogsc*=gS7~)I-)GXRx^*^T15(r{(G6bObQAMUR4F7i}00GP`L<T^}um)or`+U&PIVy+jTx-c)hp|Xdyp2bhUwF%naVm'
    '(p8z!yICQR97Z@RqS51qKs7nE-oO*tdQ-O;I(E1Ze#^A=vvMR^=EE#4CY9e9$YPo1!-R=L#~x1u;>xj85ZUMyI5wJaFT^aTE@sGMWSS2rw>;B)I7J;Z%'
    '?HA8e5XTDNN}e4fV#!dG#^z0m}x%D90DI?nvbR+QfBf5^I!=OGej0(8#Bbryc?R(3^7BP0S<evbUcN7GsJ9op}BU)6K9thZ=00?$PBpk07PcQ&GrBcmK'
    'kz03iuAi2cZd20O?^skc}R}V8oHKgAhKS#6C30q2$;T1Rx|h=CN?4lpEt?Qcbb32AP6V6O=Nd(D?2mP-cKs!R-_o!d5;Bf-3;d6c~sy_L}m-2D@cBxRn'
    'i1P^45TENDPjC@YX~09uL)^feBck^*p=`NVgt$tOV^4<?tG7*jxxN=nRB!-yBxflT9*B4Y&1R~URV?4qHY<P7$-cmkQI#R2;xRRkSYfc=pTn)xIsU<VE'
    '<bVXdx{>;iNh90$4Ua^Py=G5a?5Fu;9=>Ya+NRxTR(xhx=<`r!xyq<vv5%*RLObC>#o>BRU%-NY&G%Bff=iK?Q>AlW~O|j7`$A-@LoSUM{RG2fN4C2e|'
    'Ni5^Tc;>}F48+W($3d8FnoAFa(AA$ykAzf>LXK3d*;+;KH3ndrR}4K5HZuYqduU{44BYSqLp3v_;Mw=cFXQ03XKiOj!ixa1wPMdvaD9x1n|bK(<qp-jh'
    'T5|$P489ZBN-o<DWV3TX&z?EXd9<Vd}BQW_nH6)aV(Ch=y;)iv{sLgGng<dLN2Ikt$c9etW|{GXT4X2Np<dw8+iGpu7}El3?*abz?<B81<PlNn`5-x_Y'
    'j>^xP0z2C|+)BpVc$>+>D+T!B*~)Lgsi+J)5#1%&9794n)aku17H#C{x(nP!ogjikl<Ar}tvd;i}c6Ml^T|ots=M!tT1c?X3<#BiFtNFaz_HVC2$~Ut-'
    'X&tM$mBUzxdokwL$#^lGI2LFOFYQYRk-N7<jMcW8g?03y3}r~q&pX@9`nNK(=Mz~g7PBDDd9A8CJ(!DcslyV}r5_r=_Dl0#n;h7#(Dt_UCk^eAC#(h>?'
    '`Yf=`a5Vjf?!)S%DH49r7!d8S3p~VPWi{h4ru+=Dl`6z^~h=(vYm6W$21%$1LD?_Q4p2W!PR)tilZUtNWXoRiEnqpGH)<V%Jgso=Upsz{|!lN*H1riIT'
    'N>c!VjR5Faf?%uiU<KluqEhGu(rOL>b#H;NngbvTAgc+%9;AYKBuR*@@ko&2MCOV|Vp|2Tpz%md7NHYc?u^84^e9J0EcZ=h>WdyFJaOi`I%dmkq<$pGe'
    '(@L)H0P6|gb{eB@fh(AM{61y;I#1=i};}N7{MWIcs2AS^t=+^iHg6$5ojfek<`cvVgiS>!VrkLV<4w61TqsNwbBIlm8W9}q$m6;Lm<jFx9a6DG<FNhD|'
    't4e-hlyZwdbNVK-N_VH5i%GeS=2tBg<^H*0V|6Dk7n|cY}^)11p7!MI$kZ2rFnD0(;V+VDUtw!5$-$#Q|WBkzD1_u*P#g1t?c!t*Er-xnn~a2q;(6PJ}'
    '<!^2%Z{DKO7HrzJxM<?4C{#af^PZo5VSA&5iJctE-GnOYzvF3*}S6$uJ3xn`Ur6QrvQvD2x`3`z0pnfbLM_uX2&(9r#ySyvk7Zj4<|M9Zi$$5vX05Dbi'
    '(94phx$Y?9w#40q}qx`e)7mj4MeZO!dljr+|qo8L|@IpJm_^1I28nRi*HXXOMij03y(w8C9*bS#*q)pde=Fl@W+Ia=udJ1(`<Sq4cQ0T36NH$_`nnA(0'
    'sKn<SI%t*gSNN^xb!NujitFoB0B%4erQUJ}S`5LJF<<PLpP0Uw=|%aT^O)&IAzsh5(~m;HUZZ3gN3Wix#~YOLuF8)vy)nnG`*oT$X5bf&LTgsqfTL8tQ'
    'hnA6QmZ=mMD;7zz7ho?RKZ&UJweVKEJHjvSmXui<+pwjq4g{yQ>gS)a4m%0q5_oa)`K#c-LiZEh*l-|fyPDlPwsuC(Wq@)PJrOgJ<mziL*P?|>l^(Pvp'
    'TSQjj#uR9qMZC9y4zx%stQsL(}NidnLFVy?WzQ+6f|_?rOEUhlQwQiFcfUKoi_)qCjZJQ9zpONgJF}Bi6Aeo_8veB9DpPMqmN75$KqfpvorBF<lI?X|7'
    '+C;AliS2$4d?(N8f@71;1%8)ou)_O9QQQ2Ne}m@KUPof~M6h68ydqNvit_fD-W2Jq@xWF@_3=omm)j3B_&g>a})Ko<n&@nGzU>zT-7(nmd++lk6dI++l'
    '`DaTwMLhnpRy4uq_x^>71MfCmVB!w&XTWiWkq^@R^kjNbcD<iSoR-|wNEAaXRg{#JXvyj4NJXLjTbe4JTH$$CTUiV8WT&dqI!wI#%xlC%gd0emXUWP{('
    '?>sF?R5F`+E$2IUW@5|v&@bhAS`eiw98+5^$Uk<>ZMl&8RXb=(7fQbb2u+F0baW<}qHu`ftL1fhjt9mh1+0s%_Jt@rq^o@)>2j;nGth#RwcabTLtgXMe'
    'yxrdAVZTKV%&_m3gel++UR7J=vp#cf#^!(m`d_DS)QOQzt<g`5WAbJi;S_mxw_=2M1Rs9qM%HF={rx<KiB@=$n}#6f|C8YUvgx&-*vZ)i3?a4>UU-?-~'
    '!W$Vd?_*i;fs`7k*sJVe$g4K3~<<S~4=z7aWB%FzfHf88PjTE4dt#dH)f4WC{bBLb{L4VR)}3h+`51R=27JT3mP4na03nhFN1)t=8#TQ%_wXrTcGdr}b'
    '1S1By$}yll5-r~7Xnk;at!Klh8TbpM0zw3MC(O?p;jk!gp5-2WsuP$>=E#9@mnB<RkOQ&Iz|GVYn~0_46vhQLdAEtS`({T3DdK&1a@OX&zkTLUG4o*-J'
    'JjIJOu&crQ|=(8IkYxOWp*0UfU<o&5g0NVPc?MNGkOU=A_LL-M9e`c%n6rskfI+diJO*_x`k^*uH;#tcnNHVL{BkDu|NijL4sH_#0G44H6PZ{$<io$YD'
    'VVI=2T)!%o7??BpbGiuAFI!UEB$~XQ6t|Y^ZZYYk(ZflrXXqD8)iWxfR@6$}{p1WZe6^ouq5Rbz$SRb-+V5<X)Gv@K_O7d(uE+W*f41|)J4M$y{b-dQS'
    'Y`}e?zfY=E~%%^NK2u&7s_b$tA~oj=Y9`@vRUZxndT$rF7o)~fLbew2b+3kWoW9#Eb`fSwsPKE?$>HEF961#BPesvTyuuPD}beEFNznpDq~Q1Wu<!`bm'
    '3$0;~6il)G<odGvZZx@whMO)T(mdnr0tY0XZf;z?}NoQoFgXKNGc^?Jn(u(r~WdoJ^@}6y2j5+AXW)q&29a6SbRfos+5Ef>LQt**ueer)Npm^7_@PEKf'
    'o6<?c67&oiS39VrRO-EYu))Nb&g_+IG&#ikotUdVk`t7BMtUdrpuO1V2F_FT7y%Ivugbd=ijgKFh#XTc>U_d@I!$7<&~YHy|YQuis+qTQI^@=-9=>mpb'
    '>q#~`Ag(mwYu_AEFaAxLpaa<s6KR%gu`}O`|$8TeJ*J%1~8Xpj_3RBGr<c*cQRO09ksb!5<L#NBBUkvTFHfqoqz2=tM@&>d-uf?U7mReqFWvMN5xMdE%'
    '%;A?g{4$4M=J3lLewo8BbNFQrzs%v6Il?kWSmp@J9ATLwEOUfqj<C!TmN~*QM_A^F%N&t&Y^Hu|v;VkE5|>HhGD#c<LVT|pytG~$2gh4Qs^w85cu_6;r'
    'B+9+j?-uMU1Tx$KB|Xb$sfI%4%K|9mP7T?Yi+5P`A~I-sy|c@6PLqm<uF@0%vKJwmBVa>{LLyPZ&J-IHNVutQj1G1Ew#MV%2KOKt&Lh+=4i_tZOoy>cN'
    'el{IXR?5H6N<wP_2jRvUxeqhpIbN{h`XE?DfNR<f88`r=GZ-R4YrZF15j{OxE_ATWWr(g{8JEuMIY0woh!yd{D2pJdQ;Pkf~SeqH+$(irI`&J?Nx)8I9'
    '?OQ60Y$0CHG@zg`;$MbM{yX)KuGc|I)%CN2jiE(azq2PQ5DCN2jiE(azq2PQ5DCN2jiE(azq2PQ6M&_(mCTAi1gTWWr(g{2mkT3Tv(sg>m%76qX&m6Uu'
    'NN7f$gd|9ggpnbw(EaaAqYFesgIsUi|D=^us%aUXaM^%e519o*eJ^R!2@qY8a@ipl73LlVu->uq{>1K2JVRyH=u<EDHt^I$0b+eiNVs`K9?;Cs3J<)$='
    'LvyyDFYWPGpQrvqgIYPhM?JTPNBHq!v%UC&PgmFb&8OS@FX(i&z1;kO%Jb>j{NwTwE8Ela?e<}PJ^lFOkDbOxqke1cquY;1H=FhD;ePXL!@m3Q&G|Raz'
    'y8_a%dgYB&Hd5mhwbJ4=JII&`SxMA-d-MEZ9lKSyV|d}`<*8L{$TIN{HN>hZXfQCKHlD6Y>x2K=H~AD_KWSq{%G09Unc4O^B3noCrH0d@9x(Z`>Wm6_L'
    'C-F4&u?wuQ!)h7whXI{C>5*zS{rlp1%9$k7qxhh(As5K7ZL=+1uN`K%DZUDSp4Xy}PxMIghQ#<~aJz{pRlWes84xw_ls@>l2&p$@KN_&4YU~z4%`S`kq'
    'WL|6sVu3VUWa@x%&)lCP{V40&=oJu|3vVk8;H`rZ5KnL)d6-Xo}Eea;civ2p&0V9k^1^xqJQu@b-AP^aG_sABz5+F-BoyN&kd6t3oz={edq-n&z{kXD`'
    'pNI#k0zXeQwGM&8w2)6PM0JSzf+BV+V34rY5O~bmTuQ%79t{!eCt#A7J;ri41{%SpGq0@`MZ!SKwi3LwDH`|-_{a+_7+&1+7X1&EhVFRX<-TrpV&e(|Q'
    ')&1t$SZLCQOuxV1Y%ajj*q7<-Zhf_#v^&%9?$=vB!X`~G?r+z74Fl$RwO?Oj3T)W);>+DVc@W!X+`9i8J7V*u)2sdGhqVa6E>5QpKWuKm&z~mk=k(3P)'
    '|Q9Su&vYU_4OXTvA5Hk&HnS|9u#AIZ1eQ|VYk<2V7I4tw|zbD$kfI*51-cf5FL9zy}!Nblj8`cv+LWxZF>6;y}kY`+c=WxcURj!8ct?9`+jq|*@|tcmV'
    'TycevLUD^N=&YJ8)!B$NVEdSfDG)N+g>)<{T3%nXKxVYveZ!Fj$q9Muv6FFVX{Ka*>`4GmG?=nO0;}kU2$qOH3#-GRbTrV_Zxn>MIF;#>gp36B9dT5E+'
    'AM`hbx&FIIL;9MT(S){t=krVQz6GhfJ%Gn0k%e3&U@rl{#^25L-6a1LI{;2gY;!8v#pgLCj22It@v49>w%6gdY!P~;rEcELG#<pP(<nGxi71Iz~U+Ye>'
    '}5s&oD0wPu7nE`~S?wS2VK-n|%$1e+*_2XT6X8e#c_RRM2vjJxM66esT*{t!iY}W8eHs|2y*qno(Vsj3DhRr$n2{z~8=hvKrpI&ngcubjdz+=js10EB<'
    '6kygaa}Ib+{Emg$x||*>7V)Ht!I&xrcd8gHs$%e|iovWZ2FI!xY^!4Mu8P6HDh3y;7@VwPaI=cR(JBU4s~DWEVsN*L!Qm<fm#Y|@u3~Vziox+J2G^??o'
    'UdYVzltFNDuxuO7?Pl3NP~(Y5h{jMs2Gx=Vn~OIAt5S;l&BbzqGCvkiXky7hSaDSlA~fskBT8dDuxuP7?Px7NRx^oQ7VR1sTh)_Vn~;YAz><pl&Kh!re'
    'a8&iXm|-hSaGTlBZ%wpNb)YDuxuQ7?P-BNTZ4&kt&8%su+@~Vo0ZoA)zXMq<z`^us`5hGeVzWyWdQL&(k-j=ihvIfA-<s`!fu^{sesc>hjT_SvmaC7RN'
    'u@!avrszkj&c1G2(8{YS^iHjWXt-fnLVWb(H^{bSmHxx-&~dGY?_`T5C9O|<FDd^+c+E}pykY_@+%f86EAC(BDOU!9$uynglO`T482Z$7+y`|88Lzdj}'
    '8-(TbTSL@50-LtDpz`&1J2A6-hx!&%c;n#ot^YnSYzk7Ck{QdXeuf7kf+xt()+ubpq{mY-H-gEcY+h6Vpc08W_F*+Py+}>|ipZ7P{%X^m>`@?H@5BJx6'
    '>GI-uZ(wS7y!5-j{P>H*LuN3%_fM_M)7_Wte*MGKEhzh~7WL||DC_YIh<;4Dv#~C3uQU)sj&nVJd`=hxbUz#Cz!7l1*w$a5dv~|KKgRQmtNjK;+X%lj@'
    'K?t_8TPkd*)}}8xZkYzx1#cXqrc|!c#iL`4XLC1Znb&1zr9;qaXdO6uOG~_xi<`tO*RZ~yz%tA+siM<WUoI@$C~77w|g)z=$F^=(_?eT#v^AJpEoz_e*'
    'vr6S`MuC)$w7p-+p!cvz;El92n}izs`+szy6;mFU~)l|MA_)BaVFe{QUXZ$@#CpcK+%$miGGH2Xy_3IQ;V8I{GI@`u*wKub-d(*NpY<^!b}NKa<n{?_e'
    'iqug}bF{fTUz{BUPl39RwuXm84Dcl4W^&3=vJgOm38e)G|K4SwESJmL=2;HMs6tnD0+$7{!%?eXX9{pSzHJs&o^n;m5G|03h;<?Da{FD7GscXfPeOtZP'
    'U+LD!zKVDsLcK-wa|JQ}OpOUt_y!m1E_p7_d^u7K1?8A$<-@ZA2L|>foas1CZy*qjT>g~%9Z~yJ&{qxh)52vO%&mO-dGw}a>=k4jspT0vN_voFox8J^h'
    'aq{8So3ry*=ih=0&d}So2)ovo{`iG&-x<fA{3qD9zj${3{@WMl-@ZTjXWstH<2=TKNxwr3$@I-%kFK^yaIlVcoBOK`T)j(1(LP>nuc+}ZuReY>!*ukU-'
    'DY$2_`okW|4hdxN1L6E&%Z<V1X0Uhp7*ES(Ca6!UH<xe|4|Aq|MMeFKnr)bJNDT<{oWkxr`y}7cy<Vh{pl|wjD^|~Q$phq=Ru3n7*yon{^>6UM*a^_O9'
    'KQH000080GFX@T^ans8U+FX04D?h06qW!0AyiwVJ~xTVQ_P3Z*+4nXl!(7UvhI_aB_1pUt?r5Gc+<bV`VmBV`VrtIbk<sH7;#oZfRy^b963hb8l{4Q{8'
    'TxFc`h>Q$)Gorb{rU*aQhpB%uvznnoe5lui?Jjngotz_95qs=oV-O@=iIan5)A{c(WLtI=qj6G1o=<6tyCoy93@cvd$Yj=yz>ZF5!!LexC4ZAPc8ZK|3'
    '`usg;D5jLp{J1;nA??p*OQI&Ql7hDvS+s+i&-3Z%k{{2kKw=mvV$A<k5BdaY75pJ{9eihBFKV5ap!|g6w?5%f}XS`-K+^!C(^~S0&+AEmH8txRVb_x~;'
    '4KE6o_X=X2d7Eg^nM*CkYhC`NAkku?C#;_}^pQ?^(9uTMZF|t-?omOi$EJE-thY{M?dY*M3)XPy>Zc4lL0Zsx9I5lpb7@8=^nP6A-WGMzIdjo32B`z-('
    'M6X-rSwAuGK?+-lFws<-wk}V8~Ea2@Rxxv?*|?meQpzj8-12WPP{f|Jq<iDauQ?D^|QfE-Kc#q^o`MfYjPuJ_c-v>c$U=IKTZuj9xM7&-oMlvY?x>Y_9'
    '1O=g>oGNh$ar0IB4(S!1DqRPa*b!?_L8C0H9wyKN0AmF5gEKV0jVWb20QNew`f&!TFI;e3R3o2a!y1Aw(W^0LC+zzdDTg$Z-&wVbu34NzG4{$mAEr{{M'
    'S3n!D}|M3f;ynU7|&y*K^EtFkYy+J>^Mt>v%D@_!Ej@WgTVjuWW8uK|?HUw!=rP)h>@6aWAK2mqI%X<fR<IxBY*001;9001@s003lRbYU-ZZeeh9Xm50'
    'LFKBFZXkT+^VKFgZXmDR*GBaZ}H85r{V`gJCW;HlCW;Ho3YIARHty*hu97mG;D@5;?Jsdex)%6y<K2Q>E@l1(=I9lwUZ4e`}B@Rr^aA$^=WJCY`A|j_H'
    'N@v{*3md8~c6C)gA|o@4{Jr~iSZ@}qW%r=#C$8J?F0SXB4Q7|~?R+zA@ssQMtKk|yc{0qew^w@?^YySdyt^Himy7Kko?pG+yIrq-U0e=ptlq5d))xboA'
    'Dzt(XZXS0Ew&mid4TKlfp-T!&HC`bIsE^xu6I23*ZK7wKKi}Ov2_o6-}jTg>~}di&SIX_z?tVP#Yxb+9<#obiBFsb%%YFGS>&v?Nzg1|mP(z3%2~!N7r'
    '#59U{+F?xW-w<tl~Q#IioDCx=G?JLzdv3n`Fw8AxpqEljJ#rEFs1^Y1}hImXJc7IL#nS$bGRBGGqw_`q?g!CDe@Xdd?tAXwgri;|#Jym))e-O2`s@#&^'
    '`HK$h57tfWQ-vc%}aB=Up;Sz=1nq%M#pmJlaZ-$9mGA$8G8$dVEy^OUn>&a$6W%2FUps<BL(PJk>qI6ujGzw=(k4U+;fJ-q;%P8#<2bVBK?y`c{(q@wR'
    'UUC`h<#|gFe$;&&QN<-^O;G0T2<~C`N-8Y&MjtbbVr}cA7vPMti7aZzg{GPV2VfTjWI@cx+8}!uH`!q?=uTL~tYBFx0WwAWTd{bz!K2IXQq1ChNB|%T4'
    'H@u<I10+Zug3?;S(^%f54v-*~7A6VSNhX2pK^Vr6CyviN-#2-T5RC9bfcOMptWj1c{OAv2ir^z=BYlITGY)ga1x^MsCyA)Q!KJ~Qe8#!Nj;FtHu6MrZU'
    '?R@@940{+!z9!yjUvt9QxDU5&I)G~y>bRwd@NCrj<kf2$xSL((iA?rDhEc|!V`XlD`^a@Y?D567Pu!ufu@Opasi++ZVVKx_7=b}lLEg@s@^dyBYYow&K'
    'Q;n;T5zC!xCte?PDZGR3M9rS1JVkWX(!_d`**r9V^xGP>0a0P0O+*Zf^Vr%bM3R5C<nz1*9@b_@+@4#JIL@R3w5KGM7=049q3y(I^4qriZAlh`xe?z11'
    'CG6$sj>Lu9ZK_0CDZAW41D767!<j9Rn=A2rUniFZIgBTSYEE(-cymIzo$3E>dEgLN6QkzpD#LZNadvP``I<s$<%A!<o_a2>6S?+Sx6(ksP<iRbB2WLeT'
    'AYR{?}bkMsdKREf_i?|zTN2imxgA2d}&m(aM0+Pt`jF%8#WFpctVn6^uok)XZnL#^2yO1_Au}l3JH=!b+>^l-i198|&ktO1m$nsfa2}0EAS!9_kSUn>s'
    'X6T-{Cu0co=;c<=C`!IK^AW=+<vs`*o^h0FYl;V-MV25ZV>gjyZ`A2YTu`Wqgglum61&om)D=r}3T7|vDu|paUKLfRxMNwe2laD2k8zt^2I*?X(;@A|Q'
    'HNd}k}du`<2FSgM5l|wn0y*_XWV8c5ut~1n>}QyLLFqujtNaGBUiR+dsd~sVrHqvzIJw{27od1GlGCZdT}(OkT!{m3ePAl(TRyYqqOvKw;O~nu#ThU7^'
    'Nj!@OVaP1yg12#wcyvn(k$kHh78Y+$&05>@8XprXE<9MHQ!>G_B9Xgub_kV5A~Qvgp<`Q~@_I2eo3X`m(b&>lH*Nh=j?#cZ@PuEs{{+Fv>h+>6vgCNzQ'
    'vkQcv{aa6^4h^a3csGFZ}i@C)_siCzS{N!=5@aB3CqdZHKZNYb_^dO;9LJl|(ysBGy@IUumK&@M?F4KruXvgpgAtk2^8n&q~h=#^wids3FVOItn54j4a'
    'KMy9tx9#MrxPj@4j4ZVF8CnCyd!CtgnfHZoq7cW<&Em@E*?;X)V$*RgfUzHmBrt+_Mit)aw{DU`{6MDKG-+_I&3;hmT1HF`!e#bj#)Hq9G;cBup<rySa'
    'rG7Cq>>)PjMbROR=ZT|HrnJWFFODuj77X%x29}EFjItD<fR+H!i3yPcR5g&%9Bf85sPc5pmMf|XOxGK~)hun1AQ*u)`yG~)z3AyaGOaMN(bGM7u_)(J+'
    'X5dI?UHnQQ@XDzFdZ~vrmiY5Y2B>v$eBHI2@zT8=@3}@GC|VQNF_}rWo3aGhzXl9<cmwfOJ#wHg4W3{l?7%fh$=E+Eh=`wedN<xRP3_6(MlD&z#y?(Sz'
    'tyM`nHn%9e3<gs~9vJn^&?=0+Wd_D(MG-GuC00+JPnH6;+7{53Al(CbIr(dZ(;YjrRk&RyAm^8KsW9QX!CQ19NlJtc=AxJx#8aU6=s_xi*l!SltA2Z79'
    'qM+>=}zDzM}=6KOW9I0LyBxhtB~<XV(gS<OmdjTBiY(4K*nK}<kF)h?_YGop&6W?*H2q8xEcSyUxL1)4e#Jh}Cg$aTaHOLV53K<F@^lLG^hQ$a#>Yann'
    'czLB29O#`5t-r%4Hy{f(;KLE5=6pJ5vR8RC&D5?ut(eWsv28psLCKKbJfEuGFBzj1mIbjqOP%p_LNeyVo$W5wt;Ru!#NooKS%ppl?04`N;5+_adV4@lY'
    '=1bausRjtAEfdwK!L&_O12_{KCu-5yfwF3+5{=oLvPlPrQc=Y{k&j?<inLUeYqHhSiO5HGi6(hbu4(W-`L++cT19NAJde)UIu>ECWvdT*VXv`1&`SE-H'
    'M3x^Mg=9LP}8eZiQq<M31SZ=cvQvH6MK>EiM><Apm!n%^u!))*{c%ni9LK4j75pP+_SDAHxhd&-4RWU12HT*S7SplEMTT;D2QR9l7UYQ8<CO8(?JZ2ep'
    'D<I#IWqV==o?2OKxP9Zw!lFs6!OP=4$*GjbWRyOeDjCth2xrT~e)Ct&4HiYF0a;sS<1o<=8|dMS&2RO)`Wep++W?+#9wcHO=T0t0puFY7O;7)U!ClaBO'
    '24qc=o18&_@J3}`H45_M$iW2v2}Vgon@K__hSa)bDxhKMU;h~$*ZDG|{`4N;vq1GQ6b1?LH&PPw&O6_=cHEBJ{SEU6ZBz(3o?B#dO_Yhm6n608zU#7Jb'
    '860(D;VwjC}OcFm)g`j4aRWbW*p`I848~fcm)}@K9G*2f#rH1ZO8nJW1;mvel=YpQriA7RW(6jzw1haF2U`dr(WY}k<7nUQDVV}W<@{__IhtYgdVb4}w'
    'lR{pr=0UL+IzORG?4>If_MBobUM~q_VlN)$^^V?2$tvfl$cNOb#-k#?dh6G+)rTl`b=_n2;ZCe96g%)^)Jmcl(J5BQLNV$XyY6vzWtC)~?9_!t%L$o24'
    'y^ud7v><4u3cw2<{;Etlb$QRviNxQ#LE?w#1^I5tM<SN^lI)^d%)&V*9yAf>8-=gy~UpO7r9sLA=V{Juh`4R&{^!|5Fcc=EcWVmt#DT6jKD_LcgmcoOP'
    'FJ`WlrP<R_$4tv!O0wpO=+68#K~Il#t09kp9a;Cb~V_Q^@SuxzZBKoV_zX&dQwZb$Ef5l{widN1Z?><-LcU*VKgtv22dI^kLsYD-oHAX;oGtGLx>p_#J'
    'y{^?<qpjMgzFRhfk-$jm~TkZOu`GKCl@D9D&F3n5T|2d6j8#w>O#C(Oo(K-QOq{-=59)I#?+?{^yr&tMJuw0O7ccc@A6%EoA?*kcr4+o5;MBD~Uzdv;h'
    'Tz49766`n>9qXdu2d4(SKyu^_7{dk*1Zxr1hsaDZS-5;SQu6oxLF;SI9(ftuTEBhDSACcQhq(%2f#Rs9L=>CW+89zn$M+p(ih%dT7N+fN~@~&qihZNnK'
    '<Xu!0dKito^*be~pDWgpm#BDUFGOwWVb6@qtXgzy;(&GSsauo4CYJi?)`Z%UF^g_Zcta%%+O0{prnac+h9{YVii&P%$LgeHs~82!qLQ062<1@4-K=yfy'
    '3vIV$cIH2x@<gn5UD4T-zhC?4x+5rGBO~$TT0GayCT&^ox{Qh#a!0bjm@)Hfl;ae!wQQRS%hX}3qYeaC)!77vlwVjv=4|x&l!WOAuO`KBki;KvRPHRZV'
    'A5;pvbvd0n`Z4<eCA#5unIzNnzN$$`q)U^-jTsV5X%SZ<Z6QM+R5p)iP#I<k5JyoW_E!Is=TiCd%1Go!Pr8ab}Qz(8?l+1W8?;+O*v@O;uR6V$iByROy'
    'U#sxHGrvTKg2+wjz^S*W@WuP6f5Ht9aRMY}()f-B2DHD-dV3wHBg30K}|s52s3TeEVe3akzu(mH|FCA&ndQ9%sMFx}RuAjWrU#G-=OTk}(+E(6$zu|<q'
    '|EGM*Q*_G*4Go=jA8|)AJ-DW%A-fj468tT%$>g+0C52O(LNgbcm8t$H;M0eZmnz`MR|M=AJp45ox3za8&Qu5PVCBEI$xSvv-p2UQ^r}1ssNd?&#m)(OO'
    'yVvXCAN-xva>3thEiSI+-k)D_>6>ANf47U9q5Bz{E`A%Z-tY52P)LaQ;U!)!?U!Z0+fEqA-;T}ja*sc}8<rRE`HQsqc6hT|zvrUG@^W~`hr{lH`|$BsY'
    'MOV4%jMnty8H0q!%nb!yS~JHZ}ocbW|(j8*2CAfJ$m`#{K?_NFE;=9)9&bQz1q7S)_aTP)%?|BJ6~??yiXRmcjzPj3UKjyfuo=-kH5P2tH1GgeVe_5y|'
    'X8Wd%WTLnq0Hf+2P3-Qtf`$9bRAG+w1c~OgOsao><=xtJ~G}Vhg#h_lD(mv3<Yy=k;*AT5lK2H~;cA86F-%fc@^__dxP~_vk+mg!|p&A3(1d@D9XqgaM'
    '&;i~+=O1nCf=M`xhP{qDOL2*3UA$qNP-*31}I*yryIsQvEbugriL`EzV_@{+-X^&GL;Q~n&gJv$+x?01KpW54Go%q$pZB$NH_#VM=5{qF3Tbb|2*W)~d'
    'KIrcj{VhwiRj?}ArI9$J3+}(7hU-$6t`pta3n0KaV_vp9b;!0C0+&vzaH}mx`ohc53u5ac`VwDEyjyBuXQj2Mb?sz?1&zF~-8KnDuJq#BFAnnqf-Od-w'
    '&aBgYx1KNcibm=ltyjQ}wwZC8?fjZY&|ux8_qS^hh?%WBnQwp9LK?3-S!}QF<|Y8G*qz+H8*W~$?$&QQvuF3@ZVC0c8_n811zxzCw(Xt`+pA$siMc<`+'
    '#TL+wq^`kyL-Oc=|h`iw%*>oLEM?`yBDjQ-QjdWcXqw{b=b{+*v;p^Xik@O-z}EA-RPO_?Co$lEDg5ekT00wYv4qw(F>k<2m&Y`q9BTgB#`1E3#ND|0x'
    'BM=po)hku%g9qf-4@p0L!MsAd9D9pv6-(*y1S}aPgE4x_BxEUOZKUFP@qKn3t;tVFkQEjHljUjHh5g##1yX<0%=K@sthDcq#^HJh3^|8EQpzhT0IFp%z'
    '4E=;F?9*LiWGGjw66Gt_c~`+;FLwP7@@4Wo5!7)@-$XlENnW7{xV+=kKYHjK8nVRV2EqZe!#Jz>M>4I4&}*f4s<hS4)NjNY+f^pFjsmuwh4)$JJCyk*1'
    'aF&jp&*)V#}hS7UAj2^UM^r8);Cv6zLX~XDI8%D3%FnZR8(YrQ`9=73w;%j)f{gj3ggm1`Q&pJ!--IJ5qlXK)HWVI*H@x%EW=Huhbdw-Gg=}%s~{^Uh}'
    '>atzmUGUFY6rTU*jaaO)%9t;gtF6)|s)ygZ?fYB)Da@l6M~Abc$9B}Pv+RLu2f@vJLeYz8vcH7+u>bL+$H!-9M^BHR9nOwVpPfHGJwE^2(-V~_+iSl6W'
    '`23Id4TMRO!#_%-2U$7dbxSPpZ?{i?rOWeeK4KAefxIuHcVFQH`C>2%6EVIsoSmG{IdGvl|<y!^;0gGUaZ!`<Z63!{qeiYi|wb+ZtvFD`sngvx<x+OOh'
    '2w$U%vjs?HM-QzV8pP<Y4oDxt+f|SW>cYO{wEgN@2x=Ig-G5<OX|Pt`@ctOMX|UpLvR#Q1Az%BeVrwOuPPp;G5g|ddl|~i|xR@vBMwk{LS>UeZTz%hIn'
    'wX9_HJX$+;fvkDu-z^6Gk5Mr|eoim=-`hUvI)I=|aqt=1@HXl4{LK*zzW)#dxC5cX3ywL=!0&E0VM6-7RGoSy<EXBSt)&HPUU8<b&K9XNmB+c(osyM6o'
    '3^b1>$KlucyZ@-Sm(}(|l^k{ZI`}^~wdzgHDI6FK$ntlDV+3{1V_VoEVm);MEul#A)|6-@_Pfi~mp8Rw6dVX^F?Ahm$`af@WboTV@{K@RI=sbFN3u+=X'
    '-tTR}Sev~+-wfM1-N#DxYCXKhsu9t{#XWTJiJjK;Vh%q}$Iqt2a(aaV`Fx|2ec0S=7?=M=kh8~6|MpK7WPZDtUII45&0;AqPhT&tht0q7-@km6?@Qq}m'
    'pAVwzb$U>OL+S5?EKN`%V)ED688KfxA5hv=SMG&PamJ3{`Kg^;mOJQ2?+G;v#*p8{@ZU(Pac2%jY@)h-<+MkeDUb${P@|~?0EK)P&ngiSZUWYQTzKJyn'
    'GJW9{oSCyZi8L_TuHE*~=G4UwrM#feT-L{qy~X`(t(UiVZDD=NI&Hdn`5gHp6-`u;{&I*UOHINl6_gezw{yL_Y^_QC}Y{R|oXfrv~=NpU0+?PxsnEPSt'
    '?%!Aja6L--jP_&)(qO9KQH000080GFX@T{X!cZT0~G00;yC06zc#0AyiwVJ~xTVQ_P3Z*+4nXl!(7Uvp?-F)?3ga9?6FGh;P1FlI1gW@9vFH8?nCH90P'
    'AVQy(=Wpi{cYIARHWm8RWqc9M?_g6$Y<kD?0A0~hlt!U#msA*CKQmI|7kYk&MRTfxmda3%~*VuG%wHMx-H{%)m8NXjlCR2(9=1feZ$@Fv<r<|X#ZRe+E'
    '`mN7)?OA1nXn15<jLccr)(w_m^^6NFEZme9Ejeeu%L<FKuB=`zxhM&@>^ZRd8P=EX-_N*uU83!*XxX1-JnO1bEYV?~ujA`kNrtLrw#(wp?3LBaI8$(qG'
    '~6jz?-krUYj{zxx>ta7<!z!tSFW^(bkFrcL88S(PuM(Z7$RNrprfr`_c(wS_d5lt9-HcUDAf_tHWuYsu$D_J-(}bf^4PtxvC8*GO9Ogg@ZeMaXGNb=Zh'
    'U5Bd#U|UfehnpNJxH-41PEA^?u|x&jx=P`Rab;$mnyM7~JTyGHTGqm~|L=V$>wYJ~vMWH+5s}gQ0JY{*NX%YWBO4r^Z=QWB({MbTn1;r~H1XyK0$eOZL'
    'WS-U-s%?f}QXvcZ+@XZ9k3F!G$a3j@z~E&+%D&{r(~YzO*%J@Bv_E_~|3BE&8xZs0Cp=uzKe*r$wy#Ac35F%0O@!!`)QW3h0k>)XKbC|nHvD%|i>Epqw'
    'qu>W3Xeso*{At(STh*!WMC}24M#p`NVU3V>EdDqCN$LN<+z|3(Ywed^f$lYIj`~^@;0|XQR000O8m!WB0G%{9;<`@6~Ry+UzUH||9WMOn+FLQ2TaC2yH'
    'baO9rWM^M6IA1X^UtuyiGB;!}IXE~uIX5_FGh|_6WnnouWiex8H)dsIWHM$oW@BSBFf=)3Wi>KoGh#PkWiT=~GG#DiGB#ynGA?R!Z*JXN+j1O7a{ZNw-'
    'WMqh5Z!g_ZUk2jkc22$;Kcx_&{|X=nllYxmV@c>Tu5RSjPP&x-7otc`zQVq=Q*c)0FdH(SFta4td^ZqQ&pLjnU#4md+<y4u3fIC-8|dR+=D#Z$;Pv0wX'
    '#o>X5FmXwT;Z0H|@+uPTFQRdivz$Xwfc5<7P2kH}>tS+bqYeeSLU(c5r3`n}scHC)hA|X}8FCi+JYzexBQZ|CGBNT;4UajV<~m3w6l$eOXocj?vNR%6Z'
    'ZWbSg-fp!+$M(RFg9>uf*dt~5H@w)>&*q6?#4S(A1~`+knc9IlOdz#LbRj+oQ0O`t<0U4qUvry{L6HTTRx7vSq7=NI4`Tp|s=u?VEWH<q3>`6kT=-w?T'
    'f0lq1SE;*mP3h<4gAPv5e^Np*Il(+zuz{fcHL}@)J5BaFwKq;WZ0%O!{g&m>Sx`r&&D#vPkTVe%{=Ia1;;bf+D@&lVXVk*vM6F4_;D;%At$~`CGP<ezQ'
    'c%~wU4tS)3rsauTfK(2O%nQtQv|q$rNAp!2_0%rsLWyQEkkTQ};1c%*1!jR1c$NItSeZ9KJH{z`>d6>I`iVC_$?;oblXy~ZT*?AsdSjAmQpTgqv9mWGr'
    'RQ4XQBqoOJjz798;_C}4924>=+nld3Rn@0M^!;@!FbdK=NXThei4j^v>J>@kvk4nAZeFSgRJGEMiyS41+t`&u#wASPTF|6LT&*oIceU*Os>$OU?=DLwZ'
    '_x=Kn1{3E)=3`&=qOe%E`GsjOEOCNQ1Ae<zX)8r6AbL>9Tnk%;go;1&g^nXb$)m{hr|K==2_Tb6#hG<y=Ud58Jukc(9(0b5GdMuDfXPt*S^Gd}}WggRd'
    'Ol!;W4)8f}}x>#li_6|b^wQUv-zj36T~rfpF~nkK?Zso>Rxl}behSgFb}A;C%&t(0J;(sLBt<Rb~(ssgela4Tz$##=IR2a@1cGoU4Kt4j!(prxE-2_+R'
    'gXbBxT%23iJXr_`LK?{^&>jWkJA#p<DD3`%1PPZK)6OKU|;UqeR2?P{*LxGU#gW>=}rdQ}H5Hh_&TEVzdGRzkYEET<^U}Vv?C>UB215&}*QYp;9Yt7j-'
    'pV1}Pc**ENzKkz<t^I4J1&%YqIE5gF7_a$?F-hYLGQ2GXql}}i3Wb@Rd>Ll~(@4Qc<8>~^8lN<uDT0BgWC$uqVzdn4ROKYGVO20jVUG%;RNStDBy|Z%D'
    'ig$#1(Z=mko5X0futi;kfLJ1tH7<!ITzd%NNRAanVIT>`$C-BAtchXYY$QhfZS+(B;=?>hkP7;jYA_HN8L7L<7hSpiTIi*Z-HbS6GLKwWNhQq5R!4Mlt'
    'z$@Q!dNnUYmPyt!EsQWpJ;}Y}B;pjm(vIa7h$%3f$`=GKCbl*A}fb2lu4v$9XOLoP~QCv=&jiP%0=8r;DUq6?iqNt^%>TDCHf9)<xwsAABoW8xh-Xk=E'
    'kI2$U2E+IHzg*SuexkFagy%9gyN(g^|(!X#<lQYQN%@|p(C;XSRW6S#{-9?_7q*MAjm&s$MW>9(Sr%IFp2h|XJ4&Qf%J%d2K6(R}z#AFB&EPEiwJ6gBn'
    'J&jH`WcvIB0i_QY)v2p6KR<zW3MVfkm{}hRL0~pb0dZWu3F`a1ejfsgpC*MNjw2M-IF1rX)@VxXNm-D<m)L?$8#f$lcQOzMTSuQI{<Gg{8z%ieOC~N||'
    'SUMsNyF~F~K8>~JB<3c{!+bidWj>`{oboAs%rT!DiX+W@O8DfIPfGzI@C_y9VLpvPH25azf%!Bgje~E}8JSN#^Pposjf@J8`83g&9P?>OpyHTM%YGbuE'
    '6SPRw>~j)C-|)|1t{USUUwwHZ!My0oT!j8r$|`H{F0D7gJVcckywrPO-P@H>oiVqAC>(G?rHaw2o8g2w*~Ym5!~mZ19@?uM833dN<xT%H26BEzLXfbz4'
    'm0})_BQ`LC}>-W(@jsDw#2;+k|Xt#enF*eMGaU;5~APS6-RNAzt|&@*-Y&WmiWW;^a$+SJ;L$GN-5B9O9J^l+y(Qf5tk8c-3cJhj`^Brwh#3@+HJ8n?p'
    'ZDyz-&ap5>i!W^CHUg?d|JINCij%sbjW6{1;kM9uN^2g*RZ2Pb?EzDZylcDKw=zOcI`PR*%UoM@Z{h|UPR3o$#Gy{5N0SX~vMBXG(kFp-lS#t}Nn862@'
    'w(T^htI0;Gg)EfduALf+A)Vd%glaY)FqZi7z4n{YOqB()I#>rO~A1KQ31Z616mrmt?2W+Y4Fd0csAOf31B>)l>eci$!Z}EdPgS<4NAfKzsAyj0L2?#zF'
    '?H>?(YN|CL^wjkHP_TYrlnyT6DlIr3mUt>S9+s#(I36}?J`YO-9uRp9*s&CMR5FDLh&-kct*xju&7qR}0*HqdFJ^r6CDP3JD)p7|si?q=ADrgLZ3Yj__'
    '<_m~%=j^gR%W5h%=k&NG2^Sy3(EMyr}WIaQg&vkssfe~$?<j+j7XlkDj1O>^-wS(mBPY|NEMA<Fe24L7mP@Krb}u!Xf4|fwdOMtvh`3fBK3Ha5h)O_7m'
    'P@iFgTnu$@vR7XX^1UoYO+Gs_uZwRcC>7mWqA_oU@F=uyD?*?Dv72IAwM?XVoJgIA_&od^l$vlv#i%e4a$(M$6U?h{6}ti36hWh3+m8g;#OnfGCWn!vj'
    '%RxycIxL}4>Djz}Wt>uOeTL`tq!1;>lP_E6sMk4)Fc2_Oo)8LT>iD2!(H3PfQ-vAP7JFovumfhdNmkhPAQ7@oCG&g|@gDD0-Px@4`Bm)aBObqolewT?@'
    'wxN%<3yXINzcxF-$e@HZiPc#d6gg2DxJ-`_%-Z`IW6LlYSbt$gm6cd1FoN~MdzSI?Bp{qE>Yv%hnMMjKMicOwz%BwTNIHhFc8K;yyyyBD!X~rpm56?K2'
    'v}YyI%(a!@0y19-G>eWbUPTj}^=k>vRKN?2>Wl=VxeET8E=YjuYi0w63=h|-6(&4frxr-^aGknTp9ii}_c9QS=1zV=G-|xWfjyR_9tZJgUNGqe0@B<Qo'
    'dzP(+!1mGLegC6A`p|BCiM;TkyFefD$Ui?3WTKwr5S~YOLOIbkW&J$IGs}hSs)a-m@33m0+o!Rpah{%#FqqW85NR%>Ij7-V0rBk9#a&Fo|PPwO)D9nT`'
    'M(U04{2t!m}32sRhqksF1E$3wgEASPMC(E6@H_@1km8i9I`5y7`_hEXREB*~4;7dtS{L^@G%n>Bp4`D1-Vja!WO2h@U(=GErB6Eg9h$&z?+@<&$a*T{g'
    'P0PR_PWqHTOqZNp5XnvmEuP!DPd2I`@d!l{S8?MFS7(mnN{{!XACs(#O!SAf=5O{gqIJ?J_F)`U6F9#|9RmFBP}lzXx!6m|^Ig8@j5qXvk|tk8pjE3H*'
    'z<o?iuQ$IIA4;FGMTT%_cvp^4K3e90euC$g|mT)H^c6qe|LMpd7H87LOX96=>j}n;4a&~e8v31|vB(_$~8HlX~*MhLMAP%v06tF?~3jMiCW>6s;wp{|@'
    'U0~m(M+|Jd^oXHS!=AFS^%6?WVZ&bfshGLF#=%!j-+=HHgjs|7FSYgrU)8Rm@4#?A!dEP$TZFHevz`fR!bs2T!W5J_0+}T>c>&By2Jr}7R;twqSk|}pf'
    'yz{_MS!wOuPlJcs=l#V4sGTLL{_VX2|%W%e^dvC7$6dlsZxu?W4g7Gcuak@=#@;^A0-}BB#6Xgig}TE%=P2dHnBu?U>MdS@z_8Y+ipzcZHwZ|+ZLq?-n'
    'JAqKW;k%k+)5`KI*pVT)J(Q*4Eq3KxEXZ=zUSQE!Q|Z8+t9osH5sNGU}++8X0x!LgVVFP)<gjx^EQgmW!^p?cb<YFtJ2r)TtP0Vp2ziawgvLT;qWWP&7'
    'L~vQbj<)IRJu(JLcvn-e~d><K#}i^!(1)9a=3w&N6e+e&q@*#e{%aMuK+Bd(foV#G~LGA%Be>b{73R_Z+$u9>1=Bwl11i@0QZsTXm_N?z57E2b(h;)V('
    '8L|ibv^ozJ(6#-tv^-||PR@*WrR!omky>?<DslH|z!^;)fs1lTqY*YzWMmDNc*he<11koa~foGD9ga#_YQ^~@W_b!zzT%%|fu3pUo7Or%u#2}Nap}-&`'
    '+?5z)R8FMow$+@fJB<SdnW7NQ@}Q>~+|)FwZd=W%s<$oZROj0lV~LGky&y@@Lr~e1pa+ZMq8Z%O9H|-H=(4Hi+en?aW^j`pfRKXP^@(p^Wz90rw|(o9Z'
    '(nP@86l$T-+ZdU$6%h+;8R0K4L<QAnks9(G$DqhGbCau**)cafu!aO;dsF-oz&oy9*CoAUaiCiUrrE`h@u4d6G4=4bs~o98&z!Z<qVdI4L(8qq!(tD-s'
    '0$mndb9_S^sv2jYTzs)Zk;-PEIeb^xB6nuCzZJd;%?r4L<Ws(zp^-TS7or3o|*s7!-*s!7@3q#Li_mmDsu7DXGLJPlj|)pph8T1522s5?k`B#8yL3C3e'
    '*OwCzPChIEypi6K48r}`3`Yx@$LvZ=(@L3#kCaRMv7x#|HFWmZ$&>qVH}d1($qx@2HTSC~x#P`38tPPSS%>&*&3DzL%qO{RCl*&@@sQ4)Of*HdLr@f}W'
    '2Ra>eaC|r@Aid!mesqNZ&yS5(IvX9+U#g=kg%5SN#rQ()KTWa9JzHCoz*T!l~ZTTXzNCSxPqN**AYT&`{48LVBMBnmk7-PMvc}3V#+qJdZQiCi+5nRi>'
    '-cy4F%sZO5)fz7_zonmx@>?p1GUHAs+5S)2#j^c({KRBF#Sc^F&G~fQUK@4UcJ}XjdfjF}n<GsB)Y|M~2mf>Bj!QN;u`lQPrOPkc$DYGaTh8pu5e7Hye'
    '0+mn!8GgkvRmF@(sVv)KY%#M_Tgv!B7V*e=JQQ6%RYYm*o$>%*H`v=)Lo3O+h(;{wqM%z@$nBXKR<kRa{k@Hqfc0T@?G}$s+rD5v+4Y8I~l#}W}EAFHT'
    'nu~$iChgwVP#k_s2ODYEaT0xt=#`(?c^GHREx&nXgwPlVxhMb<5FLho=W46tG`^N!&-p(|`R$N)2UCm)&YLnsg@9dO990mfd7CUXPmf=(<_HZPzw39k('
    'mV1(V&8>1o?7y4lp$nOyCBJzd|7zFM}6Zn>V$FTX6zp^37SJ$h<kU?+S0Zx%s!vM1kLBDRNp7k08kd&HgW$R5c)9UB6%$En5Fo$T8ai=sQ(cPEITHs=h'
    '{)AsoTBIiza`~#w+Juz-u9lt`vv-x=1X3sEgyFEWfblS-d@Y(izd5o-K&og9<o$TZVvcgVwdW1}2&+m~9j6FWvey4|c4RcSUHj+JRXP46r{2S-X9&Kiq'
    '&2ow#UE!eF<3F|ID}Ka*vnTBwYR+_UQ|R)#nZuW%fb4L!?&h2fMPx_IcGk=%nG|A{Yuhm#6ne={7v^M{w3B_iZ07t5O3EHDJ98dxbB1l!%?vD{u<Y^8V'
    'rlVLYRitB^*c_4^0MRU`fAfi0BA8g-h61U-*lVhWhQ-Q-)-i0JnROwX3v@#e_saOX3yL8Rl9`5*dJ=n4mPW`lmTsLFT4Ic(7AnDt~Zx<FQxnJq`U5!!v'
    'xvstb5n?pTF-vH*fhFCds~?&imb9ne6m^J89=~TgI4ABFC4U)53pF_XPT&g96plMUC)uQX>N0)QCt&MGt_kYD7tAHKL-sqANy+#RZki8db<?jdF5Zqr4'
    'o~s36xhD$03{N^)O-967L2m0Z}UT273hA~!awkRu!A<jO{QIkQng?rap_f!=jw=|GP`5qb>T&|^@E9)n)=7?h*Opdmd5HR&<vN{>NddJJ0AW6+!)gZA_'
    'oG^od*MLh;h>M>|jk3pk)3|iG=(5xPVcJ&xEtjC~bJqAtdF=$(lLF0OSq<^*_*0*WSocI!V|G7<_w5#!Qx>)PCo(~t_oL#kd;{J4HSz|Wd%<$XIQL`G^'
    'g&QrAD3HOH!%siI!avRAk9PTN+65NQFJ{aqUys%uS2ge2Wpmk%7Tsc_udqckS}oe~^uiLuP1`JYM(^!Lx7^ftw;vYEcD1rSExlb&XER&18rdDZyUVBZ)'
    '>qwTg_!VQwEb<W%~@>T%%-a=oAZ8(DeKY17+*Itn};cVI@S^SLsZCz59`k2+2~@5+-U0-8>a8euA4}fRl9tL4A!SM<M8Ijz8JORZr)wrke|ubjK=M9jY'
    'Mt9c-l?4y~T+KqocLSJ)UjM7{9jd`pR<UmFcffn3md2!|$)A<ExQn>e=MpmtRXi%Cf6|T4OrBE8S%~H*3s{n^|RU5NOR+^KQOt(=MhE$g=f%0}Zbvp-D'
    '9;38L$CX18X%p1wn3#tv|p?)ti$&u&JWm3*cF>kl{d3+QvwzH4XQf?TB+z>Ve2fB(mSuIv<R(~gNYvrXrhBZKerNwb7Czw6%HO7oE^rI~kYuDxE&+I9P'
    '@qP=ft7k2aqqi>CM`=P<q9a?+Rwc4Rq{OGz{tw)==<v3HPIZ%7mnPT5}%eN0k&)X>yjTT|`x_R3UFbNFj?WOsm*`U3+Xrc9A)6b3_$(FVAchhA@o!coV'
    '?d7tW(2*ZJ+*^E;3Dah$PoAFV)j30w-5P_RN0U3hroR1ocEO)#^Jm%XrCA(^64~bO+r%Z?#^OXXH=Q(!20#6ht#20i4fo@d!-KQKC*srgk!-gpcJp*r#'
    'QnV55B{I<rp>n4E_!lwdV2Wm==s6f(TnHjPhK3I|M={faC|+(_Ya%N^=f}Q0jj&0nsz^2&*rOrjQ;+0cC}tF_V@PQzkmPWy?@XxFZbrFJ$(1O*I7Sv^|'
    't%<naB=%x!c2py>Vyuzgl0<w(BP2_3g67W;x@c$#~Bk?|QYjowuA^{Oa~r$ko{1xm}yGySgz$e%PHuvTt;#quWE-jQz#{&A@VHdri8jHp(BkI<NM&X2J'
    '##dVeqim&3Nu>8}XAS~SZ&d_OiYk9}>2&vt#dcW=*c9-0F7#}>O0;K9Ix9)GQWmx0B~aihzR`eD)7V{e$a*K90;F3qQ*UIXLi6}xY`$;}>J>vgs#Cezh'
    'wV<F(z4{_gM#h!V`>G-O>Zhi+Rvx6YD?dHX9Q9s<f-R+x)d!Oid7`b(&Z@%n@hd;hNJ}~;l$sf*-kDh;j$0662BzL!nZ6~{0wih-$Sgjr3(F7<o7O4C;'
    '%kI4e>)kia_-#9%>{*%BYjDpJlVw|I`exS*(606_+x6bj{GvO@$8#77e-D8k`-8>g0^*n$z|c0wpgO$(Q~&QC>HnYLFCP8N;p4OOvp>8%yd(3IgR_Iv!'
    '?Q0hdwFtrdieZ|A5QNq{OOn1R*zeG{MsV-YlCEZ&N_o$pf$AfT7MqHfzv)KE|!1!uN5*0{DgB9>h>=T*R&t)b36ZRw|4%?Zco~aW-~(}-v0H?E|#BdhM'
    '{|#y<Gm~&)Z!_7NACxzx>(uUw^(Mw!8yCd47zIyUB2F3dWU|k`Go-hJk%ccCC$e%Sj8N@}2GMAExblu6}3419OR)wD-qVH~*G*%D$a%W-}Y(iD#CwdE$'
    'LoV#^$jA1=JMO{az&PVda@*RKqs-2dWrcJLZ5aAzYlfd2lC@WuUo`ku{ikqNDsPsO@yyWc3)&!1=x)$X^`*}CoBW6J{Y_P6oCnqilN<6~}evFxsIzg~B'
    '@^u_c;JK5^Y{Ij{Sl)E+dFT=>ehvAal6!!)EP~7U|<FMni>HJyqVQ^Q7)GS^uyFY#=X|p)J>fRsBW>2wx@J$<jzPWzWE?-<c?q=O`)%)qPnZLC|Khyb!'
    'y|1R1S2O$n`jc_i%r6%AIdbsP&6A;fdHc<qjVaV>p_9$5U7fCxnlEqeiTW6S%7==pSz^;?t>toZr1n1k{5J-$^B)dRPK~$|$lm4W;rz!huRne9>g4g^`'
    '4A{?7d}6Dc3?>5{N>{>FFreZ26XZ4CD-4@qF?xP!oOjsr^hcI9UT8-_ChXuejg<N@0%T-K07`C?(7~gKD^CSBjiCV_paMDiiGCtzcm$(8-r4N!?L}0zP'
    'FlO3J#vPtLxQt;Q)$16Xo>DvmgJ#q96lL47;`0RzSny_AaL8n*WS{f4`ORbMaP_>kkk9G+o@4@x`Ol^T#h<J-_QaE>Fd$b6y^v9KCpQ{=!_;g2(x>RWG'
    'Oa*Pt`<SJ%8ae)8#>`=;xi`RMuS+0ognZB7?@JM2$?cfqTd=D3Hy!F&6Kr)MXx9-qBBIs9aGUhn4BSy9|u{qx-yU)#3Z8@x9$<o}j^61?3otTb)WJze0'
    '{%d&enUGXJAhOW_~TTPjmchSNy(6kF%-M%UL>F43?2B_^T1$<y1AA~xiKmGji<H!F2P)h>@6aWAK2mqI%X<aljR*U8s0034z001%o003lRbYU-ZZeeh9'
    'Xm50LFLPvPUobddF)&|bVl!kjV=!VeFlJ#fVK-uCI5IXaYIARH-CNso97l5fm5JULDGU(Zb?a^fR}PSbC|Kad0I1MfR3Mr&4Pcgo>G51hVik<=Z}{CW`'
    'yKly{u1Xor+Wa9;(AxHFLtb!ol{d)nU$H9c`|$OOZKi^uBP2Q+t1vCJln~}vu3rjPm^ZdtlG7W%$hgt%tlVyW;S~I<mG75E=S{LF<m$I?W)@>$E|&Rcz'
    'SklW&@jrEo~>*Fn4LU$ajl)=KOx1+kgL*yBu8JHM5N^`Xvi>$o73%Rr!w5(df!~(g}1bNSC1dIhE0Ma-{2QKjf}7I@-4Tq41&$qg`2(c1HVtj>jCXjd{'
    'QvSCNjG)2~gSLnK{-&NZhZtvNOK%t066>m%nE;2T^b4Zg7mq`^0qo;3L;%?IBQxqkt^DTppPpS%k2jiDe7zLE2dtB;hp0F}VUIQm3sJtz<PsNFy*pu+-'
    'V)NF+vq1U>GEYvE;YJ6K_1&-$H0CnMHrgicIn>u1D&SeugH*hN)ou|q@C*V+dgdupQB8Lumq=Kg9iClnG4vNeR%yqP1#9T-7RUGxyF6TmtW-*Y`A<p0u'
    '_XP!JffRU^{MJ~RH$XeaDSGP37)APtH$KVnTVs=WQg2+!0%Ce&l4?@Mqs+0hHy)+uTH{etT5mkcM7<l2k`@fcqblgr#-j>Y5sXJwL2to$)CK1mkD7iFj'
    'EA%uj7O0>4pty(mr#SO<)TIwUY-TAq>!+Y%VSR3c)3Du0V_Fa-os3;(4k-_=lZqA)A>LJz)~(0qHE9<Y1qoixjl^K%y&qGudL-^F6X5n*vsj%c^J&)71'
    'afcxjbkN_!j-1;Opr09(HqHXMyEhNSqJbx!!oNo{e))*w3!JXz;D7NE&==FB5~W9N)u^UOpOao5Jg^d5{&avTafX`az5!BQK_HQAC<1!b+*&)rFNxMFv'
    '=@$}u6qN)@e?V5QP?6x`$^3EZjzvL$dUYmUZSGI0lz;8ruBC2*@t2$`UzoMZ_l6+CDO9XZNS(j{o7k{&?|lws=xCH*0BLgFZw!75I-9U&8rK^oyCI)w='
    'Y6nR5|km-Zs079l$=qeC0y+T^SxKc9A7Yr;Fy`*4d(X}WTS`q_N!PrtM%)o2S*)*TgCD(Y#=s~`WFL|x~Yo-N`Gr~B9Ach#P`HC?~;|wyqEd`^Dqpb>s'
    'nVftXX9Ckm!ARqEF2)+4G@mJgfv02$DoA3q4B%AdB(Y&tFh*gI3Zhipu7V_W2}vpw#F7P+QALpS`YM5>BUF&0V!*4wt<E_Y+!RP^aI2Y_>Vo@1oZ2BI('
    'z9z1QVD?EXniE)s6>Z+9DR*LBOOQGHe};yHU^3Knka99WE>MiVu56A<J1t6ajcX^kc?9<%i~^~dvUF29Ft{mugz@KwC9b?m3MGS6mtsP>mo9R6u8$Gtu'
    'zPsr0U0cE&H5>dl|GAQMynnC=jQMq+AtvHL0!wvAQVb9f;ON<uo6BD_I*6+isE8;>HM+6bRaO=|tDOU!0GyZR5(8yra?y0uaI^Y2H#M`y%q12F>9;t*8'
    '^Yi$xyMkh9l+6>iU4QBLW$qMXX;72}A`TT#wZbbZUKW+>5o_)Q<H3ph?u6JQiI_0!J*-^6%R)U=Dv0_U-D>abR{)ObajdVv2FiFX4S(P(<3%Na49Xz-1'
    'Ri9ILZLgTcHQhzSH2vYF8^dFb=ygk%leyPQa`Grx<Au?GmD@o(Lfsnv4pN1%G0=rl`A`QDl@nSxWwdN$|Cd$KnI;>?rrCprzDSgZ_pBjoI&3sDu<djcK'
    '0U_`WCFNm0jX^Z{Ch39sG$oCLZ_*i=Pd)RXV?K?H3Xb_S(U%<aX-S~sm`}@o9DFOvnc%lRF>)vPtuF;A;kRCQB*AYjqHCO}kTR!8Sjha6kUWE9NKBDfj'
    'rL7QpN8u+PH-QU{Rr-9_ml_@gJ`z}^eGYC=b{67ai2uKv~Nm6h=DZtI;Osq7`eUnWaQR($&5kJl}cs|`g1CoF{s;wY-+`T=)iqMv#8)ba)?)6na3es`5'
    'y8jUU_9#M;zkhONdw4hBPv#r`{aml@FBD1p<G@I)`}GXI+PQ<t3*J%-HfJ#4DRaKSaFpq0*k^opEMt+Qo%>TVgocJu=KY+C3GbS#m_p@$?7EK)VMgd=9'
    '=zU>tV0%ul|syCqJ|saTw7oCS!^2)he0JD9zuw>el{6`&(<$|W$7lN`noI>{Lvu~X5HBM3MNN%Yhk0!AO^l*H7!ASIKLj0mF_%C`<iH;ke=fwacSR~H{'
    'B%J2kbD9M*j<$wolspc>lNlqXFn?of45)^&i!XR()gEWJ@G@>A%tI8o%WRM96J{9dB5PNE>H6ZlV^!!k;eqfXiF5fCGI3AXGDmWgNs5>|wHflZ(O9UPe'
    'c?{UG6n9iIg$al}rVy>Is5H%?lKTRPhZQeoeDfvJ%=jwxmGP;lz>FW9=ErRY56t+1$`8!=F^E=Xq0G$qNwP8HtI!L|_`;|3%(_x`W~r(ImJ!MEb`*?Ap'
    '1LX+ks|d_Fd~)0!i-22jb1P!)j}7HNPVVDYBy*t+YPnmGZM1(P%t9(c#{z+5U&@ENR==+oHNP!3pi)$@h_ayLb9svfXY>8fpeCMeg&MfjKZ*R&Z_M9ft'
    'xsGb~tC%BOf?t)n|M-XC0JTfGB*PMB_%w)((im7u1OZqVR?8E)a!RapHg|jHbf_QCPXj3j#!8Gc=A!BIxUCR&Yd0u2u!di@^3!-tUi0*T)GU3cDGsI)N'
    'yRX7vh0VM4LG1fnp8tRjIZhN_UYj+z*rwNB3L?13okrn0(Zt&^A96X$gd2%fc$ORBhWUeCMcS?hRaQV)MfG=)zz3wMM!l<Gae87kg6pJ)?xA9QsouHqC'
    'EfM=X?yavA16=I>QIK^w``#42Lj8lqDo^i^nGr~BfWaAm9lsvrRlnQCaDS;2qIF+<#CD6>ZmEQs~UkNmejx1h96P)#H3C>i&3ykWF1f#hM{+cdGfa_~!'
    '1BMI_*QpgIJY1(1Nb+!<x>TPBu2c6i5RB$denB*9yu^V$mZTmB@n~K!=>-DP+!LJ!BGTLuas@)tT<IbZlbR;=4fByx%pxky)zb=ur3R%Lg@{XY<$#b=0'
    '<SooQvz8a6uFox#8U#5jG>?ep-{w^1Zo)-l7Q+6g(P5k?GYYR6p5ad9F$Ef8J}G%HDCZPYM#Qg7Rspw&swOEu2>6swa-`!Ii@Sm{#EayYG8>yJ6O8;o-'
    'HiLeDB%Aa!h+(%^3BA)Q#!Kl?f<=`Z01#HDrjNJUcQ`SAZ=U;TX@JOp@i3Y7AXAy0K2qwoIaJd{S-0Orx5R*fdZNY6u4Ep_Ia@hraDcJ(SWt^`QPvpdP'
    'Az&ze_&)>ciZEJZ!&It12)InN$g6XuoXuqKpyvL+OE4A6rCNR6Wgh{~+cgMllpRb=G;(1TMyH$V>-aw%I<4ZyQN4`vF@VMMO9mROc>Cm?otwE{vaw>UL'
    '0lgVcSGg*%kn8|W>assh+-`pg&R?QiRtp(SDu(co#v2_%%LHG*&xk_eGAse<`0^wa?-=#+kY`pY{p;E)1va$6NO3h)zUizt+xxL20S54o5@D+qvgZeME'
    '_5@$muA%S1a6ZCUETmh6ub8u*32MSf&+Nh!lsN*KB{g{g%t{9F2wYaG)d*PDxAuX`RIWvUvP!QkfXS-9u~`mn<_JVqtAz<brlx;X2Zk6R5|61;i^OBPw'
    'UKyCeYNP7OxPbK9#bTU#AAwik$BAY<JC5?M0H>o)*|uPKo{F?Oyq5g;>+6>r3&7*6g59?I|GroO}Remw&`5DZI#y6+s;5_)T!uwQMWDEI6E79EySp!>N'
    'PUzsMQ)7b?QRn>ZnjoMxDBE6zi6YuD9*qs8%qsL}b*d7-?csM}=}G-tt`IfeBDFJ3q2fQu5S3>^RXYBW{}$K9B4PJ0gq7rm)lNrSi7p6nWc9b+Oq3q!w'
    '`31f(Obns8#oO-nK@E}H7Th<jG*Jr}N-qF*FlWEzXOWO}I=amPwt)rc#mDlg)O3F<^#FunAPxL*|kUc~iM=RQ{3GA33`k5RpLVj-!%W*NiF71^i~l#gs'
    'w306ins#MrVHmU^CBC&yIl8uB0D#BCA!j<<fl`LGNXcn$s%>ov#bg9H3ldGY?AS2wB7-UpVr0TZSoT@vG0|uF*5Y6(Sry1PTG^uV|&8e!lE$CF|+ZSVr'
    'jb6PVNzg-3*^{6Li{hdg+|(SY8Qkcyspi{AowsIilOBMOg4*?oZ(n82GS9bt>ymF@YrPpEqUzs#s=>!#p48w|Lq`oh@gkZkYrQlfhNLqjVky}@<$Qsp<'
    '_qC?!7H8A;FBJRqiSBQ#0FnZ5R!<Z1osm`lyG$-hUyztZ1Cj_mWd5MLHwi_W|iLJ=!Kc)^MzUec885cHG|aPW7tkkFRt|3hcB+QKO1}kEr|_2^GwpX5>'
    '#74KvxShIldSai7UY}IkCjfWjB@Bx!x(M#3oOMbWfm>7}5hvn4}V0@~XsELr*1k)cmyVMI?rFm7<9uJ<6y05}RxL5}UHA#MVK20HtvPE4{hu0Tg9cQ{C'
    '%DnBIA54nw+RU`SV(O#)E1_Tx^rS~u&>3O_2a!R$?@cf;8t)4Nd;eDl{+Wl!-PPES=^svam@k)DcMDs8Fl+IqXT9@est-BZPua$Cx8sj#KumP%V{;K9C'
    'ZPi@!6YD;bTBC|*Xi0`7REstv8!R`#dWiLeE@@yDmy{dUd*izfIwcApIEJP7p%e>xGg9OYwnzz*&FEGEQpNsNaDu^=UPAA#^Puaz?{dfGtWIn|YQ|8V2'
    'blqMXb=h|I?|OROW<Q%FO#jr{>|zK1bLEapHaM{_=lZ3~FWSeR!%thz?8^}bH|>0UgI~ck>-Mr+-eA&nK50LIILP+lXZ<36&JO1DO*6|re*D;rb!XRC_'
    'IcD@jIP^ewOO`b+V=7B4=z7He06gE-NB<zSbXwb_V}uq&PTK9{B1iKz3XP1>vlEz3UA21-Wj!<Wq0?-ITUJ8(jB>;H*3>FGaEJIakrVTS0j^UYO;09(N'
    '~A32O|`)Uw=v5N5#{B{X|L)WlxvgYBid4CewO49xayLWHVlmn)T?qS-x%8HZmQzE6D|u-I3{O+bz1;)Yh3??R-65-;BOmwu^4Lp3X17EX<*avXeb}YGG'
    'g}d;D(}L3gqz-&-QKhkX}zvO{~so$SaS$vzz$0<p)b#n+we+Y^hTJK1+9h@dv-4AIl}`2!;7PImkQqN6=AZd)C{Ld3KAc-m&qFmAg&KSp%g$qw+@_Ir7'
    'ZtYObHWQ?8c<OQ<APIh{POkvOOkqwMJKHGk$hj<NhPop-HJ!)r{(+&I^=gS^#W|z%!iXUC!pxNU;wc{&(#DTLX?Hp>(bZ}GX^17MBm!W{{aJBB{oD4-|'
    'N6U8B%qN)?VwP*$F&q?n$xavMWSO*+eY<St{0d6S9xpp{9&U4nZPv{UETFLL@y%js@mFfgj+^y6PK5HZ<LUZp(?|elF+1LTXs_RNo8@IDeP-Wn=5{>n2'
    'DN6-ni+pz2Hj@Q+x1nugv8h%YR(QetF@E?ZD%jL{yfmReOj(Jmv%3u`|PB<?wP{`+3Bo%*Y=;k?>{$h`57k3zMand-C&vQ^nE*N=W<)dm`@_dmz>kWe@'
    '^!V`k;dX)zd|d@N`lm0^QVzNJm8vfUaspNoO^pqPwCiMu)`(mCG7c$Z3soa$BRk9M`BI*EK52d5ubPUw|Aruu+v<*r-}gjG!VnHmZ;#8|CE6MtM22Q9<'
    'r(6yJf~b!6#4k3kW74BF6RP>CLcUi280qsO2jJq9)DG3ZK<L1B6fTGM0DoF0Sr^cXa#$Dl<$22JWQXj6|tqk0Tl)nm}C9)ouE7&NTMpk+M<P3tjeTaQ8'
    'GdVHjRwjb8FY0jMZ5_kW(O`Wu>@p8IY>$jc{7vG#+wRhtFbY)p%Hr~we+s#q48rg*#Es-dY!Ir~MKfl61&E$`E`E1$+7S1nb%qL%u)*V+h@7iT^*^U<7'
    'VxzCHMKfA0+VS+l62nc~EO$ol?MAoU)Ofca7Rz?EvOO)mT~B8-TeTY59lX2Cr}NfV-DZWD@L;t4ZK};#Y~IYKt1FxHeu^pU(Zm>EH#3`uDSbND5&1(@$'
    'cGQ>&f?kVVv5{o>lPcP@5`>6NS0N*e1{Cyr#9p8=ElAlwc~EyUEh$O$<>U;?Q)GoZOM4rO}M?qi3g*jwaGo6ZOj<Iw(a`La^;ojuTPkk+D*gnucqUxk!'
    '9-H<ldKGOFzo8tA1KzI=w61Wji-(%#52^Wo{5?%~kVmzH8GirVz-o^?CyhuOy*KH7E(9>vU$fX1t!hLt@4baG37;x|`2#Mw^v<rUB~@H}nhWbJD(RXWf'
    'Ear5C`B<;{Qp$A7Nu6l>Fti8ixM=a(ac@AFBsgf+kG-rGv^ktwB_cWbV_Ud-Bc`>Ue8Z)X>F^arDFjdlB>!PFgEd(ySqp;!Fqx?8PBo4MsUQ>Qsld)1j'
    '@-*?Nm4@S@1DH4qqVfDIs+YT@Z4Cn2o`J&mNy|`$h^<UG^jvdLCwexq=Wk;RcDJSjavYF74A3WS!e3J>&W~Wb{p6As$Lz3MZgP%u}JHMvB{dso5pJ(%D'
    '+3TfQ9EcLx=I`6YCELd0L^C&?G>Qg4{gSP37WfVK<CDXKv%@Fi)Ao^Uw<vb=bXLUuyxI@`pYW#5w%9Iua&&rn`0VKU!P(J^=jTse9G(C8?3i$TJ;V18o'
    '5}TRe>wrGyO^4GKU~k|t9^|A{&jY>UN83d_TImL|KPoU&@C_b=Bqt?_q*3wKXdiA`}LW~4tu%V!-TzYXZF8ZU(dGdCgb((vc+aO<D$uU&m8Z1wYQzOoL'
    'v0s_EyN%*xtEao3gvQF++aXokOy3bf}}-L)nb|#sJO0a%Fo>x~VqGAGkWN_O@oi1`v9GFanpuw$SOX2)$Y~%RPKQHZYHUZHLcxeYkgT&u<=@0`|uiyAj'
    '~Qz=9rst$&w+#maG`%a8hD(b!{en7G$$EP^i0r=ea0<K`8+Z@S6N9$o8owkIaj)oNoQ;MWgv-(ba_dB^Gas=aQ02Pd<GAhqr0#cokQ+`HZFn}>U!=y({'
    'pb)|2<?1zUxzC1oK`o+l~&X13te}Bgz*OnxAw}@>gyIQsvHau9Z9pBLeC^QzR{5Q+)y#?#tH_iBMJD==Xnbm7>&k~bmTWI=b*9_3E_AcA?-qHM`JIBX!'
    '7zuw5fgbyV#pD9wm>9s&Hpieky#Q1H?;h#@pWrVZ{mbFwv-7h*yga-k^OJ+KgVV#aFE4v}a(H_9{EQz??=1Z3m)BO0TY3E2BKK>9WO~jzgI=IDwDVej9'
    '>amtJ}WMkfBCN!G70>Ia}?_KFAUeTAMJBH|7^E*{>g4n+KXm0Lm}S&_02AppKXSrdz-yn{^ifxT}Bq5Mw7q%+4f(5z9Y8013-CxjE%d=aBd35m6nnZR!'
    '@e3eM@$&jdsgP3!(Cz?d%_>?R&0%XT$?@iJ7$b$5l7~mUqg&oo{9{8{>&*ma=)`eOO}49E~3?ytYlJh8#}s%<R{%45Hls;&pcL8ZK~WBQ${i{*Ca({eA'
    'kL&2Nzjt(Z^6x@^1ODAmuOXb#oxx6|3W?cHO`0`T^?@xYp4mxJSDZgR2gu5Z6ycenJ#^g}z@>dgGJxv`YHHTEyV$iau<lHC;d1^iIl>g40F<Fo1fS@U6'
    'VSBTUsUM{;oekN(NIKArLAIxS?v3~GP8-BjIe$y^rTs-b(-E!6Y>9U!>wL?GC`G&o(rk7VU`~Uiran{T)7WX-F@X^hap?i7z&6|xW)M}xV&8%IWu92EA'
    'Z|;fu7=Ox#imO>-(`T*aa&n~hKK}eS2C(xV4o^;vxD&|U<>%r2$1krxeevq#@!|OpC~p@&KX`UvNag(H<1a5hJ9-9m@$4nn-^HR|_;bR)VW+3ZFCHBn|'
    '6}$-E_;3-B>(T59iBcrJ^$|P9xy(<%~K=fK`ZyJ+ck=W=Ig&T6^<K&QhUR)y>`C0np_GFp0}&()pX$iia!(O^vSay|G}al15XUQwbxca!{PQWrskUejD'
    'LT>mGE=%R+H-w5B@Y=+?Dagqto-pFJ3*r>pLz_#iw&#9-bV%cyj*2T-1Wc`LR_mr}x*OGxAs0yf}XH>6-hd>z(=N`RUow*{f|%7kWGFPk(p8tC!}uhrh'
    'vl`-P`xC$Ao#y*fGkWOZKe=G9qI+*|$g-4|cmw%Z%LH!<Y@mVFYu-7u^)ZP7hl;ML2rdpBM2B|wI*(V|;TnV5Ic!Z6UZ3tQd3Df#K=;q3;f?JEU*U>_f'
    'ZI-@`R{PE+*{{c`-0|XQR000O8m!WB0QQ5S%>j3}&E(8DoI{*LxWMOn+FLQ2TaC2yHbaO9rWM^M6IA1X^Uu0r4WHVzhVlpshVKHGhVrDopHZE;pZfRy^'
    'b963hb8l{?Q%!H1KoGs>S6DvylEy6S00zmb{IRi7h?5%IQc_iD1A|TFfWt1mMEURC4^r0WUU+Zb%+4?m_T#G8>t~cx2D~5i`YbynXq?!n|4n9^`XtpO='
    'VumJ7NkRH>gr4dSduYLxkb-~l@*M^TTxP8RHY^5g7G3{mOVt4>~H<+{QX4B!-T8{c@6KA>7XeKKAA^*j1scjB&+FcP|~E}@HUy*%O8USRFmme!Hj5FDp'
    '+k4%%3&9C|KMpAiDA{)SxRDS|pJ^zEcotG1L>Hy@oc@B@a4U>vK;nXmRtXAl74JJ&(jXBHG2gIB}?%aPmWhQV^R<U#BWBeHI=_;lhTmWb1~^OGCb7F+o'
    'T_RUpFYwoM2=BL-h~e6{KL{Mq0y9bepcoEUxXLW3K97Df$;j9I&mhel0kyl1pGxTzaw9}InM^nWtBQL}mMcx>zv8}CnILnnPje~IU(`dl^ex`02BB@Iu'
    'd>lk78#zr@`yS2T*ae`YsbZytg_!^-AwReRbDRx|0PqB}^4Ef`+?|b8Xl+jn9{ut~u^>WBEoI81%j?k?S;9wZNdK3`$FrW;(km9yqNzd$9@kIPSaB(-2'
    '{~a*=&8o7UQZ;o7NplvL6tatLoh=>(YQbxS#g<o}e*sWS0|XQR000O8m!WB0rFF&?!wCQY`xgKJ82|tPWMOn+FLQKtWO-k3Z*X;NVRUJ4ZZ2bUc1@T~Z'
    '(X+$hWGs~XTk3wmuf6ULM`_qx(Sf(T)-#_1hEk$i~jpQFUg?{0~bDzct{Rs=AGgB;p<<&|M|!Nu3x`={^PoR{r1;ypa1;t*WbVU@6&Ie|MR!MKVB)sc*'
    'Ph(Xt`hi-Y5xK?R^}SK{4B~a&1yXdgvuMsgRT{y^m>-%&=%qOr%BzIq4#!R~Fl3KuWDOI!OUiQz~bzq+Qoqtob}in{Lrs%1*RrE6GNK{J3nj+(yaKK>{'
    'RWW(WB)-mLY!QZCahH9M%6g*4g-4%*d5?pB8byIL=!V%Y}$YF4ehgK@RiN;KO516yq)S#=NrIjw{HahcajPo?`#v96V37TL!S|JsfWr8kpxi*O{NhFVH'
    '>l0`VCbV!*b6&>~>RV4ShnMIlisAMN5V!fxcI&KV&K&t1hAyP&UK1OZ~oE86^%sr3Nb0g9gN^6T2%m-=H+D_uN5C2%E;-Sz|caQ*GuQ|vU1JZXR3II>%'
    '>YxJZuy{bbDvPEX9T=c=YiHRG=rPLm3yh0ZTj>4@7S^7Uwd{bMk|Ck^DwtQh<&u03tg9oZR&nfyosgYjDYVL&eAo%u193WZlpl6JVlK&6nv(({9m-sL='
    'c6`jHJg)m^%SkPeik!GO|jMUR)^T&prVs;^~`UIK8DTr99ttZoJ5gYLq3E(XEqp<uZ5+IiM@u1VUI38nPh6%_2|-*wIOqm1=#s;K)|w6j|Wr%PA58ObH'
    'pgcflbil`$yND97oLA!O$vYhV!E<PEK74xPf+>k;(2rJI!&5%cILpZipt!zHH+P5!DqPFWb0+%%yG)@>N<GX2U@NWa~9Kr~pIM&MWOIrnqhn3@}ElXGF'
    'hBuhnGS%QlV_!!y1QMJbzUil&!sTn(-D;jbu4V?{cBQbj2zoQIC^qn8cTK9A&NRSk0Nj8Wn$|3h;kwdtdzQJhF^I(4K(CsG^BwZt=z)P~foI)z21+8}l'
    'L%ska5*BPXfEVV)DhR;0JhPslAuRg`5wAjwfQ)~(gOh><MH#Wt(%Gx6Mv9aegWAM!#NvO9PoXC?D0W0S%xw526wQ<r!VxJ+ih|RYw%M*)qtzIhjJHef8'
    '#6FaulVE8<^$`RggQdlg9UE>1i$buf_{>;ZDBm?V=6U*QS*pdmr9-GXoO&BQld=nCm8uZ37F(oH%}@|LeFoJ_o9<DvvPe_O%RP@XH)XPQ;zOdni0Y(78'
    '_D>#5tY)?2*vz0l}f|hRu5s@W~o=XY!SB2%+Z1`uWE^`VOe16=WT6CoyG&#KW}4+c+4mcIH6RmYnJD2X+fG=P7d&C)JoL>E}b3E0eCdnDk9I@NW!aR#r'
    'yL%F^@*wZW~}B{zrd(1O{eulaq0f05m$mSHZlDu6Cu>%Y9eKL0!e2PB@dG%G#?F&LnUqISeOwGUP(WoVTjRjQZU?P4hGfyAvE4W;UMHyH?LOtzLvmDNi'
    '$q7@0^aGzkCF`yM12qVTkt!;7FpBT#(uAqY_)gC<Q~&ab;(W~$tIrtq|h_jA^BSAAX26iUT!^5?4Ou0pz=DUz7i&Ui{B31HoyDUik+XMW1#;Md1v*9fV'
    'oSy|8V8j()T=f|dz#{2wOG|C8yx{tpsds3i`%ef0AeoE;&l?C#At*0}h@WZbXU;$KhutycA1rSw&Y$(_lKvM@*Af6FO>c}Ol7A=6HjyT4dRDwFxV)(Ms'
    'Q{plu-<af-M%d0OQd1U9YWP0xxFl**mCrx!u%xQyrgZqYqmt866?A}cgDh;xK?NlAr8sC8iL8?6zyJh}ojE!HVNu2VxZ^SPnBD=#B@%g@S1^tdnClzEI'
    '7SfHzIBXa6$+k=`*|lF{f#au>zU}Ksi+G11qj}ZD4iE*N(%>kUQka;wsn72OWIO7<4x&65}gUV(lg9Z`FYD_%u@#Dv&1sdmHUK}%8YpWg7a2Uvyyu~BM'
    'Ib!1mSsGG^CI!S<l-f5}RI7z3KC|X~@MY2hZCmUJ0{>d)`)&`nAif=WUjHo>tGScB$qqDc*L`nWOR5!2kr*%3o2IsbQI_xlLn6av(Zjt4*~$J7A}{Z&u'
    '{j9gN7&P>=VzW05;ql@h$(UH_T!jF<dl$@x3#pL^HozwS6JxHF$;w4dJI&U2qNSdsmD_r@&E1bcNwjOyL|S@E>)YOk(_n$`CsE#x90<@@A4S>w!T2c3|'
    'RNG_V*on#S12&xl$GHN4j>3IwF;C@Rl*PJvF8&M7NC-2GVZDZ9n?UVOp7D*W?*(dMGsLR|g@+OiH`Vy<%SC@uR&R4hh$$K(Wn9J9nyeHGWE2(#QxVz>f'
    '`Et#}b{%;~MxPO=excauX7R9{M^u`raemm|BP4Oa0SG@aHd~E{?Z2+Zf{WM&NI%!Rq>l~{+k>4+%zT7yMihf^Hl~Y_u6OT%ozi0s)rW`e#X*>w%)M<lW'
    '^R?V+ucc;1Z_TJ@)}bWY`5)32<N2mylpo^LT&tMlXfu&2df0XZ8z$p4ncUv^hs#V%*j||di2%Rp%6k;B;TQ06gqUe7PRThw+-2L;|e7WJ&cVvT}kQO#q'
    'R#)LP9F<-R|eh1%+O!a=V`|7a`9Z+3oIkiBd|KhjRD31Xq;!K9BAa?4hdq-|l{w-~v!o&u@3XOK9VE)*6%KqVjpW`(1*7oA;mF-R}}vHyLv>?-FT!?7H'
    '3kE<uns?dPX|dTpns27N4AM8!Qwmzon*cZs_4d=V_AnX`F=TV&REBh1W+3cS%YD6<w(g>QvDp0!lsD|YoWk7_&`SftI9%p$LwFRuNv5vC5UpbET5QYTP'
    'o9N!0Jp;}WCdV6tWX))xn1!QZj^H>*<sq_Lmp9R#&s9seERe&3FanJ;GX>wo!3Ka)kfVc||sP(F8!Cz6W=g52WWmW5SFIDkhAPJCwK8I?3lmv1FZ~h62'
    'yKhmkc=JzCg)>QKgw&Z?^BJLPukRAWL6v^T{0^D`S$*c9;FrvWH9MfQE+tPLe)CUI+z!h;LjMF+DmM;)MOB0{Wc`fLPU!-oud?b$pX&1T;>PK^-5gBg;'
    'uOB87dH-7lV6@*+&D5%N%ZvM#&Kz$H^3J+j%~>ePup|)U^Fx6X?w0#eaks>3<1i{*g(e7YNWGM8An9h@Kt17)rZpPAnRKpEsjoXf0iNSsgeiKDDcLtyK'
    'nom4Bcd%pF`gEXBjey+jDV1OQTEZ`zuXNPn1m?(9#T~sGd0_Gx&|0ffLCLm$-Q&dwV=G!yR;{%;S+6JTuOiDKoUxs#|Z5M`lu-AFclnP)h*<6ay3h000'
    'O8m!WB07Uom!(E|Vg<_Q1*9RL6T0000000000fB^si003lRbYU-WZ*X;NVRUJ4ZeML>bYWy+bYU)Pb8l`?O928D0~7!N00;n=p=n)zP5jX14*&oiA^-p'
    '~00000000000001h0Sp8H0AyiwVJ~xTVQ_P3Z*+4nWMyq{UvO$+ZeM0*VP;}7G%;f}H#IgkVK+EAVKgpkb8l`?O928D0~7!N00;n=p=n*hd&wCM0ssI?'
    '1ONa#00000000000001h0YMZ10AyiwVJ~xTVQ_P3Z*+4nWMyq{UvO$+ZeM0*VP;}7G%;f}H#IgkVK+EAVKgpnVQy(=Wpi{cYIARHP)h*<6ay3h000O8m'
    '!WB0&IvzPqgwy~nBf2bVgLXD0000000000fB~l#003lRbYU-ZZeeh9Xm50LFKBFZXkTMwUtx1(axz~sHfCdHFfubaVKXo^G&e9fWo9=vG+{V0GG#M2Fg'
    '0d5GG$|CVPs}GWi>NoVPrI6Vl!kgWi&D{IWS~qG&ndeYIARHP)h*<6ay3h000O8m!WB0(3wVPN@xH81o!{|HUIzs0000000000fC0*K003lRbYU-ZZee'
    'h9Xm50LFKBFZXkTMwUtx1(axz~tWHd4}H8?P4I5s#qWMejFHDNR^YIARHP)h*<6ay3h000O8m!WB0U4~cXT>=0AaRmSXKL7v#0000000000fB|UR003l'
    'RbYU-ZZeeh9Xm50LFKBFZXkTMwUtx1(axz~tWHd4}H8?P4I5s#qWMejFHDNR^ZDDR{W@U49E^2dcZcs}B0u%!j000080GFX@UCs$VSEE}10GQzb05$*s'
    '00000000000Du7^-T(k(VRT_Hb8ca9b7*gLb1!IYbZB2=WM5%(WO6cJWn^SEGh;SpWH2!}I5c85W-v1_E^2dcZcs}B0u%!j000080GFX@UG?8TFirvh0'
    'D=Vo06zc#00000000000Du7*IspJ=VRT_Hb8ca9b7*gLb1!IYbZB2=WM5%(WO6cJWn^SEGh;SpWH2!}I5c85W-v1_E^T3MX=Y_}bS`RhZ*EXa0Rj{Q6a'
    'WAK2mqI%X<g<Bji;Vk006V<001=r0000000000004ji#5@52WMOn+FLQ2TaC2yHbaO9gY;<T}V`N`pb7XQ~Gc;pkW;QcsW@TkJVPZ2kW-~Q8E^2dcZcs'
    '}B0u%!j000080GFX@T`FQG3sM3A0DlDl06qW!00000000000Du9tmjM7|VRT_Hb8ca9b7*gLb1!IYbZB2=WM5%(WO83KG-G3CHZx{sWo0*EVly{pGc`F'
    'bZDDR{W@U49E^2dcZcs}B0u%!j000080GFX@UFHalr=D5>0JG`<0AT<C00000000000Du8wngIZ0VRT_Hb8ca9b7*gLb1!IYbZB2=WM5%(WO83OVq<18'
    'GBRRiHZVD3GB{>4Hexk7G&g2BVq#)2IAJw0W;8G~IAk|8WiVnhFf%eRG-EV0IXE~nHDx(BW@a!hYIARHP)h*<6ay3h000O8m!WB0Jk`FQWCs8M<q!Y>H'
    'vj+t0000000000fB}H@0RUuSbYU-ZZeeh9Xm50LFKBFZXkTS$X>(s;Z82qEGh=3EFk)jcF=R1iWH&Q3V>M<sE^2dcZcs}B0u%!j000080GFX@T{P5oyD'
    '9<z0DS}i06+i$00000000000Du8G{Q&@EVRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_VQn#GUo&H7W-wx7Ffn8?Wn?!qG-EYpH!f{qZfRy^b963hb8l`?O92'
    '8D0~7!N00;n=p=n*0h4ge_2LJ%)5C8x-00000000000001h0m1(P0AyiwVJ~xTVQ_P3Z*+4nXl!(7Uu9@%b6;U?F=byhHDNh3H)JtpFf}<hV=^%?Ffn2'
    '-YIARHP)h*<6ay3h000O8m!WB0nP*g-C;|WgeFOjiKmY&$0000000000fB|*~0sv%TbYU-ZZeeh9Xm50LFKBFZXkTS$X>(s;Z82qEH8o*5GdE;0W-v85'
    'H)Ap}FfcJ<E^T3MX=Y_}bS`RhZ*EXa0Rj{Q6aWAK2mqI%X<b{;Dso~60089>001`t0000000000004ji{0Ra8WMOn+FLQ2TaC2yHbaO9gY;<T}WoT)0U'
    'tw)AWnW@5F=At3Wo0rnWMMckWHvTtGGs1lb8l`?O928D0~7!N00;n=p=n)bxY%eX0ssJg1ONa)00000000000001h0kaVT0AyiwVJ~xTVQ_P3Z*+4nXl'
    '!(7Uu9@%b6;U?F=bz3GcjUgVP$19G-P2oFl071W-??hZDDR{W@U49E^2dcZcs}B0u%!j000080GFX@T^%D~HRA>V0KE?Y05<>t00000000000Du8F6ao'
    'NbVRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_Vr?;FUokQ^V>B=@H8eA1G%z_fIAJzpV=iiQZ*EXa0Rj{Q6aWAK2mqI%X<fBUU9%nn002`2002M$000000000'
    '0004jia2o;uWMOn+FLQ2TaC2yHbaO9gY;<T}WoT)0Ut(=BV_z{cHe)m}Ff}wYWHc~2HaKB6V`DCDVQy(=Wpi{cYIARHP)h*<6ay3h000O8m!WB0yBBqS'
    '-Ua{wzYhQaHvj+t0000000000fC20s0sv%TbYU-ZZeeh9Xm50LFKBFZXkTS$X>(s<Z82kCG-WhoV`5}5GB-9kGBsmlH8)~7E^2dcZcs}B0u%!j000080'
    'GFX@U5@v{IUWK408<1206+i$00000000000Du7>B?16sVRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_Vr?;FUo>SjWMg7vFfunbIWje4WHmQpI4*5rZfRy^b9'
    '63hb8l`?O928D0~7!N00;n=p=n)3N=v5N1^@uP4*&o+00000000000001h0h=cR0AyiwVJ~xTVQ_P3Z*+4nXl!(7Uu9@%b6;X@F=Jn5Wiw_oHeoR}V=!'
    'YlW@KhHGdD3VYIARHP)h*<6ay3h000O8m!WB0uOp%Y9RdIVQv?72KmY&$0000000000fC0)b0sv%TbYU-ZZeeh9Xm50LFKBFZXkTS$X>(s<Z82kCW@R&'
    'GGd5u{HDfSiIA&yKH8VFcE^T3MX=Y_}bS`RhZ*EXa0Rj{Q6aWAK2mqI%X<e<C9_b4Q008q3001`t0000000000004jiMKJ;ZWMOn+FLQ2TaC2yHbaO9g'
    'Y;<T}WoT)0UvOnPV_!2lFl1vkIb$_5WH4nmFfnB~IWjJ4b8l`?O928D0~7!N00;n=p=n)C$pDlh0ssJE1ONa)00000000000001h0i`$s0AyiwVJ~xTV'
    'Q_P3Z*+4nXl!(7Uu9@%b6;>}Ib&ZlIWS~nHaTN8Gh{GjHZU<|I5{#dZDDR{W@U49E^2dcZcs}B0u%!j000080GFX@U4eam^a%$50Ot<?05<>t0000000'
    '0000Du7{I|2Y?VRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_aAi4TUo>W8IXGisF*RdlW@a>EH)AtqH7;s%Z*EXa0Rj{Q6aWAK2mqI%X<dOPVO%5v003bG002'
    'M$0000000000004jigF*rTWMOn+FLQ2TaC2yHbaO9gY;<T}WoT)0UvOnPV_!68V>viuVKFshWoBkHVmD(mW;HHtVQy(=Wpi{cYIARHP)h*<6ay3h000O'
    '8m!WB0_Kovt4F>=K<_`b>Hvj+t0000000000fB^+X0sv%TbYU-ZZeeh9Xm50LFKBFZXkTS$X>(t2WjSMCW;8N3Gh}07WMVNeI5je1F=9A0E^2dcZcs}B'
    '0u%!j000080GFX@T@7H^p(O$U0AU0G06+i$00000000000Du8!OacI8VRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_aAi4TUuHBiHZx>nVPs-4FgP_bVKHJjG'
    '%jsnZfRy^b963hb8l`?O928D0~7!N00;n=p=n)q+HxZk2LJ#Y5C8x-00000000000001h0qRZy0AyiwVJ~xTVQ_P3Z*+4nXl!(7Uu9@%b6<07F=Jn0Wj'
    'ACoGiG5nW->A|F)}bSIbk_2YIARHP)h*<6ay3h000O8m!WB0>08P#9RdIVR|Eh6KmY&$0000000000fB{lg0sv%TbYU-ZZeeh9Xm50LFKBFZXkTS$X>('
    't5YcXSAVP!XDF*9aiHfAz1GBGkRGdW>7E^T3MX=Y_}bS`RhZ*EXa0Rj{Q6aWAK2mqI%X<bw_%YP*Y000{h001`t0000000000004ji%~%2eWMOn+FLQ2'
    'TaC2yHbaO9gY;<T}WoT)0Uvq0QV_#xnWMMHlF=k?7Wn*PDV>n`FWi&2ob8l`?O928D0~7!N00;n=p=n(y*TvT#0ssJ41ONa)00000000000001h0Y_j0'
    '0AyiwVJ~xTVQ_P3Z*+4nXl!(7Uu9@%b6<07F=Jn1VPs)3I5B2oVr64xG-EhoW@R)kZDDR{W@U49E^2dcZcs}B0u%!j000080GFX@T|rE{xDy8e02vSf0'
    '5<>t00000000000Du9+VgdkUVRT_Hb8ca9b7*gLb1!IYbZB2?XlZj_b89hUUu0xuHaIskVqs-7Vm3B8W-&A~FfM9yZ*EXa0Rj{Q6aWAK2mqI%X<axS41'
    'FL100376002M$0000000000004jiENTJ(WMOn+FLQ2TaC2yHbaO9gY;<T}WoT)0Uvq0QV_#%sWi~iBGh$(7Gh#M2Ic705GcYb~VQy(=Wpi{cYIARHP)h'
    '*<6ay3h000O8m!WB0NP-?Q+!g=;(K7%5Hvj+t0000000000fB~v(0sv%TbYU-ZZeeh9Xm50LFKBFZXkT)3Ut(l)a%VDMI5#+BGBPn{V>vQqWH(`DGB;#'
    'nE^2dcZcs}B0u%!j000080GFX@U3Y~h?d1Ug0O11w06+i$00000000000DuA8g8~3#VRT_Hb8ca9b7*gLb1!IYbZB34b6;X)b8=@gUpO~7V=^)^W@9-r'
    'Wn?#DWimHpV=irBZfRy^b963hb8l`?O928D0~7!N00;n=p=n)iRVr;JAOHXYP5=Nk00000000000001h0Um||0AyiwVJ~xTVQ_P3Z*+4nXl!(7UvhI_a'
    'B_1pUt?r5Gc+<bV`VmBV`VrtIbk<sH7;s%Z*EXa0Rj{Q6aWAK2mqI%X<Zro!Wsnv001Wh002G!0000000000004jim!<*$WMOn+FLQ2TaC2yHbaO9gY;'
    '<T}a&upBa&t0YV`MWkG%_}0Wj0}BWjHoDVK-zoE^T3MX=Y_}bS`RhZ*EXa0Rj{Q6aWAK2mqI%X<fR<IxBY*001;9001@s0000000000004ji_^1K^WMO'
    'n+FLQ2TaC2yHbaO9gY;<T}b7)~PF<)qKUt%&dV>LA}W-w!BV>D(pI5=iCIWB5*Z*EXa0Rj{Q6aWAK2mqI%X<aqRA8qyl000OC002J#0000000000004j'
    'i!@L3jWMOn+FLQ2TaC2yHbaO9gY;<T}b7)~PF<)qKUt%&dV>LA}W-w!BV>D(pI5=iCIWBEsZfRy^b963hb8l`?O928D0~7!N00;n=p=n(-GFFS`7ytlP'
    'JOBV*00000000000001h0T#ak0AyiwVJ~xTVQ_P3Z*+4nb7W^<FgRZ^FkfLZIWjk7FgZ9lIXO2tW;0}AVr5}DIAt+oV>f1HWMndCHD+UDGcYtcWo0!oW'
    'iw(oVP!BfH!@`~WHL5oV=^vkb8l`?O928D0~7!N00;n=p=n(-GFFS`7ytlPJOBVP00000000000001h0eIK~0AyiwVJ~xTVQ_P3Z*+4nb7W^<FgRZ^Fk'
    'fV1Gh{PkFk&(=W??a5H)3WuGBz%1b8l`?O928D0~7!N00;n=p=n)F*|fIn0RR9l1ONa#00000000000001h0j}`^0AyiwVJ~xTVQ_P3Z*+4nb7W^<FgR'
    'Z^FkfV1Gh{PkFk&(=W??a5H)3WuGBz%4VQy(=Wpi{cYIARHP)h*<6ay3h000O8m!WB0rFF&?!wCQY`xgKJ82|tP0000000000fC2LJ0sv%TbYU-ZbaiB'
    'TUvO`5b!=gDX>V>WV{>*;O9ci10000j04)HQ5&!_~{Q>|000'
)

def _prepare_bundled_assets():
    """Restore the included data into a versioned writable runtime cache."""
    import os

    # A per-user directory avoids writing to, or assuming write access to, GitHub.
    user_id = str(os.getuid()) if hasattr(os, 'getuid') else 'default'
    root = Path(tempfile.gettempdir()) / (
        'european_public_health_' + user_id + '_' + _BUNDLED_ASSETS_SHA256[:16]
    )
    marker = root / '.assets_verified'
    if marker.is_file() and marker.read_text() == _BUNDLED_ASSETS_SHA256:
        return root
    raw = base64.b85decode(_BUNDLED_ASSETS_BASE85.encode('ascii'))
    if hashlib.sha256(raw).hexdigest() != _BUNDLED_ASSETS_SHA256:
        raise ValueError('The bundled Eurostat query cache is incomplete or has been modified.')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for item in archive.infolist():
            destination = (root / item.filename).resolve()
            if not destination.is_relative_to(root.resolve()):
                raise ValueError('Unexpected bundled data path.')
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = archive.read(item)
            if not destination.is_file() or destination.read_bytes() != content:
                # Each writer owns a distinct temporary file; replacement is atomic.
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                    temporary.write(content)
                    temporary_path = Path(temporary.name)
                temporary_path.replace(destination)
    marker.write_text(_BUNDLED_ASSETS_SHA256)
    return root


ROOT = _prepare_bundled_assets()
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


CACHE=Path(os.environ.get('EUROHEALTH_CACHE',str(ROOT/'data/snapshots')))

@st.cache_data(show_spinner=False)
def population_inputs():
    data=pd.read_csv(ROOT/'data/study_population.csv',keep_default_na=False)
    meta=json.loads((ROOT/'data/population_metadata.json').read_text())
    meta['snapshot_path']=str(ROOT/'data/snapshots'/meta['snapshot_file'])
    return data,meta

@st.cache_data(ttl=86400,show_spinner=False)
def live_inputs(cause,source,refresh_generation=0,offline=False):
    return load_mortality([cause],CACHE,source=source,offline=offline,refresh=refresh_generation>0 and not offline)

@st.cache_data(ttl=86400,show_spinner=False)
def contextual_input(name,sex,countries,offline):
    return load_factor(name,CACHE,sex=sex,geographies=countries,offline=offline)

def remember(kind,settings,tables,inputs,metadata=None):
    key=operation_id(kind,settings,[m['sha256'] for m in st.session_state.snapshots.values()])
    st.session_state.analyses[key]={'kind':kind,'settings':copy.deepcopy(settings),'tables':{k:v.copy() for k,v in tables.items()},'inputs':{k:v.copy() for k,v in inputs.items()},'metadata':metadata or {}}
    st.session_state.analyses[key]['metadata']['available_source_snapshots']=list(st.session_state.snapshots)
    st.session_state.pop('export_archive',None)
    return key

def load_mortality_view(cause,choice,refresh_generation=0,offline=False):
    """Keep the two collections and their source manifests distinct."""
    parts=[];sources=[];issues=[]
    for code in MORTALITY_SOURCES[choice]:
        try:
            frame,meta=live_inputs(cause,code,refresh_generation,offline)
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
    generation: int
    offline: bool

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
    offline=os.environ.get('EUROHEALTH_OFFLINE')=='1'
    with st.sidebar:
        st.header('📊 Global data filters',anchor=False)
        mode=st.selectbox('Mortality data',list(MORTALITY_SOURCES),
                          help='Choose a Eurostat collection or view both on one timeline.')
        combined=mode==COMBINED_COLLECTIONS
        country_slot=st.empty()
        available=list(CAUSES)
        cause=st.selectbox('Cause of death',available,index=available.index('I60-I69'),format_func=lambda c:CAUSES.get(c,c))
        refresh_slot=st.empty()
        if not offline and refresh_slot.button('Refresh Eurostat data',width='stretch'):
            st.session_state.refresh_generation=st.session_state.get('refresh_generation',0)+1
    generation=st.session_state.get('refresh_generation',0)
    try:
        with st.spinner('Loading Eurostat mortality rates…'):
            data,sources,issues=load_mortality_view(cause,mode,generation,offline)
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
                              combined,spans_change,generation,offline),sources


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


def show_forecast(observed,prediction,title):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=observed.Year,y=observed.Rate,name='Observed',mode='lines+markers',line=dict(color=INK,width=2.5)))
    fig.add_trace(go.Scatter(x=prediction.Year,y=prediction.Upper,name='Upper limit',line=dict(width=0),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=prediction.Year,y=prediction.Lower,name='95% model interval',fill='tonexty',
                             fillcolor='rgba(17,125,119,.16)',line=dict(width=0),hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=prediction.Year,y=prediction.Forecast,name=title,mode='lines+markers',line=dict(color=TEAL,dash='dash',width=2.5)))
    fig.add_vline(x=int(observed.Year.max())+.5,line_dash='dot',line_color='#72808b')
    fig.update_layout(xaxis_title='Year',yaxis_title=RATE_LABEL,hovermode='x unified')
    plot_chart(fig,alt='Observed rates, forecast and prediction interval',key='forecast_chart')


def render_trends(selection):
    s=selection.settings;selected=selection.selected
    with st.sidebar:
        st.divider()
        st.header('🔮 Forecast settings',anchor=False)
        method=st.selectbox('Forecast model',list(MODELS),key='forecast_model',persist_state='page')
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
        st.caption(f"{method} · {horizon}-year horizon · {COLLECTION_LABELS[s['dataset']]} series, {period[0]}–{period[1]}")
        if selection.spans_change and s['dataset']==HISTORICAL_COLLECTION:
            st.caption('Projections from the historical series use the historical definition.')
        op=s|{'model':method,'horizon':horizon}
        if st.button('Generate forecast',type='primary'):
            try:
                series=selected[['Year','Rate']]
                with st.spinner('Fitting the selected forecast…'):
                    predicted,failed=forecast(series,horizon,[method])
                remember('forecast',op,{'forecasts':predicted,'failures':failed},{'series':series})
            except ValueError as exc:st.warning(str(exc))
        result=current_result('forecast',op)
        if result:
            p=result['tables']['forecasts']
            if not p.empty:
                show_forecast(result['inputs']['series'],p,method)
                with st.expander('Forecast values'):
                    st.dataframe(p,hide_index=True,width='stretch')
            if not result['tables']['failures'].empty:st.warning(result['tables']['failures'].to_string(index=False))
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
                other,meta=live_inputs(second,s['dataset'],selection.generation,selection.offline)
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
                        factor,meta=contextual_input(name,s['sex'],tuple(EU27),selection.offline)
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
                if s['dataset']==MODERN_COLLECTION and s['age']=='TOTAL' and s['sex']=='T':
                    candidate,meta=population_inputs()
                    included=table.IncludedCountries.iloc[0].split(';')
                    weights=candidate.pivot(index='Country',columns='Year',values='Population').reindex(index=included,columns=years)
                    if weights.notna().all(axis=None) and (weights>0).all(axis=None):
                        population=candidate
                        st.session_state.snapshots[meta['sha256']]=meta
                        captured['population']=population
                with st.spinner('Resampling country histories…'):
                    stability,influence=dispersion_audit(inputs,years,EU27,population)
                remember('dispersion',op,{'dispersion':table,'stability':stability,'leave_one_out':influence},captured)
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
            with st.expander('Rates and endpoint sensitivity'):
                st.dataframe(table[['Year','N_countries','P10','P90','P90_P10','IQR','LogSD','CV']],hide_index=True,width='stretch')
                st.caption('Included: '+table.IncludedCountries.iloc[0].replace(';',', '))
                if table.ExcludedCountries.iloc[0]:st.caption('Incomplete: '+table.ExcludedCountries.iloc[0].replace(';',', '))
                if table.P90_P10.isna().any():st.caption('The percentile ratio is undefined when the 10th percentile is zero.')
                st.dataframe(item['tables']['stability'],hide_index=True,width='stretch')
                st.caption('The country bootstrap resamples whole histories to assess sensitivity to country composition.')
            with st.expander('Leave one country out'):
                st.dataframe(item['tables']['leave_one_out'],hide_index=True,width='stretch')
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
    if sources:
        with st.expander('Eurostat sources'):
            for meta in sources:
                code=meta['dataset']
                st.caption(f"{COLLECTION_LABELS[code]} · {code} · Source update: {meta.get('updated',meta.get('source_updated','not provided'))} · Retrieved: {meta.get('retrieved_utc','not recorded')}")
            st.markdown('[Eurostat causes-of-death metadata](https://ec.europa.eu/eurostat/cache/metadata/en/hlth_cdeath_sims.htm)')
            st.caption(f'Dashboard version {__version__}')


if __name__ == '__main__':
    if '--archive' in sys.argv[1:]:
        _reproduce_cli()
    else:
        main()
