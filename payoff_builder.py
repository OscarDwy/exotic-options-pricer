"""
Payoff Builder — Oscar Dawny
EDHEC Business School / Centrale Lille

pip install streamlit streamlit-drawable-canvas numpy pandas plotly scipy
streamlit run payoff_builder.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
from streamlit_drawable_canvas import st_canvas
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Payoff Builder", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500&family=Roboto:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .stApp { background: #F7F6F2; color: #1a1a1a; }

    section[data-testid="stSidebar"] { background: #1a1a1a; border-right: none; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    section[data-testid="stSidebar"] label {
        font-family: 'Roboto Mono', monospace !important;
        font-size: 10px !important; letter-spacing: 1.5px !important;
        text-transform: uppercase !important; color: #555 !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background: #C8B560; color: #1a1a1a; border: none; border-radius: 2px;
        font-family: 'Roboto Mono', monospace; font-size: 11px; font-weight: 500;
        letter-spacing: 2px; text-transform: uppercase;
        padding: 12px 20px; width: 100%; margin-top: 8px; }

    .label { font-family:'Roboto Mono',monospace; font-size:9px; letter-spacing:2px;
             text-transform:uppercase; color:#999; margin-bottom:3px; }
    .value { font-family:'Roboto Mono',monospace; font-size:22px; font-weight:500; color:#1a1a1a; }
    .kpi { border-left:2px solid #e0e0e0; padding:8px 0 8px 14px; margin:4px 0; }
    .kpi.gold  { border-left-color:#C8B560; }
    .kpi.green { border-left-color:#2E6B3E; }
    .kpi.red   { border-left-color:#8B2020; }

    .leg-row { display:flex; align-items:center; padding:8px 14px;
               border-bottom:1px solid #eee; font-family:'Roboto Mono',monospace; font-size:12px; }
    .leg-row:hover { background:#EFEDE6; }
    .tag { display:inline-block; font-family:'Roboto Mono',monospace; font-size:10px;
           letter-spacing:1px; text-transform:uppercase; padding:2px 8px;
           border-radius:2px; margin-right:8px; }
    .tag.call { background:rgba(46,107,62,0.12); color:#2E6B3E; }
    .tag.put  { background:rgba(139,32,32,0.12);  color:#8B2020; }

    div[data-testid="stTabs"] button {
        font-family:'Roboto Mono',monospace; font-size:10px;
        text-transform:uppercase; letter-spacing:1.5px; color:#999; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color:#1a1a1a; border-bottom:2px solid #1a1a1a; }
    div[data-testid="stTabs"] { border-bottom:1px solid #e0e0e0; }
    [data-testid="metric-container"] { display:none; }

    /* Boutons principaux sous le canvas */
    div[data-testid="stButton"] > button {
        font-family: 'Roboto Mono', monospace;
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border-radius: 2px;
        padding: 10px 16px;
        transition: all 0.15s;
    }
    div[data-testid="stButton"]:first-child > button {
        background: #1a1a1a;
        color: white;
        border: none;
    }
    div[data-testid="stButton"]:first-child > button:hover {
        background: #333;
    }
    div[data-testid="stButton"]:nth-child(2) > button {
        background: transparent;
        color: #999;
        border: 1px solid #ddd;
    }
    div[data-testid="stButton"]:nth-child(2) > button:hover {
        border-color: #999;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

BG = '#F7F6F2'; GRID = '#e8e6e0'; MONO = 'Roboto Mono'
GOLD = '#C8B560'; DARK = '#1a1a1a'; GREEN = '#2E6B3E'
RED = '#8B2020'; MUTED = '#999999'; AMBER = '#8B6914'
CANVAS_W, CANVAS_H = 680, 320


# ── FONCTIONS ─────────────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, type_='call'):
    if T <= 0 or sigma <= 0 or K <= 0:
        return max(S - K, 0) if type_ == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type_ == 'call':  return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif type_ == 'put': return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return 0.0

def payoff_at_expiry(S_T, K, type_):
    S_T = np.asarray(S_T, dtype=float)
    if type_ == 'call': return np.maximum(S_T - K, 0)
    elif type_ == 'put': return np.maximum(K - S_T, 0)
    return np.zeros_like(S_T)

def canvas_to_payoff(pts, S_min, S_max, pf_min, pf_max, S_range):
    if not pts or len(pts) < 2:
        return None
    pts = sorted(pts, key=lambda p: p[0])
    px  = np.array([p[0] for p in pts], dtype=float)
    py  = np.array([p[1] for p in pts], dtype=float)
    Sd  = S_min + px / CANVAS_W * (S_max - S_min)
    pd_ = pf_max - py / CANVAS_H * (pf_max - pf_min)
    _, idx = np.unique(Sd, return_index=True)
    Sd, pd_ = Sd[idx], pd_[idx]
    if len(Sd) < 2:
        return None
    return interp1d(Sd, pd_, kind='linear',
                     bounds_error=False, fill_value=(pd_[0], pd_[-1]))(S_range)


def build_library(S0, S):
    lib = []

    for p1 in [0.85, 0.88, 0.90, 0.95, 1.00]:
        for p2 in [1.05, 1.08, 1.10, 1.12, 1.15, 1.20]:
            if p1 >= p2:
                continue
            K1, K2 = S0*p1, S0*p2
            lib.append({'name': 'Bull Call Spread',
                'detail': f'Long call K={K1:.0f} / Short call K={K2:.0f}',
                'payoff': np.maximum(S-K1,0) - np.maximum(S-K2,0),
                'legs': [{'type':'call','strike':K1,'weight':+1},
                         {'type':'call','strike':K2,'weight':-1}]})
            lib.append({'name': 'Bear Put Spread',
                'detail': f'Long put K={K2:.0f} / Short put K={K1:.0f}',
                'payoff': np.maximum(K2-S,0) - np.maximum(K1-S,0),
                'legs': [{'type':'put','strike':K2,'weight':+1},
                         {'type':'put','strike':K1,'weight':-1}]})
            lib.append({'name': 'Bull Put Spread',
                'detail': f'Short put K={K2:.0f} / Long put K={K1:.0f}',
                'payoff': -np.maximum(K2-S,0) + np.maximum(K1-S,0),
                'legs': [{'type':'put','strike':K2,'weight':-1},
                         {'type':'put','strike':K1,'weight':+1}]})
            lib.append({'name': 'Bear Call Spread',
                'detail': f'Short call K={K1:.0f} / Long call K={K2:.0f}',
                'payoff': -np.maximum(S-K1,0) + np.maximum(S-K2,0),
                'legs': [{'type':'call','strike':K1,'weight':-1},
                         {'type':'call','strike':K2,'weight':+1}]})

    for K in [S0*0.92, S0*0.95, S0*0.97, S0, S0*1.03, S0*1.05, S0*1.08]:
        lib.append({'name': 'Straddle',
            'detail': f'Long call + Long put K={K:.0f}',
            'payoff': np.maximum(S-K,0) + np.maximum(K-S,0),
            'legs': [{'type':'call','strike':K,'weight':+1},
                     {'type':'put', 'strike':K,'weight':+1}]})
        lib.append({'name': 'Short Straddle',
            'detail': f'Short call + Short put K={K:.0f}',
            'payoff': -np.maximum(S-K,0) - np.maximum(K-S,0),
            'legs': [{'type':'call','strike':K,'weight':-1},
                     {'type':'put', 'strike':K,'weight':-1}]})

    for Kp in [S0*0.85, S0*0.88, S0*0.90, S0*0.92, S0*0.95]:
        for Kc in [S0*1.05, S0*1.08, S0*1.10, S0*1.12, S0*1.15]:
            lib.append({'name': 'Strangle',
                'detail': f'Long call K={Kc:.0f} + Long put K={Kp:.0f}',
                'payoff': np.maximum(S-Kc,0) + np.maximum(Kp-S,0),
                'legs': [{'type':'call','strike':Kc,'weight':+1},
                         {'type':'put', 'strike':Kp,'weight':+1}]})
            lib.append({'name': 'Short Strangle',
                'detail': f'Short call K={Kc:.0f} + Short put K={Kp:.0f}',
                'payoff': -np.maximum(S-Kc,0) - np.maximum(Kp-S,0),
                'legs': [{'type':'call','strike':Kc,'weight':-1},
                         {'type':'put', 'strike':Kp,'weight':-1}]})
            lib.append({'name': 'Strip',
                'detail': f'Long call K={Kc:.0f} + 2x Long put K={Kp:.0f}',
                'payoff': np.maximum(S-Kc,0) + 2*np.maximum(Kp-S,0),
                'legs': [{'type':'call','strike':Kc,'weight':+1},
                         {'type':'put', 'strike':Kp,'weight':+2}]})
            lib.append({'name': 'Strap',
                'detail': f'2x Long call K={Kc:.0f} + Long put K={Kp:.0f}',
                'payoff': 2*np.maximum(S-Kc,0) + np.maximum(Kp-S,0),
                'legs': [{'type':'call','strike':Kc,'weight':+2},
                         {'type':'put', 'strike':Kp,'weight':+1}]})
            lib.append({'name': 'Risk Reversal',
                'detail': f'Long call K={Kc:.0f} / Short put K={Kp:.0f}',
                'payoff': np.maximum(S-Kc,0) - np.maximum(Kp-S,0),
                'legs': [{'type':'call','strike':Kc,'weight':+1},
                         {'type':'put', 'strike':Kp,'weight':-1}]})
            lib.append({'name': 'Collar',
                'detail': f'Long put K={Kp:.0f} / Short call K={Kc:.0f}',
                'payoff': np.maximum(Kp-S,0) - np.maximum(S-Kc,0),
                'legs': [{'type':'put', 'strike':Kp,'weight':+1},
                         {'type':'call','strike':Kc,'weight':-1}]})

    for Km in [S0*0.93, S0*0.95, S0*0.97, S0, S0*1.03, S0*1.05, S0*1.07]:
        for w in [0.06, 0.08, 0.10, 0.12, 0.15]:
            Kl, Kh = Km*(1-w), Km*(1+w)
            pf = np.maximum(S-Kl,0) - 2*np.maximum(S-Km,0) + np.maximum(S-Kh,0)
            lib.append({'name': 'Butterfly',
                'detail': f'Long K={Kl:.0f} / Short 2x K={Km:.0f} / Long K={Kh:.0f}',
                'payoff': pf,
                'legs': [{'type':'call','strike':Kl,'weight':+1},
                         {'type':'call','strike':Km,'weight':-2},
                         {'type':'call','strike':Kh,'weight':+1}]})
            lib.append({'name': 'Short Butterfly',
                'detail': f'Short K={Kl:.0f} / Long 2x K={Km:.0f} / Short K={Kh:.0f}',
                'payoff': -pf,
                'legs': [{'type':'call','strike':Kl,'weight':-1},
                         {'type':'call','strike':Km,'weight':+2},
                         {'type':'call','strike':Kh,'weight':-1}]})
            pf_ib = (np.maximum(Kl-S,0) - np.maximum(Km-S,0)
                     - np.maximum(S-Km,0) + np.maximum(S-Kh,0))
            lib.append({'name': 'Iron Butterfly',
                'detail': f'Long put K={Kl:.0f} / Short ATM / Long call K={Kh:.0f}',
                'payoff': pf_ib,
                'legs': [{'type':'put', 'strike':Kl,'weight':+1},
                         {'type':'put', 'strike':Km,'weight':-1},
                         {'type':'call','strike':Km,'weight':-1},
                         {'type':'call','strike':Kh,'weight':+1}]})

    for p1,p2,p3,p4 in [
        (0.82,0.90,1.10,1.18),(0.85,0.92,1.08,1.15),
        (0.88,0.94,1.06,1.12),(0.90,0.95,1.05,1.10),
        (0.85,0.95,1.05,1.15),(0.88,0.96,1.04,1.12),
    ]:
        K1,K2,K3,K4 = S0*p1,S0*p2,S0*p3,S0*p4
        pfc = (np.maximum(S-K1,0)-np.maximum(S-K2,0)
               -np.maximum(S-K3,0)+np.maximum(S-K4,0))
        lib.append({'name': 'Condor',
            'detail': f'Long K={K1:.0f} / Short K={K2:.0f} / Short K={K3:.0f} / Long K={K4:.0f}',
            'payoff': pfc,
            'legs': [{'type':'call','strike':K1,'weight':+1},
                     {'type':'call','strike':K2,'weight':-1},
                     {'type':'call','strike':K3,'weight':-1},
                     {'type':'call','strike':K4,'weight':+1}]})
        pfic = (np.maximum(K1-S,0)-np.maximum(K2-S,0)
                -np.maximum(S-K3,0)+np.maximum(S-K4,0))
        lib.append({'name': 'Iron Condor',
            'detail': f'Long put K={K1:.0f} / Short put K={K2:.0f} / Short call K={K3:.0f} / Long call K={K4:.0f}',
            'payoff': pfic,
            'legs': [{'type':'put', 'strike':K1,'weight':+1},
                     {'type':'put', 'strike':K2,'weight':-1},
                     {'type':'call','strike':K3,'weight':-1},
                     {'type':'call','strike':K4,'weight':+1}]})

    for K1_p in [0.90, 0.95, 1.00, 1.02]:
        K1, K2 = S0*K1_p, S0*(K1_p+0.08)
        lib.append({'name': 'Call Spread Ratio 1x2',
            'detail': f'Long call K={K1:.0f} / Short 2x call K={K2:.0f}',
            'payoff': np.maximum(S-K1,0) - 2*np.maximum(S-K2,0),
            'legs': [{'type':'call','strike':K1,'weight':+1},
                     {'type':'call','strike':K2,'weight':-2}]})
        lib.append({'name': 'Put Spread Ratio 1x2',
            'detail': f'Long put K={K2:.0f} / Short 2x put K={K1:.0f}',
            'payoff': np.maximum(K2-S,0) - 2*np.maximum(K1-S,0),
            'legs': [{'type':'put','strike':K2,'weight':+1},
                     {'type':'put','strike':K1,'weight':-2}]})

    for K1_p in [0.90, 0.95, 1.00]:
        K1, K2, K3 = S0*K1_p, S0*(K1_p+0.07), S0*(K1_p+0.14)
        lib.append({'name': 'Call Ladder',
            'detail': f'Long K={K1:.0f} / Short K={K2:.0f} / Short K={K3:.0f}',
            'payoff': np.maximum(S-K1,0) - np.maximum(S-K2,0) - np.maximum(S-K3,0),
            'legs': [{'type':'call','strike':K1,'weight':+1},
                     {'type':'call','strike':K2,'weight':-1},
                     {'type':'call','strike':K3,'weight':-1}]})

    for part in [0.3, 0.5, 0.7, 0.8, 1.0, 1.2]:
        for prot in [0.80, 0.85, 0.90, 0.95, 1.00]:
            lib.append({'name': 'Capital protégé',
                'detail': f'Protection {prot*100:.0f}% + participation {part*100:.0f}%',
                'payoff': prot*S0 + part*np.maximum(S-S0,0),
                'legs': [{'type':'call','strike':S0,'weight':part}]})

    for K in [S0*0.80, S0*0.85, S0*0.90, S0*0.95]:
        coupon = (S0 - K) / S0
        lib.append({'name': 'Reverse Convertible',
            'detail': f'Coupon {coupon*100:.0f}% / Barrière K={K:.0f}',
            'payoff': S0*(1+coupon) - np.maximum(K-S,0),
            'legs': [{'type':'put','strike':K,'weight':-1}]})

    for K in [S0*0.82,S0*0.85,S0*0.88,S0*0.90,S0*0.92,S0*0.95,S0*0.97,
              S0,S0*1.03,S0*1.05,S0*1.08,S0*1.10,S0*1.12,S0*1.15,S0*1.18]:
        lib.append({'name':'Call vanille','detail':f'Long call K={K:.0f}',
            'payoff':np.maximum(S-K,0),'legs':[{'type':'call','strike':K,'weight':+1}]})
        lib.append({'name':'Put vanille','detail':f'Long put K={K:.0f}',
            'payoff':np.maximum(K-S,0),'legs':[{'type':'put','strike':K,'weight':+1}]})
        lib.append({'name':'Short Call','detail':f'Short call K={K:.0f}',
            'payoff':-np.maximum(S-K,0),'legs':[{'type':'call','strike':K,'weight':-1}]})
        lib.append({'name':'Short Put','detail':f'Short put K={K:.0f}',
            'payoff':-np.maximum(K-S,0),'legs':[{'type':'put','strike':K,'weight':-1}]})

    for K in [S0*1.03,S0*1.05,S0*1.08,S0*1.10,S0*1.15]:
        lib.append({'name':'Covered Call','detail':f'Long stock / Short call K={K:.0f}',
            'payoff': S - np.maximum(S-K,0),
            'legs':[{'type':'call','strike':K,'weight':-1}]})

    for K in [S0*0.85,S0*0.88,S0*0.90,S0*0.92,S0*0.95]:
        lib.append({'name':'Protective Put','detail':f'Long stock + Long put K={K:.0f}',
            'payoff': S + np.maximum(K-S,0),
            'legs':[{'type':'put','strike':K,'weight':+1}]})

    return lib


def match_structure(target, S_range, S0, T, r, sigma):
    lib   = build_library(S0, S_range)
    t_std = np.std(target)
    if t_std < 1e-8:
        return None, 0, [], np.zeros_like(target)
    t_norm = (target - np.mean(target)) / t_std
    best_score, best_struct, best_scale = -np.inf, None, 1.0
    for s in lib:
        pf    = s['payoff']
        p_std = np.std(pf)
        if p_std < 1e-8:
            continue
        corr = float(np.corrcoef(t_norm, (pf-np.mean(pf))/p_std)[0, 1])
        if corr > best_score:
            best_score  = corr
            best_struct = s
            best_scale  = float(np.dot(target, pf) / (np.dot(pf, pf) + 1e-10))
    if best_struct is None:
        return None, 0, [], np.zeros_like(target)
    recon = best_struct['payoff'] * best_scale
    legs  = [{'type': l['type'], 'strike': l['strike'],
               'weight': l['weight']*best_scale,
               'price': bs_price(S0, l['strike'], T, r, sigma, l['type']),
               'cost':  l['weight']*best_scale*bs_price(S0,l['strike'],T,r,sigma,l['type'])}
             for l in best_struct['legs']]
    return best_struct, best_score, legs, recon


# ── SIDEBAR ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:24px 0 16px;'>
        <div style='font-family:Roboto Mono,monospace;font-size:9px;color:#555;
                    letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;'>
            Réplication Statique</div>
        <div style='font-family:Roboto Mono,monospace;font-size:22px;
                    color:#e0e0e0;font-weight:400;letter-spacing:-1px;'>Payoff Builder</div>
        <div style='font-family:Roboto Mono,monospace;font-size:9px;
                    color:#444;margin-top:4px;letter-spacing:1px;'>
            Oscar Dawny — EDHEC / Centrale Lille</div>
    </div>
    <hr style='border-color:#333;margin:0 0 20px;'>
    """, unsafe_allow_html=True)

    S0    = st.number_input("Spot (S0)", value=100.0, min_value=1.0, step=1.0)
    sigma = st.slider("Volatilité (%)", 5.0, 60.0, 20.0, 0.5) / 100
    r     = st.slider("Taux sans risque (%)", 0.0, 8.0, 3.5, 0.1) / 100
    T     = st.slider("Maturité (années)", 0.1, 3.0, 1.0, 0.1)
    st.markdown('<hr style="border-color:#333;margin:16px 0;">', unsafe_allow_html=True)
    S_min  = st.slider("Spot min (% S0)", 50, 90, 70, 5) / 100 * S0
    S_max  = st.slider("Spot max (% S0)", 110, 150, 130, 5) / 100 * S0
    pf_min = st.slider("Payoff min", -50, 0, -20, 5)
    pf_max = st.slider("Payoff max", 5, 80, 30, 5)


S_range = np.linspace(S_min, S_max, 300)

# ── EN-TÊTE ───────────────────────────────────────────────────────────────

st.markdown(f"""
<div style='padding:4px 0 20px;'>
    <div style='font-family:Roboto Mono,monospace;font-size:9px;color:#999;
                letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;'>
        Réplication statique</div>
    <div style='font-family:Roboto,sans-serif;font-size:28px;
                font-weight:300;color:#1a1a1a;letter-spacing:-1px;'>
        Payoff Builder
        <span style='font-size:15px;color:#999;'>
            &nbsp;/&nbsp;S0={S0:.0f}&nbsp;/&nbsp;σ={sigma*100:.0f}%&nbsp;/&nbsp;T={T:.1f}y
        </span>
    </div>
</div>
<hr style='border:none;border-top:1px solid #e0e0e0;margin:0 0 20px;'>
""", unsafe_allow_html=True)

st.markdown("""
<div style='font-family:Roboto Mono,monospace;font-size:11px;color:#666;
            background:#EFEDE6;padding:12px 16px;border-left:3px solid #C8B560;
            margin-bottom:20px;line-height:1.6;'>
    Dessinez le payoff à la main — axe X = prix S_T à maturité, axe Y = gain/perte.
    L'app identifie la structure standard la plus proche parmi 400+ structures.
</div>
""", unsafe_allow_html=True)

# ── CANVAS ────────────────────────────────────────────────────────────────

# Initialisation des boutons (définis sous le canvas mais utilisés dedans)
if 'clear_btn' not in st.session_state:
    st.session_state['clear_btn'] = False
if 'decompose_btn' not in st.session_state:
    st.session_state['decompose_btn'] = False

# Échelle Y + canvas côte à côte
col_y, col_c = st.columns([0.06, 1])

with col_y:
    step  = (pf_max - pf_min) / 4
    vals  = [pf_max, pf_max-step, pf_min+2*step, pf_min+step, pf_min]
    items = "".join(
        f"<span style='color:{'#2E6B3E' if v>0 else ('#8B2020' if v<0 else '#999')};display:block;"
        f"font-family:Roboto Mono,monospace;font-size:9px;text-align:right;'>{v:+.0f}</span>"
        for v in vals)
    st.markdown(
        f"<div style='height:{CANVAS_H}px;display:flex;flex-direction:column;"
        f"justify-content:space-between;padding:0 6px 0 0;'>{items}</div>",
        unsafe_allow_html=True)

with col_c:
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)", stroke_width=3, stroke_color="#1a1a1a",
        background_color="#FAFAF7", update_streamlit=True,
        height=CANVAS_H, width=CANVAS_W, drawing_mode="freedraw",
        key=f"canvas_main_{st.session_state.get('canvas_key', 0)}",
        display_toolbar=False)

    # Axe X
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val in zip([c1,c2,c3,c4,c5],
                         [S_min, S_min+(S_max-S_min)*.25, S0,
                          S_min+(S_max-S_min)*.75, S_max]):
        col.markdown(f"<div style='font-family:Roboto Mono,monospace;font-size:9px;"
                     f"color:#999;text-align:center;'>{val:.0f}</div>",
                     unsafe_allow_html=True)
    st.markdown("<div style='font-family:Roboto Mono,monospace;font-size:9px;"
                "color:#bbb;text-align:center;margin-top:2px;'>S_T →</div>",
                unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    btn_c1, btn_c2, btn_c3 = st.columns([2, 1, 1])
    with btn_c1:
        decompose_btn = st.button("Identifier la structure", use_container_width=True)
    with btn_c2:
        clear_btn = st.button("Effacer le dessin", use_container_width=True)
        if clear_btn:
            st.session_state['canvas_key'] = st.session_state.get('canvas_key', 0) + 1
            st.rerun()
    with btn_c3:
        pass

st.markdown('<hr style="border:none;border-top:1px solid #e0e0e0;margin:20px 0;">', unsafe_allow_html=True)

# ── EXTRACTION ────────────────────────────────────────────────────────────

path_points  = []
drawn_payoff = None

if canvas_result.json_data is not None:
    for obj in canvas_result.json_data.get("objects", []):
        if obj.get("type") == "path":
            for cmd in obj.get("path", []):
                if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ['M','L','Q','C']:
                    path_points.append((float(cmd[1]), float(cmd[2])))
    if len(path_points) >= 2:
        drawn_payoff = canvas_to_payoff(path_points, S_min, S_max, pf_min, pf_max, S_range)

# ── RÉSULTATS ─────────────────────────────────────────────────────────────

TL = {'call':'Call','put':'Put'}
TC = {'call':'call','put':'put'}

if drawn_payoff is not None:

    tab1, tab2 = st.tabs(["Identification", "Legs individuels"])

    with tab1:
        if decompose_btn:
            with st.spinner(""):
                res = match_structure(drawn_payoff, S_range, S0, T, r, sigma)
            st.session_state['res']        = res
            st.session_state['decomposed'] = True

        if st.session_state.get('decomposed'):
            res = st.session_state.get('res')
            if res is None or res[0] is None:
                st.warning("Dessin trop ambigu — essaie un trait plus net.")
            else:
                struct, score, legs, recon = res
                total_cost = sum(l['cost'] for l in legs)
                klass = 'green' if score > 0.85 else ('gold' if score > 0.65 else 'red')

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class='kpi gold'>
                        <div class='label'>Structure identifiée</div>
                        <div style='font-family:Roboto Mono,monospace;font-size:15px;
                                    font-weight:500;color:#1a1a1a;margin-top:4px;'>
                            {struct['name']}</div>
                        <div style='font-family:Roboto Mono,monospace;font-size:10px;
                                    color:#999;margin-top:3px;line-height:1.4;'>
                            {struct['detail']}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class='kpi {klass}'>
                        <div class='label'>Similarité</div>
                        <div class='value'>{score*100:.0f}%</div>
                        <div style='font-family:Roboto Mono,monospace;font-size:10px;
                                    color:#999;margin-top:2px;'>corrélation de forme</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class='kpi'>
                        <div class='label'>Legs</div>
                        <div class='value'>{len(legs)}</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    cc = GREEN if total_cost >= 0 else RED
                    st.markdown(f"""
                    <div class='kpi'>
                        <div class='label'>Coût</div>
                        <div class='value' style='color:{cc};'>{total_cost:+.3f}</div>
                        <div style='font-family:Roboto Mono,monospace;font-size:10px;
                                    color:#999;margin-top:2px;'>{total_cost/S0*100:+.2f}% du spot</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<hr style="border:none;border-top:1px solid #e0e0e0;margin:16px 0;">', unsafe_allow_html=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=S_range, y=np.maximum(drawn_payoff,0),
                    fill='tozeroy', fillcolor='rgba(46,107,62,0.06)',
                    line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=S_range, y=np.minimum(drawn_payoff,0),
                    fill='tozeroy', fillcolor='rgba(139,32,32,0.06)',
                    line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=S_range, y=drawn_payoff, mode='lines',
                    line=dict(color=DARK, width=2.5, dash='dot'), name='Dessin'))
                fig.add_trace(go.Scatter(x=S_range, y=recon, mode='lines',
                    line=dict(color=GOLD, width=2), name=struct['name']))
                fig.add_vline(x=S0, line_color=MUTED, line_dash='dot', line_width=1,
                               annotation_text='S0',
                               annotation_font=dict(color=MUTED, size=10, family=MONO))
                fig.add_hline(y=0, line_color=MUTED, line_width=0.8)
                fig.update_layout(
                    paper_bgcolor=BG, plot_bgcolor=BG,
                    font=dict(family=MONO, color='#666', size=10),
                    xaxis=dict(title='S_T', gridcolor=GRID, zeroline=False),
                    yaxis=dict(title='Gain / Perte', gridcolor=GRID, zeroline=False),
                    height=320,
                    legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)',
                                bordercolor=GRID, borderwidth=1,
                                orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                    margin=dict(l=60, r=20, t=50, b=50))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div style="font-family:Roboto Mono,monospace;font-size:9px;color:#999;letter-spacing:2px;text-transform:uppercase;margin:16px 0 8px;">Composition</div>', unsafe_allow_html=True)

                for leg in legs:
                    dc = GREEN if leg['weight'] > 0 else RED
                    d  = 'Long' if leg['weight'] > 0 else 'Short'
                    st.markdown(f"""
                    <div class='leg-row'>
                        <span style='width:60px;font-size:10px;color:{dc};
                                     font-family:Roboto Mono,monospace;'>{d}</span>
                        <span class='tag {TC.get(leg["type"],"call")}'>{TL.get(leg["type"],leg["type"])}</span>
                        <span style='flex:1;'>K = <b>{leg["strike"]:.1f}</b>
                            &nbsp;({leg["strike"]/S0*100:.0f}% ATM)</span>
                        <span style='width:100px;text-align:right;color:#999;'>
                            px = {leg["price"]:.4f}</span>
                        <span style='width:110px;text-align:right;color:{dc};'>
                            coût = {leg["cost"]:+.4f}</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:40px;font-family:Roboto Mono,monospace;
                        font-size:11px;color:#ccc;letter-spacing:2px;text-transform:uppercase;'>
                Clique sur "Identifier la structure"</div>
            """, unsafe_allow_html=True)

    with tab2:
        if st.session_state.get('decomposed') and st.session_state.get('res') and st.session_state['res'][0]:
            _, _, legs, recon = st.session_state['res']
            palette = [GREEN, RED, GOLD, '#1a4a7a', '#6B4C2E', '#4a2e6b']
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=S_range, y=drawn_payoff, mode='lines',
                line=dict(color=DARK, width=2, dash='dot'), name='Dessin'))
            fig2.add_trace(go.Scatter(x=S_range, y=recon, mode='lines',
                line=dict(color=GOLD, width=2), name='Structure'))
            for i, leg in enumerate(legs):
                c = leg['weight'] * payoff_at_expiry(S_range, leg['strike'], leg['type'])
                s = "+" if leg['weight'] > 0 else "−"
                fig2.add_trace(go.Scatter(x=S_range, y=c, mode='lines',
                    line=dict(color=palette[i%len(palette)], width=1, dash='dash'),
                    name=f"{s} {TL.get(leg['type'],'')[:4]} K={leg['strike']:.0f}", opacity=0.7))
            fig2.add_vline(x=S0, line_color=MUTED, line_dash='dot', line_width=1)
            fig2.add_hline(y=0, line_color=MUTED, line_width=0.8)
            fig2.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(family=MONO, color='#666', size=10),
                xaxis=dict(title='S_T', gridcolor=GRID, zeroline=False),
                yaxis=dict(title='Contribution', gridcolor=GRID, zeroline=False),
                height=380,
                legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)',
                            bordercolor=GRID, borderwidth=1, orientation='v', x=1.01, y=1),
                margin=dict(l=60, r=160, t=20, b=50))
            st.plotly_chart(fig2, use_container_width=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:60px;font-family:Roboto Mono,monospace;
                font-size:11px;color:#ccc;letter-spacing:2px;text-transform:uppercase;'>
        Dessine un payoff dans la zone ci-dessus pour commencer
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr style='border:none;border-top:1px solid #e0e0e0;margin:40px 0 16px;'>
<div style='font-family:Roboto Mono,monospace;font-size:9px;color:#ccc;
            letter-spacing:1.5px;text-transform:uppercase;'>
    Payoff Builder — Oscar Dawny — EDHEC Business School / Centrale Lille — 2026
</div>
""", unsafe_allow_html=True)
