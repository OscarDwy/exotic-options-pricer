"""
Shared utilities — Payoff Builder
"""

import numpy as np
import streamlit as st
from scipy.stats import norm

BG = '#F7F6F2'; GRID = '#e8e6e0'; MONO = 'Roboto Mono'
GOLD = '#C8B560'; DARK = '#1a1a1a'; GREEN = '#2E6B3E'
RED = '#8B2020'; MUTED = '#999999'; AMBER = '#8B6914'
CANVAS_W, CANVAS_H = 680, 320

TL = {'call': 'Call', 'put': 'Put'}
TC = {'call': 'call', 'put': 'put'}

SHARED_CSS = """
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
"""


def inject_css():
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


def render_sidebar():
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
        S_min = st.slider("Spot min (% S0)", 50, 90, 70, 5) / 100 * S0
        S_max = st.slider("Spot max (% S0)", 110, 150, 130, 5) / 100 * S0
        pf_min = st.slider("Payoff min", -50, 0, -20, 5)
        pf_max = st.slider("Payoff max", 5, 80, 30, 5)

    S_range = np.linspace(S_min, S_max, 300)
    return S0, sigma, r, T, S_min, S_max, pf_min, pf_max, S_range


def render_footer():
    st.markdown("""
    <hr style='border:none;border-top:1px solid #e0e0e0;margin:40px 0 16px;'>
    <div style='font-family:Roboto Mono,monospace;font-size:9px;color:#ccc;
                letter-spacing:1.5px;text-transform:uppercase;'>
        Payoff Builder — Oscar Dawny — EDHEC Business School / Centrale Lille — 2026
    </div>
    """, unsafe_allow_html=True)


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
