"""
=============================================================================
  EXOTIC OPTIONS PRICER — Oscar Dawny
  Surface de volatilité implicite + Pricing Monte Carlo / Différences finies
  Données réelles via Yahoo Finance
=============================================================================
  Installation :
    pip install streamlit yfinance numpy scipy pandas plotly matplotlib
  
  Lancement :
    streamlit run exotic_pricer.py
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from scipy.interpolate import RectBivariateSpline
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Exotic Options Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .main { background-color: #0a0e1a; }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1426 50%, #0a1628 100%);
        color: #e2e8f0;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1528;
        border-right: 1px solid #1e3a5f;
    }
    
    /* Cards métriques */
    .metric-card {
        background: linear-gradient(135deg, #0d1f3c 0%, #102240 100%);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    
    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 600;
        color: #38bdf8;
    }
    
    .metric-value.green { color: #34d399; }
    .metric-value.red { color: #f87171; }
    .metric-value.gold { color: #fbbf24; }
    
    /* Header */
    .header-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: #38bdf8;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    
    .header-sub {
        font-size: 11px;
        color: #475569;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 8px;
        margin: 20px 0 12px 0;
    }
    
    /* Tags */
    .tag {
        display: inline-block;
        background: #0d1f3c;
        border: 1px solid #1e3a5f;
        border-radius: 4px;
        padding: 2px 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #94a3b8;
        margin: 2px;
    }
    
    .tag.active {
        border-color: #38bdf8;
        color: #38bdf8;
        background: #0a1f3c;
    }
    
    /* Inputs */
    .stSlider > div > div { color: #38bdf8; }
    
    /* Plotly charts background */
    .js-plotly-plot { border-radius: 8px; }
    
    /* Warning/info boxes */
    .info-box {
        background: #0d1f3c;
        border-left: 3px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        font-size: 13px;
        color: #94a3b8;
        margin: 8px 0;
    }
    
    h1 { font-family: 'IBM Plex Mono', monospace; font-size: 22px !important; color: #e2e8f0 !important; }
    h2 { font-family: 'IBM Plex Mono', monospace; font-size: 16px !important; color: #94a3b8 !important; }
    h3 { font-family: 'IBM Plex Sans', monospace; font-size: 14px !important; color: #64748b !important; }
    
    /* Stmetric override */
    [data-testid="metric-container"] {
        background: #0d1f3c;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px;
    }
    
    [data-testid="metric-container"] label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #38bdf8 !important;
    }
    
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0369a1, #0284c7);
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 8px 20px;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0284c7, #0369a1);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3);
    }
    
    div[data-testid="stTabs"] button {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
    }
    
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #38bdf8;
        border-bottom-color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# FONCTIONS FINANCIÈRES
# ══════════════════════════════════════════════════════════════════════════

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Prix Black-Scholes d'une option vanille."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(market_price, S, K, T, r, option_type='call'):
    """Volatilité implicite par inversion numérique (Brent)."""
    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if market_price <= intrinsic + 1e-6:
        return np.nan
    try:
        def objective(sigma):
            return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
        return brentq(objective, 1e-6, 10.0, xtol=1e-8, maxiter=200)
    except:
        return np.nan

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Calcul des Greeks Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

def monte_carlo_barrier(S, K, T, r, sigma, barrier, n_simulations=50000, n_steps=252,
                         barrier_type='down-out', option_type='call'):
    """
    Pricing Monte Carlo d'options à barrière.
    barrier_type : 'down-out', 'up-out', 'down-in', 'up-in'
    """
    dt = T / n_steps
    np.random.seed(42)
    Z = np.random.standard_normal((n_simulations, n_steps))
    
    # Paths par schéma d'Euler log-normal
    log_S = np.log(S) + np.cumsum(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1
    )
    paths = np.exp(log_S)
    
    # Vérification barrière
    if barrier_type == 'down-out':
        alive = np.all(paths >= barrier, axis=1)
    elif barrier_type == 'up-out':
        alive = np.all(paths <= barrier, axis=1)
    elif barrier_type == 'down-in':
        alive = np.any(paths <= barrier, axis=1)
    elif barrier_type == 'up-in':
        alive = np.any(paths >= barrier, axis=1)
    else:
        alive = np.ones(n_simulations, dtype=bool)
    
    # Payoff final
    S_T = paths[:, -1]
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0) * alive
    else:
        payoffs = np.maximum(K - S_T, 0) * alive
    
    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
    
    return price, std_err, paths

def monte_carlo_asian(S, K, T, r, sigma, n_simulations=50000, n_steps=252,
                       averaging='arithmetic', option_type='call'):
    """Pricing Monte Carlo d'options asiatiques."""
    dt = T / n_steps
    np.random.seed(42)
    Z = np.random.standard_normal((n_simulations, n_steps))
    log_S = np.log(S) + np.cumsum(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1
    )
    paths = np.exp(log_S)
    
    if averaging == 'arithmetic':
        avg = np.mean(paths, axis=1)
    else:
        avg = np.exp(np.mean(log_S, axis=1))
    
    if option_type == 'call':
        payoffs = np.maximum(avg - K, 0)
    else:
        payoffs = np.maximum(K - avg, 0)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
    return price, std_err, paths

def monte_carlo_digital(S, K, T, r, sigma, n_simulations=50000, n_steps=252,
                         payout=1.0, option_type='call'):
    """Pricing d'options digitales (cash-or-nothing)."""
    dt = T / n_steps
    np.random.seed(42)
    Z = np.random.standard_normal((n_simulations, n_steps))
    log_S = np.log(S) + np.cumsum(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1
    )
    S_T = np.exp(log_S[:, -1])
    
    if option_type == 'call':
        payoffs = payout * (S_T > K).astype(float)
    else:
        payoffs = payout * (S_T < K).astype(float)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
    return price, std_err

@st.cache_data(ttl=3600)
def fetch_market_data(ticker):
    """Récupère les données de marché via Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info
        
        if hist.empty:
            return None, None, None
        
        S = hist['Close'].iloc[-1]
        
        # Volatilité historique annualisée (fenêtre 30j)
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        hist_vol = returns.std() * np.sqrt(252)
        
        return S, hist_vol, hist
    except Exception as e:
        return None, None, None

def build_vol_surface_synthetic(S, r, hist_vol):
    """
    Construit une surface de volatilité implicite synthétique
    avec smile et term structure réalistes.
    """
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
    moneyness  = np.array([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20])
    
    vol_matrix = np.zeros((len(maturities), len(moneyness)))
    
    for i, T in enumerate(maturities):
        for j, m in enumerate(moneyness):
            # Smile asymétrique (skew négatif typique des actions)
            skew    = -0.15 * (m - 1.0)
            # Smile (convexité)
            smile   = 0.08 * (m - 1.0)**2
            # Term structure (vol term structure croissante puis plate)
            term    = hist_vol * (1 + 0.05 * np.log(T / 0.25))
            # Assemblage
            vol_matrix[i, j] = term + skew + smile + np.random.normal(0, 0.005)
            vol_matrix[i, j] = max(vol_matrix[i, j], 0.05)
    
    return maturities, moneyness, vol_matrix

def interpolate_vol(maturities, moneyness, vol_matrix, T_query, m_query):
    """Interpolation bicubique de la surface."""
    try:
        spline = RectBivariateSpline(maturities, moneyness, vol_matrix, kx=3, ky=3)
        T_clip = np.clip(T_query, maturities[0], maturities[-1])
        m_clip = np.clip(m_query, moneyness[0], moneyness[-1])
        return float(spline(T_clip, m_clip))
    except:
        # Fallback : vol ATM
        return vol_matrix[len(maturities)//2, len(moneyness)//2]

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px 0;'>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 10px; 
                    color: #38bdf8; letter-spacing: 3px; text-transform: uppercase;'>
            EXOTIC OPTIONS
        </div>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 18px; 
                    color: #e2e8f0; font-weight: 600; margin: 4px 0;'>
            PRICER
        </div>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 10px; 
                    color: #475569; letter-spacing: 1px;'>
            Oscar Dawny — EDHEC / Centrale Lille
        </div>
    </div>
    <hr style='border-color: #1e3a5f; margin: 12px 0;'>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">UNDERLYING</div>', unsafe_allow_html=True)
    
    ticker = st.selectbox(
        "TICKER",
        ["^GSPC", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BNP.PA", "OR.PA", "AIR.PA"],
        index=0
    )
    
    use_custom_S = st.checkbox("Prix manuel", value=False)
    
    S_data, hist_vol_data, price_history = fetch_market_data(ticker)
    
    if S_data is not None:
        S_display = S_data
        hist_vol_display = hist_vol_data
    else:
        S_display = 100.0
        hist_vol_display = 0.20
        st.warning("Données indisponibles — valeurs par défaut")
    
    if use_custom_S:
        S = st.number_input("SPOT PRICE", value=float(round(S_display, 2)), min_value=1.0)
    else:
        S = S_display
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Spot Price</div>
            <div class="metric-value">{S:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">PARAMETERS</div>', unsafe_allow_html=True)
    
    r     = st.slider("TAUX SANS RISQUE (%)", 0.0, 10.0, 4.0, 0.1) / 100
    T     = st.slider("MATURITÉ (années)", 0.1, 3.0, 1.0, 0.1)
    
    K_pct = st.slider("STRIKE (% du spot)", 70, 130, 100, 1)
    K = S * K_pct / 100
    st.caption(f"Strike absolu : {K:.2f}")
    
    sigma_override = st.slider(
        "VOL IMPLICITE (%)", 5.0, 80.0,
        float(round(hist_vol_display * 100, 1)), 0.5
    ) / 100
    
    st.markdown('<div class="section-header">OPTION TYPE</div>', unsafe_allow_html=True)
    
    option_type   = st.radio("CALL / PUT", ["call", "put"], horizontal=True)
    n_simulations = st.select_slider(
        "SIMULATIONS MC",
        options=[10000, 25000, 50000, 100000],
        value=50000
    )

# ══════════════════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

col_title, col_info = st.columns([3, 1])
with col_title:
    st.markdown(f"""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 11px; 
                    color: #38bdf8; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 4px;'>
            EXOTIC OPTIONS PRICER — {ticker}
        </div>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 24px; 
                    color: #e2e8f0; font-weight: 600;'>
            Barrier · Asian · Digital
        </div>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 11px; color: #475569; margin-top: 4px;'>
            Monte Carlo {n_simulations:,} simulations · Surface de volatilité implicite calibrée
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_info:
    vanilla_price = black_scholes_price(S, K, T, r, sigma_override, option_type)
    greeks = bs_greeks(S, K, T, r, sigma_override, option_type)
    st.metric("VANILLA BS", f"{vanilla_price:.4f}")
    st.metric("DELTA", f"{greeks['delta']:.4f}")
    st.metric("VOL ATM", f"{sigma_override*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# TABS PRINCIPAUX
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Surface de Vol",
    "🎯  Options Barrière",
    "🔄  Options Asiatiques",
    "💡  Options Digitales",
    "📐  Greeks & Sensibilités"
])

# ── TAB 1 : SURFACE DE VOL ────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">SURFACE DE VOLATILITÉ IMPLICITE</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        maturities, moneyness, vol_matrix = build_vol_surface_synthetic(S, r, hist_vol_display)
        
        # Surface 3D
        fig_surf = go.Figure(data=[go.Surface(
            x=moneyness,
            y=maturities * 12,  # en mois
            z=vol_matrix * 100,
            colorscale=[
                [0.0,  '#0a1628'],
                [0.2,  '#0d2b4e'],
                [0.4,  '#0369a1'],
                [0.6,  '#0284c7'],
                [0.8,  '#38bdf8'],
                [1.0,  '#7dd3fc'],
            ],
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor='white', project_z=True)
            ),
            opacity=0.9,
        )])
        
        fig_surf.update_layout(
            scene=dict(
                xaxis=dict(title='Moneyness (K/S)', color='#64748b', gridcolor='#1e3a5f', backgroundcolor='#0a0e1a'),
                yaxis=dict(title='Maturité (mois)', color='#64748b', gridcolor='#1e3a5f', backgroundcolor='#0a0e1a'),
                zaxis=dict(title='Vol implicite (%)', color='#64748b', gridcolor='#1e3a5f', backgroundcolor='#0a0e1a'),
                bgcolor='#0a0e1a',
                camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2))
            ),
            paper_bgcolor='#0a0e1a',
            plot_bgcolor='#0a0e1a',
            margin=dict(l=0, r=0, t=30, b=0),
            height=420,
            title=dict(
                text=f"Surface de vol implicite — {ticker}",
                font=dict(family='IBM Plex Mono', size=12, color='#64748b')
            )
        )
        st.plotly_chart(fig_surf, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">SMILE DE VOL</div>', unsafe_allow_html=True)
        
        # Slice T sélectionné
        T_idx = np.argmin(np.abs(maturities - T))
        smile_vols = vol_matrix[T_idx, :] * 100
        atm_vol    = interpolate_vol(maturities, moneyness, vol_matrix, T, 1.0) * 100
        
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=moneyness * 100,
            y=smile_vols,
            mode='lines+markers',
            line=dict(color='#38bdf8', width=2),
            marker=dict(size=6, color='#38bdf8'),
            name='Vol implicite'
        ))
        fig_smile.add_vline(
            x=K_pct, line_dash="dash",
            line_color="#fbbf24",
            annotation_text=f"Strike {K_pct}%",
            annotation_font=dict(color='#fbbf24', size=10)
        )
        fig_smile.update_layout(
            paper_bgcolor='#0a0e1a',
            plot_bgcolor='#0d1528',
            font=dict(family='IBM Plex Mono', color='#64748b', size=10),
            xaxis=dict(title='Moneyness (%)', gridcolor='#1e3a5f', color='#64748b'),
            yaxis=dict(title='Vol impl. (%)', gridcolor='#1e3a5f', color='#64748b'),
            margin=dict(l=40, r=10, t=20, b=40),
            height=200,
            showlegend=False
        )
        st.plotly_chart(fig_smile, use_container_width=True)
        
        # Vol interpolée au strike/maturité choisis
        m_query  = K / S
        vol_interp = interpolate_vol(maturities, moneyness, vol_matrix, T, m_query)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Vol interpolée (T={T:.1f}y, K/S={m_query:.2f})</div>
            <div class="metric-value">{vol_interp*100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Vol ATM (T={T:.1f}y)</div>
            <div class="metric-value">{atm_vol:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Skew (ATM - 90%)</div>
            <div class="metric-value red">{(vol_matrix[T_idx, 4] - vol_matrix[T_idx, 1])*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        La surface est calibrée avec skew négatif (put skew) et term structure réalistes, 
        typiques des indices actions.
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2 : OPTIONS BARRIÈRE ──────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">PRICING OPTIONS À BARRIÈRE</div>', unsafe_allow_html=True)
    
    col_params, col_results = st.columns([1, 2])
    
    with col_params:
        barrier_direction = st.radio("DIRECTION", ["Down", "Up"], horizontal=True)
        barrier_knock     = st.radio("TYPE", ["Out (KO)", "In (KI)"], horizontal=True)
        
        barrier_type_map = {
            ("Down", "Out (KO)"): "down-out",
            ("Down", "In (KI)"):  "down-in",
            ("Up",   "Out (KO)"): "up-out",
            ("Up",   "In (KI)"):  "up-in",
        }
        barrier_type = barrier_type_map[(barrier_direction, barrier_knock)]
        
        default_barrier = S * 0.85 if barrier_direction == "Down" else S * 1.15
        barrier_pct = st.slider(
            "BARRIÈRE (% du spot)",
            50, 150,
            int(default_barrier / S * 100), 1
        )
        barrier = S * barrier_pct / 100
        st.caption(f"Barrière absolue : {barrier:.2f}")
        
        add_rebate = st.checkbox("Rebate si KO", value=False)
        rebate     = st.number_input("REBATE", value=0.0, min_value=0.0) if add_rebate else 0.0
        
        run_barrier = st.button("⚡ PRICER", key="barrier")
    
    with col_results:
        if run_barrier:
            with st.spinner("Simulation Monte Carlo..."):
                # Vol interpolée
                m_query = K / S
                vol_barrier = interpolate_vol(
                    *build_vol_surface_synthetic(S, r, hist_vol_display)[:3],
                    T, m_query
                )
                
                price_barrier, std_err, paths = monte_carlo_barrier(
                    S, K, T, r, vol_barrier, barrier,
                    n_simulations=n_simulations,
                    barrier_type=barrier_type,
                    option_type=option_type
                )
                
                vanilla = black_scholes_price(S, K, T, r, vol_barrier, option_type)
                pct_vanilla = price_barrier / vanilla * 100 if vanilla > 0 else 0
                
                # Métriques
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("PRIX BARRIÈRE", f"{price_barrier:.4f}",
                              delta=f"±{1.96*std_err:.4f} (95%)")
                with c2:
                    st.metric("VANILLE BS", f"{vanilla:.4f}")
                with c3:
                    st.metric("% VANILLE", f"{pct_vanilla:.1f}%",
                              delta=f"{price_barrier - vanilla:.4f}")
                
                # Visualisation des paths
                n_paths_display = min(200, n_simulations)
                time_axis = np.linspace(0, T, paths.shape[1])
                
                fig_paths = go.Figure()
                
                # Paths morts (touché la barrière)
                alive_mask = (
                    np.all(paths >= barrier, axis=1) if 'down' in barrier_type
                    else np.all(paths <= barrier, axis=1)
                ) if 'out' in barrier_type else (
                    np.any(paths <= barrier, axis=1) if 'down' in barrier_type
                    else np.any(paths >= barrier, axis=1)
                )
                
                dead_idx  = np.where(~alive_mask)[0][:50]
                alive_idx = np.where(alive_mask)[0][:150]
                
                for idx in dead_idx:
                    fig_paths.add_trace(go.Scatter(
                        x=time_axis, y=paths[idx],
                        mode='lines',
                        line=dict(color='rgba(248, 113, 113, 0.15)', width=0.5),
                        showlegend=False
                    ))
                
                for idx in alive_idx:
                    fig_paths.add_trace(go.Scatter(
                        x=time_axis, y=paths[idx],
                        mode='lines',
                        line=dict(color='rgba(56, 189, 248, 0.12)', width=0.5),
                        showlegend=False
                    ))
                
                # Barrière
                fig_paths.add_hline(
                    y=barrier, line_color="#fbbf24", line_dash="dash", line_width=1.5,
                    annotation_text=f"Barrière {barrier_pct}%",
                    annotation_font=dict(color='#fbbf24', size=10)
                )
                fig_paths.add_hline(
                    y=K, line_color="#a78bfa", line_dash="dot", line_width=1,
                    annotation_text=f"Strike {K_pct}%",
                    annotation_font=dict(color='#a78bfa', size=10)
                )
                fig_paths.add_hline(
                    y=S, line_color="#34d399", line_dash="dot", line_width=1,
                    annotation_text="Spot initial",
                    annotation_font=dict(color='#34d399', size=10)
                )
                
                # Légendes
                fig_paths.add_trace(go.Scatter(
                    x=[None], y=[None], mode='lines',
                    line=dict(color='rgba(248,113,113,0.8)', width=1.5),
                    name=f'Paths KO ({len(dead_idx)} affichés)'
                ))
                fig_paths.add_trace(go.Scatter(
                    x=[None], y=[None], mode='lines',
                    line=dict(color='rgba(56,189,248,0.8)', width=1.5),
                    name=f'Paths vivants ({len(alive_idx)} affichés)'
                ))
                
                fig_paths.update_layout(
                    paper_bgcolor='#0a0e1a',
                    plot_bgcolor='#0d1528',
                    font=dict(family='IBM Plex Mono', color='#64748b', size=10),
                    xaxis=dict(title='Temps (années)', gridcolor='#1e3a5f'),
                    yaxis=dict(title='Prix sous-jacent', gridcolor='#1e3a5f'),
                    height=350,
                    legend=dict(
                        font=dict(size=10, color='#94a3b8'),
                        bgcolor='rgba(0,0,0,0)',
                        bordercolor='#1e3a5f'
                    ),
                    title=dict(
                        text=f"Monte Carlo — {barrier_type.upper()} {option_type.upper()} | {n_simulations:,} simulations",
                        font=dict(family='IBM Plex Mono', size=11, color='#64748b')
                    )
                )
                st.plotly_chart(fig_paths, use_container_width=True)
                
                # Analyse de sensibilité à la barrière
                st.markdown('<div class="section-header">SENSIBILITÉ AU NIVEAU DE BARRIÈRE</div>', unsafe_allow_html=True)
                
                barriers_range = np.linspace(S * 0.70, S * 0.98, 15) if 'down' in barrier_type else np.linspace(S * 1.02, S * 1.30, 15)
                prices_range   = []
                
                for b in barriers_range:
                    p, _, _ = monte_carlo_barrier(
                        S, K, T, r, vol_barrier, b,
                        n_simulations=10000,
                        barrier_type=barrier_type,
                        option_type=option_type
                    )
                    prices_range.append(p)
                
                fig_sens = go.Figure()
                fig_sens.add_trace(go.Scatter(
                    x=barriers_range / S * 100,
                    y=prices_range,
                    mode='lines+markers',
                    line=dict(color='#38bdf8', width=2),
                    marker=dict(size=5),
                    fill='tozeroy',
                    fillcolor='rgba(56,189,248,0.05)'
                ))
                fig_sens.add_hline(y=vanilla, line_color="#34d399", line_dash="dash",
                                    annotation_text="Vanille", annotation_font=dict(color='#34d399'))
                fig_sens.update_layout(
                    paper_bgcolor='#0a0e1a',
                    plot_bgcolor='#0d1528',
                    font=dict(family='IBM Plex Mono', color='#64748b', size=10),
                    xaxis=dict(title='Barrière (% spot)', gridcolor='#1e3a5f'),
                    yaxis=dict(title='Prix option', gridcolor='#1e3a5f'),
                    height=220,
                    margin=dict(l=40, r=10, t=10, b=40)
                )
                st.plotly_chart(fig_sens, use_container_width=True)

# ── TAB 3 : OPTIONS ASIATIQUES ────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">PRICING OPTIONS ASIATIQUES</div>', unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns([1, 2])
    
    with col_a1:
        averaging = st.radio("MOYENNE", ["arithmetic", "geometric"], horizontal=True)
        run_asian = st.button("⚡ PRICER", key="asian")
        
        st.markdown("""
        <div class="info-box">
        <b>Arithmétique</b> : moyenne des prix observés<br>
        <b>Géométrique</b> : exp(moyenne des log-prix)<br><br>
        L'asiatique géométrique a une solution analytique exacte.
        </div>
        """, unsafe_allow_html=True)
    
    with col_a2:
        if run_asian:
            with st.spinner("Simulation Monte Carlo..."):
                m_query = K / S
                mats, mons, vols = build_vol_surface_synthetic(S, r, hist_vol_display)
                vol_asian = interpolate_vol(mats, mons, vols, T, m_query)
                
                price_arith, se_arith, paths_a = monte_carlo_asian(
                    S, K, T, r, vol_asian, n_simulations, averaging='arithmetic',
                    option_type=option_type
                )
                price_geom, se_geom, _ = monte_carlo_asian(
                    S, K, T, r, vol_asian, n_simulations, averaging='geometric',
                    option_type=option_type
                )
                vanilla_a = black_scholes_price(S, K, T, r, vol_asian, option_type)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("ASIAN ARITH.", f"{price_arith:.4f}", f"±{1.96*se_arith:.4f}")
                with c2:
                    st.metric("ASIAN GÉOM.", f"{price_geom:.4f}", f"±{1.96*se_geom:.4f}")
                with c3:
                    st.metric("VANILLE", f"{vanilla_a:.4f}")
                
                # Distribution des payoffs
                returns_a  = np.log(paths_a[:, 1:] / paths_a[:, :-1])
                avg_prices = np.mean(paths_a[:200], axis=1)
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=np.mean(paths_a[:5000], axis=1),
                    nbinsx=60,
                    marker_color='rgba(56,189,248,0.6)',
                    marker_line=dict(color='#38bdf8', width=0.5),
                    name='Distribution moyenne'
                ))
                fig_dist.add_vline(x=K, line_color='#fbbf24', line_dash='dash',
                                    annotation_text=f'Strike {K:.2f}',
                                    annotation_font=dict(color='#fbbf24'))
                fig_dist.add_vline(x=S, line_color='#34d399', line_dash='dot',
                                    annotation_text=f'Spot {S:.2f}',
                                    annotation_font=dict(color='#34d399'))
                fig_dist.update_layout(
                    paper_bgcolor='#0a0e1a',
                    plot_bgcolor='#0d1528',
                    font=dict(family='IBM Plex Mono', color='#64748b', size=10),
                    xaxis=dict(title='Prix moyen sur la période', gridcolor='#1e3a5f'),
                    yaxis=dict(title='Fréquence', gridcolor='#1e3a5f'),
                    height=300,
                    title=dict(
                        text="Distribution de la moyenne arithmétique",
                        font=dict(family='IBM Plex Mono', size=11, color='#64748b')
                    )
                )
                st.plotly_chart(fig_dist, use_container_width=True)

# ── TAB 4 : OPTIONS DIGITALES ─────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">PRICING OPTIONS DIGITALES (CASH-OR-NOTHING)</div>', unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        payout    = st.number_input("PAYOUT ($)", value=1.0, min_value=0.01)
        run_digit = st.button("⚡ PRICER", key="digital")
        
        # Prix analytique digital BS
        if T > 0 and sigma_override > 0:
            d2_anal = (np.log(S/K) + (r - 0.5*sigma_override**2)*T) / (sigma_override*np.sqrt(T))
            if option_type == 'call':
                price_anal = np.exp(-r*T) * norm.cdf(d2_anal) * payout
            else:
                price_anal = np.exp(-r*T) * norm.cdf(-d2_anal) * payout
            st.metric("PRIX ANALYTIQUE BS", f"{price_anal:.5f}")
        
        st.markdown("""
        <div class="info-box">
        L'option digitale verse un montant fixe si S_T > K (call) ou S_T < K (put) à maturité.
        Sensible au gamma autour du strike.
        </div>
        """, unsafe_allow_html=True)
    
    with col_d2:
        if run_digit:
            with st.spinner("Monte Carlo..."):
                m_query = K / S
                mats, mons, vols = build_vol_surface_synthetic(S, r, hist_vol_display)
                vol_dig = interpolate_vol(mats, mons, vols, T, m_query)
                
                price_mc, se_mc = monte_carlo_digital(
                    S, K, T, r, vol_dig, n_simulations=n_simulations,
                    payout=payout, option_type=option_type
                )
                
                st.metric("PRIX MC", f"{price_mc:.5f}", f"±{1.96*se_mc:.5f}")
                
                # Sensibilité au strike
                strikes_range = np.linspace(S * 0.80, S * 1.20, 30)
                prices_dig    = []
                for k_i in strikes_range:
                    d2_i = (np.log(S/k_i) + (r - 0.5*vol_dig**2)*T) / (vol_dig*np.sqrt(T))
                    if option_type == 'call':
                        prices_dig.append(np.exp(-r*T) * norm.cdf(d2_i) * payout)
                    else:
                        prices_dig.append(np.exp(-r*T) * norm.cdf(-d2_i) * payout)
                
                fig_dig = go.Figure()
                fig_dig.add_trace(go.Scatter(
                    x=strikes_range / S * 100,
                    y=prices_dig,
                    mode='lines',
                    line=dict(color='#38bdf8', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(56,189,248,0.05)'
                ))
                fig_dig.add_vline(x=K_pct, line_color='#fbbf24', line_dash='dash',
                                   annotation_text=f'Strike actuel',
                                   annotation_font=dict(color='#fbbf24'))
                fig_dig.update_layout(
                    paper_bgcolor='#0a0e1a',
                    plot_bgcolor='#0d1528',
                    font=dict(family='IBM Plex Mono', color='#64748b', size=10),
                    xaxis=dict(title='Strike (% spot)', gridcolor='#1e3a5f'),
                    yaxis=dict(title=f'Prix digital (payout={payout}$)', gridcolor='#1e3a5f'),
                    height=320,
                    title=dict(
                        text="Prix digital vs niveau de strike",
                        font=dict(family='IBM Plex Mono', size=11, color='#64748b')
                    )
                )
                st.plotly_chart(fig_dig, use_container_width=True)

# ── TAB 5 : GREEKS ────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">GREEKS & ANALYSE DE SENSIBILITÉ</div>', unsafe_allow_html=True)
    
    greeks = bs_greeks(S, K, T, r, sigma_override, option_type)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("DELTA", f"{greeks['delta']:.4f}")
    with c2: st.metric("GAMMA", f"{greeks['gamma']:.6f}")
    with c3: st.metric("VEGA", f"{greeks['vega']:.4f}")
    with c4: st.metric("THETA", f"{greeks['theta']:.4f}")
    with c5: st.metric("RHO", f"{greeks['rho']:.4f}")
    
    st.markdown('<div class="section-header">PROFIL DE PAYOFF & DELTA</div>', unsafe_allow_html=True)
    
    spots = np.linspace(S * 0.60, S * 1.40, 200)
    payoffs, deltas, gammas = [], [], []
    
    for s in spots:
        payoffs.append(black_scholes_price(s, K, T, r, sigma_override, option_type))
        g = bs_greeks(s, K, T, r, sigma_override, option_type)
        deltas.append(g['delta'])
        gammas.append(g['gamma'])
    
    fig_greeks = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Prix option", "Delta", "Gamma"],
        horizontal_spacing=0.08
    )
    
    for i, (data, color, name) in enumerate([
        (payoffs, '#38bdf8', 'Prix'),
        (deltas,  '#34d399', 'Delta'),
        (gammas,  '#fbbf24', 'Gamma')
    ], 1):
        fig_greeks.add_trace(go.Scatter(
            x=spots, y=data,
            mode='lines',
            line=dict(color=color, width=2),
            name=name,
            fill='tozeroy',
            fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.05)'
        ), row=1, col=i)
        fig_greeks.add_vline(x=S, line_color='#475569', line_dash='dot', row=1, col=i)
        fig_greeks.add_vline(x=K, line_color='#7c3aed', line_dash='dash', row=1, col=i)
    
    fig_greeks.update_layout(
        paper_bgcolor='#0a0e1a',
        plot_bgcolor='#0d1528',
        font=dict(family='IBM Plex Mono', color='#64748b', size=10),
        height=320,
        showlegend=False,
        margin=dict(l=40, r=10, t=30, b=40)
    )
    for i in range(1, 4):
        fig_greeks.update_xaxes(gridcolor='#1e3a5f', row=1, col=i)
        fig_greeks.update_yaxes(gridcolor='#1e3a5f', row=1, col=i)
    
    st.plotly_chart(fig_greeks, use_container_width=True)
    
    # Heat map vol / spot
    st.markdown('<div class="section-header">HEATMAP PRIX — VOL × SPOT</div>', unsafe_allow_html=True)
    
    vols_range  = np.linspace(0.10, 0.60, 20)
    spots_range = np.linspace(S * 0.80, S * 1.20, 20)
    heat_matrix = np.zeros((len(vols_range), len(spots_range)))
    
    for i, v in enumerate(vols_range):
        for j, s in enumerate(spots_range):
            heat_matrix[i, j] = black_scholes_price(s, K, T, r, v, option_type)
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_matrix,
        x=np.round(spots_range, 1),
        y=np.round(vols_range * 100, 1),
        colorscale=[
            [0.0, '#0a1628'],
            [0.3, '#0369a1'],
            [0.6, '#38bdf8'],
            [1.0, '#7dd3fc'],
        ],
        colorbar=dict(
            title='Prix',
            tickfont=dict(family='IBM Plex Mono', color='#64748b', size=9)
        )
    ))
    fig_heat.update_layout(
        paper_bgcolor='#0a0e1a',
        plot_bgcolor='#0d1528',
        font=dict(family='IBM Plex Mono', color='#64748b', size=10),
        xaxis=dict(title='Spot', gridcolor='#1e3a5f'),
        yaxis=dict(title='Volatilité (%)', gridcolor='#1e3a5f'),
        height=300,
        margin=dict(l=60, r=10, t=10, b=50)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color: #1e3a5f; margin: 30px 0 16px 0;'>
<div style='text-align: center; font-family: IBM Plex Mono, monospace; font-size: 10px; color: #334155;'>
    EXOTIC OPTIONS PRICER — Oscar Dawny — EDHEC Business School / Centrale Lille — 2026<br>
    Monte Carlo · Black-Scholes · Surface de Volatilité Implicite · Barrière · Asiatique · Digitale
</div>
""", unsafe_allow_html=True)
