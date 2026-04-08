"""
Autocall Simulator — Oscar Dawny
EDHEC Business School / Centrale Lille

pip install streamlit yfinance numpy pandas plotly scipy
streamlit run autocall_simulator.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Autocall Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500&family=Roboto:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .stApp { background: #F7F6F2; color: #1a1a1a; }

    section[data-testid="stSidebar"] {
        background: #1a1a1a;
        border-right: none;
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    section[data-testid="stSidebar"] label {
        font-family: 'Roboto Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: #666 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: #C8B560;
        color: #1a1a1a;
        border: none;
        border-radius: 2px;
        font-family: 'Roboto Mono', monospace;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 12px 20px;
        width: 100%;
        margin-top: 12px;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #b5a050;
    }

    .label {
        font-family: 'Roboto Mono', monospace;
        font-size: 9px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #999;
        margin-bottom: 2px;
    }
    .value {
        font-family: 'Roboto Mono', monospace;
        font-size: 28px;
        font-weight: 500;
        color: #1a1a1a;
        line-height: 1.1;
    }

    .kpi-block {
        border-left: 2px solid #e0e0e0;
        padding: 8px 0 8px 14px;
        margin: 6px 0;
    }
    .kpi-block.highlight { border-left-color: #C8B560; }
    .kpi-block.danger    { border-left-color: #8B2020; }

    .rule {
        font-family: 'Roboto Mono', monospace;
        font-size: 11px;
        color: #555;
        line-height: 1.8;
        background: #EFEDE6;
        padding: 14px 18px;
        border-left: 3px solid #C8B560;
        margin: 8px 0;
    }

    [data-testid="metric-container"] { display: none; }

    div[data-testid="stTabs"] button {
        font-family: 'Roboto Mono', monospace;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #999;
        padding: 8px 16px;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #1a1a1a;
        border-bottom: 2px solid #1a1a1a;
    }
    div[data-testid="stTabs"] { border-bottom: 1px solid #e0e0e0; }

    .waiting {
        text-align: center;
        padding: 80px 20px;
        color: #ccc;
        font-family: 'Roboto Mono', monospace;
        font-size: 12px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# ── CONSTANTES GRAPHIQUES ─────────────────────────────────────────────────

BG      = '#F7F6F2'
GRID    = '#e8e6e0'
MONO    = 'Roboto Mono'
GOLD    = '#C8B560'
DARK    = '#1a1a1a'
GREEN   = '#2E6B3E'
AMBER   = '#8B6914'
RED     = '#8B2020'
MUTED   = '#999999'


# ── FONCTIONS ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty:
            return 0.20, 100.0, None
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        return float(returns.std() * np.sqrt(252)), float(hist['Close'].iloc[-1]), hist
    except:
        return 0.20, 100.0, None


def simuler_autocall(S0, sigma, r, T_max, freq_obs,
                     b_rappel, b_protection, coupon_annuel,
                     nominal, n_sims):
    dt       = 1 / 252
    obs_days = [int(i * freq_obs / 12 * 252)
                for i in range(1, int(T_max * 12 / freq_obs) + 1)]
    T_days   = int(T_max * 252)
    np.random.seed(42)
    Z     = np.random.standard_normal((n_sims, T_days))
    paths = S0 * np.exp(np.cumsum(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1
    ))
    resultats = []
    rappele   = np.zeros(n_sims, dtype=bool)
    for day in obs_days:
        t   = day / 252
        day = min(day, paths.shape[1] - 1)
        S_t = paths[:, day]
        new = (~rappele) & (S_t >= b_rappel * S0)
        coupon = coupon_annuel * t * nominal
        for i in np.where(new)[0]:
            resultats.append({'simulation': i, 'scenario': 'rappel',
                               'duree': t, 'S_final': S_t[i],
                               'gain': nominal + coupon,
                               'rendement_pct': coupon / nominal * 100})
        rappele |= new
    for idx, s in zip(np.where(~rappele)[0], paths[~rappele, min(T_days-1, paths.shape[1]-1)]):
        if s >= b_protection * S0:
            resultats.append({'simulation': idx, 'scenario': 'protection',
                               'duree': T_max, 'S_final': s,
                               'gain': nominal, 'rendement_pct': 0.0})
        else:
            resultats.append({'simulation': idx, 'scenario': 'perte',
                               'duree': T_max, 'S_final': s,
                               'gain': nominal * (s / S0),
                               'rendement_pct': (s / S0 - 1) * 100})
    return pd.DataFrame(resultats), paths


# ── SIDEBAR ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:24px 0 20px;'>
        <div style='font-family:Roboto Mono,monospace;font-size:9px;
                    color:#555;letter-spacing:3px;text-transform:uppercase;
                    margin-bottom:6px;'>Produit Structuré</div>
        <div style='font-family:Roboto Mono,monospace;font-size:22px;
                    color:#e0e0e0;font-weight:400;letter-spacing:-1px;'>Autocall</div>
        <div style='font-family:Roboto Mono,monospace;font-size:9px;
                    color:#444;margin-top:4px;letter-spacing:1px;'>
            Oscar Dawny — EDHEC / Centrale Lille</div>
    </div>
    <hr style='border-color:#333;margin:0 0 20px;'>
    """, unsafe_allow_html=True)

    ticker = st.selectbox("Sous-jacent",
        ["^FCHI", "^GSPC", "^STOXX50E", "BNP.PA", "OR.PA", "AIR.PA"], index=0)

    hist_vol, spot_yf, _ = get_market_data(ticker)

    st.markdown(f"""
    <div style='display:flex;gap:20px;margin:10px 0 18px;'>
        <div>
            <div style='font-family:Roboto Mono,monospace;font-size:9px;
                        color:#555;letter-spacing:1.5px;text-transform:uppercase;'>Spot</div>
            <div style='font-family:Roboto Mono,monospace;font-size:18px;
                        color:#C8B560;'>{spot_yf:.1f}</div>
        </div>
        <div>
            <div style='font-family:Roboto Mono,monospace;font-size:9px;
                        color:#555;letter-spacing:1.5px;text-transform:uppercase;'>Vol 1Y</div>
            <div style='font-family:Roboto Mono,monospace;font-size:18px;
                        color:#C8B560;'>{hist_vol*100:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sigma    = st.slider("Volatilité (%)", 5.0, 60.0, round(hist_vol*100, 1), 0.5) / 100
    r        = st.slider("Taux sans risque (%)", 0.0, 8.0, 3.5, 0.1) / 100
    st.markdown('<hr style="border-color:#333;margin:16px 0;">', unsafe_allow_html=True)
    nominal  = st.number_input("Nominal (€)", value=1000, min_value=100, step=100)
    T_max    = st.slider("Maturité max (années)", 1, 10, 5, 1)
    freq_obs = st.select_slider("Fréquence observation (mois)", [1, 3, 6, 12], value=12)
    st.markdown('<hr style="border-color:#333;margin:16px 0;">', unsafe_allow_html=True)
    b_rappel     = st.slider("Barrière de rappel (%)", 80, 120, 100, 1) / 100
    coupon       = st.slider("Coupon annuel (%)", 1.0, 20.0, 8.0, 0.5) / 100
    b_protection = st.slider("Barrière protection (%)", 40, 100, 70, 1) / 100
    n_sims       = st.select_slider("Simulations Monte Carlo",
                                     [5000, 10000, 25000, 50000], value=10000)
    run = st.button("Lancer la simulation")


# ── EN-TÊTE ───────────────────────────────────────────────────────────────

st.markdown(f"""
<div style='padding:4px 0 20px;'>
    <div style='font-family:Roboto Mono,monospace;font-size:9px;
                color:#999;letter-spacing:3px;text-transform:uppercase;
                margin-bottom:6px;'>Simulateur — Produit Structuré</div>
    <div style='font-family:Roboto,sans-serif;font-size:28px;
                font-weight:300;color:#1a1a1a;letter-spacing:-1px;'>
        Autocall <span style='color:#C8B560;'>{ticker}</span>
        <span style='font-size:16px;color:#999;'>
            &nbsp;/&nbsp;{T_max} ans
            &nbsp;/&nbsp;rappel {b_rappel*100:.0f}%
            &nbsp;/&nbsp;coupon {coupon*100:.1f}%/an
            &nbsp;/&nbsp;protection {b_protection*100:.0f}%
        </span>
    </div>
</div>
<hr style='border:none;border-top:1px solid #e0e0e0;margin:0 0 20px;'>
""", unsafe_allow_html=True)

# Mécanique
with st.expander("Mécanique du produit", expanded=False):
    st.markdown(f"""
    <div class='rule'>
        Rappel anticipé — observation tous les {freq_obs} mois<br>
        Si S(t) &gt;= {b_rappel*100:.0f}% x S0 : remboursement nominal + coupon {coupon*100:.1f}% x durée écoulée
    </div>
    <div class='rule' style='border-left-color:#8B6914;'>
        Protection du capital — à maturité ({T_max} ans)<br>
        Si S(T) &gt;= {b_protection*100:.0f}% x S0 : remboursement nominal, sans coupon
    </div>
    <div class='rule' style='border-left-color:#8B2020;'>
        Perte en capital — à maturité ({T_max} ans)<br>
        Si S(T) &lt; {b_protection*100:.0f}% x S0 : remboursement proportionnel à la performance
    </div>
    """, unsafe_allow_html=True)


# ── SIMULATION ────────────────────────────────────────────────────────────

if run:
    with st.spinner(""):
        df, paths = simuler_autocall(
            spot_yf, sigma, r, T_max, freq_obs,
            b_rappel, b_protection, coupon, nominal, n_sims
        )

    p_re = (df['scenario'] == 'rappel').sum()
    p_pr = (df['scenario'] == 'protection').sum()
    p_lo = (df['scenario'] == 'perte').sum()

    pct_re = p_re / n_sims * 100
    pct_pr = p_pr / n_sims * 100
    pct_lo = p_lo / n_sims * 100

    dur_moy   = df[df['scenario']=='rappel']['duree'].mean() if p_re > 0 else 0
    rend_re   = df[df['scenario']=='rappel']['rendement_pct'].mean() if p_re > 0 else 0
    perte_moy = df[df['scenario']=='perte']['rendement_pct'].mean() if p_lo > 0 else 0
    esp_gain  = df['gain'].mean()
    esp_rend  = (esp_gain / nominal - 1) * 100
    color_rend = GREEN if esp_rend >= 0 else RED

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='kpi-block highlight'>
            <div class='label'>Rappel anticipé</div>
            <div class='value' style='color:#C8B560;'>{pct_re:.1f}%</div>
            <div style='font-family:Roboto Mono,monospace;font-size:11px;
                        color:#999;margin-top:4px;'>durée moy. {dur_moy:.1f} ans</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='kpi-block'>
            <div class='label'>Protection capital</div>
            <div class='value'>{pct_pr:.1f}%</div>
            <div style='font-family:Roboto Mono,monospace;font-size:11px;
                        color:#999;margin-top:4px;'>remboursement {nominal:,}€</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='kpi-block danger'>
            <div class='label'>Perte en capital</div>
            <div class='value' style='color:#8B2020;'>{pct_lo:.1f}%</div>
            <div style='font-family:Roboto Mono,monospace;font-size:11px;
                        color:#999;margin-top:4px;'>perte moy. {abs(perte_moy):.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='kpi-block'>
            <div class='label'>Rendement espéré</div>
            <div class='value' style='color:{color_rend};'>{esp_rend:+.2f}%</div>
            <div style='font-family:Roboto Mono,monospace;font-size:11px;
                        color:#999;margin-top:4px;'>gain moy. {esp_gain:,.0f}€</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #e0e0e0;margin:20px 0;">', unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Trajectoires",
        "Probabilités par date",
        "Distribution des gains",
        "Sensibilité"
    ])

    with tab1:
        time_ax = np.linspace(0, T_max, paths.shape[1])
        fig = go.Figure()

        shown  = {'rappel': 0, 'protection': 0, 'perte': 0}
        limits = {'rappel': 120, 'protection': 60, 'perte': 120}
        c_alpha = {
            'rappel':     'rgba(46,107,62,0.15)',
            'protection': 'rgba(139,105,20,0.15)',
            'perte':      'rgba(139,32,32,0.15)'
        }
        for _, row in df.sample(min(300, len(df))).iterrows():
            sc = row['scenario']
            if shown[sc] < limits[sc]:
                fig.add_trace(go.Scatter(
                    x=time_ax, y=paths[int(row['simulation'])],
                    mode='lines',
                    line=dict(color=c_alpha[sc], width=0.6),
                    showlegend=False
                ))
                shown[sc] += 1

        fig.add_hline(y=spot_yf * b_rappel, line_color=GREEN,
                       line_width=1.2, line_dash='dash',
                       annotation_text=f"Rappel  {b_rappel*100:.0f}%",
                       annotation_font=dict(color=GREEN, size=10, family=MONO))
        fig.add_hline(y=spot_yf * b_protection, line_color=RED,
                       line_width=1.2, line_dash='dash',
                       annotation_text=f"Protection  {b_protection*100:.0f}%",
                       annotation_font=dict(color=RED, size=10, family=MONO))
        fig.add_hline(y=spot_yf, line_color=MUTED, line_width=0.8, line_dash='dot',
                       annotation_text="S0",
                       annotation_font=dict(color=MUTED, size=10, family=MONO))

        for i in range(1, int(T_max * 12 / freq_obs) + 1):
            t = i * freq_obs / 12
            if t <= T_max:
                fig.add_vline(x=t, line_color='#ddd', line_width=0.8, line_dash='dot')

        for col, lbl in [
            (GREEN, f"Rappel anticipé ({pct_re:.0f}%)"),
            (AMBER, f"Protection capital ({pct_pr:.0f}%)"),
            (RED,   f"Perte ({pct_lo:.0f}%)")
        ]:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                      line=dict(color=col, width=1.5), name=lbl))

        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=MONO, color='#666', size=10),
            xaxis=dict(title='Années', gridcolor=GRID, zeroline=False),
            yaxis=dict(title='Prix', gridcolor=GRID, zeroline=False),
            height=400,
            legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)',
                        bordercolor=GRID, borderwidth=1,
                        orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dates_obs = sorted([
            round(i * freq_obs / 12, 4)
            for i in range(1, int(T_max * 12 / freq_obs) + 1)
            if i * freq_obs / 12 <= T_max
        ])
        rpar = df[df['scenario']=='rappel']['duree'].round(2).value_counts()

        prob_d = []
        for d in dates_obs:
            if len(rpar) > 0:
                key = min(rpar.index, key=lambda x: abs(x - d))
                prob_d.append(rpar.get(key, 0) / n_sims * 100)
            else:
                prob_d.append(0)
        prob_c = np.cumsum(prob_d)

        fig2 = make_subplots(rows=1, cols=2,
                              subplot_titles=["P(rappel) par date", "P(rappel) cumulée"],
                              horizontal_spacing=0.1)
        fig2.add_trace(go.Bar(
            x=[f"A{d:.1f}" for d in dates_obs], y=prob_d,
            marker_color=GOLD, marker_line=dict(color=GOLD, width=0)
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=[f"A{d:.1f}" for d in dates_obs], y=prob_c,
            mode='lines+markers',
            line=dict(color=DARK, width=1.5),
            marker=dict(size=5, color=DARK),
            fill='tozeroy', fillcolor='rgba(200,181,96,0.12)'
        ), row=1, col=2)

        fig2.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=MONO, color='#666', size=10),
            height=320, showlegend=False,
            margin=dict(l=50, r=20, t=40, b=60)
        )
        for i in range(1, 3):
            fig2.update_xaxes(gridcolor=GRID, row=1, col=i, tickangle=45)
            fig2.update_yaxes(gridcolor=GRID, row=1, col=i, title_text='%')
        st.plotly_chart(fig2, use_container_width=True)

        table = pd.DataFrame({
            'Date': [f"Année {d:.1f}" for d in dates_obs],
            'P(rappel)': [f"{p:.1f}%" for p in prob_d],
            'P(rappel cumulé)': [f"{p:.1f}%" for p in prob_c],
            'Coupon': [f"{coupon*d*100:.1f}%" for d in dates_obs],
            'Gain si rappel (€)': [f"{nominal + nominal*coupon*d:,.0f}" for d in dates_obs]
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

    with tab3:
        fig3 = go.Figure()
        for sc, col, lbl in [
            ('rappel', GREEN, 'Rappel anticipé'),
            ('protection', AMBER, 'Protection capital'),
            ('perte', RED, 'Perte en capital')
        ]:
            data = df[df['scenario'] == sc]['gain']
            if len(data) > 0:
                rv, gv, bv = int(col[1:3],16), int(col[3:5],16), int(col[5:7],16)
                fig3.add_trace(go.Histogram(
                    x=data, name=lbl, nbinsx=40,
                    marker_color=f'rgba({rv},{gv},{bv},0.6)',
                    marker_line=dict(color=col, width=0.5)
                ))
        fig3.add_vline(x=nominal, line_color=MUTED, line_dash='dash', line_width=1,
                        annotation_text=f'Nominal  {nominal:,}€',
                        annotation_font=dict(color=MUTED, size=10, family=MONO))
        fig3.add_vline(x=esp_gain, line_color=DARK, line_dash='dash', line_width=1.5,
                        annotation_text=f'Espérance  {esp_gain:,.0f}€',
                        annotation_font=dict(color=DARK, size=10, family=MONO))
        fig3.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=MONO, color='#666', size=10),
            xaxis=dict(title='Gain final (€)', gridcolor=GRID),
            yaxis=dict(title='Fréquence', gridcolor=GRID),
            height=380, barmode='overlay',
            legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)',
                        bordercolor=GRID, borderwidth=1),
            margin=dict(l=50, r=20, t=20, b=50)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("""
        <div style='font-family:Roboto Mono,monospace;font-size:11px;color:#999;
                    margin-bottom:16px;'>
            Grille 10x10 — 2 000 simulations par point.
            Impact de la volatilite et de la barriere de rappel.
        </div>
        """, unsafe_allow_html=True)

        vols_g = np.linspace(0.10, 0.50, 10)
        bars_g = np.linspace(0.85, 1.10, 10)
        prob_g = np.zeros((len(vols_g), len(bars_g)))
        rend_g = np.zeros((len(vols_g), len(bars_g)))

        with st.spinner("Calcul de la grille..."):
            for i, v in enumerate(vols_g):
                for j, b in enumerate(bars_g):
                    df_g, _ = simuler_autocall(
                        spot_yf, v, r, T_max, freq_obs,
                        b, b_protection, coupon, nominal, 2000
                    )
                    prob_g[i, j] = (df_g['scenario']=='rappel').sum() / 2000 * 100
                    rend_g[i, j] = (df_g['gain'].mean() / nominal - 1) * 100

        xl = [f"{b*100:.0f}%" for b in bars_g]
        yl = [f"{v*100:.0f}%" for v in vols_g]

        cg1, cg2 = st.columns(2)
        with cg1:
            fig_g1 = go.Figure(go.Heatmap(
                z=prob_g, x=xl, y=yl,
                colorscale=[[0, BG], [0.5, '#d4c068'], [1, GREEN]],
                colorbar=dict(title='%', tickfont=dict(family=MONO, size=9))
            ))
            fig_g1.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(family=MONO, color='#666', size=10),
                xaxis=dict(title='Barrière de rappel'),
                yaxis=dict(title='Volatilité'),
                height=300, margin=dict(l=60, r=20, t=30, b=50),
                title=dict(text="P(rappel anticipé)",
                           font=dict(family=MONO, size=11, color='#666'))
            )
            st.plotly_chart(fig_g1, use_container_width=True)
        with cg2:
            fig_g2 = go.Figure(go.Heatmap(
                z=rend_g, x=xl, y=yl,
                colorscale=[[0, RED], [0.5, BG], [1, GREEN]],
                zmid=0,
                colorbar=dict(title='%', tickfont=dict(family=MONO, size=9))
            ))
            fig_g2.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(family=MONO, color='#666', size=10),
                xaxis=dict(title='Barrière de rappel'),
                yaxis=dict(title='Volatilité'),
                height=300, margin=dict(l=60, r=20, t=30, b=50),
                title=dict(text="Rendement espéré",
                           font=dict(family=MONO, size=11, color='#666'))
            )
            st.plotly_chart(fig_g2, use_container_width=True)

else:
    st.markdown("""
    <div class='waiting'>
        Configurer les parametres et lancer la simulation
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr style='border:none;border-top:1px solid #e0e0e0;margin:40px 0 16px;'>
<div style='font-family:Roboto Mono,monospace;font-size:9px;color:#ccc;
            letter-spacing:1.5px;text-transform:uppercase;'>
    Autocall Simulator — Oscar Dawny — EDHEC Business School / Centrale Lille — 2026
</div>
""", unsafe_allow_html=True)
