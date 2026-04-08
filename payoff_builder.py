"""
Leg Builder — Oscar Dawny
EDHEC Business School / Centrale Lille

pip install streamlit streamlit-drawable-canvas numpy pandas plotly scipy
streamlit run payoff_builder.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import (inject_css, render_sidebar, render_footer, bs_price,
                   payoff_at_expiry, BG, GRID, MONO, GOLD, DARK, GREEN,
                   RED, MUTED, AMBER, TL, TC)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Leg Builder — Payoff Builder", layout="centered",
                   initial_sidebar_state="expanded")
inject_css()
S0, sigma, r, T, S_min, S_max, pf_min, pf_max, S_range = render_sidebar()

st.markdown(f"""
<div style='padding:4px 0 10px;'>
    <div style='font-family:Roboto Mono,monospace;font-size:9px;color:#999;
                letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;'>
        Construction manuelle</div>
    <div style='font-family:Roboto,sans-serif;font-size:28px;
                font-weight:300;color:#1a1a1a;letter-spacing:-1px;'>
        Leg Builder
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
    Ajoutez des options une par une — le payoff, le coût et les breakevens
    se mettent à jour en temps réel.
</div>
""", unsafe_allow_html=True)

if 'legs_manual' not in st.session_state:
    st.session_state['legs_manual'] = []

st.markdown("""
<div style='font-family:Roboto Mono,monospace;font-size:9px;color:#999;
            letter-spacing:2px;text-transform:uppercase;margin:16px 0 8px;'>
    Ajouter un leg</div>
""", unsafe_allow_html=True)

lb_c1, lb_c2, lb_c3, lb_c4, lb_c5 = st.columns([1.2, 1.2, 1.5, 1, 1])
with lb_c1:
    lb_type = st.selectbox("Type", ["Call", "Put"], key="lb_type", label_visibility="collapsed")
with lb_c2:
    lb_dir = st.selectbox("Direction", ["Long", "Short"], key="lb_dir", label_visibility="collapsed")
with lb_c3:
    lb_strike = st.number_input("Strike", value=float(S0), min_value=0.1, step=1.0,
                                 key="lb_strike", label_visibility="collapsed")
with lb_c4:
    lb_qty = st.number_input("Quantité", value=1, min_value=1, max_value=10, step=1,
                              key="lb_qty", label_visibility="collapsed")
with lb_c5:
    lb_add = st.button("＋ Ajouter", key="lb_add_btn", use_container_width=True)

if lb_add:
    w = lb_qty if lb_dir == "Long" else -lb_qty
    st.session_state['legs_manual'].append({
        'type': lb_type.lower(),
        'strike': lb_strike,
        'weight': w,
    })
    st.rerun()

lb_clear_col, _ = st.columns([1, 4])
with lb_clear_col:
    if st.button("Tout effacer", key="lb_clear"):
        st.session_state['legs_manual'] = []
        st.rerun()

st.markdown('<hr style="border:none;border-top:1px solid #e0e0e0;margin:16px 0;">', unsafe_allow_html=True)

legs_m = st.session_state['legs_manual']

if legs_m:
    st.markdown("""
    <div style='font-family:Roboto Mono,monospace;font-size:9px;color:#999;
                letter-spacing:2px;text-transform:uppercase;margin:10px 0 8px;'>
        Composition actuelle</div>
    """, unsafe_allow_html=True)

    total_cost_m = 0.0
    total_payoff_m = np.zeros_like(S_range)

    for i, leg in enumerate(legs_m):
        price = bs_price(S0, leg['strike'], T, r, sigma, leg['type'])
        cost = leg['weight'] * price
        total_cost_m += cost
        pf_leg = leg['weight'] * payoff_at_expiry(S_range, leg['strike'], leg['type'])
        total_payoff_m += pf_leg

        dc = GREEN if leg['weight'] > 0 else RED
        d = 'Long' if leg['weight'] > 0 else 'Short'
        w_str = f" {abs(leg['weight'])}x" if abs(leg['weight']) > 1 else ""

        col_leg, col_rm = st.columns([6, 1])
        with col_leg:
            st.markdown(f"""
            <div class='leg-row'>
                <span style='width:60px;font-size:10px;color:{dc};
                             font-family:Roboto Mono,monospace;'>{d}{w_str}</span>
                <span class='tag {leg["type"]}'>{TL.get(leg["type"],leg["type"])}</span>
                <span style='flex:1;'>K = <b>{leg["strike"]:.1f}</b>
                    &nbsp;({leg["strike"]/S0*100:.0f}% ATM)</span>
                <span style='width:110px;text-align:right;color:{dc};'>
                    coût = {cost:+.4f}</span>
            </div>""", unsafe_allow_html=True)
        with col_rm:
            if st.button("✕", key=f"rm_{i}"):
                st.session_state['legs_manual'].pop(i)
                st.rerun()

    max_gain_m = float(np.max(total_payoff_m))
    max_loss_m = float(np.min(total_payoff_m))
    be_indices = np.where(np.diff(np.sign(total_payoff_m)))[0]
    breakevens = [float(S_range[idx] + (S_range[idx+1]-S_range[idx]) *
                  (-total_payoff_m[idx])/(total_payoff_m[idx+1]-total_payoff_m[idx]+1e-15))
                  for idx in be_indices]

    st.markdown('<hr style="border:none;border-top:1px solid #e0e0e0;margin:16px 0;">', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        cc_total = GREEN if total_cost_m >= 0 else RED
        st.markdown(f"""
        <div class='kpi gold'>
            <div class='label'>Coût net de la structure</div>
            <div class='value' style='color:{cc_total};'>{total_cost_m:+.4f}</div>
            <div style='font-family:Roboto Mono,monospace;font-size:10px;
                        color:#999;margin-top:2px;'>{total_cost_m/S0*100:+.2f}% du spot</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class='kpi green'>
            <div class='label'>Gain max</div>
            <div class='value' style='color:{GREEN};'>{max_gain_m:+.1f}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class='kpi red'>
            <div class='label'>Perte max</div>
            <div class='value' style='color:{RED};'>{max_loss_m:+.1f}</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        be_str = " / ".join(f"{b:.1f}" for b in breakevens) if breakevens else "—"
        st.markdown(f"""
        <div class='kpi'>
            <div class='label'>Breakeven(s)</div>
            <div class='value' style='font-size:18px;'>{be_str}</div>
        </div>""", unsafe_allow_html=True)

    fig_lb = go.Figure()
    fig_lb.add_trace(go.Scatter(x=S_range, y=np.maximum(total_payoff_m, 0),
        fill='tozeroy', fillcolor='rgba(46,107,62,0.06)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_lb.add_trace(go.Scatter(x=S_range, y=np.minimum(total_payoff_m, 0),
        fill='tozeroy', fillcolor='rgba(139,32,32,0.06)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_lb.add_trace(go.Scatter(x=S_range, y=total_payoff_m, mode='lines',
        line=dict(color=DARK, width=2.5), name='Payoff total'))

    palette_lb = [GOLD, GREEN, RED, '#1a4a7a', '#6B4C2E', '#4a2e6b', AMBER, '#2a6b6b']
    for i, leg in enumerate(legs_m):
        pf_i = leg['weight'] * payoff_at_expiry(S_range, leg['strike'], leg['type'])
        s = "+" if leg['weight'] > 0 else "−"
        w_s = f"{abs(leg['weight'])}x " if abs(leg['weight']) > 1 else ""
        fig_lb.add_trace(go.Scatter(x=S_range, y=pf_i, mode='lines',
            line=dict(color=palette_lb[i % len(palette_lb)], width=1.2, dash='dash'),
            name=f"{s} {w_s}{TL.get(leg['type'],'')} K={leg['strike']:.0f}", opacity=0.7))

    for be in breakevens:
        fig_lb.add_vline(x=be, line_color=AMBER, line_dash='dash', line_width=1,
                         annotation_text=f'BE {be:.0f}',
                         annotation_font=dict(color=AMBER, size=9, family=MONO))

    fig_lb.add_vline(x=S0, line_color=MUTED, line_dash='dot', line_width=1,
                     annotation_text='S₀', annotation_font=dict(color=MUTED, size=10, family=MONO))
    fig_lb.add_hline(y=0, line_color=MUTED, line_width=0.8)
    fig_lb.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=MONO, color='#666', size=10),
        xaxis=dict(title='S_T', gridcolor=GRID, zeroline=False),
        yaxis=dict(title='Gain / Perte', gridcolor=GRID, zeroline=False),
        height=380,
        legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)',
                    bordercolor=GRID, borderwidth=1, orientation='h',
                    yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=60, r=20, t=50, b=50))
    st.plotly_chart(fig_lb, use_container_width=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:60px;font-family:Roboto Mono,monospace;
                font-size:11px;color:#ccc;letter-spacing:2px;text-transform:uppercase;'>
        Ajoute un leg pour commencer la construction</div>
    """, unsafe_allow_html=True)

render_footer()
