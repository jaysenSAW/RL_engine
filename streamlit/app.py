import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy
import sys
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="Rocket RL Tutorial", layout="wide", page_icon="🚀")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, "../src/utils/")

from agent import Environment
from Q_learning import QLearningTrainer
from deep_q_learning import DQNTrainer

# ── Palette ───────────────────────────────────────────────────────────────────
C_ORANGE = "#E8572A"
C_BLUE   = "#2563EB"
C_TEAL   = "#0D9488"
C_GRAY   = "#6B7280"
C_BORDER = "#E2E8F0"
C_LIGHT  = "#F8FAFC"

# ── CSS (white background) ────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stApp, [data-testid="stAppViewContainer"] {{ background: #ffffff; }}
  section[data-testid="stMain"] {{ background: #ffffff; }}
  [data-testid="stSidebar"] {{ background: #F8FAFC; }}
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: #1e293b; }}

  /* force all standard text elements to be dark on white */
  p, li, span, label, div {{ color: #1e293b; }}

  .page-title {{
    font-size: 2rem; font-weight: 800; color: {C_ORANGE};
    letter-spacing: -0.5px; margin-bottom: 2px;
  }}
  .page-sub {{ font-size: 1rem; color: {C_GRAY}; margin-bottom: 1rem; }}

  .sec-title {{
    font-size: 1.05rem; font-weight: 700; color: {C_ORANGE};
    border-left: 4px solid {C_ORANGE}; padding-left: 10px;
    margin: 1.4rem 0 0.6rem 0;
  }}
  .info-card {{
    background: {C_LIGHT}; border: 1px solid {C_BORDER};
    border-radius: 10px; padding: 16px 20px; margin: 8px 0;
    color: #1e293b;
  }}
  .formula-box {{
    background: #FFF7ED; border-left: 4px solid {C_ORANGE};
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-family: monospace; font-size: 0.95rem; margin: 6px 0 12px 0;
    color: #1e293b;
  }}
  .step-badge {{
    display: inline-block; background: {C_ORANGE}; color: white;
    border-radius: 50%; width: 26px; height: 26px; text-align: center;
    line-height: 26px; font-weight: 700; font-size: 0.85rem; margin-right: 8px;
  }}

  /* tabs */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 4px; background: {C_LIGHT}; border-radius: 10px; padding: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 7px; padding: 6px 16px; font-weight: 600;
    color: #374151 !important; background: transparent;
  }}
  .stTabs [aria-selected="true"] {{
    background: {C_ORANGE} !important; color: white !important;
  }}

  /* metric label + value */
  [data-testid="stMetricLabel"] {{ color: #6B7280 !important; }}
  [data-testid="stMetricValue"] {{ color: {C_ORANGE} !important; font-weight: 700; }}

  /* number inputs, sliders, select labels */
  .stNumberInput label, .stSlider label, .stSelectbox label,
  .stMultiSelect label, .stToggle label, .stFileUploader label {{
    color: #1e293b !important; font-weight: 500;
  }}

  /* code blocks */
  .stCode, code {{ background: #F1F5F9 !important; color: #1e293b !important; }}

  /* expander */
  div[data-testid="stExpander"] {{
    border: 1px solid {C_BORDER}; border-radius: 8px; background: {C_LIGHT};
  }}
  div[data-testid="stExpander"] summary span {{
    color: #1e293b !important;
  }}

  /* dataframe */
  [data-testid="stDataFrame"] {{ color: #1e293b; }}

  /* alerts / info boxes */
  [data-testid="stAlert"] {{ color: #1e293b; }}

  /* markdown inside st.markdown */
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stMarkdownContainer"] td,
  [data-testid="stMarkdownContainer"] th {{ color: #1e293b !important; }}

  hr {{ border-color: {C_BORDER}; }}
  .stProgress > div > div {{ background: {C_ORANGE}; }}
</style>
""", unsafe_allow_html=True)

# ── Default config ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "states_variables": ["pos_x", "pos_y", "angle", "speed_x", "speed_y", "weight_rocket"],
    "agent_variables": ["booster", "alpha"],
    "initial_values": {
        "pos_x": [75.0], "pos_y": [175.0], "angle": [0.0],
        "speed_x": [0.0], "speed_y": [0.0], "weight_rocket": [305],
        "booster": [0.0], "alpha": [0.0],
        "acceleration_x": [0.0], "acceleration_y": [0.0],
        "m_fuel": [300], "futur_pos_x": [75.0], "futur_pos_y": [175.0],
        "weight_dry_rocket": [5], "G": [1.62], "m_fuel_ini": [300.0],
        "pos_x_star": [140.0], "pos_y_star": [0.0],
        "pos_x_ini": [75.0], "pos_y_ini": [175.0],
        "upper_boundary": [0.0], "lower_boundary": [0.0],
        "acceleration_limit_x": [10], "acceleration_limit_y": [10],
        "speed_limit_x": [10], "speed_limit_y": [10],
        "distance_x_reward": [1.0], "distance_y_reward": [1.0],
        "speed_x_reward": [0.0], "speed_y_reward": [0.0],
        "ratio_fuel": [1.0], "dt": [4], "time": [0]
    },
    "_limit": ["min", "max", "n_bins"],
    "limit": {
        "pos_x": [50.0, 200, 61], "pos_y": [0.0, 250.0, 81],
        "angle": [-0.8, 0.8, 3], "speed_x": [-30.0, 30.0, 13],
        "speed_y": [-30.0, 30.0, 13], "weight_rocket": [0.0, 305, 62],
        "booster": [0.0, 2.0, 3], "alpha": [-0.8, 0.8, 3],
        "acceleration_x": [-20.0, 20.0, 21], "acceleration_y": [-20.0, 20.0, 21],
        "m_fuel": [0.0, 400, 801]
    },
    "n_action": {
        "booster": {"0": 0.0, "1": 1.0, "2": 2.0},
        "alpha": {"0": -0.8, "1": 0.0, "2": 0.8}
    },
    "action_to_take": {
        "booster": {"$booster$": "$action$"},
        "alpha": {"$alpha$": "$action$"}
    },
    "equations_variables": {
        "$F$": "600", "$time$": "$time$ + $dt$", "$theta$": "0.0",
        "$x_0$": "$pos_x$", "$y_0$": "$pos_y$",
        "$Vx_0$": "$speed_x$", "$Vy_0$": "$speed_y$",
        "$angle$": "$angle$ + $alpha$",
        "$m_fuel$": "$m_fuel$ - $booster$ *5 -np.ceil( np.abs($alpha$) ) *5",
        "$weight_rocket$": "$weight_dry_rocket$ + $m_fuel$",
        "$acceleration_x$": "($F$/(5+$weight_rocket$) * np.sin($angle$)) * $booster$",
        "$acceleration_y$": "($F$/(5+$weight_rocket$) * np.cos($angle$)) * $booster$ - $G$",
        "$speed_x$": "($F$/(5+$weight_rocket$) * np.sin($angle$)) * $booster$ * $dt$ + $Vx_0$",
        "$speed_y$": "($F$/(5+$weight_rocket$) * np.cos($angle$)) * $booster$ * $dt$ - $G$ * $dt$ + $Vy_0$",
        "$pos_x$": "(0.5 * $F$/(5+$weight_rocket$) * np.sin($angle$)) * $booster$ * $dt$**2 + $Vx_0$ * $dt$ + $x_0$",
        "$pos_y$": "(0.5 * $F$/(5+$weight_rocket$) * np.cos($angle$)) * $booster$ * $dt$**2 - $G$ * $dt$**2 + $Vy_0$ * $dt$ + $y_0$",
        "$distance_y_reward$": "np.abs( ($pos_y$ - $pos_y_star$)/($pos_y_ini$ - $pos_y_star$) )",
        "$distance_x_reward$": "np.abs( ($pos_x$ - $pos_x_star$)/($pos_x_ini$ - $pos_x_star$) )",
        "$speed_x_reward$": "np.exp(1) - np.exp( np.max([ np.max( np.abs($speed_x$)/$speed_limit_x$ ), 1 ]) )",
        "$speed_y_reward$": "np.exp(1) - np.exp( np.max([ np.max( np.abs($speed_y$)/$speed_limit_y$ ), 1 ]) )",
        "$futur_pos_y$": "$pos_y$ + 3 * $speed_y$",
        "$futur_pos_x$": "$pos_x$ + 3 * $speed_x$",
        "$ratio_fuel$": "$m_fuel$/$m_fuel_ini$",
        "y_lower_limit": "0", "y_upper_limit": "200",
        "$upper_boundary$": "-np.exp(0) + np.exp(np.min([ np.min(-$futur_pos_y$ + y_upper_limit), 0]))",
        "$lower_boundary$": "-np.exp(0) + np.exp(np.min([ np.min($futur_pos_y$ -y_lower_limit), 0]))"
    },
    "equations_rewards": {
        "distance_y_reward": "np.abs( ($pos_y$ - $pos_y_star$)/($pos_y_ini$ - $pos_y_star$) )",
        "distance_x_reward": "np.abs( ($pos_x$ - $pos_x_star$)/($pos_x_ini$ - $pos_x_star$) )",
        "speed_x_reward": "np.exp(1) - np.exp( np.max([ np.max( np.abs($speed_x$)/$speed_limit_x$ ), 1 ]) )",
        "speed_y_reward": "np.exp(1) - np.exp( np.max([ np.max( np.abs($speed_y$)/$speed_limit_y$ ), 1 ]) )",
        "y_lower_limit": "0", "y_upper_limit": "200",
        "upper_boundary": "-np.exp(0) + np.exp(np.min([ np.min(-$futur_pos_y$ + y_upper_limit), 0]))",
        "lower_boundary": "-np.exp(0) + np.exp(np.min([ np.min($futur_pos_y$ -y_lower_limit), 0]))",
        "height_boundaries": "-2 + lower_boundary + upper_boundary",
        "_ratio_fuel": "$m_fuel$/$m_fuel_ini$",
        "$booster$": "2*(-distance_y_reward) + speed_y_reward + 0.5 * $ratio_fuel$",
        "$alpha$": "2*(-distance_x_reward)  + speed_x_reward  - 0.2*np.sin(np.abs($angle$)) + 0.5 * $ratio_fuel$"
    },
    "stop_episode": {
        "pos_x": [135, 145], "pos_y": [0, 5],
        "acceleration_y": [-2, 2], "speed_x": [-10, 10], "speed_y": [-10, 10]
    },
    "condition_stop_episode": "all"
}

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("config", copy.deepcopy(DEFAULT_CONFIG)),
             ("training_done", False),
             ("trainer", None),
             ("selected_episode", 0),
             ("algo_mode", "tabular")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🚀 Rocket Landing — Reinforcement Learning Tutorial</div>',
            unsafe_allow_html=True)
st.markdown('<div class="page-sub">An interactive Q-learning sandbox for a 2D rocket landing problem</div>',
            unsafe_allow_html=True)

# ── Shared plot layout helper ──────────────────────────────────────────────────
def light_layout(**extra):
    base = dict(
        template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#1e293b"),
        margin=dict(l=50, r=20, t=45, b=45),
        xaxis=dict(showgrid=True, gridcolor=C_BORDER, zeroline=True, zerolinecolor="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor=C_BORDER, zeroline=True, zerolinecolor="#94a3b8"),
    )
    base.update(extra)
    return base

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_intro, tab_env, tab_reward, tab_train, tab_results, tab_json = st.tabs([
    "📖 Introduction",
    "⚙️ Environment",
    "📐 Reward Explorer",
    "🎯 Training",
    "📊 Results",
    "📄 JSON Export",
])

# ══════════════════════════════════════════════════════════════════════════════
# 0 — INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_intro:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="sec-title">🎯 Goal</div>', unsafe_allow_html=True)
        st.markdown("""
The purpose of this tutorial is to **land a 2-D rocket at a specific ground coordinate** by
learning to control two actuators simultaneously:

- 🔥 **Booster** — main engine thrust; governs altitude and vertical deceleration
- 🔄 **Alpha** — lateral thruster / gimbal angle; governs horizontal motion and orientation

Instead of hand-crafting a controller, we use **tabular Q-learning** to let the agent discover
an optimal policy purely through trial and error across thousands of simulation steps.
        """)

        st.markdown('<div class="sec-title">🧩 The Toy Model</div>', unsafe_allow_html=True)
        st.markdown("""
The rocket dynamics are governed by **Newtonian kinematics** discretised at each timestep `dt`:

- Thrust force **F = 600 N** applied along the current rocket angle
- Moon gravity **G = 1.62 m/s²** acts downward at all times
- Fuel is consumed proportionally to booster level and angle corrections
- Equations are compiled directly from the JSON config at runtime — no hard-coded physics

The **state space** is discretised into bins to build a finite Q-table. Each state is a
tuple of binned values for `(pos_x, pos_y, angle, speed_x, speed_y, weight_rocket)`.

The **joint action space** combines both agents:

| Agent | Discrete Actions |
|-------|-----------------|
| Booster | 0 (off) · 1 (half thrust) · 2 (full thrust) |
| Alpha   | −0.8 (tilt left) · 0 (straight) · +0.8 (tilt right) |

This gives **9 joint actions** per state.
        """)

        st.markdown('<div class="sec-title">⚠️ Episode Termination</div>', unsafe_allow_html=True)
        st.markdown("""
An episode ends when **all** stop conditions are simultaneously satisfied:

| Condition | Target range |
|-----------|-------------|
| `pos_x` in landing zone | 135 – 145 |
| `pos_y` near ground | 0 – 5 |
| `acceleration_y` low | −2 to +2 |
| `speed_x` low | −10 to +10 |
| `speed_y` low | −10 to +10 |

It also ends if the rocket flies out of the defined state-space bounds.
        """)

    with right:
        st.markdown('<div class="sec-title">🔄 Q-Learning Update Rule</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card"><b>Bellman equation:</b></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="formula-box">Q(s,a) ← Q(s,a) + α·[ R + γ·maxQ(s',a') − Q(s,a) ]</div>""",
                    unsafe_allow_html=True)
        st.markdown("""
| Symbol | Meaning |
|--------|---------|
| `Q(s,a)` | Estimated value of state `s`, action `a` |
| `α` | Learning rate |
| `R` | Immediate reward received |
| `γ` | Discount factor (importance of future rewards) |
| `maxQ(s',a')` | Best achievable value from next state |
        """)

        st.markdown('<div class="sec-title">🗺️ Workflow</div>', unsafe_allow_html=True)
        steps = [
            ("Configure", "Set rocket start, target, physics & reward weights in **Environment**"),
            ("Explore", "Visualise reward components live in **Reward Explorer**"),
            ("Train", "Run Q-learning with live episode logs in **Training**"),
            ("Analyse", "Inspect trajectories, state plots & convergence in **Results**"),
            ("Export", "Download the JSON config for reproducibility in **JSON Export**"),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(f'<span class="step-badge">{i}</span><b>{title}</b> — {desc}',
                        unsafe_allow_html=True)
            st.write("")

        st.markdown('<div class="sec-title">📏 Simulation Space</div>', unsafe_allow_html=True)
        fig_map = go.Figure()
        fig_map.add_shape(type="rect", x0=50, x1=200, y0=0, y1=250,
                          fillcolor="rgba(219,234,254,0.35)", line_width=0)
        fig_map.add_shape(type="rect", x0=50, x1=200, y0=-8, y1=0,
                          fillcolor="#d4a373", line_width=0)
        fig_map.add_vrect(x0=135, x1=145, fillcolor="rgba(34,197,94,0.25)",
                          line_width=1, line_color="#16a34a",
                          annotation_text="🎯 Target", annotation_position="top left",
                          annotation_font_size=10)
        fig_map.add_trace(go.Scatter(
            x=[75], y=[175], mode="markers+text",
            marker=dict(color=C_ORANGE, size=15, symbol="triangle-up"),
            text=["Start"], textposition="top center",
            showlegend=False
        ))
        fig_map.add_annotation(
            x=140, y=2, ax=75, ay=175, xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowsize=1.2, arrowcolor=C_ORANGE, arrowwidth=2
        )
        fig_map.update_layout(
            height=255, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="X", range=[45, 205], showgrid=True, gridcolor=C_BORDER),
            yaxis=dict(title="Altitude", range=[(-12), 260], showgrid=True, gridcolor=C_BORDER),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(color="#1e293b")
        )
        st.plotly_chart(fig_map, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — ENVIRONMENT CONFIG
# ══════════════════════════════════════════════════════════════════════════════
with tab_env:
    cfg = st.session_state.config

    st.markdown('<div class="sec-title">Rocket & Mission Parameters</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Initial Position**")
        pos_x   = st.number_input("pos_x (initial)", value=float(cfg["initial_values"]["pos_x"][0]), key="pos_x")
        pos_y   = st.number_input("pos_y (initial)", value=float(cfg["initial_values"]["pos_y"][0]), key="pos_y")
        angle_0 = st.number_input("angle (initial, rad)", value=float(cfg["initial_values"]["angle"][0]), key="angle_init")

    with col2:
        st.markdown("**Target (Landing Zone)**")
        pos_x_star = st.number_input("Target X", value=float(cfg["initial_values"]["pos_x_star"][0]), key="px_star")
        pos_y_star = st.number_input("Target Y", value=float(cfg["initial_values"]["pos_y_star"][0]), key="py_star")
        st.markdown("**Stop Zone X range**")
        stop_x_min = st.number_input("stop pos_x min", value=float(cfg["stop_episode"]["pos_x"][0]), key="stop_xmin")
        stop_x_max = st.number_input("stop pos_x max", value=float(cfg["stop_episode"]["pos_x"][1]), key="stop_xmax")

    with col3:
        st.markdown("**Fuel & Physics**")
        m_fuel = st.number_input("m_fuel (initial)", value=float(cfg["initial_values"]["m_fuel"][0]), key="mfuel")
        dt_val = st.number_input("dt (timestep)", value=float(cfg["initial_values"]["dt"][0]),
                                  min_value=0.1, max_value=20.0, key="dt_val")
        G_val  = st.number_input("G (gravity)", value=float(cfg["initial_values"]["G"][0]), key="grav")

    st.divider()
    st.markdown('<div class="sec-title">Reward Weights</div>', unsafe_allow_html=True)
    rcol1, rcol2 = st.columns(2)

    with rcol1:
        st.markdown("**🔥 Booster reward** — vertical / altitude control")
        w_dist_y  = st.slider("Distance Y penalty weight", -5.0, 0.0, -2.0, 0.1, key="w_dy")
        w_speed_y = st.slider("Speed Y reward weight", 0.0, 3.0, 1.0, 0.1, key="w_sy")
        w_fuel_b  = st.slider("Fuel ratio weight (booster)", 0.0, 2.0, 0.5, 0.1, key="w_fb")
        st.markdown(
            f'<div class="formula-box">R_booster = {w_dist_y}·(−dist_y) + {w_speed_y}·speed_y_rew + {w_fuel_b}·fuel</div>',
            unsafe_allow_html=True)

    with rcol2:
        st.markdown("**🔄 Alpha reward** — horizontal / angle control")
        w_dist_x  = st.slider("Distance X penalty weight", -5.0, 0.0, -2.0, 0.1, key="w_dx")
        w_speed_x = st.slider("Speed X reward weight", 0.0, 3.0, 1.0, 0.1, key="w_sx")
        w_angle   = st.slider("Angle penalty weight", -1.0, 0.0, -0.2, 0.05, key="w_ang")
        w_fuel_a  = st.slider("Fuel ratio weight (alpha)", 0.0, 2.0, 0.5, 0.1, key="w_fa")
        st.markdown(
            f'<div class="formula-box">R_alpha = {w_dist_x}·(−dist_x) + {w_speed_x}·speed_x_rew + {w_angle}·sin(|angle|) + {w_fuel_a}·fuel</div>',
            unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sec-title">Speed & Acceleration Limits</div>', unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    with lc1:
        sl_x = st.number_input("speed_limit_x", value=float(cfg["initial_values"]["speed_limit_x"][0]), key="slx")
        sl_y = st.number_input("speed_limit_y", value=float(cfg["initial_values"]["speed_limit_y"][0]), key="sly")
    with lc2:
        al_x = st.number_input("acceleration_limit_x", value=float(cfg["initial_values"]["acceleration_limit_x"][0]), key="alx")
        al_y = st.number_input("acceleration_limit_y", value=float(cfg["initial_values"]["acceleration_limit_y"][0]), key="aly")

    st.divider()
    st.markdown('<div class="sec-title">Obstacle (optional)</div>', unsafe_allow_html=True)
    use_obstacle = st.toggle("Enable obstacle avoidance", value=False, key="use_obs")
    obs_x = obs_y = obs_r = excl = None
    if use_obstacle:
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            obs_x = st.number_input("Obstacle X", value=130.0, key="obs_x")
            obs_y = st.number_input("Obstacle Y", value=90.0,  key="obs_y")
        with oc2:
            obs_r = st.number_input("Radius", value=5.0,  key="obs_r")
            excl  = st.number_input("Exclusion zone", value=2.0, key="obs_excl")
        with oc3:
            st.info("A proximity penalty is added to both agent rewards when the rocket enters the exclusion zone.")

    if st.button("✅ Apply Configuration", type="primary", key="apply_cfg"):
        new_cfg = copy.deepcopy(DEFAULT_CONFIG)
        new_cfg["initial_values"].update({
            "pos_x": [pos_x], "pos_y": [pos_y], "angle": [angle_0],
            "m_fuel": [m_fuel], "m_fuel_ini": [float(m_fuel)],
            "pos_x_ini": [pos_x], "pos_y_ini": [pos_y],
            "pos_x_star": [pos_x_star], "pos_y_star": [pos_y_star],
            "futur_pos_x": [pos_x], "futur_pos_y": [pos_y],
            "dt": [dt_val], "G": [G_val],
            "speed_limit_x": [sl_x], "speed_limit_y": [sl_y],
            "acceleration_limit_x": [al_x], "acceleration_limit_y": [al_y],
            "weight_rocket": [5 + m_fuel],
        })
        new_cfg["stop_episode"]["pos_x"] = [stop_x_min, stop_x_max]
        b_eq = f"{w_dist_y}*(-distance_y_reward) + {w_speed_y}*speed_y_reward + {w_fuel_b} * $ratio_fuel$"
        a_eq = f"{w_dist_x}*(-distance_x_reward) + {w_speed_x}*speed_x_reward + {w_angle}*np.sin(np.abs($angle$)) + {w_fuel_a} * $ratio_fuel$"
        new_cfg["equations_rewards"]["$booster$"] = b_eq
        new_cfg["equations_rewards"]["$alpha$"]   = a_eq
        if use_obstacle and obs_x is not None:
            new_cfg["initial_values"].update({
                "pos_x_obstacle": [obs_x], "pos_y_obstacle": [obs_y],
                "obstacle_radius": [obs_r], "exclusion_zone": [excl],
                "obstacle_penalty": [0.0],
                "distance2obstacle_squarred": [(obs_x - pos_x)**2 + (obs_y - pos_y)**2]
            })
            new_cfg["equations_variables"]["$computed_distance2obstacle_squarred$"] = \
                "($pos_y$ - $pos_y_obstacle$)**2 + ($pos_x$ - $pos_x_obstacle$)**2"
            new_cfg["equations_variables"]["$obstacle_penalty$"] = \
                "np.minimum($computed_distance2obstacle_squarred$ -($obstacle_radius$ + $exclusion_zone$)**2, 0)"
            new_cfg["equations_rewards"]["$computed_distance2obstacle_squarred$"] = \
                "($pos_y$ - $pos_y_obstacle$)**2 + ($pos_x$ - $pos_x_obstacle$)**2"
            new_cfg["equations_rewards"]["$booster$"] = b_eq + " + $obstacle_penalty$"
            new_cfg["equations_rewards"]["$alpha$"]   = a_eq + " + $obstacle_penalty$"
        st.session_state.config = new_cfg
        st.session_state.training_done = False
        st.success("Configuration applied! Explore the Reward Explorer or go straight to Training.")


# ══════════════════════════════════════════════════════════════════════════════
# 2 — REWARD EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_reward:
    cfg = st.session_state.config
    iv  = cfg["initial_values"]

    speed_lim_x  = float(iv.get("speed_limit_x",  [10])[0])
    speed_lim_y  = float(iv.get("speed_limit_y",  [10])[0])
    pos_x_ini_v  = float(iv.get("pos_x_ini",  [75])[0])
    pos_y_ini_v  = float(iv.get("pos_y_ini",  [175])[0])
    pos_x_star_v = float(iv.get("pos_x_star", [140])[0])
    pos_y_star_v = float(iv.get("pos_y_star", [0])[0])
    m_fuel_ini_v = float(iv.get("m_fuel_ini", [300])[0])

    # pull weights set in environment tab
    w_dy_v  = st.session_state.get("w_dy",  -2.0)
    w_sy_v  = st.session_state.get("w_sy",   1.0)
    w_fb_v  = st.session_state.get("w_fb",   0.5)
    w_dx_v  = st.session_state.get("w_dx",  -2.0)
    w_sx_v  = st.session_state.get("w_sx",   1.0)
    w_ang_v = st.session_state.get("w_ang", -0.2)
    w_fa_v  = st.session_state.get("w_fa",   0.5)

    st.markdown('<div class="sec-title">1 · Distance Reward Component</div>', unsafe_allow_html=True)
    st.markdown("""
`distance_reward = |pos − pos_star| / |pos_ini − pos_star|`

Always non-negative. Equals **0 at the target**, **1 at the start**. Entered as a *negative* weighted penalty.
    """)

    dr1, dr2 = st.columns(2)
    with dr1:
        # Centre the x-axis on pos_y_star so both arms of the V are always visible.
        # Half-span = max(distance from target to start, target to plot edge) × 1.3
        half_span_y = max(abs(pos_y_ini_v - pos_y_star_v), 50) * 1.3
        py_range    = np.linspace(pos_y_star_v - half_span_y,
                                   pos_y_star_v + half_span_y, 400)
        norm_y      = max(abs(pos_y_ini_v - pos_y_star_v), 1e-6)
        dy_rew      = np.abs((py_range - pos_y_star_v) / norm_y)

        fig = go.Figure()
        # Shaded area under the V
        fig.add_trace(go.Scatter(
            x=py_range, y=dy_rew, mode="lines",
            fill="tozeroy", fillcolor="rgba(232,87,42,0.08)",
            line=dict(color=C_ORANGE, width=2.5), name="distance_y_reward"
        ))
        # Minimum point at target (reward = 0)
        fig.add_trace(go.Scatter(
            x=[pos_y_star_v], y=[0], mode="markers",
            marker=dict(color=C_TEAL, size=12, symbol="circle",
                        line=dict(color="white", width=2)),
            name="Target (reward=0)", showlegend=True
        ))
        # Start point (reward = 1 by definition)
        start_reward_y = abs(pos_y_ini_v - pos_y_star_v) / norm_y
        fig.add_trace(go.Scatter(
            x=[pos_y_ini_v], y=[start_reward_y], mode="markers",
            marker=dict(color=C_GRAY, size=12, symbol="diamond",
                        line=dict(color="white", width=2)),
            name=f"Start (reward={start_reward_y:.2f})", showlegend=True
        ))
        fig.add_vline(x=pos_y_star_v, line_dash="dash", line_color=C_TEAL,
                       annotation_text=f"Target Y={pos_y_star_v:.0f}",
                       annotation_position="top right",
                       annotation_font=dict(color=C_TEAL))
        fig.add_vline(x=pos_y_ini_v, line_dash="dot", line_color=C_GRAY,
                       annotation_text=f"Start Y={pos_y_ini_v:.0f}",
                       annotation_position="top left",
                       annotation_font=dict(color=C_GRAY))
        fig.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8",
                       annotation_text="reward=1 (start level)",
                       annotation_position="bottom right",
                       annotation_font=dict(color="#94a3b8", size=11))
        fig.update_layout(
            title="Distance Y reward — V-shape centred on target",
            xaxis_title="pos_y (altitude)",
            yaxis_title="distance_y_reward",
            height=320, legend=dict(orientation="h", y=-0.25),
            **light_layout(yaxis=dict(rangemode="tozero", showgrid=True,
                                      gridcolor=C_BORDER, zeroline=True,
                                      zerolinecolor="#94a3b8"))
        )
        st.plotly_chart(fig, use_container_width=True)

    with dr2:
        half_span_x = max(abs(pos_x_ini_v - pos_x_star_v), 30) * 1.3
        px_range    = np.linspace(pos_x_star_v - half_span_x,
                                   pos_x_star_v + half_span_x, 400)
        norm_x      = max(abs(pos_x_ini_v - pos_x_star_v), 1e-6)
        dx_rew      = np.abs((px_range - pos_x_star_v) / norm_x)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=px_range, y=dx_rew, mode="lines",
            fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
            line=dict(color=C_BLUE, width=2.5), name="distance_x_reward"
        ))
        fig.add_trace(go.Scatter(
            x=[pos_x_star_v], y=[0], mode="markers",
            marker=dict(color=C_TEAL, size=12, symbol="circle",
                        line=dict(color="white", width=2)),
            name="Target (reward=0)", showlegend=True
        ))
        start_reward_x = abs(pos_x_ini_v - pos_x_star_v) / norm_x
        fig.add_trace(go.Scatter(
            x=[pos_x_ini_v], y=[start_reward_x], mode="markers",
            marker=dict(color=C_GRAY, size=12, symbol="diamond",
                        line=dict(color="white", width=2)),
            name=f"Start (reward={start_reward_x:.2f})", showlegend=True
        ))
        fig.add_vline(x=pos_x_star_v, line_dash="dash", line_color=C_TEAL,
                       annotation_text=f"Target X={pos_x_star_v:.0f}",
                       annotation_position="top right",
                       annotation_font=dict(color=C_TEAL))
        fig.add_vline(x=pos_x_ini_v, line_dash="dot", line_color=C_GRAY,
                       annotation_text=f"Start X={pos_x_ini_v:.0f}",
                       annotation_position="top left",
                       annotation_font=dict(color=C_GRAY))
        fig.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8",
                       annotation_text="reward=1 (start level)",
                       annotation_position="bottom right",
                       annotation_font=dict(color="#94a3b8", size=11))
        fig.update_layout(
            title="Distance X reward — V-shape centred on target",
            xaxis_title="pos_x (horizontal)",
            yaxis_title="distance_x_reward",
            height=320, legend=dict(orientation="h", y=-0.25),
            **light_layout(yaxis=dict(rangemode="tozero", showgrid=True,
                                      gridcolor=C_BORDER, zeroline=True,
                                      zerolinecolor="#94a3b8"))
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-title">2 · Speed Reward Component</div>', unsafe_allow_html=True)
    st.markdown("""
`speed_reward = exp(1) − exp( max( |speed| / speed_limit, 1 ) )`

Returns **0** when speed is within the limit, drops sharply negative beyond it.
    """)

    sp1, sp2 = st.columns(2)
    speed_rng = np.linspace(0, 40, 400)
    with sp1:
        rew_sy = np.exp(1) - np.exp(np.maximum(speed_rng / speed_lim_y, 1))
        fig    = go.Figure()
        fig.add_trace(go.Scatter(x=speed_rng, y=rew_sy, mode="lines",
                                  line=dict(color=C_ORANGE, width=2.5), name="speed_y_reward"))
        fig.add_vline(x=speed_lim_y, line_dash="dash", line_color=C_TEAL,
                       annotation_text=f"limit={speed_lim_y}")
        fig.add_hline(y=0, line_dash="dot", line_color=C_GRAY)
        fig.update_layout(title="Speed Y reward vs |speed_y|",
                           xaxis_title="|speed_y|", yaxis_title="reward",
                           height=280, **light_layout())
        st.plotly_chart(fig, use_container_width=True)

    with sp2:
        rew_sx = np.exp(1) - np.exp(np.maximum(speed_rng / speed_lim_x, 1))
        fig    = go.Figure()
        fig.add_trace(go.Scatter(x=speed_rng, y=rew_sx, mode="lines",
                                  line=dict(color=C_BLUE, width=2.5), name="speed_x_reward"))
        fig.add_vline(x=speed_lim_x, line_dash="dash", line_color=C_TEAL,
                       annotation_text=f"limit={speed_lim_x}")
        fig.add_hline(y=0, line_dash="dot", line_color=C_GRAY)
        fig.update_layout(title="Speed X reward vs |speed_x|",
                           xaxis_title="|speed_x|", yaxis_title="reward",
                           height=280, **light_layout())
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-title">3 · Angle Penalty (Alpha Agent)</div>', unsafe_allow_html=True)
    st.markdown("`angle_penalty = w_angle · sin(|angle|)` — penalises angular deviation from vertical.")
    ang_range   = np.linspace(-0.8, 0.8, 300)
    ang_penalty = w_ang_v * np.sin(np.abs(ang_range))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ang_range, y=ang_penalty, mode="lines",
                              line=dict(color=C_ORANGE, width=2.5)))
    fig.add_vline(x=0, line_dash="dot", line_color=C_GRAY)
    fig.update_layout(title=f"Angle penalty (w={w_ang_v}) vs angle",
                       xaxis_title="angle (rad)", yaxis_title="penalty",
                       height=250, **light_layout())
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-title">4 · Fuel Ratio Reward</div>', unsafe_allow_html=True)
    st.markdown("`ratio_fuel = m_fuel / m_fuel_ini` — linear bonus encouraging fuel efficiency.")
    fuel_range = np.linspace(0, m_fuel_ini_v, 300)
    fuel_ratio = fuel_range / m_fuel_ini_v
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fuel_range, y=w_fb_v * fuel_ratio, mode="lines",
                              line=dict(color=C_ORANGE, width=2.5), name=f"booster (w={w_fb_v})"))
    fig.add_trace(go.Scatter(x=fuel_range, y=w_fa_v * fuel_ratio, mode="lines",
                              line=dict(color=C_BLUE, width=2.5, dash="dash"), name=f"alpha (w={w_fa_v})"))
    fig.update_layout(title="Fuel reward vs remaining fuel",
                       xaxis_title="m_fuel", yaxis_title="reward",
                       height=270, legend=dict(orientation="h"), **light_layout())
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-title">5 · Combined Reward Landscape (Heatmap)</div>', unsafe_allow_html=True)
    st.markdown("""
Total reward `R_booster + R_alpha` at every (pos_x, pos_y) cell, assuming
**zero speed, full fuel, angle = 0**. Both distance penalties are active simultaneously,
so the landscape varies in **both** X and Y directions — the minimum (darkest red) is
the combined optimum closest to the landing target ☆.
    """)

    # ── Build combined reward grid ────────────────────────────────────────────
    px_g = np.linspace(50, 200, 100)
    py_g = np.linspace(0,  250, 100)
    PX, PY = np.meshgrid(px_g, py_g)   # shape (100, 100)

    norm_y = max(abs(pos_y_ini_v - pos_y_star_v), 1e-3)
    norm_x = max(abs(pos_x_ini_v - pos_x_star_v), 1e-3)

    dy_map = np.abs((PY - pos_y_star_v) / norm_y)
    dx_map = np.abs((PX - pos_x_star_v) / norm_x)

    speed_reward_at_zero = 0.0  # exp(1) - exp(max(0/limit, 1)) = 0

    booster_map = (w_dy_v * (-dy_map) + w_sy_v * speed_reward_at_zero + w_fb_v * 1.0)
    alpha_map   = (w_dx_v * (-dx_map) + w_sx_v * speed_reward_at_zero + w_fa_v * 1.0)

    # Combined: sum of both agent rewards
    combined_map = -(booster_map + alpha_map)

    fig_hm = go.Figure()

    # ── Heatmap layer ─────────────────────────────────────────────────────────
    fig_hm.add_trace(go.Heatmap(
        x=px_g, y=py_g, z=combined_map,
        colorscale="RdYlGn",
        zmin=float(combined_map.min()),
        zmax=float(combined_map.max()),
        colorbar=dict(
            title=dict(text="R_booster + R_alpha", font=dict(color="#1e293b")),
            tickfont=dict(color="#1e293b"),
            thickness=16, len=0.8,
        ),
        hovertemplate=(
            "pos_x=%{x:.0f}  pos_y=%{y:.0f}<br>"
            "combined reward=%{z:.3f}<extra></extra>"
        ),
    ))

    # ── Iso-reward contour lines ──────────────────────────────────────────────
    fig_hm.add_trace(go.Contour(
        x=px_g, y=py_g, z=combined_map,
        showscale=False,
        contours=dict(
            coloring="none",
            showlabels=True,
            labelfont=dict(size=10, color="#374151"),
        ),
        line=dict(color="rgba(30,30,30,0.35)", width=1),
        hoverinfo="skip",
    ))

    # ── Target ☆ ─────────────────────────────────────────────────────────────
    fig_hm.add_trace(go.Scatter(
        x=[pos_x_star_v], y=[pos_y_star_v], mode="markers+text",
        marker=dict(color="white", size=16, symbol="star",
                    line=dict(color="#1e293b", width=1.5)),
        text=["Target"], textposition="top center",
        textfont=dict(color="#1e293b", size=11),
        name="Target",
        hovertemplate=f"Target  pos_x={pos_x_star_v:.0f}, pos_y={pos_y_star_v:.0f}<extra></extra>",
    ))

    # ── Start ● ──────────────────────────────────────────────────────────────
    fig_hm.add_trace(go.Scatter(
        x=[pos_x_ini_v], y=[pos_y_ini_v], mode="markers+text",
        marker=dict(color="#3b82f6", size=13, symbol="circle",
                    line=dict(color="white", width=1.5)),
        text=["Start"], textposition="top center",
        textfont=dict(color="#1e293b", size=11),
        name="Start",
        hovertemplate=f"Start  pos_x={pos_x_ini_v:.0f}, pos_y={pos_y_ini_v:.0f}<extra></extra>",
    ))

    fig_hm.update_layout(
        title=dict(
            text="Combined reward landscape  <i>(R_booster + R_alpha, speed=0, fuel=max, angle=0)</i>",
            font=dict(size=13, color="#1e293b"),
        ),
        xaxis=dict(title="pos_x (horizontal)", showgrid=False, range=[px_g[0], px_g[-1]]),
        yaxis=dict(title="pos_y (altitude)",    showgrid=False, range=[py_g[0], py_g[-1]]),
        height=480,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#1e293b"),
        margin=dict(l=55, r=20, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:

    # ── Algorithm selector ────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">Algorithm</div>', unsafe_allow_html=True)
    algo_col1, algo_col2 = st.columns([1, 2])
    with algo_col1:
        algo_choice = st.radio(
            "Choose training algorithm",
            options=["🗂️ Tabular Q-Learning", "🧠 Deep Q-Network (DQN)"],
            key="algo_radio",
            horizontal=False,
        )
        is_dqn = algo_choice.startswith("🧠")
        if is_dqn != (st.session_state.algo_mode == "dqn"):
            st.session_state.algo_mode     = "dqn" if is_dqn else "tabular"
            st.session_state.training_done = False

    with algo_col2:
        if is_dqn:
            st.markdown("""
<div class="info-card">
<b>Deep Q-Network (DQN)</b> replaces the Q-table with a neural network that approximates
<code>Q(s, a)</code> for all joint actions simultaneously. Key advantages:<br><br>
• Works on <b>continuous state vectors</b> — no discretisation or bin explosion<br>
• <b>Experience replay</b> — past transitions are stored and randomly re-sampled,
  breaking temporal correlations that destabilise tabular updates<br>
• <b>Target network</b> — a frozen copy of the network used to compute stable
  Bellman targets; synced every N gradient steps<br>
• <b>Double DQN</b> — the online network selects the best next action while the
  target network evaluates it, reducing Q-value over-estimation
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="info-card">
<b>Tabular Q-Learning</b> maintains an explicit table of Q-values indexed by
discretised state tuples. Simple and interpretable, but the table grows
with the number of bins and state variables, and cannot generalise between
states. Best for small, low-dimensional problems.
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Shared hyperparameters ────────────────────────────────────────────────
    st.markdown('<div class="sec-title">Shared Hyperparameters</div>', unsafe_allow_html=True)
    sh1, sh2, sh3 = st.columns(3)
    with sh1:
        num_episodes = st.number_input("Number of episodes", min_value=1, max_value=50000,
                                        value=50, step=10, key="n_ep")
        run_limit    = st.number_input("Run limit (steps/episode)", min_value=10, max_value=50000,
                                        value=300, step=50, key="rl")
    with sh2:
        gamma      = st.slider("Discount factor (γ)", 0.5, 1.0, 0.99, 0.01, key="gam")
        decay_type = st.selectbox("Epsilon decay type", ["exponential", "linear"], key="decay")
    with sh3:
        decrease_prob = st.number_input("Epsilon decay rate", value=0.005, min_value=0.0001,
                                         max_value=1.0, format="%.4f", key="dp")
        conv_crit     = st.number_input("Convergence criterion", value=0.1, min_value=0.001,
                                         format="%.4f", key="cc")

    # ── Algorithm-specific hyperparameters ────────────────────────────────────
    if is_dqn:
        st.markdown('<div class="sec-title">DQN-Specific Hyperparameters</div>', unsafe_allow_html=True)
        dq1, dq2, dq3 = st.columns(3)
        with dq1:
            dqn_lr         = st.number_input("Learning rate (Adam)", value=0.001, min_value=1e-5,
                                              max_value=0.1, format="%.5f", key="dqn_lr")
            batch_size     = st.number_input("Batch size", min_value=8, max_value=512,
                                              value=64, step=8, key="bs")
        with dq2:
            buffer_cap     = st.number_input("Replay buffer capacity", min_value=100,
                                              max_value=100_000, value=10_000, step=500, key="buf")
            target_update  = st.number_input("Target network sync (steps)", min_value=1,
                                              max_value=1000, value=50, step=10, key="tu")
        with dq3:
            h1 = st.number_input("Hidden layer 1 width", min_value=16, max_value=512,
                                   value=128, step=16, key="h1")
            h2 = st.number_input("Hidden layer 2 width", min_value=0, max_value=512,
                                   value=128, step=16, key="h2",
                                   help="Set to 0 to use only one hidden layer")
            hidden_sizes = [int(h1)] + ([int(h2)] if h2 > 0 else [])
            grad_steps = st.number_input("Gradient updates per step", min_value=1, max_value=16,
                                          value=4, step=1, key="grad_steps",
                                          help="How many gradient updates to run per environment step. Higher = more sample-efficient but slower per step.")

        st.markdown(f"""
<div class="formula-box">
Network: input({len(st.session_state.config["states_variables"]) +
               len(st.session_state.config["agent_variables"])})
→ ReLU({h1}) {"→ ReLU(" + str(h2) + ") " if h2 > 0 else ""}
→ Linear({len(list(__import__("itertools").product(*[list(st.session_state.config["n_action"][k].keys())
          for k in st.session_state.config["n_action"]])))} joint actions)
</div>""", unsafe_allow_html=True)

    else:
        lr = st.slider("Learning rate (α)", 0.01, 1.0, 0.1, 0.01, key="lr")

    st.divider()

    # ── Launch ────────────────────────────────────────────────────────────────
    if st.button("🚀 Start Training", type="primary", key="train_btn"):
        progress_bar = st.progress(0, text="Initialising environment…")
        log_area     = st.empty()
        try:
            env = Environment(st.session_state.config)

            if is_dqn:
                # ── DQN path ─────────────────────────────────────────────────
                trainer = DQNTrainer(
                    env=env,
                    num_episodes=int(num_episodes),
                    hidden_sizes=hidden_sizes,
                    lr=float(dqn_lr),
                    discount_factor=float(gamma),
                    batch_size=int(batch_size),
                    buffer_capacity=int(buffer_cap),
                    target_update=int(target_update),
                    exploration_prob=[0.05, 1.0],
                    decrease_prob_exp=float(decrease_prob),
                    decay_type=decay_type,
                    run_limit=int(run_limit),
                    convergence_criterion=float(conv_crit),
                )
                trainer.gradient_steps = int(grad_steps)
                epsilons = trainer.get_epsilon()
                logs = []

                for episode in range(int(num_episodes)):
                    pct = int((episode + 1) / int(num_episodes) * 100)
                    progress_bar.progress(
                        pct,
                        text=f"[DQN] Episode {episode+1}/{int(num_episodes)} "
                             f"| ε={epsilons[episode]:.3f} "
                             f"| buffer={len(trainer.replay)}"
                    )
                    steps = trainer.training_dqn(epsilons[episode], episode)

                    ep_df = pd.concat([
                        pd.DataFrame(trainer.env.all_states()),
                        pd.DataFrame(trainer.env.rewards).add_prefix("reward_"),
                    ], axis=1)
                    trainer.states_for_all_episodes.append(ep_df)
                    trainer.q_table_for_all_episodes.append(None)
                    trainer.loss_episodes.append(trainer.loss_train)
                    trainer.monitor_iter.append(steps)

                    ls = f"{trainer.loss_train:.4f}" if (
                        trainer.loss_train is not None and not np.isnan(trainer.loss_train)
                    ) else "—"
                    logs.append(
                        f"Ep {episode+1:>4} | ε={epsilons[episode]:.3f} "
                        f"| steps={steps} | mse_loss={ls} "
                        f"| buf={len(trainer.replay)}"
                    )
                    log_area.code("\n".join(logs[-20:]))

                    if len(trainer.loss_episodes) > 7:
                        recent = [l for l in trainer.loss_episodes[-6:]
                                  if l is not None and not np.isnan(l)]
                        if len(recent) == 6 and all(
                            abs(a - b) < float(conv_crit) for a, b in zip(recent, recent[1:])
                        ):
                            logs.append("→ Early stopping: loss converged.")
                            log_area.code("\n".join(logs[-20:]))
                            break

            else:
                # ── Tabular Q-Learning path ───────────────────────────────────
                trainer = QLearningTrainer(
                    env=env,
                    num_episodes=int(num_episodes),
                    learning_rate=float(lr),
                    discount_factor=float(gamma),
                    run_limit=int(run_limit),
                    decrease_prob_exp=float(decrease_prob),
                    convergence_criterion=float(conv_crit),
                    decay_type=decay_type,
                )
                epsilons = trainer.get_epsilon()
                logs = []

                for episode in range(int(num_episodes)):
                    pct = int((episode + 1) / int(num_episodes) * 100)
                    progress_bar.progress(
                        pct,
                        text=f"[Q-Table] Episode {episode+1}/{int(num_episodes)} "
                             f"| ε={epsilons[episode]:.3f}"
                    )
                    iters = trainer.training_q_learning(epsilons[episode], episode)
                    trainer.states_for_all_episodes.append(
                        pd.concat([pd.DataFrame(trainer.env.all_states()),
                                   pd.DataFrame(trainer.env.rewards).add_prefix("reward_")], axis=1)
                    )
                    trainer.q_table_for_all_episodes.append(trainer.q_table.copy())
                    trainer.loss_episodes.append(trainer.loss_train)
                    trainer.monitor_iter.append(iters)
                    ls = f"{trainer.loss_train:.4f}" if trainer.loss_train is not None else "—"
                    logs.append(
                        f"Ep {episode+1:>4} | ε={epsilons[episode]:.3f} "
                        f"| steps={iters} | q_loss={ls} "
                        f"| states={len(trainer.q_table)}"
                    )
                    log_area.code("\n".join(logs[-20:]))

                    if len(trainer.loss_episodes) > 7:
                        recent = [l for l in trainer.loss_episodes[-6:] if l is not None]
                        if len(recent) == 6 and all(np.abs(np.diff(recent)) < float(conv_crit)):
                            logs.append("→ Early stopping: Q-table converged.")
                            log_area.code("\n".join(logs[-20:]))
                            break

            progress_bar.progress(100, text="Training complete!")
            st.session_state.trainer       = trainer
            st.session_state.training_done = True
            st.session_state.algo_mode     = "dqn" if is_dqn else "tabular"
            st.session_state.selected_episode = len(trainer.states_for_all_episodes) - 1
            st.success(
                f"{'DQN' if is_dqn else 'Q-Learning'} training complete — "
                f"{len(trainer.states_for_all_episodes)} episodes run."
            )
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not st.session_state.training_done or st.session_state.trainer is None:
        st.info("Run training first — go to the **Training** tab.")
    else:
        trainer = st.session_state.trainer
        cfg     = st.session_state.config
        n_ep    = len(trainer.states_for_all_episodes)

        st.markdown('<div class="sec-title">Episode Explorer</div>', unsafe_allow_html=True)
        ep_idx  = st.slider("Select episode to inspect", 0, n_ep - 1,
                             st.session_state.selected_episode, key="ep_slider")
        st.session_state.selected_episode = ep_idx
        ep_data = trainer.states_for_all_episodes[ep_idx]
        final   = ep_data.iloc[-1]

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Steps", len(ep_data))
        mc2.metric("Final pos_x", f"{final['pos_x']:.1f}")
        mc3.metric("Final pos_y", f"{final['pos_y']:.1f}")
        mc4.metric("Fuel remaining", f"{final.get('m_fuel', 0):.0f}")

        st.divider()

        # ── Trajectory ────────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">🗺️ Rocket Trajectory</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_shape(type="rect", x0=50, x1=200, y0=-8, y1=0,
                       fillcolor="#d4a373", line_width=0)
        fig.add_shape(type="line", x0=50, x1=200, y0=0, y1=0,
                       line=dict(color="#78716c", width=1.5))
        stop_x = cfg["stop_episode"]["pos_x"]
        fig.add_vrect(x0=stop_x[0], x1=stop_x[1], fillcolor="rgba(34,197,94,0.18)",
                       line_width=1, line_color="#16a34a",
                       annotation_text="Landing Zone", annotation_position="top left",
                       annotation_font_size=11)
        if "pos_x_obstacle" in cfg["initial_values"]:
            ox = cfg["initial_values"]["pos_x_obstacle"][0]
            oy = cfg["initial_values"]["pos_y_obstacle"][0]
            r  = cfg["initial_values"].get("obstacle_radius", [5])[0]
            th = np.linspace(0, 2 * np.pi, 60)
            fig.add_trace(go.Scatter(
                x=ox + r * np.cos(th), y=oy + r * np.sin(th),
                fill="toself", fillcolor="rgba(239,68,68,0.25)",
                line=dict(color="#dc2626"), name="Obstacle"
            ))
        n_steps = len(ep_data)
        fig.add_trace(go.Scatter(
            x=ep_data["pos_x"], y=ep_data["pos_y"], mode="lines+markers",
            marker=dict(color=list(range(n_steps)), colorscale="Plasma", size=5,
                        colorbar=dict(
                            title=dict(text="Step", font=dict(color="#1e293b")),
                            tickfont=dict(color="#1e293b"),
                            thickness=12, len=0.5,
                        )),
            line=dict(width=2, color="rgba(100,100,200,0.4)"),
            name="Trajectory",
            hovertemplate="Step %{marker.color}<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(x=[ep_data["pos_x"].iloc[0]], y=[ep_data["pos_y"].iloc[0]],
                                  mode="markers",
                                  marker=dict(color=C_TEAL, size=13, symbol="circle"),
                                  name="Start"))
        fig.add_trace(go.Scatter(x=[ep_data["pos_x"].iloc[-1]], y=[ep_data["pos_y"].iloc[-1]],
                                  mode="markers",
                                  marker=dict(color=C_ORANGE, size=13, symbol="star"),
                                  name="End"))
        fig.add_trace(go.Scatter(
            x=[cfg["initial_values"]["pos_x_star"][0]],
            y=[cfg["initial_values"]["pos_y_star"][0]],
            mode="markers",
            marker=dict(color="#16a34a", size=13, symbol="x"),
            name="Target"
        ))
        fig.update_layout(
            height=420, paper_bgcolor="white", plot_bgcolor="white",
            font=dict(color="#1e293b"),
            xaxis=dict(title="X Position", range=[45, 210],
                        showgrid=True, gridcolor=C_BORDER),
            yaxis=dict(title="Altitude (Y)", range=[-12, 260],
                        showgrid=True, gridcolor=C_BORDER),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                         bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=50, r=20, t=30, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── State variables ───────────────────────────────────────────────────
        st.markdown('<div class="sec-title">📈 State Variables Over Time</div>', unsafe_allow_html=True)
        sv_opts   = [c for c in ep_data.columns if not c.startswith("reward_")]
        sv_def    = [v for v in ["pos_x", "pos_y", "speed_x", "speed_y", "angle", "m_fuel"] if v in sv_opts]
        sel_vars  = st.multiselect("Variables to plot", sv_opts, default=sv_def, key="sv_select")
        sv_colors = [C_ORANGE, C_BLUE, C_TEAL, "#7c3aed", "#db2777", "#ca8a04"]

        if sel_vars:
            ncols = min(3, len(sel_vars))
            nrows = (len(sel_vars) + ncols - 1) // ncols
            fig_sv = make_subplots(rows=nrows, cols=ncols,
                                    subplot_titles=sel_vars,
                                    vertical_spacing=0.12, horizontal_spacing=0.08)
            for i, var in enumerate(sel_vars):
                r, c = divmod(i, ncols)
                fig_sv.add_trace(
                    go.Scatter(y=ep_data[var], mode="lines", name=var, showlegend=False,
                               line=dict(color=sv_colors[i % len(sv_colors)], width=2),
                               hovertemplate=f"{var}: %{{y:.3f}}<extra></extra>"),
                    row=r+1, col=c+1
                )
            fig_sv.update_layout(
                height=270 * nrows, paper_bgcolor="white", plot_bgcolor="white",
                font=dict(color="#1e293b"), margin=dict(l=50, r=20, t=50, b=40)
            )
            fig_sv.update_xaxes(showgrid=True, gridcolor=C_BORDER)
            fig_sv.update_yaxes(showgrid=True, gridcolor=C_BORDER)
            st.plotly_chart(fig_sv, use_container_width=True)

        # ── Rewards ───────────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">🏆 Rewards Over Time</div>', unsafe_allow_html=True)
        rew_cols = [c for c in ep_data.columns if c.startswith("reward_")]
        if rew_cols:
            fig_r = make_subplots(rows=1, cols=len(rew_cols),
                                   subplot_titles=[c.replace("reward_", "Agent: ") for c in rew_cols],
                                   horizontal_spacing=0.1)
            pal = [C_ORANGE, C_BLUE]
            fill_pal = ["rgba(232,87,42,0.12)", "rgba(37,99,235,0.12)"]
            for i, rc in enumerate(rew_cols):
                fig_r.add_trace(
                    go.Scatter(y=ep_data[rc], mode="lines", name=rc,
                               line=dict(color=pal[i % 2], width=2),
                               fill="tozeroy", fillcolor=fill_pal[i % 2]),
                    row=1, col=i+1
                )
            fig_r.update_layout(
                height=300, paper_bgcolor="white", plot_bgcolor="white",
                font=dict(color="#1e293b"), margin=dict(l=50, r=20, t=50, b=40)
            )
            fig_r.update_xaxes(showgrid=True, gridcolor=C_BORDER)
            fig_r.update_yaxes(showgrid=True, gridcolor=C_BORDER,
                                zeroline=True, zerolinecolor="#94a3b8")
            st.plotly_chart(fig_r, use_container_width=True)

        st.divider()

        # ── Convergence ───────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">📉 Training Convergence</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            valid = [(i, l) for i, l in enumerate(trainer.loss_episodes) if l is not None]
            if valid:
                ix, lx = zip(*valid)
                fig_l = go.Figure(go.Scatter(
                    x=list(ix), y=list(lx), mode="lines+markers",
                    line=dict(color=C_ORANGE, width=2),
                    marker=dict(size=4, color=C_ORANGE)
                ))
                fig_l.update_layout(title="Loss per Episode",
                                     xaxis_title="Episode", yaxis_title="Loss",
                                     height=300, **light_layout())
                st.plotly_chart(fig_l, use_container_width=True)

        with cc2:
            if trainer.monitor_iter:
                fig_s = go.Figure(go.Bar(
                    x=list(range(len(trainer.monitor_iter))),
                    y=trainer.monitor_iter,
                    marker_color=C_BLUE
                ))
                fig_s.update_layout(title="Steps per Episode",
                                     xaxis_title="Episode", yaxis_title="Steps",
                                     height=300, **light_layout())
                st.plotly_chart(fig_s, use_container_width=True)

        # ── Cumulative reward ─────────────────────────────────────────────────
        st.markdown('<div class="sec-title">🎯 Cumulative Reward per Episode</div>', unsafe_allow_html=True)
        cum = {}
        for ep_d in trainer.states_for_all_episodes:
            for rc in [c for c in ep_d.columns if c.startswith("reward_")]:
                cum.setdefault(rc.replace("reward_", ""), []).append(ep_d[rc].sum())
        if cum:
            fig_c = go.Figure()
            for i, (agent, vals) in enumerate(cum.items()):
                fig_c.add_trace(go.Scatter(
                    x=list(range(len(vals))), y=vals, mode="lines",
                    name=f"Agent: {agent}",
                    line=dict(color=[C_ORANGE, C_BLUE][i % 2], width=2)
                ))
            fig_c.update_layout(title="Cumulative Reward per Episode",
                                  xaxis_title="Episode", yaxis_title="Cumulative Reward",
                                  height=280, legend=dict(orientation="h"),
                                  **light_layout())
            st.plotly_chart(fig_c, use_container_width=True)

        with st.expander("📋 Raw episode data"):
            st.dataframe(ep_data.round(4), use_container_width=True, height=300)

        # ── DQN-only: Q-value landscape & network summary ─────────────────────
        if st.session_state.algo_mode == "dqn" and hasattr(trainer, "online_net"):
            st.divider()
            st.markdown('<div class="sec-title">🧠 DQN Diagnostics</div>', unsafe_allow_html=True)

            dq_col1, dq_col2 = st.columns(2)

            with dq_col1:
                st.markdown("**Network architecture**")
                arch_rows = trainer.get_network_summary()
                st.dataframe(pd.DataFrame(arch_rows), use_container_width=True, hide_index=True)
                total_params = sum(r["Parameters"] for r in arch_rows)
                st.caption(f"Total trainable parameters: **{total_params:,}**")
                st.caption(f"Device: **{str(trainer.device).upper()}**")

            with dq_col2:
                st.markdown("**Training stats**")
                buf = trainer.replay
                st.metric("Buffer",          f"{len(buf):,} / {buf.buffer.maxlen:,}")
                st.metric("Gradient steps",  f"{trainer._grad_steps:,}")
                st.metric("Target syncs",    f"{trainer._grad_steps // trainer.target_update:,}")

            # Q-value heatmap — vectorised using get_q_values() + torch.no_grad
            st.markdown(
                "**Max Q-value landscape** — scanned across (pos_x, pos_y); "
                "all other state variables fixed at last episode's final values."
            )
            last_state = ep_data.iloc[-1]
            px_g = np.linspace(50, 200, 40)
            py_g = np.linspace(0, 250, 40)
            max_q_grid = np.zeros((len(py_g), len(px_g)))

            for ri, py_v in enumerate(py_g):
                for ci, px_v in enumerate(px_g):
                    vec = np.array([
                        float(last_state.get(k, 0.0)) if k not in ("pos_x", "pos_y")
                        else (px_v if k == "pos_x" else py_v)
                        for k in trainer.state_keys
                    ], dtype=np.float32)
                    max_q_grid[ri, ci] = float(np.max(trainer.get_q_values(vec)))

            fig_qv = go.Figure(go.Heatmap(
                x=px_g, y=py_g, z=max_q_grid,
                colorscale="RdYlGn",
                colorbar=dict(
                    title=dict(text="max Q", font=dict(color="#1e293b")),
                    tickfont=dict(color="#1e293b"),
                ),
                hovertemplate="pos_x=%{x:.0f}, pos_y=%{y:.0f}<br>max Q=%{z:.3f}<extra></extra>"
            ))
            stop_x = cfg["stop_episode"]["pos_x"]
            fig_qv.add_vrect(x0=stop_x[0], x1=stop_x[1],
                              fillcolor="rgba(34,197,94,0.2)", line_width=1,
                              line_color="#16a34a",
                              annotation_text="Landing", annotation_font_size=10)
            fig_qv.update_layout(
                title="Max Q-value landscape (other variables fixed at episode end)",
                xaxis=dict(title="pos_x", showgrid=False),
                yaxis=dict(title="pos_y", showgrid=False),
                height=380, paper_bgcolor="white", plot_bgcolor="white",
                font=dict(color="#1e293b"), margin=dict(l=50, r=20, t=45, b=45)
            )
            st.plotly_chart(fig_qv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5 — JSON EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_json:
    st.markdown('<div class="sec-title">Current Configuration</div>', unsafe_allow_html=True)
    st.markdown("The JSON below reflects exactly what was (or will be) used for training.")
    json_str = json.dumps(st.session_state.config, indent=2)
    st.code(json_str, language="json")
    st.download_button("⬇️ Download config.json", data=json_str,
                        file_name="rocket_config.json", mime="application/json",
                        key="dl_json")

    st.divider()
    st.markdown('<div class="sec-title">Load Custom JSON</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a .json config file", type=["json"], key="json_upload")
    if uploaded is not None:
        try:
            loaded = json.load(uploaded)
            st.session_state.config        = loaded
            st.session_state.training_done = False
            st.success("Config loaded! Head to Training to run.")
            st.json(loaded)
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")

    st.divider()
    st.markdown('<div class="sec-title">Presets</div>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    with pc1:
        if st.button("Load default (rocket_tuto_3)", key="load_tuto3"):
            st.session_state.config        = copy.deepcopy(DEFAULT_CONFIG)
            st.session_state.training_done = False
            st.success("Loaded default config.")
    with pc2:
        if st.button("Reset all", key="reset_cfg"):
            st.session_state.config        = copy.deepcopy(DEFAULT_CONFIG)
            st.session_state.training_done = False
            st.success("Reset to default.")