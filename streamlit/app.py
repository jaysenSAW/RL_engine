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
sys.path.insert(1, "../src/utils/")
from agent import Environment
from Q_learning import QLearningTrainer
sys.path.insert(1, "plotly_graph/")
st.set_page_config(page_title="Rocket RL Tutorial", layout="wide", page_icon="🚀")

# ─── Inject source path ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #FF6B35; margin-bottom: 0; }
    .section-title { font-size: 1.2rem; font-weight: 600; color: #FF6B35; margin-top: 1rem; }
    .metric-card { background: #1e2533; border-radius: 8px; padding: 12px 16px; margin: 4px 0; border-left: 3px solid #FF6B35; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background: #1e2533; border-radius: 6px 6px 0 0; color: #ccc; }
    .stTabs [aria-selected="true"] { background: #FF6B35 !important; color: white !important; }
    div[data-testid="stExpander"] { border: 1px solid #333; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Default JSON config (rocket_tuto_3) ─────────────────────────────────────
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

# ─── Session state init ───────────────────────────────────────────────────────
if "config" not in st.session_state:
    st.session_state.config = copy.deepcopy(DEFAULT_CONFIG)
if "training_done" not in st.session_state:
    st.session_state.training_done = False
if "trainer" not in st.session_state:
    st.session_state.trainer = None
if "selected_episode" not in st.session_state:
    st.session_state.selected_episode = 0

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚀 Rocket RL Tutorial</div>', unsafe_allow_html=True)
st.markdown("Configure your rocket environment, train a Q-learning agent, and visualize the results.")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Environment Config", "🎯 Training", "📊 Results", "📄 JSON Export"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Environment Config
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Rocket & Mission Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cfg = st.session_state.config

    with col1:
        st.markdown("**Initial Position**")
        pos_x = st.number_input("pos_x (initial)", value=float(cfg["initial_values"]["pos_x"][0]), key="pos_x")
        pos_y = st.number_input("pos_y (initial)", value=float(cfg["initial_values"]["pos_y"][0]), key="pos_y")
        angle = st.number_input("angle (initial)", value=float(cfg["initial_values"]["angle"][0]), key="angle_init")

    with col2:
        st.markdown("**Target (Landing Zone)**")
        pos_x_star = st.number_input("pos_x_star (target X)", value=float(cfg["initial_values"]["pos_x_star"][0]), key="px_star")
        pos_y_star = st.number_input("pos_y_star (target Y)", value=float(cfg["initial_values"]["pos_y_star"][0]), key="py_star")
        st.markdown("**Stop Zone (X range)**")
        stop_x_min = st.number_input("stop pos_x min", value=float(cfg["stop_episode"]["pos_x"][0]), key="stop_xmin")
        stop_x_max = st.number_input("stop pos_x max", value=float(cfg["stop_episode"]["pos_x"][1]), key="stop_xmax")

    with col3:
        st.markdown("**Fuel & Rocket**")
        m_fuel = st.number_input("m_fuel (initial)", value=float(cfg["initial_values"]["m_fuel"][0]), key="mfuel")
        dt = st.number_input("dt (timestep)", value=float(cfg["initial_values"]["dt"][0]), min_value=0.1, max_value=20.0, key="dt_val")
        st.markdown("**Gravity**")
        G = st.number_input("G (gravity)", value=float(cfg["initial_values"]["G"][0]), key="grav")

    st.divider()
    st.markdown('<div class="section-title">Reward Weights</div>', unsafe_allow_html=True)
    st.markdown("The reward is a weighted sum of components. Adjust the multipliers below.")

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown("**Booster reward** (controls vertical landing)")
        w_dist_y = st.slider("Distance Y weight (neg)", -5.0, 0.0, -2.0, 0.1, key="w_dy")
        w_speed_y = st.slider("Speed Y reward weight", 0.0, 3.0, 1.0, 0.1, key="w_sy")
        w_fuel_b = st.slider("Fuel ratio weight (booster)", 0.0, 2.0, 0.5, 0.1, key="w_fb")

    with rcol2:
        st.markdown("**Alpha reward** (controls horizontal / angle)")
        w_dist_x = st.slider("Distance X weight (neg)", -5.0, 0.0, -2.0, 0.1, key="w_dx")
        w_speed_x = st.slider("Speed X reward weight", 0.0, 3.0, 1.0, 0.1, key="w_sx")
        w_angle = st.slider("Angle penalty weight", -1.0, 0.0, -0.2, 0.05, key="w_ang")
        w_fuel_a = st.slider("Fuel ratio weight (alpha)", 0.0, 2.0, 0.5, 0.1, key="w_fa")

    st.divider()
    st.markdown('<div class="section-title">Speed & Acceleration Limits</div>', unsafe_allow_html=True)
    lcol1, lcol2 = st.columns(2)
    with lcol1:
        sl_x = st.number_input("speed_limit_x", value=float(cfg["initial_values"]["speed_limit_x"][0]), key="slx")
        sl_y = st.number_input("speed_limit_y", value=float(cfg["initial_values"]["speed_limit_y"][0]), key="sly")
    with lcol2:
        al_x = st.number_input("acceleration_limit_x", value=float(cfg["initial_values"]["acceleration_limit_x"][0]), key="alx")
        al_y = st.number_input("acceleration_limit_y", value=float(cfg["initial_values"]["acceleration_limit_y"][0]), key="aly")

    st.divider()
    st.markdown('<div class="section-title">Obstacle (optional)</div>', unsafe_allow_html=True)
    use_obstacle = st.toggle("Enable obstacle avoidance", value=False, key="use_obs")
    if use_obstacle:
        ocol1, ocol2, ocol3 = st.columns(3)
        with ocol1:
            obs_x = st.number_input("Obstacle X", value=130.0, key="obs_x")
            obs_y = st.number_input("Obstacle Y", value=90.0, key="obs_y")
        with ocol2:
            obs_r = st.number_input("Obstacle radius", value=5.0, key="obs_r")
            excl = st.number_input("Exclusion zone", value=2.0, key="obs_excl")
        with ocol3:
            st.info("An obstacle penalty is added to both agent rewards based on proximity.")

    # Build updated config
    if st.button("✅ Apply Configuration", type="primary", key="apply_cfg"):
        new_cfg = copy.deepcopy(DEFAULT_CONFIG)
        # Initial values
        new_cfg["initial_values"].update({
            "pos_x": [pos_x], "pos_y": [pos_y], "angle": [angle],
            "m_fuel": [m_fuel], "m_fuel_ini": [float(m_fuel)],
            "pos_x_ini": [pos_x], "pos_y_ini": [pos_y],
            "pos_x_star": [pos_x_star], "pos_y_star": [pos_y_star],
            "futur_pos_x": [pos_x], "futur_pos_y": [pos_y],
            "dt": [dt], "G": [G],
            "speed_limit_x": [sl_x], "speed_limit_y": [sl_y],
            "acceleration_limit_x": [al_x], "acceleration_limit_y": [al_y],
            "weight_rocket": [5 + m_fuel],
        })
        # Stop episode
        new_cfg["stop_episode"]["pos_x"] = [stop_x_min, stop_x_max]

        # Reward equations
        b_reward = f"{w_dist_y}*(-distance_y_reward) + {w_speed_y}*speed_y_reward + {w_fuel_b} * $ratio_fuel$"
        a_reward = f"{w_dist_x}*(-distance_x_reward) + {w_speed_x}*speed_x_reward + {w_angle}*np.sin(np.abs($angle$)) + {w_fuel_a} * $ratio_fuel$"
        new_cfg["equations_rewards"]["$booster$"] = b_reward
        new_cfg["equations_rewards"]["$alpha$"] = a_reward

        # Obstacle
        if use_obstacle:
            new_cfg["initial_values"].update({
                "pos_x_obstacle": [obs_x], "pos_y_obstacle": [obs_y],
                "obstacle_radius": [obs_r], "exclusion_zone": [excl],
                "obstacle_penalty": [0.0], "distance2obstacle_squarred": [(obs_x - pos_x)**2 + (obs_y - pos_y)**2]
            })
            new_cfg["equations_variables"]["$computed_distance2obstacle_squarred$"] = "($pos_y$ - $pos_y_obstacle$)**2 + ($pos_x$ - $pos_x_obstacle$)**2"
            new_cfg["equations_variables"]["$obstacle_penalty$"] = "np.minimum($computed_distance2obstacle_squarred$ -($obstacle_radius$ + $exclusion_zone$)**2, 0)"
            new_cfg["equations_rewards"]["$computed_distance2obstacle_squarred$"] = "($pos_y$ - $pos_y_obstacle$)**2 + ($pos_x$ - $pos_x_obstacle$)**2"
            new_cfg["equations_rewards"]["$booster$"] = b_reward + " + $obstacle_penalty$"
            new_cfg["equations_rewards"]["$alpha$"] = a_reward + " + $obstacle_penalty$"

        st.session_state.config = new_cfg
        st.session_state.training_done = False
        st.success("Configuration applied! Go to the Training tab to run Q-learning.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Training
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Q-Learning Hyperparameters</div>', unsafe_allow_html=True)

    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        num_episodes = st.number_input("Number of episodes", min_value=1, max_value=5000, value=50, step=10, key="n_ep")
        run_limit = st.number_input("Run limit (steps/episode)", min_value=10, max_value=5000, value=300, step=50, key="rl")
    with hcol2:
        lr = st.slider("Learning rate (α)", 0.01, 1.0, 0.1, 0.01, key="lr")
        gamma = st.slider("Discount factor (γ)", 0.5, 1.0, 0.99, 0.01, key="gam")
    with hcol3:
        decay_type = st.selectbox("Epsilon decay type", ["exponential", "linear"], key="decay")
        decrease_prob = st.number_input("Decay rate", value=0.005, min_value=0.0001, max_value=1.0, format="%.4f", key="dp")
        conv_crit = st.number_input("Convergence criterion", value=0.05, min_value=0.001, format="%.4f", key="cc")

    st.divider()

    if st.button("🚀 Start Training", type="primary", key="train_btn"):
        from agent import Environment
        from Q_learning import QLearningTrainer

        progress_bar = st.progress(0, text="Initializing environment...")
        log_area = st.empty()

        try:
            env = Environment(st.session_state.config)
            trainer = QLearningTrainer(
                env=env,
                num_episodes=int(num_episodes),
                learning_rate=float(lr),
                discount_factor=float(gamma),
                run_limit=int(run_limit),
                decrease_prob_exp=float(decrease_prob),
                convergence_criterion=float(conv_crit),
                decay_type=decay_type
            )

            logs = []
            proba_schedule = trainer.get_epsilon()

            for episode in range(int(num_episodes)):
                progress_pct = int((episode + 1) / int(num_episodes) * 100)
                progress_bar.progress(progress_pct, text=f"Episode {episode+1}/{int(num_episodes)} | ε={proba_schedule[episode]:.3f}")
                iters = trainer.training_q_learning(proba_schedule[episode], episode)
                trainer.states_for_all_episodes.append(
                    pd.concat([
                        pd.DataFrame(trainer.env.all_states()),
                        pd.DataFrame(trainer.env.rewards).add_prefix('reward_')
                    ], axis=1)
                )
                trainer.q_table_for_all_episodes.append(trainer.q_table.copy())
                trainer.loss_episodes.append(trainer.loss_train)
                trainer.monitor_iter.append(iters)
                logs.append(f"Ep {episode+1:>4} | ε={proba_schedule[episode]:.3f} | steps={iters} | loss={trainer.loss_train:.4f}" if trainer.loss_train is not None else f"Ep {episode+1:>4} | ε={proba_schedule[episode]:.3f} | steps={iters}")
                log_area.code("\n".join(logs[-15:]))

                # early stop
                if len(trainer.loss_episodes) > 7:
                    recent = [l for l in trainer.loss_episodes[-6:] if l is not None]
                    if len(recent) == 6:
                        diffs = np.abs(np.array(recent[:-1]) - np.array(recent[1:]))
                        if all(diffs < float(conv_crit)):
                            logs.append("→ Early stopping: converged.")
                            log_area.code("\n".join(logs[-15:]))
                            break

            progress_bar.progress(100, text="Training complete!")
            st.session_state.trainer = trainer
            st.session_state.training_done = True
            st.session_state.selected_episode = len(trainer.states_for_all_episodes) - 1
            st.success(f"Training complete! {len(trainer.states_for_all_episodes)} episodes run.")
        except Exception as e:
            st.error(f"Error during training: {e}")
            import traceback; st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Results
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if not st.session_state.training_done or st.session_state.trainer is None:
        st.info("Run training first (go to the Training tab).")
    else:
        trainer = st.session_state.trainer
        cfg = st.session_state.config
        n_ep_done = len(trainer.states_for_all_episodes)

        # Episode selector
        st.markdown('<div class="section-title">Episode Explorer</div>', unsafe_allow_html=True)
        ep_idx = st.slider("Select episode", 0, n_ep_done - 1, st.session_state.selected_episode, key="ep_slider")
        st.session_state.selected_episode = ep_idx
        ep_data = trainer.states_for_all_episodes[ep_idx]

        # ── Summary metrics ───────────────────────────────────────────────────
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        final = ep_data.iloc[-1]
        mcol1.metric("Steps", len(ep_data))
        mcol2.metric("Final pos_x", f"{final['pos_x']:.1f}")
        mcol3.metric("Final pos_y", f"{final['pos_y']:.1f}")
        mcol4.metric("Fuel remaining", f"{final.get('m_fuel', 0):.0f}" if 'fuel' not in final else "—")

        st.divider()

        # ── Trajectory plot ───────────────────────────────────────────────────
        st.markdown('<div class="section-title">🗺️ Rocket Trajectory</div>', unsafe_allow_html=True)
        fig_traj = go.Figure()

        # Ground
        fig_traj.add_shape(type="line", x0=50, x1=200, y0=0, y1=0,
                           line=dict(color="#888", width=2, dash="dot"))
        # Target zone
        stop_x = cfg["stop_episode"]["pos_x"]
        fig_traj.add_vrect(x0=stop_x[0], x1=stop_x[1], fillcolor="rgba(0,255,100,0.15)",
                            line_width=1, line_color="green",
                            annotation_text="Landing Zone", annotation_position="top left")

        # Obstacle
        if "pos_x_obstacle" in cfg["initial_values"]:
            ox = cfg["initial_values"]["pos_x_obstacle"][0]
            oy = cfg["initial_values"]["pos_y_obstacle"][0]
            r = cfg["initial_values"].get("obstacle_radius", [5])[0]
            theta_obs = np.linspace(0, 2 * np.pi, 60)
            fig_traj.add_trace(go.Scatter(
                x=ox + r * np.cos(theta_obs), y=oy + r * np.sin(theta_obs),
                fill="toself", fillcolor="rgba(255,80,80,0.3)",
                line=dict(color="red"), name="Obstacle", showlegend=True
            ))

        # Trajectory colored by episode step
        n_steps = len(ep_data)
        colors = [f"hsl({int(200 + 160 * i / max(n_steps-1, 1))},80%,55%)" for i in range(n_steps)]
        fig_traj.add_trace(go.Scatter(
            x=ep_data["pos_x"], y=ep_data["pos_y"],
            mode="lines+markers",
            marker=dict(color=list(range(n_steps)), colorscale="Plasma", size=6,
                        colorbar=dict(title="Step", thickness=12, len=0.5)),
            line=dict(width=2, color="rgba(200,200,255,0.5)"),
            name="Trajectory",
            hovertemplate="Step %{marker.color}<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>"
        ))
        # Start & End
        fig_traj.add_trace(go.Scatter(x=[ep_data["pos_x"].iloc[0]], y=[ep_data["pos_y"].iloc[0]],
                                       mode="markers", marker=dict(color="cyan", size=14, symbol="circle"),
                                       name="Start"))
        fig_traj.add_trace(go.Scatter(x=[ep_data["pos_x"].iloc[-1]], y=[ep_data["pos_y"].iloc[-1]],
                                       mode="markers", marker=dict(color="orange", size=14, symbol="star"),
                                       name="End"))
        # Target point
        fig_traj.add_trace(go.Scatter(
            x=[cfg["initial_values"]["pos_x_star"][0]],
            y=[cfg["initial_values"]["pos_y_star"][0]],
            mode="markers", marker=dict(color="lime", size=14, symbol="x"),
            name="Target"
        ))
        fig_traj.update_layout(
            template="plotly_dark", height=420,
            xaxis=dict(title="X Position", range=[45, 210]),
            yaxis=dict(title="Altitude (Y)", range=[-10, 260]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=30, b=40)
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        # ── State variables over time ─────────────────────────────────────────
        st.markdown('<div class="section-title">📈 State Variables Over Time</div>', unsafe_allow_html=True)
        state_vars = [c for c in ep_data.columns if not c.startswith("reward_")]
        selected_vars = st.multiselect(
            "Select state variables to plot",
            options=state_vars,
            default=[v for v in ["pos_x", "pos_y", "speed_x", "speed_y", "angle", "m_fuel"] if v in state_vars],
            key="sv_select"
        )

        if selected_vars:
            n_plots = len(selected_vars)
            ncols = min(3, n_plots)
            nrows = (n_plots + ncols - 1) // ncols
            fig_states = make_subplots(
                rows=nrows, cols=ncols,
                subplot_titles=selected_vars,
                vertical_spacing=0.12, horizontal_spacing=0.08
            )
            for i, var in enumerate(selected_vars):
                r, c = divmod(i, ncols)
                fig_states.add_trace(
                    go.Scatter(y=ep_data[var], mode="lines", name=var,
                               line=dict(width=2), showlegend=False,
                               hovertemplate=f"{var}: %{{y:.3f}}<extra></extra>"),
                    row=r+1, col=c+1
                )
            fig_states.update_layout(
                template="plotly_dark", height=280 * nrows,
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_states, use_container_width=True)

        # ── Rewards over time ─────────────────────────────────────────────────
        st.markdown('<div class="section-title">🏆 Rewards Over Time</div>', unsafe_allow_html=True)
        reward_cols = [c for c in ep_data.columns if c.startswith("reward_")]

        if reward_cols:
            fig_rew = make_subplots(rows=1, cols=len(reward_cols),
                                     subplot_titles=[c.replace("reward_", "Agent: ") for c in reward_cols],
                                     horizontal_spacing=0.1)
            palette = ["#FF6B35", "#4CC9F0"]
            for i, rc in enumerate(reward_cols):
                fig_rew.add_trace(
                    go.Scatter(y=ep_data[rc], mode="lines", name=rc,
                               line=dict(color=palette[i % len(palette)], width=2),
                               fill="tozeroy", fillcolor="rgba(255,107,53,0.15)" if i == 0 else "rgba(76,201,240,0.15)"),
                    row=1, col=i+1
                )
            fig_rew.update_layout(
                template="plotly_dark", height=320,
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_rew, use_container_width=True)

        st.divider()

        # ── Training convergence ───────────────────────────────────────────────
        st.markdown('<div class="section-title">📉 Training Convergence</div>', unsafe_allow_html=True)
        conv_col1, conv_col2 = st.columns(2)

        with conv_col1:
            valid_losses = [(i, l) for i, l in enumerate(trainer.loss_episodes) if l is not None]
            if valid_losses:
                idxs, losses = zip(*valid_losses)
                fig_loss = go.Figure(go.Scatter(
                    x=list(idxs), y=list(losses), mode="lines+markers",
                    line=dict(color="#FF6B35", width=2), marker=dict(size=4),
                    name="Episode loss"
                ))
                fig_loss.update_layout(
                    template="plotly_dark", title="Loss per Episode",
                    xaxis_title="Episode", yaxis_title="Loss",
                    height=320, margin=dict(l=40, r=20, t=50, b=40)
                )
                st.plotly_chart(fig_loss, use_container_width=True)

        with conv_col2:
            if trainer.monitor_iter:
                fig_steps = go.Figure(go.Bar(
                    x=list(range(len(trainer.monitor_iter))),
                    y=trainer.monitor_iter,
                    marker_color="#4CC9F0", name="Steps per episode"
                ))
                fig_steps.update_layout(
                    template="plotly_dark", title="Steps per Episode",
                    xaxis_title="Episode", yaxis_title="Steps",
                    height=320, margin=dict(l=40, r=20, t=50, b=40)
                )
                st.plotly_chart(fig_steps, use_container_width=True)

        # ── Cumulative reward across episodes ─────────────────────────────────
        st.markdown('<div class="section-title">🎯 Cumulative Reward per Episode</div>', unsafe_allow_html=True)
        cum_rewards = {}
        for ep_d in trainer.states_for_all_episodes:
            for rc in [c for c in ep_d.columns if c.startswith("reward_")]:
                agent = rc.replace("reward_", "")
                cum_rewards.setdefault(agent, []).append(ep_d[rc].sum())

        if cum_rewards:
            fig_cum = go.Figure()
            for i, (agent, vals) in enumerate(cum_rewards.items()):
                fig_cum.add_trace(go.Scatter(
                    x=list(range(len(vals))), y=vals,
                    mode="lines", name=f"Agent: {agent}",
                    line=dict(color=["#FF6B35", "#4CC9F0"][i % 2], width=2)
                ))
            fig_cum.update_layout(
                template="plotly_dark", title="Cumulative Reward per Episode",
                xaxis_title="Episode", yaxis_title="Cumulative Reward",
                height=300, margin=dict(l=40, r=20, t=50, b=40),
                legend=dict(orientation="h")
            )
            st.plotly_chart(fig_cum, use_container_width=True)

        # ── Raw data table ─────────────────────────────────────────────────────
        with st.expander("📋 View raw episode data"):
            st.dataframe(ep_data.round(4), use_container_width=True, height=300)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — JSON Export
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Current Configuration JSON</div>', unsafe_allow_html=True)
    st.markdown("This is the live JSON config that will be used for training. You can copy it or download it.")

    json_str = json.dumps(st.session_state.config, indent=2)
    st.code(json_str, language="json")
    st.download_button(
        label="⬇️ Download config.json",
        data=json_str,
        file_name="rocket_config.json",
        mime="application/json",
        key="dl_json"
    )

    st.divider()
    st.markdown('<div class="section-title">Load Custom JSON</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a JSON config file", type=["json"], key="json_upload")
    if uploaded is not None:
        try:
            loaded = json.load(uploaded)
            st.session_state.config = loaded
            st.session_state.training_done = False
            st.success("Config loaded successfully! Go to Training tab to run.")
            st.json(loaded)
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")

    st.divider()
    st.markdown('<div class="section-title">Preset Configs</div>', unsafe_allow_html=True)
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        if st.button("Load rocket_tuto_3 (default)", key="load_tuto3"):
            st.session_state.config = copy.deepcopy(DEFAULT_CONFIG)
            st.session_state.training_done = False
            st.success("Loaded rocket_tuto_3 config.")
    with pcol2:
        if st.button("Reset to default", key="reset_cfg"):
            st.session_state.config = copy.deepcopy(DEFAULT_CONFIG)
            st.session_state.training_done = False
            st.success("Config reset to default.")