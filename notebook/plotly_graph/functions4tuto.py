import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
sys.path.insert(1, "../src/utils/")
from agent import Environment

def plotly_trajectory(df_traj):
    fig = make_subplots(rows=3, cols=1)
    fig1 = px.line(df_traj.set_index('iter')[['pos_y', 'futur_pos_y']] )
    fig2 = px.line(df_traj.set_index('iter')[['acceleration_y', 'speed_y']].rename(
        columns = {'acceleration_y' : 'acceleration_y (y/time²)',
                'speed_y': 'speed_y (y/time)'
                }))
    fig3 = px.line(df_traj.set_index('iter')[['m_fuel']] )


    for d in fig1.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  name = d['name'])), row=1, col=1)

    for d in fig2.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  name = d['name'])), row=2, col=1)

    for d in fig3.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  name = d['name'])), row=3, col=1)

    # Update xaxis properties
    for i in range(32):
        fig.update_xaxes(title_text="time", row=i+1, col=1)
    fig.update_yaxes(title_text="Height (y)", row=1, col=1)
    fig.update_yaxes(title_text="Unit", row=2, col=1)
    fig.update_yaxes(title_text="fuel mass", row=3, col=1)
    fig.update_layout(height=600, width=600, title_text="Rocket first attempt")
    fig.show()

def plotly_all_reward(dt, df_penalty):
    fig = make_subplots(rows=3, cols=2)

    fig1 = px.line(dt.set_index('iter')[['pos_y', 'futur_pos_y']] )
    fig2 = px.line(df_penalty.set_index('iter')[['dist_star', 'y_lim_constraint']] )

    fig3 = px.line(dt.set_index('iter')[['futur_pos_y', 'acceleration_y', 'speed_y']] )
    fig4 = px.line(df_penalty.set_index('iter')[['acceleration_constraint', 'speed_constraint']] )

    fig5 = px.line(dt.set_index('iter')[['weight_rocket']] )
    fig6 = px.line(df_penalty.set_index('iter')[['ratio_fuel']] )

    for d in fig1.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  name = d['name'])), row=1, col=1)

    for d in fig2.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  name = d['name'])), row=2, col=1)

    for d in fig3.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  name = d['name'])), row=1, col=2)

    for d in fig4.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  name = d['name'])), row=2, col=2)

    for d in fig5.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  name = d['name'])), row=3, col=1)

    for d in fig6.data:
        fig.add_trace((go.Scatter(x=d['x'], y=d['y'],  name = d['name'])), row=3, col=2)

    # Update xaxis properties
    for i in range(3):
        fig.update_xaxes(title_text="time", row=i+1, col=1)
    for i in range(3):
        fig.update_xaxes(title_text="time", row=i+1, col=2)
    fig.update_yaxes(title_text="Height (y)", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration and speed", row=1, col=2)
    fig.update_yaxes(title_text="Distance penality", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration and speed penalities", row=2, col=2)
    fig.update_yaxes(title_text="Mass fuel", row=3, col=1)
    fig.update_yaxes(title_text="Reward fuel", row=3, col=2)
    fig.update_layout(height=1000, width=1200, title_text="Design reward with simple simulation")
    fig.show()

def booster_reward(states, acceleration_y_constraint, speed_y_limit, y_lower_limit, y_upper_limit = 200):
    dist_star = np.abs((states["pos_y"] - states["pos_y_star"])/(states["pos_y_ini"] - states["pos_y_star"]) )
    acceleration_y_constraint =  np.array([ np.exp(1) - np.exp( np.max([val, 1]) ) for val in np.abs(states["acceleration_y"])/acceleration_y_constraint ])
    speed_y_constraint =np.array([ np.exp(1) - np.exp( np.max([val, 1]) ) for val in np.abs(states["speed_y"])/speed_y_limit ])
    y_lim_constraint = np.array([-2 + np.exp(np.min([val1, 0])) + np.exp(np.min([val2, 0])) for val1, val2 in zip(states["futur_pos_y"] - y_lower_limit, - states["futur_pos_y"] + y_upper_limit) ])
    dt = pd.DataFrame({
        "dist_star" : -dist_star,
        "acceleration_constraint" : acceleration_y_constraint,
        "speed_constraint" : speed_y_constraint,
        "y_lim_constraint" : y_lim_constraint,
        "ratio_fuel" : states["m_fuel"]/states["m_fuel_ini"]
    })
    # dt["sum_penalty"] = dt.sum(1)
    dt["iter"] =  np.arange(0, dt.shape[0])
    return dt

def control_fall_simulation(JOSN_file, 
                            acceleration_y_constraint = 10, 
                            speed_y_limit = 4, 
                            y_lower_limit = 0) -> [pd.DataFrame, pd.DataFrame]:
    """
    Simulate controled rocket, by using simplist rule
    turn on/turn off engine when speed or acceleration is out of limit
    """
    # Create an environment object with the rules defined previously
    env = Environment(JOSN_file, check_model = False)
    flag = "0"
    flag_to_continue = True
    # monitor action takes for each iteration
    actions = {"action_booster" : []} 
    while flag_to_continue:    
        env.step([flag, 1]) 
        actions["action_booster"].append(flag)
        if env.futur_pos_y[-1] < 0 and env.m_fuel[-1] > 0:
            flag = "1"
        # stop engine if there is no fuel
        elif env.m_fuel[-1] <= 0:
            flag = "0"
        elif np.abs(env.speed_y[-1]) > speed_y_limit:
            # print("speed limit")
            if env.speed_y[-1] > 0:
                flag = "0"
            else:
                flag = "1"
        elif np.abs(env.acceleration_y[-1]) > acceleration_y_constraint:
            # print("acceleration limit")
            if env.acceleration_y[-1] > 0:
                flag = "0"
            else:
                flag = "1" 
        # stop simulation
        if env.pos_y[-1] < 0:
            flag_to_continue = False
            # delete last state because rocket is bellow to the ground
            env.delete_last_states()

    # save all frame as data Frame
    dt = pd.DataFrame(env.all_states())
    dt["iter"] = np.arange(0, len(env.all_states()["pos_y"]))
    df_penalty = booster_reward(env.all_states(), acceleration_y_constraint, speed_y_limit, y_lower_limit)
    df_penalty["ratio_fuel"] = dt["m_fuel"]/dt["m_fuel_ini"]
    return dt, df_penalty

def rocket_simulation(env_ref, acceleration_y_constraint = 10, speed_y_limit = 5, timestep = 0):
    flag_booster = "0"
    flag_to_continue = True
    # monitor action takes for each iteration
    actions = {"action_booster" : []} 
    while flag_to_continue:    
        current_state, rewards, done, problem, info = env_ref.step([flag_booster])
        actions["action_booster"].append(flag_booster)
        # stop simulation
        if env_ref.pos_y[-1] < 0:
            flag_to_continue = False
            print("orcket is bellow the ground")
            # delete last state because rocket is bellow to the ground
            # env_ref.delete_last_states()
            continue
        elif env_ref.pos_y[-1] + timestep * env_ref.speed_y[-1] < 0 and env_ref.m_fuel[-1] > 0:
            # to avoid futur crash, turn on rocket's engine
            flag_booster = "1"
            continue
        # stop engine if there is no fuel
        elif env_ref.m_fuel[-1] <= 0:
            flag_booster = "0"
            continue
        elif np.abs(env_ref.speed_y[-1]) > speed_y_limit and env_ref.m_fuel[-1] > 0:
            print("speed limit")
            if env_ref.speed_y[-1] > 0:
                print("turn off engine to reduce speed")
                flag_booster = "0"
            else:
                print("turn on engine to increase speed")
                flag_booster = "1"
        elif np.abs(env_ref.acceleration_y[-1]) > acceleration_y_constraint and env_ref.m_fuel[-1] > 0:
            if env_ref.acceleration_y[-1] > 0:
                print("turn off engine to reduce acceleration")
                flag_booster = "0"
            else:
                print("turn on engine to increase acceleration")
                flag_booster = "1" 
        elif any(done) and info[0] == "Reach goal":
            print({val : current_state[val] for val in ['pos_y', 'acceleration_y', 'speed_y', 'booster']})
            flag_to_continue = False
            continue
    tmp = pd.DataFrame(env_ref.all_states())
    tmp["futur_position_dt+{0}".format(timestep)] = env_ref.pos_y + timestep * env_ref.speed_y
    return tmp

def plot_rocket_altitude(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.time, y=df.pos_y,
                        mode='lines+markers',
                        name='Height'))
    for col in df.filter(like='futur_position_dt').columns:
        fig.add_trace(go.Scatter(x=df.time, y=df[col],
                            mode='lines+markers',
                            name=col))
    # Edit the layout
    fig.update_layout(
            title=dict(
                text='Rocket\'s altitude'
            ),
            xaxis=dict(
                title=dict(
                    text='Time step'
                )
            ),
            yaxis=dict(
                title=dict(
                    text='Altitude'
                )
            ),
    )
    fig.show()

def plot_reward_rocket_monoagent(env):
    """Plot reward component

    Args:
        env (environment): system 
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=env.all_states()["time"], y=env.all_states()["distance_y_reward"],
                        mode='lines+markers',
                        name='distance_y_reward'))
    fig.add_trace(go.Scatter(x=env.all_states()["time"], y=env.all_states()["speed_y_reward"],
                        mode='lines+markers',
                        name='speed_y_reward'))
    fig.add_trace(go.Scatter(x=env.all_states()["time"], y=env.all_states()["ratio_fuel"],
                        mode='lines+markers',
                        name='ratio_fuel'))
    fig.add_trace(go.Scatter(x=env.all_states()["time"], y=env.rewards['booster'],
                        mode='lines+markers',
                        name='Sum of rewards'))
    # Edit the layout
    fig.update_layout(
            title=dict(
                text='Reward and its components'
            ),
            xaxis=dict(
                title=dict(
                    text='Time step'
                )
            ),
            yaxis=dict(
                title=dict(
                    text='Score'
                )
            ),
    )
    fig.show()