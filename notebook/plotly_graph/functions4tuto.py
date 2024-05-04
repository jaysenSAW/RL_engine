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
    fig2 = px.line(df_penalty.set_index('iter')[['futur_dist_star', 'y_lim_constraint']] )

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

def booster_reward(states, acceleration_y_constraint, speed_y_limit, y_lower_limit):
    dist_squared = np.square(states["pos_y"] - states["pos_y_star"])
    acceleration_y_constraint =  -np.array([np.max([val, 0]) for val in np.abs(states["acceleration_y"]) - acceleration_y_constraint ])
    speed_y_constraint =-np.array([np.max([val, 0]) for val in np.abs(states["speed_y"]) - speed_y_limit ])
    y_lim_constraint = np.array([np.min([val, 0]) for val in states["futur_pos_y"] - y_lower_limit ])
    # return -dist_squared + acceleration_y_constraint + speed_y_constraint +y_lim_constraint + states["m_fuel"]/states["m_fuel_ini"]
    dt = pd.DataFrame({
        "futur_dist_star" : -dist_squared,
        "acceleration_constraint" : acceleration_y_constraint,
        "speed_constraint" : speed_y_constraint,
        "y_lim_constraint" : y_lim_constraint,
        "ratio_fuel" : states["m_fuel"]/states["m_fuel_ini"]
    })
    dt["sum_penalty"] = dt.sum(1)
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

