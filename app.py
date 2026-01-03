import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from PIL import Image
import base64
from volleyball import Volleyball
from target_window import HittingWindow
import pandas as pd

def get_encoded_image():
    # Load and encode the volleyball court image
    img_path = 'Volleyball_Shoot_Processed.png'
    with open(img_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f'data:image/png;base64,{encoded_string}'

# Initialize the Dash app
app = dash.Dash(__name__)

# Create the layout
app.layout = html.Div([
    html.H1("Volleyball Trajectory Simulator", 
            style={'textAlign': 'center', 'color': '#000000', 'marginBottom': 30}),
    
    html.Div([
        html.Div([
            html.Label("Start X (m)", style={'fontSize': '12px'}),
            dcc.Input(
                id='start-x',
                type='number',
                value=0,
                step=0.1,
                style={'marginBottom': '6px', 'width': '100%', 'fontSize': '12px', 'padding': '2px'}
            ),

            html.Label("Start Y (m)", style={'fontSize': '12px'}),
            dcc.Input(
                id='start-y',
                type='number',
                value=0,
                step=0.1,
                style={'marginBottom': '6px', 'width': '100%', 'fontSize': '12px', 'padding': '2px'}
            ),

            html.Label("Target X (m)", style={'fontSize': '12px'}),
            dcc.Input(
                id='target-x',
                type='number',
                value=6.5,
                step=0.1,
                style={'marginBottom': '6px', 'width': '100%', 'fontSize': '12px', 'padding': '2px'}
            ),

            html.Label("Target Y (m)", style={'fontSize': '12px'}),
            dcc.Input(
                id='target-y',
                type='number',
                value=0.3,
                step=0.1,
                style={'marginBottom': '6px', 'width': '100%', 'fontSize': '12px', 'padding': '2px'}
            ),

            html.Label("Air Time (seconds)", style={'fontSize': '12px'}),
            dcc.Input(
                id='time-input',
                type='number',
                value=1.6,
                step=0.1,
                style={'marginBottom': '6px', 'width': '100%', 'fontSize': '12px', 'padding': '2px'}
            ),

            html.Button('Update Trajectory', 
                id='update-button', 
                n_clicks=0,
                style={
                    'marginTop': 10,
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'border': 'none',
                    'padding': '6px 10px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '12px'
                }),

            html.H3("Trajectory Statistics", style={'marginTop': 20}),
            html.Div(id='trajectory-stats', style={'marginTop': 10, 'fontSize': '12px'}),
        ], style={
            'width': '15%',
            'padding': 8,
            'backgroundColor': '#ffffff',
            'borderRadius': '10px',
            'minWidth': '120px',
            'maxWidth': '180px'
        }),
        
        html.Div([
            dcc.Graph(id='trajectory-plot', style={'height': '80vh'}),
        ], style={'width': '80%'}),
        
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
], style={'padding': 20, 'fontFamily': 'Arial'})

@app.callback(
    [Output('trajectory-plot', 'figure'),
     Output('trajectory-stats', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('start-x', 'value'),
     State('start-y', 'value'),
     State('target-x', 'value'),
     State('target-y', 'value'),
     State('time-input', 'value')]
)
def update_trajectory(n_clicks, x_start, y_start, x_target, y_target, time_max):
    optimizer = Volleyball(drag_coefficient = 0.48)
    result = optimizer.find_objective([10,40], x_start, y_start, x_target, y_target, time_max)
    initial_velocity, angle = result.x[0], result.x[1]
    t, solution = optimizer.simulate(initial_velocity, angle, np.array([x_start, y_start]), time_max + 0.1)
    
    hw = HittingWindow(np.array([x_target, y_target]))

    df_sol = pd.DataFrame.from_dict(dict(zip(["x", "y", "dx", "dy"], solution.T)), orient='columns')
    df_sol["time"] = t

    df_sol_window = df_sol[df_sol.apply(hw.in_hitting_window, axis=1)]  # windshield wiper shape

    if len(df_sol_window) > 0:
        max_t = df_sol_window.loc[df_sol_window['time'].idxmax()]
        min_t = df_sol_window.loc[df_sol_window['time'].idxmin()]
        
        if max_t is not None and min_t is not None:
            print(max_t)
            print(min_t)
            time_in_window = max_t['time'] - min_t['time']
            print(f"Time in window: {time_in_window:.4f}s")
        else:
            print("No valid window found")


    # Create main trajectory plot
    fig = go.Figure()
    
    # Add background image

    
    # Add trajectory line
    fig.add_trace(go.Scatter(
        x=solution[:, 0],
        y=solution[:, 1],
        mode='lines',
        name='Trajectory',
        line=dict(color='red', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title='Volleyball Trajectory',
        xaxis_title='Distance (m)',
        yaxis_title='Height (m)',
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='rgba(255,255,255,0.9)',
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            gridcolor='lightgray',
            range=[-5.5, 2.5],  # Adjust height range to match image
            showgrid=False,
        ),
        xaxis=dict(
            gridcolor='lightgray',
            range=[8, -3],   # <-- Set x-axis range here
            showgrid=False,
        ),
    )

    fig.add_layout_image(
        dict(
            source=get_encoded_image(),
            xref="x",
            yref="y",
            x=9.22,      # Start at x=0
            y=2.5,      # Adjust this value to match your image height
            sizex=-14.22, # Width of the court in meters
            sizey=8,  # Height of the image in meters
            sizing="contain",
            opacity=1,
            layer="below")
    )

    hw.create_hitting_window_figure(fig)
    
    # # Calculate statistics
    # max_height = np.max(solution[:, 1])
    # max_distance = np.max(solution[:, 0])
    # impact_idx = np.where(solution[:, 1] <= 0)[0]
    # impact_time = t[impact_idx[0]] if len(impact_idx) > 0 else time_max
    # impact_distance = solution[impact_idx[0], 0] if len(impact_idx) > 0 else max_distance

    # Final velocity at last point
    vx_final = solution[-1, 2]
    vy_final = solution[-1, 3]
    v_final = np.sqrt(vx_final**2 + vy_final**2)

    stats = html.Div([
        # html.P(f"Maximum Height: {max_height:.2f} m"),
        # html.P(f"Maximum Distance: {max_distance:.2f} m"),
        html.P(f"Instantaneous Velocity: {v_final:.2f} m/s"),
        html.P(f"Final vx: {vx_final:.2f} m/s"),
        html.P(f"Final vy: {vy_final:.2f} m/s"),
        html.P(f"Time in window: {time_in_window:.4f}s"),
        
    ])
    
    return fig, stats

if __name__ == '__main__':
    # app.run(debug=False, host='0.0.0.0', port=8050)
    app.run(debug=True, host='0.0.0.0', port=8050)
