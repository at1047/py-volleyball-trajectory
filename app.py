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

class ProjectileOptimizer:
    def __init__(self):
        self.g = 9.81
        self.mass = 0.27
        self.Cd = 0.48
        self.rho = 1.225
        self.A = 0.0346
        
    # def equations_of_motion(self, state, t):
    #     x, y, vx, vy = state
    #     v = np.sqrt(vx**2 + vy**2)
    #     Fd = -0.5 * self.Cd * self.rho * self.A * v
        
    #     ax = (Fd * vx / self.mass) if v > 0 else 0
    #     ay = (Fd * vy / self.mass) if v > 0 else 0
    #     ay -= self.g
        
    #     return [vx, vy, ax, ay]
    
    def drag_force(self, velocity):
        """Calculate drag force using F = 1/2 * ρ * v² * Cd * A"""
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude == 0:
            return np.zeros_like(velocity)
        
        drag_magnitude = (0.5 * self.rho * velocity_magnitude**2 * 
                         self.Cd * self.A)
        
        # Drag force acts in opposite direction of velocity
        return -drag_magnitude * velocity / velocity_magnitude

    def derivatives(self, state, t):
        """Calculate derivatives for position and velocity"""
        # state = [x, y, vx, vy]
        vx, vy = state[2], state[3]
        velocity = np.array([vx, vy])
        
        # Calculate drag force
        drag_force = self.drag_force(velocity)
        
        # Acceleration components (F = ma)
        ax = drag_force[0] / self.mass
        ay = drag_force[1] / self.mass - self.g
        
        return [vx, vy, ax, ay]

    # def simulate(self, initial_velocity, angle_degrees, time_span, dt=0.01):
    #     """
    #     Simulate projectile motion
        
    #     Parameters:
    #     - initial_velocity: speed in m/s
    #     - angle_degrees: launch angle in degrees
    #     - time_span: simulation duration in seconds
    #     - dt: time step in seconds
    #     """
    #     # Convert angle to radians
    #     angle = np.deg2rad(angle_degrees)
        
    #     # Initial conditions [x, y, vx, vy]
    #     initial_state = [0, 0,
    #                     initial_velocity * np.cos(angle),
    #                     initial_velocity * np.sin(angle)]
        
    #     # Time points
    #     t = np.arange(0, time_span, dt)
        
    #     # Solve ODE
    #     solution = odeint(self.derivatives, initial_state, t)
        
    #     return t, solution
    
    def objective(self, params, x_start, y_start, x_target, y_target, time_span, dt=0.01):
        initial_velocity, angle_degrees = params
        angle = np.deg2rad(angle_degrees)
        
        x1, y1 = x_target, y_target

        initial_state = [x_start, y_start,
                        initial_velocity * np.cos(angle),
                        initial_velocity * np.sin(angle)]
        
        t = np.arange(0, time_span, dt)

        solution = odeint(self.derivatives, initial_state, t)

        xT, yT = solution[-1][0], solution[-1][1]
        return (xT - x1)**2 + (yT - y1)**2  # squared distance error


    def find_objective(self, v0_guess, x_start, y_start, x_target, y_target, t):
        result = minimize(self.objective,
                        v0_guess,
                        args=(x_start, y_start, x_target, y_target, t),
                        method='SLSQP')
        return result

    def simulate(self, v0, angle_deg, start_x=0, start_y=0, t_max=2.0, dt=0.01):
        angle = np.deg2rad(angle_deg)
        initial_state = [start_x, start_y, v0 * np.cos(angle), v0 * np.sin(angle)]
        t = np.arange(0, t_max, dt)
        solution = odeint(self.derivatives, initial_state, t)
        return t, solution

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
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
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
            'backgroundColor': '#f8f9fa',
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
    optimizer = ProjectileOptimizer()
    result = optimizer.find_objective([10,40], x_start, y_start, x_target, y_target, time_max)
    initial_velocity, angle = result.x[0], result.x[1]
    t, solution = optimizer.simulate(initial_velocity, angle, x_start, y_start, time_max)
    
    # Create main trajectory plot
    fig = go.Figure()
    
    # Add background image

    
    # Add trajectory line
    fig.add_trace(go.Scatter(
        x=solution[:, 0],
        y=solution[:, 1],
        mode='lines',
        name='Trajectory',
        line=dict(color='red', width=3)
    ))

    # fig.update_xaxes(autorange='reversed')
    
    # fig.add_layout_image(
    #     dict(
    #         source=get_encoded_image(),
    #         xref="x",
    #         yref="y",
    #         x=16,      # Start at x=0
    #         y=7,      # Adjust this value to match your image height
    #         sizex=-16, # Width of the court in meters
    #         sizey=10,  # Height of the image in meters
    #         sizing="contain",
    #         opacity=1,
    #         layer="below")
    # )

    # Add time markers
    # skip = len(t) // 10
    # if skip < 1:
    #     skip = 1
        
    # fig.add_trace(go.Scatter(
    #     x=solution[::skip, 0],
    #     y=solution[::skip, 1],
    #     mode='markers+text',
    #     name='Time Points',
    #     text=[f't={t:.1f}s' for t in t[::skip]],
    #     textposition="top center",
    #     marker=dict(size=8, color='blue')
    # ))
    
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
    
    # Calculate statistics
    max_height = np.max(solution[:, 1])
    max_distance = np.max(solution[:, 0])
    impact_idx = np.where(solution[:, 1] <= 0)[0]
    impact_time = t[impact_idx[0]] if len(impact_idx) > 0 else time_max
    impact_distance = solution[impact_idx[0], 0] if len(impact_idx) > 0 else max_distance

    # Final velocity at last point
    vx_final = solution[-1, 2]
    vy_final = solution[-1, 3]
    v_final = np.sqrt(vx_final**2 + vy_final**2)

    stats = html.Div([
        html.P(f"Maximum Height: {max_height:.2f} m"),
        html.P(f"Maximum Distance: {max_distance:.2f} m"),
        html.P(f"Instantaneous Velocity: {v_final:.2f} m/s"),
        html.P(f"Final vx: {vx_final:.2f} m/s"),
        html.P(f"Final vy: {vy_final:.2f} m/s"),
    ])
    
    return fig, stats

if __name__ == '__main__':
    # app.run(debug=False, host='0.0.0.0', port=8050)
    app.run(debug=True, port=8050)