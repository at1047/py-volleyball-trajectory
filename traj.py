import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import pandas as pd

class ProjectileMotion:
    def __init__(self, 
                 mass=0.27,          # kg (volleyball mass)
                 diameter=0.21,       # m (volleyball diameter)
                 drag_coefficient=0.88, # dimensionless 0.48
                 air_density=1.225):  # kg/m³ (at sea level, 15°C)
        
        self.mass = mass
        self.diameter = diameter
        self.area = np.pi * (diameter/2)**2  # cross-sectional area
        self.drag_coefficient = drag_coefficient
        self.air_density = air_density
        self.gravity = 9.81  # m/s²
        
    def derivatives(self, state, t):
        """Calculate derivatives for position and velocity"""
        # state = [x, y, vx, vy]
        vx, vy = state[2], state[3]
        velocity = np.array([vx, vy])
        
        # Calculate drag force
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude == 0:
            return np.zeros_like(velocity)
        
        drag_magnitude = (0.5 * self.air_density * velocity_magnitude**2 * 
                         self.drag_coefficient * self.area)
        
        drag_force = -drag_magnitude * velocity / velocity_magnitude
        
        # Acceleration components (F = ma)
        ax = drag_force[0] / self.mass
        ay = drag_force[1] / self.mass - self.gravity
        
        return [vx, vy, ax, ay]
    
    def simulate(self, initial_velocity, angle_degrees, initial_pos, time_span, dt=0.001):
        """
        Simulate projectile motion
        
        Parameters:
        - initial_velocity: speed in m/s
        - angle_degrees: launch angle in degrees
        - time_span: simulation duration in seconds
        - dt: time step in seconds
        """
        angle = np.deg2rad(angle_degrees)
        initial_state = [initial_pos[0], initial_pos[1],
                        initial_velocity * np.cos(angle),
                        initial_velocity * np.sin(angle)]
        
        t = np.arange(0, time_span, dt)
        
        solution = odeint(self.derivatives, initial_state, t)
        
        return t, solution
    
    def objective(self, params, x_initial, y_initial, x_target, y_target, time_span, dt=0.01):
        initial_velocity, angle_degrees = params
        angle = np.deg2rad(angle_degrees)
        
        x1, y1 = x_target, y_target

        initial_state = [x_initial, y_initial,
                        initial_velocity * np.cos(angle),
                        initial_velocity * np.sin(angle)]
        
        t = np.arange(0, time_span, dt)

        solution = odeint(self.derivatives, initial_state, t)

        xT, yT = solution[-1][0], solution[-1][1]
        return (xT - x1)**2 + (yT - y1)**2  # squared distance error


    def find_objective(self, v0_guess, x_initial, y_initial, x_target, y_target, t):
        result = minimize(self.objective,
                        v0_guess,
                        args=(x_initial, y_initial, x_target, y_target, t),
                        method='SLSQP')
        return result


    def in_hitting_window(self, row, target):
        """
        Check if position is within a windshield wiper shaped hitting window.
        
        Parameters:
        - row: DataFrame row with 'x' and 'y' coordinates
        - target: target position [x, y] (apex of the wiper)
        - wiper_angle_degrees: angular width of the wiper in degrees (default 60°)
        - wiper_radius: radius of the wiper (uses global hitting_radius if None)
        """

        wiper_radius_outer = 1
        wiper_radius_inner = 0.8
        wiper_angle_degrees = 30
        wiper_offset_top = 0.1

        wiper_center = target - np.array([0, wiper_radius_outer - wiper_offset_top])
            
        x = row["x"]
        y = row["y"]
        pos = np.array([x, y])
        
        direction_vector = pos - wiper_center
        print(f"direction_vector {direction_vector}")

        angle_rad = np.arctan(abs(direction_vector[1]/direction_vector[0]))
        angle_degrees = abs(np.degrees(angle_rad)-90)
        half_angle = wiper_angle_degrees / 2
        print(f"half angle {half_angle}")
        print(f"angle_degrees {angle_degrees}")
        print(f"distance from base {np.linalg.norm(pos - wiper_center)}")
        print(f"{wiper_radius_inner} {wiper_radius_outer}")
        print(f"{wiper_radius_inner <= np.linalg.norm(pos - wiper_center) <= wiper_radius_outer}")
        
        return (abs(angle_degrees) <= half_angle and 
                wiper_radius_inner <= np.linalg.norm(pos - wiper_center) <= wiper_radius_outer)

# Example usage
if __name__ == "__main__":
    # Create projectile instance (using volleyball properties)
    proj = ProjectileMotion(drag_coefficient = 0.48)
    
    # Cd = 0.48
    # rho = 1.225
    # A = 0.0346
    # m = 0.27

    time_span = 0.7 # 1.08, 1.4
    time_past_span = 0.1

    initial = np.array([0, 0])
    target = np.array([6.5, 0.3])
    hitting_radius = 0.1 # meters

    result = proj.find_objective([10,40], initial[0], initial[1], target[0], target[1], time_span)

    initial_velocity, angle = result.x[0], result.x[1]
    # Run simulation to generate trajectory
    t, solution = proj.simulate(initial_velocity, angle, initial, time_span + time_past_span, dt = 0.0001)

    
    # Trajectory plot
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(solution[:, 0], solution[:, 1], 'b-', label='Trajectory')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Projectile Trajectory')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.set_xlim([-3,8])
    ax1.set_xticks(range(-3, 8))
    ax1.vlines([-2.5, 6.5], 0, 1)
    ax1.hlines([0.2], 6, 6.5)
    ax1.invert_xaxis()
    img = plt.imread("Volleyball_Shoot_Processed.png")

    img_height = 8
    img_width = img_height*16/9

    img_pos = [-5, -5.5]
    
    df_sol = pd.DataFrame.from_dict(dict(zip(["x", "y", "dx", "dy"], solution.T)), orient='columns')
    df_sol["time"] = t

    window_target = target - np.array([0.2, 0])

    df_sol_window = df_sol[df_sol.apply(proj.in_hitting_window, axis=1, args=(window_target,))]  # windshield wiper shape
    print(f"Points in wiper window: {len(df_sol_window)}")
    print(f"Total trajectory points: {len(df_sol)}")

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

    ax1.imshow(img, extent=[img_pos[0]+img_width, img_pos[0], img_pos[1], img_pos[1]+img_height], alpha=0.6)
    
    # Draw the hitting window (windshield wiper shape)
    def draw_hitting_window(ax, target, wiper_radius_outer=1, wiper_radius_inner=0.8, 
                           wiper_angle_degrees=30, wiper_offset_top=0.1, num_points=100):
        """Draw the windshield wiper hitting window"""
        wiper_base = target - np.array([0, wiper_radius_outer - wiper_offset_top])
        half_angle = wiper_angle_degrees / 2
        
        # Create angles for the wiper arc
        angles = np.linspace(-half_angle, half_angle, num_points)
        
        # Outer arc
        outer_x = wiper_base[0] + wiper_radius_outer * np.sin(np.radians(angles))
        outer_y = wiper_base[1] + wiper_radius_outer * np.cos(np.radians(angles))
        
        # Inner arc
        inner_x = wiper_base[0] + wiper_radius_inner * np.sin(np.radians(angles))
        inner_y = wiper_base[1] + wiper_radius_inner * np.cos(np.radians(angles))
        
        # Create the wiper shape by connecting outer and inner arcs
        wiper_x = np.concatenate([outer_x, inner_x[::-1]])
        wiper_y = np.concatenate([outer_y, inner_y[::-1]])
        
        # Plot the wiper shape
        ax.fill(wiper_x, wiper_y, alpha=0.3, color='red', label='Hitting Window')
        ax.plot(wiper_x, wiper_y, 'r-', linewidth=0.5)
        
        # Mark the target
        ax.plot(target[0], target[1], 'ro', markersize=2, label='Target')
        
        # Mark the wiper base
        ax.plot(wiper_base[0], wiper_base[1], 'bo', markersize=2, label='Wiper Base')
    
    # Draw the hitting window
    draw_hitting_window(ax1, window_target)
    ax1.legend()

    plt.show()