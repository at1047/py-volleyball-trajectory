import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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
        
    def drag_force(self, velocity):
        """Calculate drag force using F = 1/2 * ρ * v² * Cd * A"""
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude == 0:
            return np.zeros_like(velocity)
        
        drag_magnitude = (0.5 * self.air_density * velocity_magnitude**2 * 
                         self.drag_coefficient * self.area)
        
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
        ay = drag_force[1] / self.mass - self.gravity
        
        return [vx, vy, ax, ay]
    
    def simulate(self, initial_velocity, angle_degrees, time_span, dt=0.01):
        """Simulate projectile motion"""
        angle = np.deg2rad(angle_degrees)
        initial_state = [0, 0,
                        initial_velocity * np.cos(angle),
                        initial_velocity * np.sin(angle)]
        t = np.arange(0, time_span, dt)
        solution = odeint(self.derivatives, initial_state, t)
        return t, solution
    
    def objective(self, params, x_target, y_target, time_span, dt=0.01):
        initial_velocity, angle_degrees = params
        angle = np.deg2rad(angle_degrees)
        x1, y1 = x_target, y_target
        initial_state = [0, 0,
                        initial_velocity * np.cos(angle),
                        initial_velocity * np.sin(angle)]
        t = np.arange(0, time_span, dt)
        solution = odeint(self.derivatives, initial_state, t)
        xT, yT = solution[-1][0], solution[-1][1]
        return (xT - x1)**2 + (yT - y1)**2

    def find_objective(self, v0_guess, x_target, y_target, t):
        result = minimize(self.objective,
                        v0_guess,
                        args=(x_target, y_target, t),
                        method='SLSQP')
        return result

    def plot_extended_trajectory(self, t, solution, target, target_time):
        """Plot the trajectory with target point and extended path highlighted"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Find the index closest to target time
        target_idx = find_nearest(t, target_time)
        
        # Plot full trajectory
        ax1.plot(solution[:, 0], solution[:, 1], 'b-', label='Full Trajectory', linewidth=2)
        
        # Highlight trajectory past target point
        ax1.plot(solution[target_idx:, 0], solution[target_idx:, 1], 'r-', 
                label='Trajectory Past Target', linewidth=3, alpha=0.8)
        
        # Mark target point
        ax1.plot(target[0], target[1], 'go', markersize=10, label='Target Point')
        
        # Mark where ball is at target time
        ax1.plot(solution[target_idx, 0], solution[target_idx, 1], 'ro', 
                markersize=8, label=f'Ball at t={target_time:.2f}s')
        
        # Mark final position
        ax1.plot(solution[-1, 0], solution[-1, 1], 'ko', 
                markersize=8, label=f'Final Position at t={t[-1]:.2f}s')
        
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Height (m)')
        ax1.set_title('Volleyball Trajectory: 0.5 seconds past target point')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim([-1, 8])
        ax1.set_ylim([-1, 3])
        
        # Add court markings
        ax1.vlines([-2.5, 6.5], 0, 1, colors='gray', linestyles='--', alpha=0.5)
        ax1.hlines([0.2], 6, 6.5, colors='gray', linestyles='--', alpha=0.5)
        
        # Velocity plot
        ax2.plot(t, solution[:, 2], 'r-', label='Horizontal velocity (m/s)')
        ax2.plot(t, solution[:, 3], 'g-', label='Vertical velocity (m/s)')
        ax2.plot(t, np.sqrt(solution[:, 2]**2 + solution[:, 3]**2), 'b-', 
                label='Total velocity (m/s)')
        
        # Mark target time on velocity plot
        ax2.axvline(x=target_time, color='orange', linestyle='--', 
                   label=f'Target time: {target_time:.2f}s')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity Components Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig

# Calculate extended trajectory
if __name__ == "__main__":
    # Create projectile instance
    proj = ProjectileMotion(drag_coefficient=0.48)
    
    # Parameters
    target = [6.5, 0.3]  # Target point
    target_time = 1.08   # Time to reach target
    extended_time = 1.58 # Total simulation time (0.5s past target)
    
    print("Calculating trajectory for 0.5 seconds past target point...")
    print(f"Target point: {target} meters")
    print(f"Target time: {target_time} seconds")
    print(f"Extended time: {extended_time} seconds")
    print(f"Extension: {extended_time - target_time} seconds past target")
    
    # Find optimal launch parameters to hit target
    result = proj.find_objective([10, 40], target[0], target[1], target_time)
    initial_velocity, angle = result.x[0], result.x[1]
    
    print(f"\nOptimal launch parameters:")
    print(f"Initial velocity: {initial_velocity:.2f} m/s")
    print(f"Launch angle: {angle:.2f} degrees")
    
    # Run extended simulation
    t, solution = proj.simulate(initial_velocity, angle, extended_time)
    
    # Find positions at key times
    target_idx = find_nearest(t, target_time)
    final_idx = len(t) - 1
    
    print(f"\nTrajectory analysis:")
    print(f"Position at target time ({target_time:.2f}s): ({solution[target_idx, 0]:.2f}, {solution[target_idx, 1]:.2f}) m")
    print(f"Position at final time ({t[final_idx]:.2f}s): ({solution[final_idx, 0]:.2f}, {solution[final_idx, 1]:.2f}) m")
    print(f"Distance traveled past target: {solution[final_idx, 0] - solution[target_idx, 0]:.2f} m")
    print(f"Height change past target: {solution[final_idx, 1] - solution[target_idx, 1]:.2f} m")
    
    # Calculate velocities
    target_velocity = np.sqrt(solution[target_idx, 2]**2 + solution[target_idx, 3]**2)
    final_velocity = np.sqrt(solution[final_idx, 2]**2 + solution[final_idx, 3]**2)
    
    print(f"\nVelocity analysis:")
    print(f"Velocity at target time: {target_velocity:.2f} m/s")
    print(f"Velocity at final time: {final_velocity:.2f} m/s")
    print(f"Velocity change: {final_velocity - target_velocity:.2f} m/s")
    
    # Create visualization
    fig = proj.plot_extended_trajectory(t, solution, target, target_time)
    plt.show()
    
    print(f"\nTrajectory calculation complete!")
    print(f"The ball continues for {extended_time - target_time:.2f} seconds past the target point.")

