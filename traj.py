import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import pandas as pd

from volleyball import Volleyball
from target_window import HittingWindow

# Example usage
if __name__ == "__main__":
    # Create projectile instance (using volleyball properties)
    proj = Volleyball(drag_coefficient = 0.48)
    
    # Cd = 0.48
    # rho = 1.225
    # A = 0.0346
    # m = 0.27

    time_span = 0.7 # 1.08, 1.4
    time_past_span = 0.1

    initial = np.array([0, 0])
    target = np.array([6.3, 0.3])
    hitting_radius = 0.1 # meters

    hw = HittingWindow(target)

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

    # df_sol_window = df_sol[df_sol.apply(hitting_window.in_hitting_window, axis=1, args=(window_target,))]  # windshield wiper shape
    df_sol_window = df_sol[df_sol.apply(hw.in_hitting_window, axis=1)]  # windshield wiper shape

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
    
    
    # Draw the hitting window
    hw.draw_hitting_window(ax1)
    ax1.legend()

    plt.show()