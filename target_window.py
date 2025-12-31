
import numpy as np
import plotly.graph_objects as go

class HittingWindow:
    def __init__(self, target):
        self.target = target
        self.wiper_radius_outer=1
        self.wiper_radius_inner=0.8
        self.wiper_angle_degrees=30
        self.wiper_offset_top=0.1
        self.num_points=100

    def in_hitting_window(self, row):
        """
        Check if position is within a windshield wiper shaped hitting window.
        
        Parameters:
        - row: DataFrame row with 'x' and 'y' coordinates
        - target: target position [x, y] (apex of the wiper)
        - wiper_angle_degrees: angular width of the wiper in degrees (default 60°)
        - wiper_radius: radius of the wiper (uses global hitting_radius if None)
        """
        target = self.target
        wiper_radius_outer = self.wiper_radius_outer
        wiper_radius_inner = self.wiper_radius_inner
        wiper_angle_degrees = self.wiper_angle_degrees
        wiper_offset_top = self.wiper_offset_top

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


    # Draw the hitting window (windshield wiper shape)
    def draw_hitting_window(self, ax):
        """Draw the windshield wiper hitting window"""

        target = self.target
        wiper_radius_outer = self.wiper_radius_outer
        wiper_radius_inner = self.wiper_radius_inner
        wiper_angle_degrees = self.wiper_angle_degrees
        wiper_offset_top = self.wiper_offset_top
        num_points = self.num_points=100

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

    def create_hitting_window_figure(self, fig):
        """
        Returns a Plotly Figure object containing the windshield wiper hitting window
        """
        target = self.target
        wiper_radius_outer = self.wiper_radius_outer
        wiper_radius_inner = self.wiper_radius_inner
        wiper_angle_degrees = self.wiper_angle_degrees
        wiper_offset_top = self.wiper_offset_top
        num_points = self.num_points=100

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

        # The Hitting Window (Filled Shape)
        fig.add_trace(go.Scatter(
            x=wiper_x, 
            y=wiper_y,
            fill='toself',   # This creates the filled polygon
            fillcolor='rgba(255, 0, 0, 0.3)', # Red with 0.3 opacity
            line=dict(color='red', width=1),
            name='Hitting Window'
        ))

        # The Target Marker
        fig.add_trace(go.Scatter(
            x=[target[0]], 
            y=[target[1]],
            mode='markers',
            marker=dict(color='red', size=8),
            name='Target'
        ))

        # The Wiper Base Marker
        fig.add_trace(go.Scatter(
            x=[wiper_base[0]], 
            y=[wiper_base[1]],
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Wiper Base'
        ))

        # Ensure aspect ratio is equal so the circle doesn't look oval
        fig.update_layout(
            yaxis_scaleanchor="x", 
            yaxis_scaleratio=1,
            showlegend=True
        )