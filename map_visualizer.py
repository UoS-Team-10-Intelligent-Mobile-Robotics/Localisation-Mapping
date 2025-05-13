import pygame
import numpy as np
import threading
import time
from math import sin, cos
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

class MapVisualizer:
    def __init__(self):
        # Create output directory
        self.output_dir = "visualization_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data storage
        self.particles = []
        self.robot_pose = [0, 0, 0]  # [x, y, theta]
        self.lidar_data = None
        self.running = True
        self.gpr_points = None
        self.persistent_map_points = None  # For persistent map visualization
        
        # Configure plot settings
        self.fig_size = (10, 10)
        self.plot_range = (-4, 4)  # meters
        self.frame_count = 0
        self.save_interval = 0.5  # Save every 0.5 seconds
        self.last_save_time = time.time()
        
        # Start the visualization thread
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
    
    def update_particles(self, particle_filter):
        """Update particle data"""
        try:
            # First attempt - if particles have x, y, theta attributes
            self.particles = [(p.x, p.y, p.theta) for p in particle_filter.particles]
        except AttributeError:
            try:
                # Second attempt - if particles have pose attribute with x, y, theta
                self.particles = [(p.pose[0], p.pose[1], p.pose[2]) for p in particle_filter.particles]
            except (AttributeError, IndexError):
                try:
                    # Third attempt - if particles have pose.x, pose.y, pose.theta
                    self.particles = [(p.pose.x, p.pose.y, p.pose.theta) for p in particle_filter.particles]
                except AttributeError:
                    # Fallback - just use position if available, default orientation to 0
                    try:
                        self.particles = [(p.x, p.y, 0) for p in particle_filter.particles]
                    except AttributeError:
                        print("Warning: Could not extract particle positions.")
                        self.particles = []
    
    def update_robot_pose(self, x, y, theta):
        """Update robot pose"""
        self.robot_pose = [x, y, theta]
    
    def update_lidar_scan(self, lidar_data):
        """Update lidar scan data"""
        if lidar_data is not None:
            # Filter out NaN values
            if isinstance(lidar_data, np.ndarray):
                valid_idx = ~np.isnan(lidar_data).any(axis=1)
                self.lidar_data = lidar_data[valid_idx].copy()
            else:
                self.lidar_data = lidar_data.copy()
    
    def update_gpr_continuous_map(self, points):
        """Update the GPR continuous map points"""
        if points is not None and len(points) > 0:
            self.gpr_points = points.copy()
    
    def update_persistent_scans(self, new_points):
        """Add new points to the persistent map visualization"""
        if not hasattr(self, 'persistent_map_points'):
            self.persistent_map_points = new_points
        else:
            self.persistent_map_points = np.vstack((self.persistent_map_points, new_points))
        
        # Optional: color coding based on age
        self.persistent_scan_colors = self._generate_point_colors(self.persistent_map_points)
    
    def _generate_point_colors(self, points):
        """Generate colors based on point age or other attributes"""
        if points.shape[1] > 2:  # If we have a timestamp column
            # Normalize timestamp to 0-1 range for color mapping
            timestamps = points[:, 2]
            normalized = (timestamps - np.min(timestamps)) / (np.max(timestamps) - np.min(timestamps) + 1e-10)
            # Create a colormap from blue (oldest) to red (newest)
            return plt.cm.cool(normalized)
        else:
            # Default color
            return np.full((len(points), 4), [0, 0, 1, 0.5])  # Blue with alpha
    
    def render_map(self):
        """Render the complete map"""
        # Create a new figure
        plt.figure(figsize=self.fig_size)
        
        # Set up axes
        plt.xlim(self.plot_range)
        plt.ylim(self.plot_range)
        plt.grid(True)
        plt.title(f"Robot SLAM Visualization - Frame {self.frame_count}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        
        # Draw coordinate grid
        plt.axhline(y=0, color='k', linestyle='-', linewidth=1)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=1)
        
        # Draw particles
        if self.particles:
            x_coords = [p[0] for p in self.particles]
            y_coords = [p[1] for p in self.particles]
            plt.scatter(x_coords, y_coords, color='blue', s=5, alpha=0.5, label="Particles")
            
            # Draw particle headings (for a subset to avoid clutter)
            particles_sample = self.particles[::5]  # Every 5th particle
            for x, y, theta in particles_sample:
                plt.arrow(x, y, 0.05*cos(theta), 0.05*sin(theta), 
                          head_width=0.02, head_length=0.03, fc='blue', ec='blue', alpha=0.5)
        
        # Draw robot
        x, y, theta = self.robot_pose
        plt.scatter(x, y, color='red', s=100, label="Robot")
        plt.arrow(x, y, 0.2*cos(theta), 0.2*sin(theta), 
                  head_width=0.05, head_length=0.1, fc='red', ec='red')
        
        # Draw lidar data
        if self.lidar_data is not None and len(self.lidar_data) > 0:
            if isinstance(self.lidar_data, np.ndarray):
                plt.scatter(self.lidar_data[:, 0], self.lidar_data[:, 1], 
                            color='green', s=10, label="LIDAR")
        
        # Draw GPR continuous map if available
        if self.gpr_points is not None and len(self.gpr_points) > 0:
            scatter = plt.scatter(
                self.gpr_points[:, 0],
                self.gpr_points[:, 1],
                c=self.gpr_points[:, 2],  # Color by uncertainty
                cmap='viridis',
                alpha=0.5,
                s=5,
                marker='o',
                label='GPR Map'
            )
            plt.colorbar(scatter, label="Uncertainty")
        
        # Draw persistent scan points if available
        if hasattr(self, 'persistent_map_points') and len(self.persistent_map_points) > 0:
            if hasattr(self, 'persistent_scan_colors'):
                plt.scatter(
                    self.persistent_map_points[:, 0],  # eastings
                    self.persistent_map_points[:, 1],  # northings
                    s=1,
                    c=self.persistent_scan_colors,
                    alpha=0.5,
                    label='Map Points'
                )
            else:
                plt.scatter(
                    self.persistent_map_points[:, 0],  # eastings
                    self.persistent_map_points[:, 1],  # northings
                    s=1,
                    c='blue',
                    alpha=0.5,
                    label='Map Points'
                )
        
        # Draw the current scan if available (optional - can remove if you only want persistent)
        if self.current_scan is not None:
            plt.scatter(
                self.current_scan[:, 0],  # eastings
                self.current_scan[:, 1],  # northings
                s=2,
                c='red',
                label='Current Scan'
            )
        
        # Add a legend
        plt.legend(loc="upper right")
        
        # Add position text
        plt.text(
            0.02, 0.98, 
            f"Robot: ({self.robot_pose[0]:.2f}, {self.robot_pose[1]:.2f}, {self.robot_pose[2]:.2f})",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )
        
        # Save the figure to a file
        filename = f"{self.output_dir}/slam_frame_{self.frame_count:04d}.png"
        plt.savefig(filename, dpi=100)
        plt.close()
        
        # Also save the latest frame to a consistent filename
        latest_filename = f"{self.output_dir}/slam_latest.png"
        plt.savefig(latest_filename, dpi=100)
        
        print(f"Saved visualization to {filename}")
        self.frame_count += 1
    
    def run(self):
        """Main visualization loop"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_save_time >= self.save_interval:
                self.render_map()
                self.last_save_time = current_time
            time.sleep(0.1)  # Sleep to prevent CPU overuse
    
    def close(self):
        """Close the visualizer"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)