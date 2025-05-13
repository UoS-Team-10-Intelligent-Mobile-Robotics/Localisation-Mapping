import pygame
import numpy as np
from math import sin, cos
import threading
import time

class RealtimeVisualizer:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        self.width, self.height = 800, 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Real-time Particle Visualization')
        
        # Set up drawing parameters
        self.background_color = (240, 240, 240)
        self.grid_color = (200, 200, 200)
        self.particle_color = (100, 100, 255, 128)  # Blue with alpha
        self.robot_color = (255, 0, 0)  # Red
        self.lidar_color = (0, 200, 0)  # Green
        
        # Scale and offset for coordinates (meters to pixels)
        self.scale = 100  # pixels per meter
        self.offset_x = self.width // 2
        self.offset_y = self.height // 2
        
        # Data storage
        self.particles = []
        self.robot_pose = [0, 0, 0]  # [x, y, theta]
        self.lidar_data = None
        self.running = True
        
        # Start the visualization thread
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates"""
        screen_x = int(x * self.scale + self.offset_x)
        screen_y = int(self.offset_y - y * self.scale)  # Flip Y for screen coords
        return (screen_x, screen_y)
    
    def update_particles(self, particle_filter):
        """Update particle data"""
        try:
            # First attempt - if particles have x, y, theta attributes
            self.particles = [(p.x, p.y, p.theta) for p in particle_filter.particles]
        except AttributeError:
            try:
                # Second attempt - if particles have pose attribute
                self.particles = [(p.pose[0], p.pose[1], p.pose[2]) for p in particle_filter.particles]
            except (AttributeError, IndexError):
                try:
                    # Third attempt - if particles have pose.x, pose.y, pose.theta
                    self.particles = [(p.pose.x, p.pose.y, p.pose.theta) for p in particle_filter.particles]
                except AttributeError:
                    # Fallback - just use position if available
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
    
    def draw_grid(self):
        """Draw coordinate grid"""
        # Fill background
        self.screen.fill(self.background_color)
        
        # Draw grid lines
        grid_spacing = 50  # pixels
        for x in range(0, self.width, grid_spacing):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, grid_spacing):
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.width, y))
            
        # Draw coordinate axes
        pygame.draw.line(self.screen, (0, 0, 0), 
                         (0, self.offset_y), (self.width, self.offset_y), 1)
        pygame.draw.line(self.screen, (0, 0, 0), 
                         (self.offset_x, 0), (self.offset_x, self.height), 1)
    
    def draw_particles(self):
        """Draw particles"""
        for p in self.particles:
            # Skip particles with NaN values
            if any(np.isnan(coord) for coord in p):
                continue
                
            x, y, theta = p
            screen_x, screen_y = self.world_to_screen(x, y)
            
            # Draw particle position
            pygame.draw.circle(self.screen, self.particle_color, (screen_x, screen_y), 3)
            
            # Draw particle heading (every 10th particle to avoid clutter)
            if self.particles.index(p) % 10 == 0:
                end_x = screen_x + int(10 * cos(theta))
                end_y = screen_y - int(10 * sin(theta))  # Flip Y for screen
                pygame.draw.line(self.screen, self.particle_color, 
                                (screen_x, screen_y), (end_x, end_y), 1)
    
    def draw_robot(self):
        """Draw robot pose"""
        x, y, theta = self.robot_pose
        screen_x, screen_y = self.world_to_screen(x, y)
        
        # Draw robot position
        pygame.draw.circle(self.screen, self.robot_color, (screen_x, screen_y), 8)
        
        # Draw robot heading
        end_x = screen_x + int(20 * cos(theta))
        end_y = screen_y - int(20 * sin(theta))  # Flip Y for screen
        pygame.draw.line(self.screen, self.robot_color, 
                        (screen_x, screen_y), (end_x, end_y), 3)
    
    def draw_lidar(self):
        """Draw lidar scan data"""
        if self.lidar_data is None or len(self.lidar_data) == 0:
            return
            
        for point in self.lidar_data:
            if len(point) >= 2:
                # Skip points with NaN values
                if np.isnan(point[0]) or np.isnan(point[1]):
                    continue
                    
                # Convert to screen coordinates
                screen_x, screen_y = self.world_to_screen(point[0], point[1])
                pygame.draw.circle(self.screen, self.lidar_color, (screen_x, screen_y), 2)
    
    def draw_text(self):
        """Draw text information"""
        font = pygame.font.SysFont('Arial', 16)
        
        # Draw robot position text
        robot_text = f"Robot: ({self.robot_pose[0]:.2f}, {self.robot_pose[1]:.2f}, {self.robot_pose[2]:.2f})"
        text_surface = font.render(robot_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))
        
        # Draw particle count
        count_text = f"Particles: {len(self.particles)}"
        text_surface = font.render(count_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 30))
    
    def run(self):
        """Main visualization loop"""
        clock = pygame.time.Clock()
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Draw everything
            self.draw_grid()
            self.draw_particles()
            self.draw_lidar()
            self.draw_robot()
            self.draw_text()
            
            # Update display
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
            
    def close(self):
        """Close the visualizer"""
        self.running = False
        pygame.quit()