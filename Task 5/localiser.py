import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

class OccupancyMap:
    def __init__(self, image_path, resolution=0.09, theta_deg=-15.0, offset_x=18.0, offset_y=-2.0):
        self.map_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, self.map_img = cv2.threshold(self.map_img, 127, 255, cv2.THRESH_BINARY)
        
        # Your perfect calibration parameters
        self.resolution = resolution
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.theta_rad = np.radians(theta_deg)
        
        # Precompute sine and cosine for speed
        self.c = np.cos(self.theta_rad)
        self.s = np.sin(self.theta_rad)
        
        self.height, self.width = self.map_img.shape
        
    def is_free(self, x, y):
        """Check if a coordinate (in raw odom frame) is in the white path."""
        px, py = self.world_to_map(x, y)
        if 0 <= px < self.width and 0 <= py < self.height:
            return self.map_img[py, px] == 255
        return False

    def world_to_map(self, x, y):
        """Transforms raw odom coordinates to image pixels using your calibration."""
        # 1. Rotate
        rot_x = x * self.c - y * self.s
        rot_y = x * self.s + y * self.c
        
        # 2. Translate
        trans_x = rot_x + self.offset_x
        trans_y = rot_y + self.offset_y
        
        # 3. Scale to pixels
        px = int(trans_x / self.resolution + self.width / 2)
        py = int(-trans_y / self.resolution + self.height / 2)
        return px, py

class ParticleFilter:
    def __init__(self, num_particles, occupancy_map):
        self.num_particles = num_particles
        self.map = occupancy_map
        # We will initialize this later when we get the first ODOM reading
        self.particles = None 
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles(self, start_x, start_y, start_theta):
        """Scatter particles in a 1-meter radius around the starting pose."""
        particles = []
        for _ in range(self.num_particles):
            x = start_x + np.random.normal(0, 0.5) 
            y = start_y + np.random.normal(0, 0.5)
            # Small angular spread (about 11 degrees)
            theta = start_theta + np.random.normal(0, 0.2) 
            particles.append([x, y, theta])
        self.particles = np.array(particles)

    def predict(self, delta_x, delta_y, delta_theta):
        """Apply odometry to particles with added Gaussian noise."""
        if self.particles is None: return

        noise_x = np.random.normal(0, 0.05, self.num_particles) 
        noise_y = np.random.normal(0, 0.05, self.num_particles)
        noise_theta = np.random.normal(0, 0.02, self.num_particles) 
        
        self.particles[:, 0] += delta_x + noise_x
        self.particles[:, 1] += delta_y + noise_y
        self.particles[:, 2] += delta_theta + noise_theta
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

    def update_weights(self, laser_ranges, laser_angles):
        """Evaluate using a softer, additive likelihood model."""
        if self.particles is None: return

        for i, particle in enumerate(self.particles):
            x, y, theta = particle
            
            # If particle drifts outside the map, penalize but don't instantly kill it
            if not self.map.is_free(x, y):
                self.weights[i] = 0.01 
                continue
                
            hits = 0
            valid_rays = 0
            for r, angle in zip(laser_ranges, laser_angles):
                if r > 80.0: continue 
                
                valid_rays += 1
                end_x = x + r * math.cos(theta + angle)
                end_y = y + r * math.sin(theta + angle)
                
                # Count how many rays actually hit a wall
                if not self.map.is_free(end_x, end_y):
                    hits += 1
            
            # Weight is based on the proportion of successful wall hits
            if valid_rays > 0:
                hit_ratio = hits / valid_rays
                # math.exp helps amplify good matches without causing instant zero-collapse
                self.weights[i] = math.exp(hit_ratio * 5.0) 
            else:
                self.weights[i] = 1.0
            
        # Normalize
        self.weights += 1.e-300 
        self.weights /= sum(self.weights)

    def resample(self):
        if self.particles is None: return
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimated_pose(self):
        if self.particles is None: return [0,0,0]
        return np.mean(self.particles, axis=0)

# ==========================================
# Execution Loop
# ==========================================
def parse_clf_and_run(clf_file, map_file):
    occ_map = OccupancyMap(map_file)
    pf = ParticleFilter(num_particles=300, occupancy_map=occ_map)
    
    prev_odom = None
    estimated_path = []
    
    with open(clf_file, 'r') as f:
        for line in f:
            data = line.split()
            if not data: continue
            
            if data[0] == 'ODOM':
                current_odom = np.array([float(data[1]), float(data[2]), float(data[3])])
                
                # --- NEW: Initialize particles on the very first odometry reading ---
                if pf.particles is None:
                    pf.initialize_particles(current_odom[0], current_odom[1], current_odom[2])
                    prev_odom = current_odom
                    continue
                # --------------------------------------------------------------------

                delta = current_odom - prev_odom
                pf.predict(delta[0], delta[1], delta[2])
                prev_odom = current_odom
                
            elif data[0] == 'FLASER':
                if pf.particles is None: continue # Skip lasers until we have an odom pose
                
                num_readings = int(data[1])
                skip = 10
                laser_ranges = [float(x) for x in data[2:2+num_readings:skip]]
                laser_angles = np.linspace(-np.pi/2, np.pi/2, num_readings)[::skip]
                
                pf.update_weights(laser_ranges, laser_angles)
                pf.resample()
                
                estimated_path.append(pf.get_estimated_pose())

    return estimated_path, occ_map
    
if __name__ == "__main__":
    clf_file_path = "/home/aces.clf"
    map_file_path = "clean_map.png"
    
    print("Running Particle Filter (Localizing...). Please wait.")
    estimated_path, occ_map = parse_clf_and_run(clf_file_path, map_file_path)
    print("Localization complete!")
    
    # --- Visualization ---
    map_img = cv2.imread(map_file_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert estimated poses to pixels using your calibrated function
    path_px, path_py = [], []
    for pose in estimated_path:
        px, py = occ_map.world_to_map(pose[0], pose[1])
        path_px.append(px)
        path_py.append(py)
        
    plt.figure(figsize=(10, 10))
    plt.imshow(map_img, cmap='gray')
    plt.plot(path_px, path_py, 'g-', linewidth=2, label='Corrected Path (MCL)')
    plt.title("Robot Strictly Localized using Particle Filter")
    plt.legend()
    plt.axis('off')
    plt.savefig("localised_mcl.png", dpi=150)
    plt.show()