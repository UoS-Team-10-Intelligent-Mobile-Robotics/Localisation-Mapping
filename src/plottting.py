import numpy as np
import matplotlib.pyplot as plt
from math_feeg6043 import Vector, Matrix, HomogeneousTransformation, polar2cartesian
from plot_feeg6043 import show_observation, plot_2dframe, show_scan
from model_feeg6043 import build_square_environment, lidar_scan, RangeAngleKinematics

# --- Load data from CSV ---
# Assuming the CSV file is in the same directory as your script or provide the full path.
csv_file_path = "particle_path_slam.csv"
# Load the data, skipping the header row.
# The header in laptop.py is: "Time, Northings KDE, Eastings KDE, Yaw KDE, Northings STD, Eastings STD, Northings GT, Eastings GT, Yaw GT"
# This means 9 data columns.
data = np.loadtxt(csv_file_path, delimiter=",", skiprows=1)

# Extract matrices based on the column structure
T = data[:, 0].reshape(-1, 1)  # Time (1 column)
P_KDE = data[:, 1:4]  # Northings KDE, Eastings KDE, Yaw KDE (3 columns)
P_STD = data[:, 4:6]  # Northings STD, Eastings STD (2 columns)
P_GT = data[:, 6:9]  # Northings GT, Eastings GT, Yaw GT (3 columns)

num_timesteps = T.shape[0]

# --- Define lidar object (ensure parameters match those used during saving in laptop.py) ---
# Parameters from laptop.py:
# lidar_xb = 0.1, lidar_yb = 0
# distance_range=[0.1, 1], scan_fov=np.deg2rad(90), n_beams=30
lidar = RangeAngleKinematics(
    x_bl=0.1,
    y_bl=0,
    gamma_bl=0,
    distance_range=[0.1, 1.0],
    scan_fov=np.deg2rad(90),  # Corrected to match laptop.py
    n_beams=30,
)

# --- Define environment map ---
environment_map = build_square_environment() + np.ones((800, 2))
m_x = environment_map[:, 0]
m_y = environment_map[:, 1]

# --- Define observation noise for lidar_scan (from laptop.py) ---
# lidar_scan primarily uses sigma_observe[0,0] as a factor for range standard deviation.
sigma_observe_param = Matrix(2, 2)
sigma_observe_param[0, 0] = (
    0.01**2
)  # As in laptop.py, used by lidar_scan for range noise std dev factor
# Other elements are not directly used by lidar_scan's noise generation but are part of the full matrix.
sigma_observe_param[0, 1] = 0.0
sigma_observe_param[1, 0] = np.deg2rad(0.1) ** 2  # As in laptop.py
sigma_observe_param[1, 1] = 0.0  # As in laptop.py
# --- End of definitions ---


# --- Plot Ground Truth Path and Generated Observations ---
fig_gt, ax_gt = plt.subplots()
print("Ground truth path and generated observations")
for i in range(num_timesteps):
    p_gt_current = P_GT[[i], 0:3].T
    ax_gt.scatter(m_y, m_x, s=0.1, c="gray", alpha=0.5)

    # Generate observations for the current GT pose
    observations_gt, _ = lidar_scan(
        p_gt_current, environment_map, lidar, sigma_observe_param
    )

    if observations_gt.size > 0 and not np.all(np.isnan(observations_gt[:, 0])):
        show_scan(p_gt_current, lidar, observations_gt, ax_gt, show_lines=False)
    else:  # If no valid observations, just plot the pose
        plot_2dframe(
            ["pose_gt", "p_gt" + str(i), "p_gt" + str(i)],
            [HomogeneousTransformation(p_gt_current[0:2], p_gt_current[2]).H],
            edge_flag=False,
            legend_flag=False,
            compact_flag=True,
        )

ax_gt.scatter(P_GT[[0], 1], P_GT[[0], 0], c="r", label="Start GT")
ax_gt.scatter(P_GT[[-1], 1], P_GT[[-1], 0], c="c", label="End GT")
ax_gt.set_xlabel("Eastings (m)")
ax_gt.set_ylabel("Northings (m)")
ax_gt.set_title("Ground Truth Path and Generated Observations")
ax_gt.legend()
ax_gt.axis("equal")
plt.show()

# --- Plot Particle Path SLAM (KDE) Path and Generated Observations ---
fig_kde, ax_kde = plt.subplots()
print("Particle Path SLAM (KDE) path and generated observations")
for i in range(num_timesteps):
    p_kde_current = P_KDE[[i], 0:3].T
    ax_kde.scatter(m_y, m_x, s=0.1, c="gray", alpha=0.5)

    # Generate observations for the current KDE pose
    observations_kde, _ = lidar_scan(
        p_kde_current, environment_map, lidar, sigma_observe_param
    )

    if observations_kde.size > 0 and not np.all(np.isnan(observations_kde[:, 0])):
        show_scan(p_kde_current, lidar, observations_kde, ax_kde, show_lines=False)
    else:  # If no valid observations, just plot the pose
        plot_2dframe(
            ["pose", "p_kde" + str(i), "p_kde" + str(i)],
            [HomogeneousTransformation(p_kde_current[0:2], p_kde_current[2]).H],
            edge_flag=False,
            legend_flag=False,
            compact_flag=True,
        )

ax_kde.scatter(P_KDE[[0], 1], P_KDE[[0], 0], c="r", label="Start KDE")
ax_kde.scatter(P_KDE[[-1], 1], P_KDE[[-1], 0], c="c", label="End KDE")
ax_kde.set_xlabel("Eastings (m)")
ax_kde.set_ylabel("Northings (m)")
ax_kde.set_title("Particle Path SLAM (KDE) Path and Generated Observations")
ax_kde.legend()
ax_kde.axis("equal")
plt.show()

# --- Plot Pose Standard Deviation (KDE) ---
plt.figure()
plt.plot(T, P_STD[:, 0], label="northings std, m")
plt.plot(T, P_STD[:, 1], label="eastings std, m")
plt.xlabel("Time, s")
plt.ylabel("Standard deviation, m")
plt.title("Pose Standard Deviation (KDE)")
plt.legend()
plt.grid(True)
plt.show()
