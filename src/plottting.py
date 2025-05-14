import numpy as np
import matplotlib.pyplot as plt
from math_feeg6043 import Vector, Matrix, HomogeneousTransformation, polar2cartesian
from plot_feeg6043 import show_observation, plot_2dframe, show_scan
from model_feeg6043 import build_square_environment, lidar_scan, RangeAngleKinematics

# --- Load data from CSV ---
# Assuming the CSV file is in the same directory as your script or provide the full path.
csv_file_path = "particle_path_slam.csv"
# Load the data, skipping the header row
data = np.loadtxt(csv_file_path, delimiter=",", skiprows=1)

# Extract the first four matrices based on the column structure
# Header: "Time, Northings KDE, Eastings KDE, Yaw KDE, Northings STD, Eastings STD, Northings GT, Eastings GT, Yaw GT, Lidar observations"
T = data[:, 0].reshape(-1, 1)  # Time (1 column)
P_KDE = data[:, 1:4]  # Northings KDE, Eastings KDE, Yaw KDE (3 columns)
P_STD = data[:, 4:6]  # Northings STD, Eastings STD (2 columns)
P_GT = data[:, 6:9]  # Northings GT, Eastings GT, Yaw GT (3 columns)

# --- Define lidar object (ensure parameters match those used during saving) ---
# Parameters from laptop.py:
# lidar_xb = 0.1, lidar_yb = 0
# distance_range=[0.1, 1], scan_fov=np.deg2rad(120), n_beams=30
lidar = RangeAngleKinematics(
    x_bl=0.1,
    y_bl=0,
    gamma_bl=0,  # Default, matches laptop.py implicit assumption
    distance_range=[0.1, 1.0],
    scan_fov=np.deg2rad(120),
    n_beams=30,
)

# --- Extract and reshape observations ---
# The first 9 columns are T, P_KDE, P_STD, P_GT
num_preceding_cols = (
    T.shape[1] + P_KDE.shape[1] + P_STD.shape[1] + P_GT.shape[1]
)  # Should be 1+3+2+3 = 9

num_timesteps = T.shape[0]

if data.shape[1] > num_preceding_cols:
    observations_flat = data[:, num_preceding_cols:]

    # Expected number of columns for observations is num_beams * 2 (for range and bearing)
    expected_obs_cols = lidar.n_beams * 2

    if observations_flat.shape[1] == expected_obs_cols:
        observations_reshaped = observations_flat.reshape(
            num_timesteps, lidar.n_beams, 2
        )
        # Convert to a list of 2D arrays, one for each timestep, to match plotting loop usage
        observations_all = [
            observations_reshaped[i, :, :] for i in range(num_timesteps)
        ]
        print(
            f"Successfully loaded observations for {num_timesteps} timesteps, with {lidar.n_beams} beams each."
        )
    else:
        print(
            f"Warning: Number of observation columns in CSV ({observations_flat.shape[1]}) "
            f"does not match expected ({expected_obs_cols} based on {lidar.n_beams} beams). "
            f"Observations might be incorrect or missing."
        )
        observations_all = [
            np.array([]) for _ in range(num_timesteps)
        ]  # Fallback to empty observations
else:
    print("No observation data found in CSV after the first 9 columns.")
    observations_all = [
        np.array([]) for _ in range(num_timesteps)
    ]  # Fallback to empty observations
# --- End of CSV loading ---

# For plotting context, ensure m_x, m_y, and ax are defined
m_x, m_y = build_square_environment().T + np.ones((800, 2))
fig, ax = plt.subplots()


print("Ground truth path and observations")
for i in range(num_timesteps):  # Use num_timesteps
    p = P_GT[[i], 0:3].T
    plt.scatter(m_y, m_x, s=0.1, c="gray", alpha=0.5)  # Added color and alpha for map
    if (
        observations_all[i].size > 0
    ):  # Check if there are observations for this timestep
        show_scan(p, lidar, observations_all[i], ax, show_lines=False)
    else:  # If no observations, just plot the pose
        plot_2dframe(
            ["pose_gt", "p_gt" + str(i), "p_gt" + str(i)],
            [HomogeneousTransformation(p[0:2], p[2]).H],
            edge_flag=False,
            legend_flag=False,
            compact_flag=True,
        )


plt.scatter(P_GT[[0], 1], P_GT[[0], 0], c="r", label="Start GT")
plt.scatter(P_GT[[-1], 1], P_GT[[-1], 0], c="c", label="End GT")
plt.xlabel("Eastings (m)")
plt.ylabel("Northings (m)")
plt.title("Ground Truth Path and Observations")
plt.legend()
plt.axis("equal")
plt.show()

fig, ax = plt.subplots()  # New figure and axis for the next plot
print("Particle Path SLAM path and observations")
for i in range(num_timesteps):  # Use num_timesteps
    p = P_KDE[[i], 0:3].T
    plt.scatter(m_y, m_x, s=0.1, c="gray", alpha=0.5)  # Added color and alpha for map
    if (
        observations_all[i].size > 0
    ):  # Check if there are observations for this timestep
        show_scan(p, lidar, observations_all[i], ax, show_lines=False)
    else:  # If no observations, just plot the pose
        plot_2dframe(
            ["pose", "p_kde" + str(i), "p_kde" + str(i)],
            [HomogeneousTransformation(p[0:2], p[2]).H],
            edge_flag=False,
            legend_flag=False,
            compact_flag=True,
        )


plt.scatter(P_KDE[[0], 1], P_KDE[[0], 0], c="r", label="Start KDE")
plt.scatter(P_KDE[[-1], 1], P_KDE[[-1], 0], c="c", label="End KDE")
plt.xlabel("Eastings (m)")
plt.ylabel("Northings (m)")
plt.title("Particle Path SLAM (KDE) Path and Observations")
plt.legend()
plt.axis("equal")
plt.show()

plt.figure()  # New figure for the STD plot
plt.plot(T, P_STD[:, 0], label="northings std, m")
plt.plot(T, P_STD[:, 1], label="eastings std, m")
plt.xlabel("Time, s")
plt.ylabel("Standard deviation, m")
plt.title("Pose Standard Deviation (KDE)")
plt.legend()
plt.grid(True)
plt.show()
