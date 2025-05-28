import numpy as np
import matplotlib.pyplot as plt

# Create a range of values for knee_vel and knee_angles
knee_velocities = np.linspace(-10, 10, 400)  # Range of -10 to 10 rad/s
knee_angles = np.linspace(-45, 95, 400)  # Range of -45 to 95 degrees

# Function to calculate sigmoid-based spasticity coefficients for angles
def calculate_spas_coef(k_ang, L_ang, a0_flexion, a0_extension, knee_angles):
    sigmoid_flexion = 1 / (1 + np.exp(-k_ang * (knee_angles - a0_flexion)))
    sigmoid_extension = 1 / (1 + np.exp(np.clip(k_ang * (knee_angles - a0_extension), -500, 500)))
    return L_ang + (L_ang * (1 - (sigmoid_flexion + sigmoid_extension)))

# Function to calculate sigmoid-based spasticity coefficients for velocities
def calculate_spas_coef_velocity(L_vel, k, v0_flexion, v0_extension, knee_velocities):
    sigmoid_flexion = 1 / (1 + np.exp(-k * (knee_velocities - v0_flexion)))
    sigmoid_extension = 1 / (1 + np.exp(k * (knee_velocities - v0_extension)))
    return L_vel + (L_vel * (1 - (sigmoid_flexion + sigmoid_extension)))

# Adjust shadow effect with gradient around each line
def add_gradient(ax, x, y, color, buffer_ratio=0.3, n_layers=20):
    # Calculate the 10% buffer above and below the line
    upper = y * (1 + buffer_ratio)
    lower = y * (1 - buffer_ratio)

    # Create gradient layers across the entire plot range
    for i in range(n_layers):
        alpha = 0.1 * (1 - i / n_layers)  # Decrease opacity with each layer
        ax.fill_between(x,
                        y * (1 + buffer_ratio * (i + 1) / n_layers),
                        y * (1 - buffer_ratio * (i + 1) / n_layers),
                        color=color, alpha=alpha)

# Spasticity coefficients for different levels
spas_coef_ang_lv0 = calculate_spas_coef(0.5, 0, -35, 80, knee_angles)
spas_coef_ang_lv1 = calculate_spas_coef(0.3, 0.2, -35, 80, knee_angles)
spas_coef_ang_lv2 = calculate_spas_coef(0.05, 0.3, -30, 75, knee_angles)
spas_coef_ang_lv3 = calculate_spas_coef(0.05, 0.5, -15, 70, knee_angles)

# Velocity-based spasticity coefficients for different levels
spas_coef_vel_lv0 = calculate_spas_coef_velocity(0, 5, -3, 3, knee_velocities)
spas_coef_vel_lv1 = calculate_spas_coef_velocity(0.2, 12, -1, 1, knee_velocities)
spas_coef_vel_lv2 = calculate_spas_coef_velocity(0.2, 12, -0.5, 0.5, knee_velocities)
spas_coef_vel_lv3 = calculate_spas_coef_velocity(0.3, 12, -0.25, 0.25, knee_velocities)

# Plot Results with gradient shadow effect
fig, axs = plt.subplots(2, 1, figsize=(6, 6))

# Plot Angular Spasticity Levels with gradient shadow
colors = ['g', '#FFDD44', 'orange', '#FF4500']
levels = [spas_coef_ang_lv0, spas_coef_ang_lv1, spas_coef_ang_lv2, spas_coef_ang_lv3]

for i, (color, spas_coef) in enumerate(zip(colors, levels)):
    add_gradient(axs[0], knee_angles, spas_coef, color=color)
    axs[0].plot(knee_angles, spas_coef, color=color, linewidth=3, label=f'lv_{i}')

axs[0].axvline(x=0, color='darkgray', linestyle='--', linewidth=2, label='Lower limit')
axs[0].axvline(x=90, color='darkgray', linestyle='--', linewidth=2, label='Upper limit')
axs[0].set_ylim([0, 1])
axs[0].set_title('Angular Spasticity Levels')
axs[0].set_xlim([-45, 95])
axs[0].set_xlabel('Knee Angle (Degrees)')
axs[0].set_ylabel('Spasticity Coefficient')
axs[0].grid(True)
axs[0].legend()

# Plot Velocity Spasticity Levels with gradient shadow across the entire plot
levels = [spas_coef_vel_lv0, spas_coef_vel_lv1, spas_coef_vel_lv2, spas_coef_vel_lv3]

for i, (color, spas_coef) in enumerate(zip(colors, levels)):
    add_gradient(axs[1], knee_velocities, spas_coef, color=color)
    axs[1].plot(knee_velocities, spas_coef, color=color, linewidth=3, label=f'lv_{i}')

axs[1].axvline(x=-10, color='darkgray', linestyle='--', linewidth=2, label='Lower limit')
axs[1].axvline(x=10, color='darkgray', linestyle='--', linewidth=2, label='Upper limit')
axs[1].set_ylim([0, 1])
axs[1].set_title('Velocity Spasticity Levels')
axs[1].set_xlim([-3, 3])
axs[1].set_xlabel('Knee Velocity (rad/s)')
axs[1].set_ylabel('Spasticity Coefficient')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()



"""
axs[0, 2].plot(knee_angles, spas_coef_ang_lv2, color='#FFDD44', linewidth=3)
axs[0, 2].axvline(x=10, color='darkgray', linestyle='--', linewidth=2, label='Inf limit (10º)')
axs[0, 2].axvline(x=80, color='darkgray', linestyle='--', linewidth=2, label='Sup limit (80º)')
axs[0, 2].set_ylim([0, 1])
axs[0, 2].set_title('Level 2 Angular Spasticity')
axs[0, 2].set_xlim([0, 90])
axs[0, 2].set_xlabel('Knee Angle (Degrees)')
axs[0, 2].grid(True)

axs[0, 3].plot(knee_angles, spas_coef_ang_lv3, color='#FFDD44', linewidth=3)
axs[0, 3].axvline(x=10, color='darkgray', linestyle='--', linewidth=2, label='Inf limit (10º)')
axs[0, 3].axvline(x=80, color='darkgray', linestyle='--', linewidth=2, label='Sup limit (80º)')
axs[0, 3].set_ylim([0, 1])
axs[0, 3].set_title('Level 3 Angular Spasticity')
axs[0, 3].set_xlim([0, 90])
axs[0, 3].set_xlabel('Knee Angle (Degrees)')
axs[0, 3].grid(True)

# Velocity Spasticity Plots with gray limits, y-axis from 0 to 1, and adjusted slopes
axs[1, 0].plot(knee_velocities, spas_coef_vel_lv0_adjusted, color='#FFDD44', linewidth=3)
axs[1, 0].axvline(x=-10, color='darkgray', linestyle='--', linewidth=2, label='Inf limit (-10 rad/s)')
axs[1, 0].axvline(x=10, color='darkgray', linestyle='--', linewidth=2, label='Sup limit (10 rad/s)')
axs[1, 0].set_ylim([0, 1])
axs[1, 0].set_title('Level 0 Velocity Spasticity')
axs[1, 0].set_xlim([-10, 10])
axs[1, 0].set_xlabel('Knee Velocity (rad/s)')
axs[1, 0].set_ylabel('Spasticity Coefficient')
axs[1, 0].grid(True)

axs[1, 1].plot(knee_velocities, spas_coef_vel_lv1_adjusted, color='#FFDD44', linewidth=3)
axs[1, 1].axvline(x=-4, color='darkgray', linestyle='--', linewidth=2, label='Inf limit (-4 rad/s)')
axs[1, 1].axvline(x=4, color='darkgray', linestyle='--', linewidth=2, label='Sup limit (4 rad/s)')
axs[1, 1].set_ylim([0, 1])
axs[1, 1].set_title('Level 1 Velocity Spasticity')
axs[1, 1].set_xlim([-10, 10])
axs[1, 1].set_xlabel('Knee Velocity (rad/s)')
axs[1, 1].grid(True)

axs[1, 2].plot(knee_velocities, spas_coef_vel_lv2_adjusted, color='#FFDD44', linewidth=3)
axs[1, 2].axvline(x=-2, color='darkgray', linestyle='--', linewidth=2, label='Inf limit (-2 rad/s)')
axs[1, 2].axvline(x=2, color='darkgray', linestyle='--', linewidth=2, label='Sup limit (2 rad/s)')
axs[1, 2].set_ylim([0, 1])
axs[1, 2].set_title('Level 2 Velocity Spasticity')
axs[1, 2].set_xlim([-10, 10])
axs[1, 2].set_xlabel('Knee Velocity (rad/s)')
axs[1, 2].grid(True)

axs[1, 3].plot(knee_velocities, spas_coef_vel_lv3_adjusted, color='#FFDD44', linewidth=3)
axs[1, 3].axvline(x=-1, color='darkgray', linestyle='--', linewidth=2, label='Inf limit (-1 rad/s)')
axs[1, 3].axvline(x=1, color='darkgray', linestyle='--', linewidth=2, label='Sup limit (1 rad/s)')
axs[1, 3].set_ylim([0, 1])
axs[1, 3].set_title('Level 3 Velocity Spasticity')
axs[1, 3].set_xlim([-10, 10])
axs[1, 3].set_xlabel('Knee Velocity (rad/s)')
axs[1, 3].grid(True)
"""
plt.tight_layout()
plt.show()
