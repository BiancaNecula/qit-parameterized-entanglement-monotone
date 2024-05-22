import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def xi(F, q, d):
    gamma = np.sqrt(F / d) + np.sqrt((d - 1) * (1 - F) / d)
    delta = np.sqrt(F / d) - np.sqrt((1 - F) / ((d - 1) * d))
    return 1 - gamma**(2*q) - (d-1) * delta**(2*q)

def C2_rhoF(F, d):
    C2 = np.zeros_like(F)
    region1 = (F <= 1 / d)
    region2 = (1 / d < F) & (F <= 4 * (d - 1) / (d ** 2))
    region3 = (F > 4 * (d - 1) / (d ** 2))
    
    C2[region1] = 0
    C2[region2] = xi(F[region2], 2, d[region2])
    C2[region3] = (d[region3] * F[region3] - d[region3]) / (d[region3] - 1) + (d[region3] - 1) / d[region3]
    
    return C2

def lower_bound(F, d):
    return (d * F - 1) ** 2 / (d ** 2 - d)

# Parameters given in the second example
alpha = beta = 1 / np.sqrt(2)
q = 4

# Define the function ΔC_q(|Γ⟩)
def delta_C_q(theta, phi):
    cos2q_theta = np.cos(theta) ** (2 * q)
    sin2q_theta = np.sin(theta) ** (2 * q)
    cos2q_phi = np.cos(phi) ** (2 * q)
    sin2q_phi = np.sin(phi) ** (2 * q)
    
    term1 = (np.abs(alpha)**2 - np.abs(alpha)**(2*q)) * (cos2q_theta + sin2q_theta)
    term2 = (np.abs(beta)**2 - np.abs(beta)**(2*q)) * (cos2q_phi + sin2q_phi)
    
    return term1 + term2

# Create a meshgrid for theta and phi
theta = np.linspace(0, np.pi/2, 100)
phi = np.linspace(0, np.pi/2, 100)
theta, phi = np.meshgrid(theta, phi)

# Compute ΔC_q(|Γ⟩) on the grid
delta_Cq_values = delta_C_q(theta, phi)

# Plotting
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot_surface(theta, phi, delta_Cq_values, cmap='viridis')

ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\phi$')
ax.set_zlabel(r'$\Delta C_q(|\Gamma\rangle)$')

plt.savefig('plot_2.png')
plt.show()


# Generate data for plot (a)
d_values_a = np.linspace(3, 10, 100)
#F_values_a = np.linspace(0, 4*(d_values_a-1)/(d_values_a**2), 100)
F_values_a = np.linspace(1/10, 4*(10-1)/(10**2), 100)
D_a, F_a = np.meshgrid(d_values_a, F_values_a)
C2_values_a = C2_rhoF(F_a, D_a)
lower_bound_values_a = lower_bound(F_a, D_a)

# Generate data for plot (b)
d_values_b = np.linspace(3, 10, 100)
#F_values_b = np.linspace(4*(d_values_b-1)/(d_values_b**2), 1, 100)
F_values_b = np.linspace(4*(3-1)/(3**2), 1, 100)
D_b, F_b = np.meshgrid(d_values_b, F_values_b)
C2_values_b = C2_rhoF(F_b, D_b)
lower_bound_values_b = lower_bound(F_b, D_b)

# Plotting
fig = plt.figure(figsize=(14, 7))

# Subplot (a)
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(D_a, F_a, C2_values_a, color='green', alpha=0.6, label='$C_2(\\rho_F)$')
ax1.plot_surface(D_a, F_a, lower_bound_values_a, color='blue', alpha=0.6, label='Lower bound')
ax1.set_xlabel('$d$')
ax1.set_ylabel('$F$')
ax1.set_zlabel('$C_2(\\rho_F)$')
ax1.set_title('(a) 3 ≤ d ≤ 10 and $1/d ≤ F ≤ 4(d-1)/d^2$')
ax1.set_xlim(3, 10)
#ax1.set_ylim(0, 1)
ax1.set_ylim(1/10, 4*(10-1)/(10**2))
ax1.set_zlim(0, 1)
ax1.view_init(elev=30, azim=60)
ax1.legend()

# Subplot (b)
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(D_b, F_b, C2_values_b, color='green', alpha=0.6, label='$C_2(\\rho_F)$')
ax2.plot_surface(D_b, F_b, lower_bound_values_b, color='blue', alpha=0.6, label='Lower bound')
ax2.set_xlabel('$d$')
ax2.set_ylabel('$F$')
ax2.set_zlabel('$C_2(\\rho_F)$')
ax2.set_title('(b) 3 ≤ d ≤ 10 and $4(d-1)/d^2 ≤ F ≤ 1$')
ax2.set_xlim(3, 10)
#ax2.set_ylim(0, 1)
ax2.set_ylim(4*(3-1)/(3**2), 1)
ax2.set_zlim(0, 1)
ax2.view_init(elev=30, azim=60)
ax2.legend()

# Save the plot as a file
plt.tight_layout()
plt.savefig('plot_1.png')
plt.show()

alpha = beta = 1 / np.sqrt(2)
q = 6

def C_q_Phi(theta):
    return 1 - np.cos(theta)**(2*q) - np.sin(theta)**(2*q)

def C_q_Psi(phi):
    return 1 - np.cos(phi)**(2*q) - np.sin(phi)**(2*q)

def C_q_Gamma(theta, phi):
    return 1 - (np.abs(alpha)**2 * np.cos(theta)**2 + np.abs(beta)**2 * np.cos(phi)**2)**q - (np.abs(alpha)**2 * np.sin(theta)**2 + np.abs(beta)**2 * np.sin(phi)**2)**q

def delta_C_q_Gamma(theta, phi):
    return C_q_Gamma(theta, phi) - (np.abs(alpha)**2 * C_q_Phi(theta) + np.abs(beta)**2 * C_q_Psi(phi))

# Create a meshgrid for theta and phi
theta = np.linspace(0, np.pi/2, 100)
phi = np.linspace(0, np.pi/2, 100)
theta, phi = np.meshgrid(theta, phi)

# Compute ΔC_q(|Γ⟩) on the grid
delta_Cq_values = delta_C_q_Gamma(theta, phi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta, phi, delta_Cq_values, cmap='viridis')

ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\phi$')
ax.set_zlabel(r'$\Delta C_q(|\Gamma\rangle)$')

ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
ax.set_xticklabels([r'$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])

ax.set_yticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
ax.set_yticklabels([r'$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])

ax.set_zticks([0, 0.3, 0.6, 0.9])
ax.set_zticklabels([r'$0$', r'$0.3$', r'$0.6$', r'$0.9$'])

ax.view_init(elev=30, azim=225)

plt.show()

plt.savefig('plot_3.png')
plt.show()

q_values = np.linspace(2, 4, 100)
theta_values = np.linspace(0, np.pi/2, 100)
Q, Theta = np.meshgrid(q_values, theta_values)

# Functions to compute the q-concurrence and upper bound
def C_q(theta, q):
    return 1 - np.cos(theta)**(2*q) - 2**(1-q) * np.sin(theta)**(2*q)

def upper_bound(theta, q):
    return 1 - np.cos(theta)**(2*q)

# Compute the values
C_q_values = C_q(Theta, Q)
upper_bound_values = upper_bound(Theta, Q)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface plot for C_q
surf1 = ax.plot_surface(Q, Theta, C_q_values, color='blue', alpha=0.7, label='C_q(|Γ\')')

# Surface plot for upper bound
surf2 = ax.plot_surface(Q, Theta, upper_bound_values, color='green', alpha=0.3, label='Upper Bound')

# Labels
ax.set_xlabel('q')
ax.set_ylabel('θ')
ax.set_zlabel('C_q or Upper Bound')
ax.set_title('The q-concurrence and upper bounds of the superposition state')

# Add legend
legend1 = ax.legend([surf1], ['C_q(|Γ\')'], loc='upper left')
legend2 = ax.legend([surf2], ['Upper Bound'], loc='upper right')
ax.add_artist(legend1)


plt.savefig('plot_4.png')
plt.show()

# # Define the functions for the new example with corrections
# def C_q_Phi_example4(theta, q):
#     return 1 - np.cos(theta)**(2*q) - 2**(1-q) * np.sin(theta)**(2*q)

# def C_q_Psi_example4(phi, q):
#     return 1 - np.cos(phi)**(2*q) - 2**(1-q) * np.sin(phi)**(2*q)

# def C_q_Gamma_example4(theta, phi, q):
#     c_plus = np.sqrt(1 + np.sin(theta) * np.sin(phi))
#     c_minus = np.sqrt(1 - np.sin(theta) * np.sin(phi))
    
#     with np.errstate(divide='ignore', invalid='ignore'):
#         term_plus = (2**q * (np.cos(theta)**2 + np.cos(phi)**2)**q + 2 * (np.sin(theta) + np.sin(phi))**(2*q)) / (4 * c_plus**(2*q))
#         term_minus = (2**q * (np.cos(theta)**2 + np.cos(phi)**2)**q + 2 * (np.sin(theta) - np.sin(phi))**(2*q)) / (4 * c_minus**(2*q))
        
#         term_plus = np.nan_to_num(term_plus, nan=1.0, posinf=1.0, neginf=0.0)
#         term_minus = np.nan_to_num(term_minus, nan=1.0, posinf=1.0, neginf=0.0)
    
#     return 1 - term_plus, 1 - term_minus

# # Compute upper bound functions
# def F_q_rhoA(theta, phi, q):
#     return 1 - ((np.cos(theta)**2 + np.cos(phi)**2)**q / 2**q) - ((np.sin(theta)**2 + np.sin(phi)**2)**q / 2**(2*q-1))

# def F_q_rhoB(theta, phi, q):
#     return 1 - ((np.cos(theta)**2 + np.cos(phi)**2)**q / 2**q) - ((np.sin(theta)**2 + np.sin(phi)**2)**q / 2**(2*q-1))

# # Create a meshgrid for theta, phi, and q
# theta = np.linspace(0, np.pi/2, 50)
# phi = np.linspace(0, np.pi/2, 50)
# q_values = np.linspace(2, 4, 50)

# theta_mesh, phi_mesh, q_mesh = np.meshgrid(theta, phi, q_values, indexing='ij')

# # Compute the values for C_q(|Γ⟩) and the upper bound
# C_q_values_plus, C_q_values_minus = C_q_Gamma_example4(theta_mesh, phi_mesh, q_mesh)
# F_q_rhoA_values = F_q_rhoA(theta_mesh, phi_mesh, q_mesh)
# F_q_rhoB_values = F_q_rhoB(theta_mesh, phi_mesh, q_mesh)

# # Plotting the q-concurrence and upper bound
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot C_q(|Γ⟩)
# ax.plot_surface(q_mesh[:, :, 0], theta_mesh[:, :, 0], C_q_values_plus[:, :, 0], color='blue', alpha=0.7, label=r'$C_q(|\Gamma\rangle)$')
# ax.plot_surface(q_mesh[:, :, 0], theta_mesh[:, :, 0], C_q_values_minus[:, :, 0], color='blue', alpha=0.7)

# # Plot the upper bound
# ax.plot_surface(q_mesh[:, :, 0], theta_mesh[:, :, 0], F_q_rhoA_values[:, :, 0], color='green', alpha=0.3, label='upper bound')
# ax.plot_surface(q_mesh[:, :, 0], theta_mesh[:, :, 0], F_q_rhoB_values[:, :, 0], color='green', alpha=0.3)

# ax.set_xlabel(r'$q$')
# ax.set_ylabel(r'$\theta$')
# ax.set_zlabel(r'$C_q$ and upper bound')


# plt.savefig('plot_4.png')
# plt.show()