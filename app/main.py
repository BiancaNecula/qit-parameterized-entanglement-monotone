import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, fractional_matrix_power
from itertools import product
import qutip

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

def C_q(theta, q):
    return 1 - np.cos(theta)**(2*q) - 2**(1-q) * np.sin(theta)**(2*q)

def upper_bound(theta, q):
    return 1 - np.cos(theta)**(2*q)

C_q_values = C_q(Theta, Q)
upper_bound_values = upper_bound(Theta, Q)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf1 = ax.plot_surface(Q, Theta, C_q_values, color='blue', alpha=0.7, label='C_q(|Γ\')')

surf2 = ax.plot_surface(Q, Theta, upper_bound_values, color='green', alpha=0.3, label='Upper Bound')

ax.set_xlabel('q')
ax.set_ylabel('θ')
ax.set_zlabel('C_q or Upper Bound')
ax.set_title('The q-concurrence and upper bounds of the superposition state')

legend1 = ax.legend([surf1], ['C_q(|Γ\')'], loc='upper left')
legend2 = ax.legend([surf2], ['Upper Bound'], loc='upper right')
ax.add_artist(legend1)


plt.savefig('plot_4.png')
plt.show()


def pauli_matrices():
    """Return the Pauli matrices as a list."""
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    return [I, X, Y, Z]

def kron_n(matrices):
    """Return the Kronecker product of a list of matrices."""
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result

def specific_heisenberg_hamiltonian(n, h):
    """Construct the Heisenberg Hamiltonian for specific qubit pairs."""
    _, X, Y, Z = pauli_matrices()
    H = np.zeros((2**n, 2**n), dtype=complex)
    
    pairs = {
        5: [(0, 4), (0, 2), (1, 3), (2, 4)],
        6: [(0, 4), (0, 2), (1, 3), (3, 5), (1, 5)],
        7: [(0, 4), (0, 2), (1, 3), (3, 5), (1, 5), (4, 6)]
    }
    
    weights = {
        5: [5, 2, 4, 6],
        6: [5, 2, 4, 1, 6],
        7: [5, 2, 4, 1, 6, 3]
    }
    
    for i, (p1, p2) in enumerate(pairs[n]):
        H += weights[n][i] * (kron_n([Z if k == p1 or k == p2 else np.eye(2) for k in range(n)]) +
                              kron_n([X if k == p1 or k == p2 else np.eye(2) for k in range(n)]) +
                              kron_n([Y if k == p1 or k == p2 else np.eye(2) for k in range(n)]))
    
    for i in range(n):
        H += h[i] * kron_n([Z if k == i else np.eye(2) for k in range(n)])
    
    return H

def plus_state(n):
    """Generate the |+> state for n qubits."""
    plus = np.array([1, 1]) / np.sqrt(2)
    state = plus
    for _ in range(n-1):
        state = np.kron(state, plus)
    return state

def density_matrix(state):
    """Create the density matrix from a state vector."""
    return np.outer(state, state.conj())

def reduced_density_matrix(rho, keep):
    """Calculate the reduced density matrix by tracing out all qubits except the ones in 'keep'."""
    total_qubits = int(np.log2(rho.shape[0]))
    dims = [2] * total_qubits
    rho_qobj = qutip.Qobj(rho, dims=[dims, dims])
    traced_rho = rho_qobj.ptrace(keep)
    return traced_rho.full()

def q_concurrence(rho_A, q):
    """Calculate the q-concurrence."""
    return 1 - np.trace(fractional_matrix_power(rho_A, q))

def tsallis_q_entanglement(rho, q):
    """Calculate the Tsallis-q entanglement."""
    return (1 - np.trace(np.linalg.matrix_power(rho, q))) / (q - 1)

def evolve_state(state, H, t):
    """Evolve the state under the Hamiltonian H for time t."""
    U = expm(-1j * H * t)
    return U @ state

# Parameters
n_values = [5, 6, 7]
q_values = np.arange(2, 21)
evolution_time = 500

# Initialize results
C_q_results = {n: [] for n in n_values}
T_q_results = {n: [] for n in n_values}

# Calculate for each n
for n in n_values:
    h = np.random.uniform(-10, 10, size=n)
    H = specific_heisenberg_hamiltonian(n, h)
    state = plus_state(n) 
    
    evolved_state = evolve_state(state, H, evolution_time)
    rho = density_matrix(evolved_state)
    
    rho_A = reduced_density_matrix(rho, list(range(n//2)))
    
    for q in q_values:
        C_q_results[n].append(q_concurrence(rho_A, q))
        T_q_results[n].append(tsallis_q_entanglement(rho_A, q))

plt.figure(figsize=(10, 6))
markers = ['*', 'd', 's']
colors = ['r', 'g', 'b']
lines = ['-', '--']
labels = [r'$C_q, n=5$', r'$T_q, n=5$', r'$C_q, n=6$', r'$T_q, n=6$', r'$C_q, n=7$', r'$T_q, n=7$']

for i, n in enumerate(n_values):
    plt.plot(q_values, C_q_results[n], color=colors[i], marker=markers[i], linestyle='-', label=labels[2*i])
    plt.plot(q_values, T_q_results[n], color=colors[i], marker=markers[i], linestyle='--', label=labels[2*i+1])

plt.xlabel(r'$q$')
plt.ylabel('Value')
plt.title('Comparison of the $q$-concurrence and Tsallis-$q$ entanglement in terms of $q$')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('plot_5.png')
plt.show()

n = 5  
q_values = np.arange(2, 21)
t_values = np.linspace(0, 500, 100)


C_q_heatmap = np.zeros((len(t_values), len(q_values)))
T_q_heatmap = np.zeros((len(t_values), len(q_values)))

h = np.random.uniform(-10, 10, size=n)
H = specific_heisenberg_hamiltonian(n, h)
state = plus_state(n)  # Use the |+> state for initialization

for t_idx, t in enumerate(t_values):
    evolved_state = evolve_state(state, H, t)
    rho = density_matrix(evolved_state)
    rho_A = reduced_density_matrix(rho, list(range(n//2)))
    
    for q_idx, q in enumerate(q_values):
        C_q_heatmap[t_idx, q_idx] = q_concurrence(rho_A, q)
        T_q_heatmap[t_idx, q_idx] = tsallis_q_entanglement(rho_A, q)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

c1 = ax[0].imshow(C_q_heatmap, aspect='auto', cmap='gray', extent=[q_values.min(), q_values.max(), t_values.min(), t_values.max()])
ax[0].set_xlabel(r'$q$')
ax[0].set_ylabel(r'$t$')
ax[0].set_title(r'$q$-concurrence Heatmap')
fig.colorbar(c1, ax=ax[0])

c2 = ax[1].imshow(T_q_heatmap, aspect='auto', cmap='gray', extent=[q_values.min(), q_values.max(), t_values.min(), t_values.max()])
ax[1].set_xlabel(r'$q$')
ax[1].set_ylabel(r'$t$')
ax[1].set_title(r'Tsallis-$q$ Entanglement Heatmap')
fig.colorbar(c2, ax=ax[1])

plt.savefig('plot_6.png')
plt.show()
