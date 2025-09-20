#Q3: Overdamped case:
#Change the damping value from 200 N/s to 20000 N/s.

import numpy as np
import matplotlib.pyplot as plt

# Given data
m = 100  # kg
Tn = 0.62832  # sec (natural period)
zeta = 10  # damping ratio, overdamped
fn = 1 / Tn  # natural frequency (Hz)
omega_n = 2 * np.pi * fn  # natural angular frequency
k = m * omega_n**2  # stiffness
c = (2 * zeta * np.sqrt(k * m))  # damping coefficient

# Newmark-beta parameters for constant acceleration
beta = 1/4
gamma = 1/2
# Time step and total simulation time
dt = 0.01  # time step (should be small for accuracy)
t_max = 40  # simulation duration
time = np.arange(0, t_max+dt, dt)  # Include the last step
n = len(time)

# Initialization
u = np.zeros(n)  # displacement
v = np.zeros(n)  # velocity
a = np.zeros(n)  # acceleration
p = np.zeros(n)  # force

# Set initial conditions
u[0] = 0.02
v[0] = 0
a[0] =((c*v[0])-(k*u[0]))/m

# Initial conditions
a[0] = (- c * v[0] - k * u[0]) / m
keff = k + (m / (beta * dt**2)) + (gamma * c) / (beta * dt)
a_coeff = (m / (beta * dt)) + (gamma * c / beta)
b_coeff = (0.5 * m / beta) + (dt * ((gamma / (2 * beta)) - 1) * c)

for i in range(n - 1):
    # Effective force
    p_eff =  m*(u[i]/(beta*dt**2) + v[i]/(beta*dt) + (1/(2*beta)-1)*a[i]) + c*(gamma*u[i]/(beta*dt) + (gamma/beta - 1)*v[i] + dt*(gamma/(2*beta) - 1)*a[i])

    # Solve for displacement
    u[i+1] = p_eff / keff

    # Update acceleration
    a[i+1] = ((u[i+1] - u[i])/(beta*dt**2)) - (v[i]/(beta*dt)) - ((1/(2*beta))-1)*a[i]

    # Update velocity
    v[i+1] = v[i] + dt*((1-gamma)*a[i] + gamma*a[i+1])

# Plot displacement vs. time
plt.figure(figsize=(9, 4))
plt.plot(time, u, label="Displacement (m)", color='b')
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Displacement vs Time (Newmark-beta Method)")
plt.legend()
plt.grid()
plt.margins(0)  
plt.show()
