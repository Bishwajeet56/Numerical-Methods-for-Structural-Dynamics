#Q4: A sinusoidal acceleration xg_dot_dot  = sin (2t) is applied to a mass of 100 kg. 
# The mass is connected with a spring of stiffness 10 kN/m and viscous damper of damping 200 Ns/m.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Given data
m = 100  # kg
Tn = 0.62832  # sec (natural period)
zeta = 0.1  # damping ratio
fn = 1 / Tn  # natural frequency (Hz)
omega_n = 2 * np.pi * fn  # natural angular frequency
k = m * omega_n**2  # stiffness
c = (2 * zeta * np.sqrt(k * m))  # damping coefficient

# Newmark-beta parameters for constant acceleration
beta = 1/4
gamma = 1/2

# Time step and total simulation time
dt = 0.01  # time step (should be small for accuracy)
t_max = 5  # simulation duration
time = np.arange(0, t_max+dt, dt)  # Include the last step
n = len(time)

# Initialization
u = np.zeros(n)  # displacement
v = np.zeros(n)  # velocity
a = np.zeros(n)  # acceleration
f = np.zeros(n)  # force

# Base acceleration ag(t) = sin(2*pi*t)  -> external term f(t) = -m*ag(t)
ag = np.sin(2*np.pi*time)
f  = -m*ag

# Set initial conditions
u[0] = 0
v[0] = 0
a[0] =(f[0]-(c*v[0])-(k*u[0]))/m

# Initial conditions
a[0] = (f[0]- c * v[0] - k * u[0]) / m
keff = k + (m / (beta * dt**2)) + (gamma * c) / (beta * dt)
a_coeff = (m / (beta * dt)) + (gamma * c / beta)
b_coeff = (0.5 * m / beta) + (dt * ((gamma / (2 * beta)) - 1) * c)

# Time stepping using Newmark-beta
for i in range(n - 1):
    # Effective force
    p_eff = f[i+1] + m*(u[i]/(beta*dt**2) + v[i]/(beta*dt) + (1/(2*beta)-1)*a[i]) + c*(gamma*u[i]/(beta*dt) + (gamma/beta - 1)*v[i] + dt*(gamma/(2*beta) - 1)*a[i])

    # Solve for displacement
    u[i+1] = p_eff / keff

    # Update acceleration
    a[i+1] = ((u[i+1] - u[i])/(beta*dt**2)) - (v[i]/(beta*dt)) - ((1/(2*beta))-1)*a[i]

    # Update velocity
    v[i+1] = v[i] + dt*((1-gamma)*a[i] + gamma*a[i+1])

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, u, label="Displacement (m)", color='b')
plt.ylabel("u (m)")
#plt.ylim(disp_limits)   # custom y-limits
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, v, label="Velocity (m/s)", color='g')
plt.ylabel("v (m/s)")
#plt.ylim(vel_limits)    # custom y-limits
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, a, label="Acceleration (m/s²)", color='r')
plt.xlabel("Time (s)")
plt.ylabel("a (m/s²)")
#plt.ylim(acc_limits)    # custom y-limits
plt.grid()
plt.legend()

plt.suptitle("Dynamic Response (Newmark-beta Method)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

