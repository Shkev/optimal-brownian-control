"""Animation of Brownian Motion (BM) as controlled by a banded control policy
to minimize average long-run holding cost and adjustment costs."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters for base BM process
N = 1000  # Number of steps
T = 1.0  # Total time
dt = T / N  # Time step size
sigma = 1  # vol of BM
mu = 0  # drift

# Time array
t = np.linspace(0.0, T, N)

# adjustment cost params
k = 0.1  # proportional upward adjustment cost
K = 1  # fixed per upward adjustment cost
l = 0.1  # proportional down adjustment cost
L = 1  # fixed per down adjustment cost

# holding cost function (convex)
h = lambda x: x**2

# Control parameters (need to compute these as soln to free boundary problem)
u, U, D, d = 0.5, 0.2, -0.2, -0.5


# the Brownian motion (updated in the `animate` function below)
W = np.zeros(N)

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim((0, T))
ax.set_ylim(d - 0.2, u + 0.2)
(line,) = ax.plot([], [], lw=2)

# also display/update adjustment and holding costs in the animation
adj_cost = np.zeros_like(W)
holding_cost = np.zeros_like(W)
adj_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
hold_text = ax.text(0.02, 0.90, "", transform=ax.transAxes)


# Initialization function: plot the background of each frame
def init():
    """Initialize animation frame and related data elements."""
    line.set_data([], [])

    # control policy bands
    ax.axhline(y=u, color="r", linestyle="-")
    ax.axhline(y=d, color="r", linestyle="-")
    ax.axhline(y=U, color="b", linestyle="--")
    ax.axhline(y=D, color="b", linestyle="--")

    # start fresh, zero them out
    adj_cost[:] = 0
    holding_cost[:] = 0
    W[:] = 0

    adj_text.set_text("Adjustment Cost: 0")
    hold_text.set_text("Holding Cost: 0")

    return line, adj_text, hold_text


def animate(i):
    """Animation function: update the data for each frame"""
    dW = sigma * np.random.normal(0, np.sqrt(dt)) + mu * dt
    W[i] = W[i - 1] + dW
    adj_cost[i] = adj_cost[i - 1]

    # if hits upper band, adjust down
    if W[i] >= u:
        W[i] = U
        adj_cost[i] += L + (u - U) * l
    # if hits lower band, adjust up
    if W[i] <= d:
        W[i] = D
        adj_cost[i] += K + (u - U) * k

    # compute holding cost post-adjustment
    holding_cost[i] = holding_cost[i - 1] + h(W[i])

    line.set_data(t[:i], W[:i])
    adj_text.set_text(f"Adjustment Cost: {adj_cost[i]:.2f}")
    hold_text.set_text(f"Holding Cost: {holding_cost[i]:.2f}")

    return line, adj_text, hold_text


# Create the animation
ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=N, interval=20, repeat_delay=2000, blit=True
)

ani.save("bm_control_animation.gif", writer="imagemagik", fps=30)
