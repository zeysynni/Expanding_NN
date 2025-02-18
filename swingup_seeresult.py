import gym
import gym_cartpole_swingup
import numpy as np
import torch
import matplotlib.pyplot as plt
# Shows animation and plot median returns at the end of each policy update
env = gym.make("CartPoleSwingUp-v1")


def show(weights, nn, index):
    """Show animation."""
    for each_game in range(1):
        prev_obs = []
        done = False
        env.reset()
        new_observation = None
        while not done:
            env.render()
            if len(prev_obs) == 0:
                action = env.action_space.sample()
            else:
                prev_obs = torch.tensor(new_observation, dtype=torch.float32).reshape([1, 5])
                m = nn.forward(prev_obs, weights, index)
                action = np.array(torch.normal(mean=float(m), std=torch.tensor(0.)))
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation


def diagram(length, m, upper, lower):
    """Plot return changes over iterations."""
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    m = np.array(m)
    upper = np.array(upper)
    lower = np.array(lower)
    np.save("m.npy", m)
    np.save("upper.npy", upper)
    np.save("lower.npy", lower)

    ax1.plot(range(0, length - 1), m)
    plt.show()
    return
