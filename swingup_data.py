import gym
import gym_cartpole_swingup
import numpy as np
import torch
import copy
import math
# Create a CartPoleSwingUp-v1 environment.
# Collect episode sample to calculate v function for each state.
env = gym.make("CartPoleSwingUp-v1")


class CollectData(object):
    """Run the agent to sample trajectories. States and rewards will be recorded"""
    def __init__(self, trajectories, decay_for_v, seed):
        """
        Args:
            trajectories(int): Number of trajectories that should be done in one iteration.
            decay_for_v(float): Decay for future steps when calculate reward.
        """
        self.tra = trajectories
        self.decay = decay_for_v
        self.seed = seed

    def letsgobrandon(self, weights, nn, index, var):
        """
        Run the agent.

        Args:
            weights: Weights of the network.
            nn: Network.
            index: Parameters for expanded layer.
            var(float): External variance.
        Return:
            training_data(list[list]): Observations for all steps of all trajectories.
            rewards(list[list]):Rewards from all steps of all trajectories.
            actions(list[list]): Actions from all steps for all trajectories.
            mean(list[list]): Means of action distributions from all steps of all trajectories.
            variance(list[list]): Variances from all steps of all trajectories.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rewards = []  # List of list (received reward) for one trajectory after each step
        training_data = []  # training_data for states
        actions = []
        mean = []
        variance = []
        while not training_data:
            for i in range(self.tra):
                done = False
                pre_obs = []
                # Reward for current episode of each time step
                reward = []
                game_memo = []
                env.reset()
                while not done:
                    if len(pre_obs) != 0:
                        # Action ranges from -1 to 1
                        m = nn.forward(pre_obs, weights, index)
                        act = torch.normal(mean=float(m), std=nn.var+var)  # Sample current action
                        if act >= 1:
                            act = torch.tensor(1)
                        elif act <= -1:
                            act = torch.tensor(-1)
                        game_memo.append([pre_obs, act, m, float(nn.var+var)])
                        act = np.array(act)
                    else:
                        # At initial state of [0 0 cos(pi) sin(pi) 0], sample an action
                        act = env.action_space.sample()
                    # State transition
                    # observation, reward, if_traj_end, _ for run one action
                    obs, rew, done, info = env.step(act)
                    obs = torch.tensor(obs, dtype=torch.float32).reshape([1, 5])
                    if len(pre_obs) != 0:
                        reward.append(rew)
                    pre_obs = obs
                # Append the reward of each episode
                rewards.append(reward)
                # Append obs, sampled action, mean of action distribution, variance
                training_data.append([i[0] for i in game_memo])
                actions.append([i[1] for i in game_memo])
                mean.append([i[2] for i in game_memo])
                variance.append([i[3] for i in game_memo])
        return training_data, rewards, actions, mean, variance

    def train_data_v(self, training_data, rewards):
        """Calculate value function for each step in each episode"""
        v_func = []
        tem = np.ones([499, 1])
        for i in range(499):
            tem[i] = self.decay**i
        for epi in range(len(training_data)):
            epi_rew = (np.array(rewards[epi], dtype=object)).reshape([len(rewards[epi]), 1])
            epi_tem = tem[:len(rewards[epi])]
            for step in range(len(training_data[epi])):
                s = training_data[epi][step]
                re = epi_rew[step:]
                step_tem = epi_tem[:len(rewards[epi]) - step]
                v = torch.tensor(float(step_tem.T.dot(re)))
                v_func.append([s, v])
        return v_func

    @staticmethod
    def best_episodes(training_data, rewards, actions, mean, variance, part):
        """Pick episodes with best reward"""
        sum_each_epi = [sum(a) for a in rewards]
        t = copy.deepcopy(sum_each_epi)
        max_training_data = []
        max_reward = []
        max_actions = []
        max_mean = []
        max_variance = []
        max_index = []
        for _ in range(part):
            number = max(t)  # Best reward
            index = t.index(number)  # Corresponding index
            t[index] = 0
            max_training_data.append(training_data[index])
            max_reward.append(rewards[index])
            max_actions.append(actions[index])
            max_mean.append(mean[index])
            max_variance.append(variance[index])
            max_index.append(index)
        # t = []
        return max_training_data, max_reward, max_actions, max_mean, max_variance

    @staticmethod
    def save(rewards, tra, avr_reward, best, worst, m, upper, lower):
        best.append(max([sum(a) for a in rewards]))  # Best episode
        worst.append(min([sum(a) for a in rewards]))  # Worst episode
        avr = np.median([sum(a) for a in rewards])
        sum_each_epi = np.array([sum(a) for a in rewards])  # Total reward for each episode
        avr_reward.append(math.floor(avr*10)/10)
        m.append(np.median(sum_each_epi))
        sq = 0
        for i in sum_each_epi:
            sq += (i - np.mean(sum_each_epi))**2
        sigma = ((sq/len(sum_each_epi))**0.5)/tra**0.5
        upper.append(np.median(sum_each_epi)+2*sigma)
        lower.append(np.median(sum_each_epi)-2*sigma)
        np.save("m.npy", np.array(m))
        np.save("upper.npy", np.array(upper))
        np.save("lower.npy", np.array(lower))
        np.save("best.npy", np.array(best))
        np.save("worst.npy", np.array(worst))
        np.save("avr_reward.npy", np.array(avr_reward))
        return avr_reward, best, worst, avr, m, upper, lower

    @staticmethod
    def load_w():
        weights = torch.load("weights.pt")
        size = (np.load("size.npy")).tolist()
        v = torch.load("vless.pt")
        index = np.load("activation.npy", allow_pickle=True).item()
        #var = torch.load("var.pt")
        return weights, size, v, index#, var

    @staticmethod
    def save_weights(weights, v, size, index, var_best):
        torch.save(weights, "weights.pt")
        torch.save(v, "vless.pt")
        np.save("size.npy", np.array(size))
        np.save("activation", index)
        np.save("var", var_best)
        return
