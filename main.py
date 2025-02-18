import numpy as np
import torch
from swingup_data import CollectData  # sample episode
from mlp_expanding import MLP
from swingup_seeresult import show, diagram
from mlp_v import ValueApprox  # state value estimation
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore")

# This code does the learning of policy and expanding of the policy network during training.
# Only mean value of the action distribution is learned.
# Variance is determined externally


def if_reset(average, average_reward, thr, red=False, res=False, ft=False):
    """
    Decides whether you need to reduce, reset or finetune regarding lr and variance.

    Args:
        average: Average return of current iteration.
        average_reward(list): History of returns over iterations.
        thr:
        red:
        res:
        ft:
    :return:
    Check weather reduce or reset lr, in case certain amount of rewards are achieved, stop training
    """
    global count, monitor, count_1, l_rate, nn
    # sometimes it stays long at the beginning for some reason
    if average > 400:
        if not ft:
            l_rate = min(l_rate, 0.00001)
            # l_rate = 0.2 * l_rate
            nn.learning_rate = l_rate
            ft = True
    if average > thr[0] or monitor == 1:
        monitor = 1
        if average < np.max(average_reward):
            count += 1
        else:
            count = 0
        if count > thr[1]:
            red = True
            count = 0
            count_1 += 1
        if count_1 > thr[2] and not average > 400:
            res = True
            nn.reset_acc_grad()
            count_1 = 0
    return red, res, ft


seed = 1
l_rate = 1
size = [[5, 1], [1, 1]]  # Initial policy network size.
threshold_node, threshold_layer = 1.5, 1.5  # Threshold for expanding node & layer.
expanding_node, expanding_layer = True, True
act_para = torch.tensor((1., 0., 0.)).requires_grad_(True)  # Parameters for new activation function.
index = {}  # Activation function parameter for each new layer, {layer_index:[parameters]}.

start = 1
T = 300  # max iteration amount.

# Number of trajectories to make in each iteration.
tra = 100
lr_min = 0.000001  # Min. of lr.
threshold = [0, 8, 3]  # Threshold for setting lr: begin to count/reduce/reset.
decay = 0.7  # lr decay.
extern_var = var_best = 0.995  # Extern std, to be reduced during training.
var_decay = 0  # std decay factor
decay_value = 0.97  # Decay for calculating value function.
avr_reward, best, worst, m, upper, lower, avr_var = [[] for x in range(7)]
data_collector = CollectData(tra, decay_value, seed)  # Initialize a collector to sample trajectories.
nn = MLP(tra, l_rate, seed, size, variance=0.2)  # # Initialize the MLP network.
torch.manual_seed(seed)
weights = nn.weights()  # Initialize weights.
monitor, count, count_1 = 0, 0, 0  # Counter, for iterations
fine_tune, reset = False, False
min_var = []

# Preparation
if start > 1:
    # If not start from iteration 1, load the existing network and display it first.
    weights, size, v, index, var_best = data_collector.load_w()
    show(weights, MLP(tra, l_rate, seed, size, variance=.668), index)
else:
    # Otherwise define a network to estimate value function.
    v = ValueApprox(5, 124, 264, 16, 1, epoch=8, learning_rate=0.0002)


# Iterate until the max. iteration is reached.
for epo in range(start, T):
    print(epo)

    # Case neither the first iteration nor expected returns are achieved.
    if not reset and avr_reward != [] and max(avr_reward) < 410:
        var = extern_var ** var_decay
        var_decay += 1
    # Case of decay in rewards, restore the best variance that has been achieved.
    else:
        var_decay = var_best
        var = extern_var ** var_decay

    # Sample trajectories.
    training_data, R, actions, mean, variance = data_collector.letsgobrandon(weights, nn, index, var)
    print("extern std: ", var, "variance: ", float(nn.var + var))
    # Compute statistics regarding sampled trajectories.
    avr_reward, best, worst, avr, m, upper, lower = data_collector.save(R, tra, avr_reward, best, worst, m, upper,
                                                                        lower)

    # When a very good avr reward (manuel set threshold) is reached, save the variance.
    if avr > 415:
        min_var.append(nn.var + var)
        show(weights, nn, index)
    print("average reward", avr)
    print(avr_reward)

    # Compute the value function
    v_func = data_collector.train_data_v(training_data, R)
    # rain the value function estimator
    v.train_v_func_in_loop(v_func, v, 0.002)  # threshold
    print("used episodes", len(training_data))
    # if the current avr reward is better than before, save the network and var
    if avr > max(avr_reward):
        print("save policy")
        data_collector.save_weights(weights, v, size, index, var_best)
        var_best = deepcopy(var_decay)
        print(size)
    # check the training state
    reduce, reset, fine_tune = if_reset(avr, avr_reward, threshold, ft=fine_tune)
    # reset: fall back to the last best model
    if reset and avr < 400:
        weights, size, v, index = data_collector.load_w()
        var_decay = var_best
        var = extern_var ** var_decay
        print("reset both")
    # reduce: decrease lr & decide whether to expand the width and depth of the network
    elif reduce:
        l_rate = max(decay * l_rate, lr_min)
        nn.learning_rate = l_rate
        print("decrease learning rate")
        old_size = deepcopy(size)
        if expanding_node and len(size) > 1 and avr < 410:
            weights, size = nn.choose_node(weights, nn, size, threshold_node, index, v, training_data, R, actions,
                                           variance)
        if expanding_layer and old_size == size and avr < 410:
            weights, size, index = nn.choose_layer(weights, nn, size, threshold_layer, index, v,
                                                   training_data, R, actions, variance, act_para)
    # otherwise, simply train new weights
    else:
        weights, index = nn.backward_ng2_fast(training_data, actions, R, weights, v, variance, index)
    print("best trajectory:", best[epo - start],
          "worst trajectory:", worst[epo - start],
          "size:", size,
          "index:", index)
print(max(avr_reward), "\n", avr_reward.index(max(avr_reward)) + 1)

# when a desired good reward is reached, display the corresponding variance
if min_var:
    print("var: ", min(min_var))
# show result and save
diagram(T, m, upper, lower)
# render
show(weights, MLP(tra, l_rate, seed, size, variance=extern_var ** var_decay), index)
