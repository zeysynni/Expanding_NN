import numpy as np
import torch


# This class implements a customized adam optimizer
class AdamOptim:
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, t, w, dw):
        # dw, db are from current minibatch
        # momentum beta 1
        # *** weights *** #
        if t == 1:
            self.m_dw = [self.m_dw * torch.ones(dw[i].shape) for i in range(len(dw))]
        self.m_dw = [self.beta1 * self.m_dw[i] + (1 - self.beta1) * dw[i] for i in range(len(dw))]

        # rms beta 2
        # *** weights *** #
        if t == 1:
            self.v_dw = [self.v_dw * torch.ones(dw[i].shape) for i in range(len(dw))]
        self.v_dw = [self.beta2 * self.v_dw[i] + (1 - self.beta2) * (dw[i] ** 2) for i in range(len(dw))]

        # bias correction
        m_dw_corr = [self.m_dw[i] / (1 - self.beta1 ** t) for i in range(len(dw))]
        v_dw_corr = [self.v_dw[i] / (1 - self.beta2 ** t) for i in range(len(dw))]

        # update weights and biases
        w = [(w[i] + self.eta * (m_dw_corr[i] / (np.sqrt(v_dw_corr[i]) + self.epsilon))).requires_grad_(True)
             for i in range(len(w))]
        return w

    def update_for_activation(self, t, w, dw):
        # dw, db are from current minibatch
        # momentum beta 1
        # *** weights *** #
        # only a 3-dim tensor
        if t == 1:
            self.m_dw = self.m_dw * torch.ones(dw.shape)
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        # rms beta 2
        # *** weights *** #
        if t == 1:
            self.v_dw = self.v_dw * torch.ones(dw.shape)
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)
        # bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** t)
        # update weights and biases
        w = (w + self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))).requires_grad_(True)
        return w

    @staticmethod
    def if_converge(w_old, w_new, index_old=None, index=None):
        if index_old is None:
            index_old = {}
        if index is None:
            index = {}
        if index:
            diff_index = sum([torch.norm((index_old[key] - index[key]), p="fro") for key in index.keys()])
            diff_w = sum([torch.norm((w_old[i] - w_new[i]), p="fro") for i in range(len(w_new))])
            return diff_w + diff_index
        else:
            return sum([torch.norm((w_old[i] - w_new[i]), p="fro") for i in range(len(w_new))])