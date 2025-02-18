import torch
import torch.nn.functional as functions
import torch.nn as nn
import random
#Network for value function estimation with a fix size


class ValueApprox(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_hidden3, n_output, epoch, learning_rate):
        super(ValueApprox, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.predict = torch.nn.Linear(n_hidden3, n_output)
        self.epo = epoch  # training epoch inside a regression training loop
        self.lr = learning_rate

    def forward(self, inputs):
        """
        Forward propagation.
        """
        out = self.hidden1(inputs)
        out = functions.relu(out)
        out = self.hidden2(out)
        out = functions.relu(out)
        out = self.hidden3(out)
        out = functions.relu(out)
        out = self.predict(out)
        return out
    
    def train_v(self, v, v_func):
        """
        One entire training process.
        """
        optimizer = torch.optim.Adam(v.parameters(), lr=self.lr)
        loss_func = torch.nn.MSELoss()
        for epoch in range(self.epo):
            for batch in v_func:
                x = torch.cat([a[0] for a in batch])
                y = torch.cat([a[1].view([1, 1]) for a in batch])
                prediction = v(x)
                loss = (loss_func(prediction, y))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        return

    @staticmethod
    def train_v_func_in_loop(v_func, v, threshold):
        """
        The regression training process will be repeated until the mse is small enough.
        """
        mse_test = 0
        mse_min = float("inf")
        e = 0
        random.shuffle(v_func)
        test = v_func[:int(0.2*len(v_func))]
        train = v_func[int(0.2*len(v_func)):]
        training_data = [train[a:a+512] for a in range(0, len(train)-len(train) % 512+1, 512)]
        if [] in training_data:
            training_data.remove([])
        while mse_min - mse_test >= threshold:
            if mse_min > mse_test and e != 0:
                mse_min = float(mse_test)
            v.train_v(v, training_data)
            x = torch.cat([a[0] for a in test])
            y = torch.cat([a[1].view([1, 1]) for a in test])
            prediction = v(x)
            mse_test = ((sum((prediction-y)**2))/len(y))**0.5
            e += 1
        return 
