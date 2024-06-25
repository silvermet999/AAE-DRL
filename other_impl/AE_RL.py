import torch
from torch.nn import Module, Sequential, Linear, ReLU


class RandomSampler():
    def __init__(self, n):
        super(RandomSampler, self).__init__()
        pass


class AgentClf(Module):
    def __init__(self):
        super(AgentClf, self).__init__()
        self.model = Sequential(
            Linear(122, 100),
            ReLU(),
            Linear(100, 100),
            ReLU(),
            Linear(100, 100),
            ReLU(),
            Linear(100, 5),
            ReLU()

        )

    def forward(self, x):
        return self.model(x)



class AgentEnv(Module):
    def __init__(self):
        super(AgentEnv, self).__init__()
        self.model = Sequential(
            Linear(122, 100),
            ReLU(),
            Linear(100, 23),
            ReLU()

        )
    def forward(self, x):
        return self.model(x)