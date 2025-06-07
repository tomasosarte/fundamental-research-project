import torch as th

class MLP(th.nn.Module):

    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], output_size=10):
        
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(th.nn.Linear(in_size, hidden_size))
            layers.append(th.nn.ReLU())
            in_size = hidden_size
        
        layers.append(th.nn.Linear(in_size, output_size))
        
        self.model = th.nn.Sequential(*layers)

    def forward(self, x: th.Tensor):
        return self.model(x.view(x.size(0), -1))