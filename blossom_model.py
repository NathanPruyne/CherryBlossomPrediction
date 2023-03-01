import functools

import torch

class Convolution(torch.nn.Sequential):

    def __init__(
        self,
        input_channels=3,
        output_channels=1,
        hidden_channels=8,
        kernel_size=5,
        num_convs=4):
        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')
        super().__init__()
        #Generate convolutional layers up to the designated number of layers
        if num_convs == 1:
            self.add_module("conv_1", conv_fn(input_channels, output_channels))
        else:
            self.add_module("conv_1", conv_fn(input_channels, hidden_channels))
            self.add_module("relu_1", torch.nn.ReLU())
            for i in range(1, num_convs - 1):
                self.add_module(f"conv_{i + 1}", conv_fn(hidden_channels, hidden_channels))
                self.add_module(f"relu_{i + 1}", torch.nn.ReLU())
            self.add_module(f"conv_{num_convs}", conv_fn(hidden_channels, output_channels))

    def forward(self, features, *_):
        return super().forward(features)
        