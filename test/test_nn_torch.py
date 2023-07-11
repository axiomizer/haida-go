from nn import neural_net as nn
import numpy as np
import torch
import unittest


class TestTorch(unittest.TestCase):
    def test_conv(self):
        input_channels = 7
        output_channels = 16
        board_size = 19

        torch_conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, 3, padding=1, dtype=torch.float64),
            torch.nn.ReLU()
        )

        class LastLayer:
            @staticmethod
            def sgd(a, pi, _):
                return [-2*(pi[i] - a[i]) for i in range(len(a))]

            @staticmethod
            def feedforward(in_activations):
                return in_activations

        my_conv = nn.ConvolutionalBlock(input_channels, output_channels)
        my_conv.kernels = torch.transpose(torch_conv[0].weight, 0, 1).detach().numpy()
        my_conv.biases = torch_conv[0].bias.detach().numpy()
        my_conv.to = LastLayer()

        np_in = np.random.randn(1, input_channels, board_size, board_size)
        torch_results = torch_conv(torch.tensor(np_in, dtype=torch.float64))
        my_results = my_conv.feedforward(np_in)

        self.assertTrue(np.allclose(torch_results.detach().numpy(), my_results))
