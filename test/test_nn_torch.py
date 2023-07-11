from nn import neural_net as nn
import numpy as np
import torch
import hyperparams as hp
import unittest


class TestTorch(unittest.TestCase):
    def test_conv(self):
        input_channels = 7
        output_channels = 16
        board_size = 19
        minibatch_size = 4

        torch_conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, 3, padding=1, dtype=torch.float64),
            torch.nn.ReLU()
        )

        class LastLayer:
            @staticmethod
            def sgd(a, pi, _):
                return [(-2 / np.array(pi[i]).size) * (pi[i] - a[i]) for i in range(len(a))]

            @staticmethod
            def feedforward(in_activations):
                return in_activations

        my_conv = nn.ConvolutionalBlock(input_channels, output_channels)
        my_conv.kernels = np.copy(torch.transpose(torch_conv[0].weight, 0, 1).detach().numpy())
        my_conv.biases = np.copy(torch_conv[0].bias.detach().numpy())
        my_conv.to = LastLayer()

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(minibatch_size, input_channels, board_size, board_size)
        torch_results = torch_conv(torch.tensor(np_in, dtype=torch.float64))
        my_results = my_conv.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results.detach().numpy(), my_results))

        # do one step of SGD for both nets and compare results
        np_target = np.random.randn(minibatch_size, output_channels, board_size, board_size)
        loss = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_conv.parameters(), lr=hp.LEARNING_RATE)
        optimizer.zero_grad()
        loss(torch_results, torch.tensor(np_target, dtype=torch.float64)).backward()
        optimizer.step()
        updated_torch_weights = torch.transpose(torch_conv[0].weight, 0, 1).detach().numpy()
        my_conv.sgd(np_in, np_target, None)
        self.assertTrue(np.allclose(updated_torch_weights, my_conv.kernels))
        updated_torch_biases = torch_conv[0].bias.detach().numpy()
        self.assertTrue(np.allclose(updated_torch_biases, my_conv.biases))
