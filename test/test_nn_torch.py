from nn import neural_net as nn
import numpy as np
import torch
import hyperparams as hp
import unittest


class LastLayer:
    @staticmethod
    def sgd(a, pi, _):
        return [(-2 / (pi[i].size * len(a))) * (pi[i] - a[i]) for i in range(len(a))]

    @staticmethod
    def feedforward(in_activations):
        return in_activations


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

        my_conv = nn.ConvolutionalBlock(input_channels, output_channels)
        my_conv.kernels = np.copy(torch.transpose(torch_conv[0].weight, 0, 1).detach().numpy())
        my_conv.biases = np.copy(torch_conv[0].bias.detach().numpy())
        my_conv.to = LastLayer()

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(minibatch_size, input_channels, board_size, board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_conv(torch_in)
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
        my_input_grads = my_conv.sgd(np_in, np_target, None)
        self.assertTrue(np.allclose(updated_torch_weights, my_conv.kernels))
        updated_torch_biases = torch_conv[0].bias.detach().numpy()
        self.assertTrue(np.allclose(updated_torch_biases, my_conv.biases))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, my_input_grads))

    def test_res(self):
        channels = 8
        board_size = 19
        minibatch_size = 4

        class TorchResidualBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1, dtype=torch.float64)
                self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1, dtype=torch.float64)

            def forward(self, x):
                skip = x
                x = torch.nn.functional.relu(self.conv1(x))
                return torch.nn.functional.relu(self.conv2(x) + skip)

        torch_res = TorchResidualBlock()

        my_res = nn.ResidualBlock(channels)
        my_res.kernels1 = np.copy(torch.transpose(torch_res.conv1.weight, 0, 1).detach().numpy())
        my_res.biases1 = np.copy(torch_res.conv1.bias.detach().numpy())
        my_res.kernels2 = np.copy(torch.transpose(torch_res.conv2.weight, 0, 1).detach().numpy())
        my_res.biases2 = np.copy(torch_res.conv2.bias.detach().numpy())
        my_res.to = [LastLayer()]

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(minibatch_size, channels, board_size, board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_res(torch_in)
        my_results = my_res.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results.detach().numpy(), my_results))

        # do one step of SGD for both nets and compare results
        np_target = np.random.randn(minibatch_size, channels, board_size, board_size)
        loss = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_res.parameters(), lr=hp.LEARNING_RATE)
        optimizer.zero_grad()
        loss(torch_results, torch.tensor(np_target, dtype=torch.float64)).backward()
        optimizer.step()
        updated_torch_weights1 = torch.transpose(torch_res.conv1.weight, 0, 1).detach().numpy()
        updated_torch_biases1 = torch_res.conv1.bias.detach().numpy()
        updated_torch_weights2 = torch.transpose(torch_res.conv2.weight, 0, 1).detach().numpy()
        updated_torch_biases2 = torch_res.conv2.bias.detach().numpy()
        my_input_grads = my_res.sgd(np_in, np_target, None)
        self.assertTrue(np.allclose(updated_torch_weights1, my_res.kernels1))
        self.assertTrue(np.allclose(updated_torch_biases1, my_res.biases1))
        self.assertTrue(np.allclose(updated_torch_weights2, my_res.kernels2))
        self.assertTrue(np.allclose(updated_torch_biases2, my_res.biases2))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, my_input_grads))

    def test_pol(self):
        input_channels = 16
        board_size = 19
        minibatch_size = 4

        torch_pol = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 2, 1, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear((board_size ** 2) * 2, (board_size ** 2) + 1, dtype=torch.float64)
        )

        my_pol = nn.PolicyHead(in_filters=input_channels, board_size=board_size)
        my_pol.kernels = np.copy(torch.transpose(torch_pol[0].weight, 0, 1).detach().numpy())
        my_pol.biases1 = np.copy(torch_pol[0].bias.detach().numpy())
        my_pol.weights = np.copy(torch_pol[3].weight.detach().numpy())
        my_pol.biases2 = np.copy(torch_pol[3].bias.detach().numpy())

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(minibatch_size, input_channels, board_size, board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_pol(torch_in)
        my_results = my_pol.feedforward(np_in, raw=True)
        self.assertTrue(np.allclose(torch_results.detach().numpy(), my_results))

        # do one step of SGD for both nets and compare results
        raw_target = torch.randn(minibatch_size, (board_size ** 2) + 1, dtype=torch.float64)
        target = torch.nn.functional.softmax(raw_target, dim=1, dtype=torch.float64)
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(torch_pol.parameters(), lr=hp.LEARNING_RATE)
        optimizer.zero_grad()
        loss(torch_results, target).backward()
        optimizer.step()
        updated_torch_weights1 = torch.transpose(torch_pol[0].weight, 0, 1).detach().numpy()
        updated_torch_biases1 = torch_pol[0].bias.detach().numpy()
        updated_torch_weights2 = torch_pol[3].weight.detach().numpy()
        updated_torch_biases2 = torch_pol[3].bias.detach().numpy()
        my_input_grads = my_pol.sgd(np_in, target.detach().numpy(), None)
        self.assertTrue(np.allclose(updated_torch_weights1, my_pol.kernels))
        self.assertTrue(np.allclose(updated_torch_biases1, my_pol.biases1))
        self.assertTrue(np.allclose(updated_torch_weights2, my_pol.weights))
        self.assertTrue(np.allclose(updated_torch_biases2, my_pol.biases2))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, my_input_grads))

    def test_val(self):
        input_channels = 16
        board_size = 19
        minibatch_size = 4

        torch_val = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 1, 1, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(board_size ** 2, 256, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1, dtype=torch.float64),
            torch.nn.Tanh()
        )

        my_val = nn.ValueHead(in_filters=input_channels, board_size=board_size)
        my_val.l1_kernels = np.ndarray.flatten(torch_val[0].weight.detach().numpy())
        my_val.l1_bias = torch_val[0].bias.detach().numpy()[0]
        my_val.l2_weights = np.copy(torch_val[3].weight.detach().numpy())
        my_val.l2_biases = np.copy(torch_val[3].bias.detach().numpy())
        my_val.l3_weights = np.ndarray.flatten(torch_val[5].weight.detach().numpy())
        my_val.l3_bias = torch_val[5].bias.detach().numpy()[0]

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(minibatch_size, input_channels, board_size, board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_val(torch_in)
        my_results = my_val.feedforward(np_in)
        self.assertTrue(np.allclose(np.ndarray.flatten(torch_results.detach().numpy()), my_results))

        # do one step of SGD for both nets and compare results
        target = torch.randn(minibatch_size, 1, dtype=torch.float64)
        loss = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_val.parameters(), lr=hp.LEARNING_RATE)
        optimizer.zero_grad()
        loss(torch_results, target).backward()
        optimizer.step()
        updated_torch_weights1 = np.ndarray.flatten(torch_val[0].weight.detach().numpy())
        updated_torch_bias1 = torch_val[0].bias.detach().numpy()[0]
        updated_torch_weights2 = torch_val[3].weight.detach().numpy()
        updated_torch_biases2 = torch_val[3].bias.detach().numpy()
        updated_torch_weights3 = np.ndarray.flatten(torch_val[5].weight.detach().numpy())
        updated_torch_bias3 = torch_val[5].bias.detach().numpy()[0]
        my_input_grads = my_val.sgd(np_in, None, np.ndarray.flatten(target.detach().numpy()))
        self.assertTrue(np.allclose(updated_torch_weights1, my_val.l1_kernels))
        self.assertTrue(np.allclose(updated_torch_bias1, my_val.l1_bias))
        self.assertTrue(np.allclose(updated_torch_weights2, my_val.l2_weights))
        self.assertTrue(np.allclose(updated_torch_biases2, my_val.l2_biases))
        self.assertTrue(np.allclose(updated_torch_weights3, my_val.l3_weights))
        self.assertTrue(np.allclose(updated_torch_bias3, my_val.l3_bias))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, my_input_grads))
