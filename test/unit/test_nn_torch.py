from src.nn.neural_net import ConvolutionalBlock, ResidualBlock, PolicyHead, ValueHead, NeuralNet
from test.mse_stub import MseStub
import numpy as np
import torch
import src.nn.hyperparams as hp
import unittest


def generate_torch_conv_block(input_channels, output_channels):
    return torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, 3, padding=1, dtype=torch.float64),
            torch.nn.ReLU()
        )


class TorchResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1, dtype=torch.float64)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1, dtype=torch.float64)

    def forward(self, x):
        skip = x
        x = torch.nn.functional.relu(self.conv1(x))
        return torch.nn.functional.relu(self.conv2(x) + skip)


def generate_torch_pol_head(input_channels, board_size):
    return torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 2, 1, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear((board_size ** 2) * 2, (board_size ** 2) + 1, dtype=torch.float64)
        )


def generate_torch_val_head(input_channels, board_size):
    return torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 1, 1, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(board_size ** 2, 256, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1, dtype=torch.float64),
            torch.nn.Tanh()
        )


def copy_conv(torch_conv, haida_conv):
    haida_conv.kernels = np.copy(torch.transpose(torch_conv[0].weight, 0, 1).detach().numpy())
    haida_conv.biases = np.copy(torch_conv[0].bias.detach().numpy())


def copy_res(torch_res, haida_res):
    haida_res.kernels1 = np.copy(torch.transpose(torch_res.conv1.weight, 0, 1).detach().numpy())
    haida_res.biases1 = np.copy(torch_res.conv1.bias.detach().numpy())
    haida_res.kernels2 = np.copy(torch.transpose(torch_res.conv2.weight, 0, 1).detach().numpy())
    haida_res.biases2 = np.copy(torch_res.conv2.bias.detach().numpy())


def copy_pol(torch_pol, haida_pol):
    haida_pol.kernels = np.copy(torch.transpose(torch_pol[0].weight, 0, 1).detach().numpy())
    haida_pol.biases1 = np.copy(torch_pol[0].bias.detach().numpy())
    haida_pol.weights = np.copy(torch_pol[3].weight.detach().numpy())
    haida_pol.biases2 = np.copy(torch_pol[3].bias.detach().numpy())


def copy_val(torch_val, haida_val):
    haida_val.l1_kernels = np.ndarray.flatten(torch_val[0].weight.detach().numpy())
    haida_val.l1_bias = torch_val[0].bias.detach().numpy()[0]
    haida_val.l2_weights = np.copy(torch_val[3].weight.detach().numpy())
    haida_val.l2_biases = np.copy(torch_val[3].bias.detach().numpy())
    haida_val.l3_weights = np.ndarray.flatten(torch_val[5].weight.detach().numpy())
    haida_val.l3_bias = torch_val[5].bias.detach().numpy()[0]


class TestTorch(unittest.TestCase):
    def test_conv(self):
        input_channels = 7
        output_channels = 16
        board_size = 19
        minibatch_size = 4

        torch_conv = generate_torch_conv_block(input_channels, output_channels)
        haida_conv = ConvolutionalBlock(input_channels, output_channels)
        copy_conv(torch_conv, haida_conv)
        haida_conv.to = [MseStub()]

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(minibatch_size, input_channels, board_size, board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_conv(torch_in)
        haida_results = haida_conv.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results.detach().numpy(), haida_results))

        # do one step of SGD for both nets and compare results
        np_target = np.random.randn(minibatch_size, output_channels, board_size, board_size)
        loss = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_conv.parameters(), lr=hp.LEARNING_RATE)
        optimizer.zero_grad()
        loss(torch_results, torch.tensor(np_target, dtype=torch.float64)).backward()
        optimizer.step()
        updated_torch_weights = torch.transpose(torch_conv[0].weight, 0, 1).detach().numpy()
        haida_input_grads = haida_conv.sgd(np_in, np_target, None)
        self.assertTrue(np.allclose(updated_torch_weights, haida_conv.kernels))
        updated_torch_biases = torch_conv[0].bias.detach().numpy()
        self.assertTrue(np.allclose(updated_torch_biases, haida_conv.biases))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_res(self):
        channels = 8
        board_size = 19
        minibatch_size = 4

        torch_res = TorchResidualBlock(channels)
        haida_res = ResidualBlock(channels)
        copy_res(torch_res, haida_res)
        haida_res.to = [MseStub()]

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(minibatch_size, channels, board_size, board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_res(torch_in)
        haida_results = haida_res.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results.detach().numpy(), haida_results))

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
        haida_input_grads = haida_res.sgd(np_in, np_target, None)
        self.assertTrue(np.allclose(updated_torch_weights1, haida_res.kernels1))
        self.assertTrue(np.allclose(updated_torch_biases1, haida_res.biases1))
        self.assertTrue(np.allclose(updated_torch_weights2, haida_res.kernels2))
        self.assertTrue(np.allclose(updated_torch_biases2, haida_res.biases2))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_pol(self):
        input_channels = 16
        board_size = 19
        minibatch_size = 4

        torch_pol = generate_torch_pol_head(input_channels, board_size)
        haida_pol = PolicyHead(in_filters=input_channels, board_size=board_size, raw=True)
        copy_pol(torch_pol, haida_pol)

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(minibatch_size, input_channels, board_size, board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_pol(torch_in)
        haida_results = haida_pol.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results.detach().numpy(), haida_results))

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
        haida_input_grads = haida_pol.sgd(np_in, target.detach().numpy(), None)
        self.assertTrue(np.allclose(updated_torch_weights1, haida_pol.kernels))
        self.assertTrue(np.allclose(updated_torch_biases1, haida_pol.biases1))
        self.assertTrue(np.allclose(updated_torch_weights2, haida_pol.weights))
        self.assertTrue(np.allclose(updated_torch_biases2, haida_pol.biases2))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_val(self):
        input_channels = 16
        board_size = 19
        minibatch_size = 4

        torch_val = generate_torch_val_head(input_channels, board_size)
        haida_val = ValueHead(in_filters=input_channels, board_size=board_size)
        copy_val(torch_val, haida_val)

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(minibatch_size, input_channels, board_size, board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_val(torch_in)
        haida_results = haida_val.feedforward(np_in)
        self.assertTrue(np.allclose(np.ndarray.flatten(torch_results.detach().numpy()), haida_results))

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
        haida_input_grads = haida_val.sgd(np_in, None, np.ndarray.flatten(target.detach().numpy()))
        self.assertTrue(np.allclose(updated_torch_weights1, haida_val.l1_kernels))
        self.assertTrue(np.allclose(updated_torch_bias1, haida_val.l1_bias))
        self.assertTrue(np.allclose(updated_torch_weights2, haida_val.l2_weights))
        self.assertTrue(np.allclose(updated_torch_biases2, haida_val.l2_biases))
        self.assertTrue(np.allclose(updated_torch_weights3, haida_val.l3_weights))
        self.assertTrue(np.allclose(updated_torch_bias3, haida_val.l3_bias))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_full_net(self):
        input_channels = 17
        filters = 8  # 256
        residual_blocks = 19
        board_size = 19
        minibatch_size = 4  # 32 per worker

        class TorchNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_block = generate_torch_conv_block(input_channels, filters)
                self.res_blocks = [TorchResidualBlock(filters) for _ in range(residual_blocks)]
                self.pol_head = generate_torch_pol_head(filters, board_size)
                self.val_head = generate_torch_val_head(filters, board_size)

            def forward(self, x):
                x = self.conv_block.forward(x)
                for res_block in self.res_blocks:
                    x = res_block.forward(x)
                p = self.pol_head.forward(x)
                v = self.val_head.forward(x)
                return p, v

        torch_net = TorchNet()
        haida_net = NeuralNet(residual_blocks=residual_blocks,
                              input_channels=input_channels,
                              filters=filters,
                              board_size=board_size,
                              raw=True)
        curr = haida_net.head
        copy_conv(torch_net.conv_block, curr)
        for torch_res in torch_net.res_blocks:
            curr = curr.to[0]
            copy_res(torch_res, curr)
        copy_pol(torch_net.pol_head, curr.to[0])
        copy_val(torch_net.val_head, curr.to[1])

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(minibatch_size, input_channels, board_size, board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_net(torch_in)
        haida_results = haida_net.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results[0].detach().numpy(), haida_results[0]))
        self.assertTrue(np.allclose(np.ndarray.flatten(torch_results[1].detach().numpy()), haida_results[1]))

        # TODO: finish this test
