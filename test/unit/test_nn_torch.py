from test.torchnet import blocks
import numpy as np
import torch
import src.nn.hyperparams as hp
import unittest


def mean_squared_error(a, pi):
    return [(-2 / (pi[i].size * len(a))) * (pi[i] - a[i]) for i in range(len(a))]


class TestTorch(unittest.TestCase):
    input_channels = 17
    filters = 16  # 256
    residual_blocks = 19
    board_size = 19
    minibatch_size = 4  # 32 per worker

    def test_conv(self):
        torch_conv = blocks.TorchConvBlock(self.input_channels, self.filters)
        haida_conv = torch_conv.copy_to_haida()

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(self.minibatch_size, self.input_channels, self.board_size, self.board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_conv(torch_in)
        haida_results = haida_conv.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results.detach().numpy(), haida_results))

        # do one step of SGD for both nets and compare results
        np_target = np.random.randn(self.minibatch_size, self.filters, self.board_size, self.board_size)
        loss = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_conv.parameters(), lr=hp.LEARNING_RATE)
        optimizer.zero_grad()
        loss(torch_results, torch.tensor(np_target, dtype=torch.float64)).backward()
        optimizer.step()
        updated_torch_weights = torch.transpose(torch_conv[0].weight, 0, 1).detach().numpy()
        haida_input_grads = haida_conv.backprop(mean_squared_error(haida_results, np_target))
        self.assertTrue(np.allclose(updated_torch_weights, haida_conv.kernels))
        updated_torch_biases = torch_conv[0].bias.detach().numpy()
        self.assertTrue(np.allclose(updated_torch_biases, haida_conv.biases))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_res(self):
        torch_res = blocks.TorchResBlock(self.filters)
        haida_res = torch_res.copy_to_haida()

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(self.minibatch_size, self.filters, self.board_size, self.board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_res(torch_in)
        haida_results = haida_res.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results.detach().numpy(), haida_results))

        # do one step of SGD for both nets and compare results
        np_target = np.random.randn(self.minibatch_size, self.filters, self.board_size, self.board_size)
        loss = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_res.parameters(), lr=hp.LEARNING_RATE)
        optimizer.zero_grad()
        loss(torch_results, torch.tensor(np_target, dtype=torch.float64)).backward()
        optimizer.step()
        updated_torch_weights1 = torch.transpose(torch_res.conv1.weight, 0, 1).detach().numpy()
        updated_torch_biases1 = torch_res.conv1.bias.detach().numpy()
        updated_torch_weights2 = torch.transpose(torch_res.conv2.weight, 0, 1).detach().numpy()
        updated_torch_biases2 = torch_res.conv2.bias.detach().numpy()
        haida_input_grads = haida_res.backprop(mean_squared_error(haida_results, np_target))
        self.assertTrue(np.allclose(updated_torch_weights1, haida_res.kernels1))
        self.assertTrue(np.allclose(updated_torch_biases1, haida_res.biases1))
        self.assertTrue(np.allclose(updated_torch_weights2, haida_res.kernels2))
        self.assertTrue(np.allclose(updated_torch_biases2, haida_res.biases2))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_pol(self):
        torch_pol = blocks.TorchPolHead(self.filters, self.board_size)
        haida_pol = torch_pol.copy_to_haida()

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(self.minibatch_size, self.filters, self.board_size, self.board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_pol(torch_in)
        haida_results = haida_pol.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results.detach().numpy(), haida_results))

        # do one step of SGD for both nets and compare results
        raw_target = torch.randn(self.minibatch_size, (self.board_size ** 2) + 1, dtype=torch.float64)
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
        haida_input_grads = haida_pol.backprop(target.detach().numpy())
        self.assertTrue(np.allclose(updated_torch_weights1, haida_pol.kernels))
        self.assertTrue(np.allclose(updated_torch_biases1, haida_pol.biases1))
        self.assertTrue(np.allclose(updated_torch_weights2, haida_pol.weights))
        self.assertTrue(np.allclose(updated_torch_biases2, haida_pol.biases2))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_val(self):
        torch_val = blocks.TorchValHead(self.filters, self.board_size)
        haida_val = torch_val.copy_to_haida()

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(self.minibatch_size, self.filters, self.board_size, self.board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_val(torch_in)
        haida_results = haida_val.feedforward(np_in)
        self.assertTrue(np.allclose(np.ndarray.flatten(torch_results.detach().numpy()), haida_results))

        # do one step of SGD for both nets and compare results
        target = torch.randn(self.minibatch_size, 1, dtype=torch.float64)
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
        haida_input_grads = haida_val.backprop(np.ndarray.flatten(target.detach().numpy()))
        self.assertTrue(np.allclose(updated_torch_weights1, haida_val.l1_kernels))
        self.assertTrue(np.allclose(updated_torch_bias1, haida_val.l1_bias))
        self.assertTrue(np.allclose(updated_torch_weights2, haida_val.l2_weights))
        self.assertTrue(np.allclose(updated_torch_biases2, haida_val.l2_biases))
        self.assertTrue(np.allclose(updated_torch_weights3, haida_val.l3_weights))
        self.assertTrue(np.allclose(updated_torch_bias3, haida_val.l3_bias))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_full_net(self):
        torch_net = blocks.TorchNet(self.residual_blocks, self.input_channels, self.filters, self.board_size)
        haida_net = torch_net.copy_to_haida()

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(self.minibatch_size, self.input_channels, self.board_size, self.board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_net(torch_in)
        haida_results = haida_net.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results[0].detach().numpy(), haida_results[0]))
        self.assertTrue(np.allclose(np.ndarray.flatten(torch_results[1].detach().numpy()), haida_results[1]))

        # TODO: finish this test
