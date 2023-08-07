from test import torchnet
import numpy as np
import torch
import src.nn.hyperparams as hp
import unittest


def mse_derivative(a, pi):
    return [(-2 / (pi[i].size * len(a))) * (pi[i] - a[i]) for i in range(len(a))]


class TestTorch(unittest.TestCase):
    input_channels = 17
    filters = 16  # 256
    residual_blocks = 19
    board_size = 19
    minibatch_size = 4  # 32 per worker

    def test_conv(self):
        torch_conv = torchnet.TorchConvBlock(self.input_channels, self.filters)
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
        haida_input_grads = haida_conv.backprop(mse_derivative(haida_results, np_target))
        self.assertTrue(torch_conv.compare_params(haida_conv))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_res(self):
        torch_res = torchnet.TorchResBlock(self.filters)
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
        haida_input_grads = haida_res.backprop(mse_derivative(haida_results, np_target))
        self.assertTrue(torch_res.compare_params(haida_res))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_pol(self):
        torch_pol = torchnet.TorchPolHead(self.filters, self.board_size)
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
        haida_input_grads = haida_pol.backprop(target.detach().numpy())
        self.assertTrue(torch_pol.compare_params(haida_pol))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_val(self):
        torch_val = torchnet.TorchValHead(self.filters, self.board_size)
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
        haida_input_grads = haida_val.backprop(np.ndarray.flatten(target.detach().numpy()))
        self.assertTrue(torch_val.compare_params(haida_val))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_full_net(self):
        torch_net = torchnet.TorchNet(self.residual_blocks, self.input_channels, self.filters, self.board_size)
        haida_net = torch_net.copy_to_haida()

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(self.minibatch_size, self.input_channels, self.board_size, self.board_size)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_net(torch_in)
        haida_results = haida_net.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results[0].detach().numpy(), haida_results[0]))
        self.assertTrue(np.allclose(np.ndarray.flatten(torch_results[1].detach().numpy()), haida_results[1]))

        # do one step of SGD for both nets and compare results
        pi_raw = torch.randn(self.minibatch_size, (self.board_size ** 2) + 1, dtype=torch.float64)
        pi = torch.nn.functional.softmax(pi_raw, dim=1, dtype=torch.float64)
        z = torch.randn(self.minibatch_size, 1, dtype=torch.float64)
        loss1 = torch.nn.CrossEntropyLoss()
        loss2 = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_net.parameters(), lr=hp.LEARNING_RATE)
        optimizer.zero_grad()
        total_loss = loss1(torch_results[0], pi) + loss2(torch_results[1], z)
        total_loss.backward()
        optimizer.step()
        haida_input_grads = haida_net.backprop(pi.detach().numpy(), np.ndarray.flatten(z.detach().numpy()))
        self.assertTrue(torch_net.compare_params(haida_net))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))
