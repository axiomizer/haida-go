from test.torch_net import TorchConvBlock, TorchResBlock, TorchPolHead, TorchValHead, TorchNet
import numpy as np
import torch
import unittest
from src.nn.operations import op
from src.nn.haida_net import ConvolutionalBlock, ResidualBlock, PolicyHead, ValueHead, HaidaNet
from test.unit.config import *


class TestNN(unittest.TestCase):
    def test_conv(self):
        torch_conv = TorchConvBlock(INPUT_CHANNELS, FILTERS)
        haida_conv = ConvolutionalBlock(INPUT_CHANNELS, FILTERS)
        haida_conv.configure(learning_rate=LEARNING_RATE)
        torch_conv.copy_trainable_params(haida_conv)

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(MINIBATCH_SIZE, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_conv(torch_in)
        haida_results = haida_conv.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results.detach().numpy(), haida_results))

        # do one step of SGD for both nets and compare results
        np_target = np.random.randn(MINIBATCH_SIZE, FILTERS, BOARD_SIZE, BOARD_SIZE)
        loss = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_conv.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        loss(torch_results, torch.tensor(np_target, dtype=torch.float64)).backward()
        optimizer.step()
        haida_input_grads = haida_conv.backprop(mse_derivative(haida_results, np_target))
        self.assertTrue(torch_conv.compare_params(haida_conv))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_res(self):
        torch_res = TorchResBlock(FILTERS)
        haida_res = ResidualBlock(FILTERS)
        haida_res.configure(learning_rate=LEARNING_RATE)
        torch_res.copy_trainable_params(haida_res)

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(MINIBATCH_SIZE, FILTERS, BOARD_SIZE, BOARD_SIZE)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_res(torch_in)
        haida_results = haida_res.feedforward(np_in)
        self.assertTrue(np.allclose(torch_results.detach().numpy(), haida_results))

        # do one step of SGD for both nets and compare results
        np_target = np.random.randn(MINIBATCH_SIZE, FILTERS, BOARD_SIZE, BOARD_SIZE)
        loss = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_res.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        loss(torch_results, torch.tensor(np_target, dtype=torch.float64)).backward()
        optimizer.step()
        haida_input_grads = haida_res.backprop(mse_derivative(haida_results, np_target))
        self.assertTrue(torch_res.compare_params(haida_res))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_pol(self):
        torch_pol = TorchPolHead(FILTERS, BOARD_SIZE)
        haida_pol = PolicyHead(FILTERS, BOARD_SIZE)
        haida_pol.configure(learning_rate=LEARNING_RATE)
        torch_pol.copy_trainable_params(haida_pol)

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(MINIBATCH_SIZE, FILTERS, BOARD_SIZE, BOARD_SIZE)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_pol(torch_in)
        haida_results = haida_pol.feedforward(np_in)
        self.assertTrue(np.allclose([op.softmax(a) for a in torch_results.detach().numpy()], haida_results))

        # do one step of SGD for both nets and compare results
        raw_target = torch.randn(MINIBATCH_SIZE, (BOARD_SIZE ** 2) + 1, dtype=torch.float64)
        target = torch.nn.functional.softmax(raw_target, dim=1, dtype=torch.float64)
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(torch_pol.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        loss(torch_results, target).backward()
        optimizer.step()
        haida_err = haida_pol.error(target.detach().numpy())
        haida_input_grads = haida_pol.backprop(haida_err)
        self.assertTrue(torch_pol.compare_params(haida_pol))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_val(self):
        torch_val = TorchValHead(FILTERS, BOARD_SIZE)
        haida_val = ValueHead(FILTERS, BOARD_SIZE)
        haida_val.configure(learning_rate=LEARNING_RATE)
        torch_val.copy_trainable_params(haida_val)

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(MINIBATCH_SIZE, FILTERS, BOARD_SIZE, BOARD_SIZE)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_val(torch_in)
        haida_results = haida_val.feedforward(np_in)
        self.assertTrue(np.allclose(np.ndarray.flatten(torch_results.detach().numpy()), haida_results))

        # do one step of SGD for both nets and compare results
        target = torch.randn(MINIBATCH_SIZE, 1, dtype=torch.float64)
        loss = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_val.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        loss(torch_results, target).backward()
        optimizer.step()
        haida_err = haida_val.error(np.ndarray.flatten(target.detach().numpy()))
        haida_input_grads = haida_val.backprop(haida_err)
        self.assertTrue(torch_val.compare_params(haida_val))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_full_net(self):
        torch_net = TorchNet(RESIDUAL_BLOCKS, INPUT_CHANNELS, FILTERS, BOARD_SIZE)
        haida_net = HaidaNet(BOARD_SIZE, RESIDUAL_BLOCKS, INPUT_CHANNELS, FILTERS)
        haida_net.configure(learning_rate=LEARNING_RATE)
        torch_net.copy_trainable_params(haida_net)

        # feedforward for both neural nets and compare results
        np_in = np.random.randn(MINIBATCH_SIZE, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        torch_results = torch_net(torch_in)
        haida_results = haida_net.feedforward(np_in)
        self.assertTrue(np.allclose([op.softmax(a) for a in torch_results[0].detach().numpy()], haida_results[0]))
        self.assertTrue(np.allclose(np.ndarray.flatten(torch_results[1].detach().numpy()), haida_results[1]))

        # do one step of SGD for both nets and compare results
        pi_raw = torch.randn(MINIBATCH_SIZE, (BOARD_SIZE ** 2) + 1, dtype=torch.float64)
        pi = torch.nn.functional.softmax(pi_raw, dim=1, dtype=torch.float64)
        z = torch.randn(MINIBATCH_SIZE, 1, dtype=torch.float64)
        loss1 = torch.nn.CrossEntropyLoss()
        loss2 = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_net.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        total_loss = loss1(torch_results[0], pi) + loss2(torch_results[1], z)
        total_loss.backward()
        optimizer.step()
        haida_err = haida_net.error([pi.detach().numpy(), np.ndarray.flatten(z.detach().numpy())])
        haida_input_grads = haida_net.backprop(haida_err)
        self.assertTrue(torch_net.compare_params(haida_net))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    # just see if we can train haida net without hitting any bugs
    # all the other tests copy trainable params from pytorch, which has proven capable of hiding bugs
    def test_no_copy(self):
        residual_blocks = 3
        haida_net = HaidaNet(BOARD_SIZE, residual_blocks, INPUT_CHANNELS, FILTERS)
        for _ in range(3):
            np_in = np.random.randn(MINIBATCH_SIZE, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            pi_raw = torch.randn(MINIBATCH_SIZE, (BOARD_SIZE ** 2) + 1, dtype=torch.float64)
            pi = torch.nn.functional.softmax(pi_raw, dim=1, dtype=torch.float64)
            z = torch.randn(MINIBATCH_SIZE, 1, dtype=torch.float64)
            minibatch = [np_in, pi.detach().numpy(), np.ndarray.flatten(z.detach().numpy())]

            haida_net.train(minibatch)
