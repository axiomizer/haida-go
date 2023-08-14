from test.torch_net import TorchResBlock, TorchNet
import random
import numpy as np
import torch
import unittest
from src.nn.haida_net import ResidualBlock, HaidaNet
from test.unit.config import *


class TestNNConfig(unittest.TestCase):
    def test_learning_rate(self):
        filters = 4
        learning_rates = [10 ** (5 * random.uniform(-1, 0)) for _ in range(10)]
        for lr in learning_rates:
            torch_res = TorchResBlock(filters)
            haida_res = ResidualBlock(filters)
            haida_res.configure(learning_rate=lr)
            torch_res.copy_trainable_params(haida_res)

            # feedforward for both neural nets and compare results
            np_in = np.random.randn(MINIBATCH_SIZE, filters, BOARD_SIZE, BOARD_SIZE)
            torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
            torch_results = torch_res(torch_in)
            haida_results = haida_res.feedforward(np_in)
            self.assertTrue(np.allclose(torch_results.detach().numpy(), haida_results))

            # do one step of SGD for both nets and compare results
            np_target = np.random.randn(MINIBATCH_SIZE, filters, BOARD_SIZE, BOARD_SIZE)
            loss = torch.nn.MSELoss(reduction='mean')
            optimizer = torch.optim.SGD(torch_res.parameters(), lr=lr)
            optimizer.zero_grad()
            loss(torch_results, torch.tensor(np_target, dtype=torch.float64)).backward()
            optimizer.step()
            haida_input_grads = haida_res.backprop(mse_derivative(haida_results, np_target))
            self.assertTrue(torch_res.compare_params(haida_res))

            # compare gradient with respect to input
            self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_weight_decay(self):
        filters = 4
        learning_rate = 0.1
        # use values for weight_decay large enough to have a notable effect
        weight_decays = [10 ** (3 * random.uniform(-1, 0)) for _ in range(10)]
        for wd in weight_decays:
            torch_res = TorchResBlock(filters)
            haida_res = ResidualBlock(filters)
            haida_res.configure(learning_rate=learning_rate, weight_decay=wd)
            torch_res.copy_trainable_params(haida_res)

            # feedforward for both neural nets and compare results
            np_in = np.random.randn(MINIBATCH_SIZE, filters, BOARD_SIZE, BOARD_SIZE)
            torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
            torch_results = torch_res(torch_in)
            haida_results = haida_res.feedforward(np_in)
            self.assertTrue(np.allclose(torch_results.detach().numpy(), haida_results))

            # do one step of SGD for both nets and compare results
            np_target = np.random.randn(MINIBATCH_SIZE, filters, BOARD_SIZE, BOARD_SIZE)
            loss = torch.nn.MSELoss(reduction='mean')
            optimizer = torch.optim.SGD(torch_res.parameters(), lr=learning_rate, weight_decay=wd*2)
            optimizer.zero_grad()
            loss(torch_results, torch.tensor(np_target, dtype=torch.float64)).backward()
            optimizer.step()
            haida_input_grads = haida_res.backprop(mse_derivative(haida_results, np_target))
            self.assertTrue(torch_res.compare_params(haida_res))

            # compare gradient with respect to input
            self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_lr_sched(self):
        residual_blocks = 3
        filters = 4
        torch_net = TorchNet(residual_blocks, INPUT_CHANNELS, filters, BOARD_SIZE)
        haida_net = HaidaNet(BOARD_SIZE, residual_blocks, INPUT_CHANNELS, filters)
        haida_net.configure(lr_sched=[(0, 0.05), (5, 0.005), (10, 0.0005)])
        torch_net.copy_trainable_params(haida_net)

        loss1 = torch.nn.CrossEntropyLoss()
        loss2 = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_net.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
        for _ in range(15):
            # feedforward
            np_in = np.random.randn(MINIBATCH_SIZE, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
            torch_results = torch_net(torch_in)

            # do one step of SGD for torch net
            pi_raw = torch.randn(MINIBATCH_SIZE, (BOARD_SIZE ** 2) + 1, dtype=torch.float64)
            pi = torch.nn.functional.softmax(pi_raw, dim=1, dtype=torch.float64)
            z = torch.randn(MINIBATCH_SIZE, 1, dtype=torch.float64)
            optimizer.zero_grad()
            total_loss = loss1(torch_results[0], pi) + loss2(torch_results[1], z)
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # train haida and compare parameters
            minibatch = [np_in, pi.detach().numpy(), np.ndarray.flatten(z.detach().numpy())]
            haida_net.train(minibatch)
            self.assertTrue(torch_net.compare_params(haida_net))
