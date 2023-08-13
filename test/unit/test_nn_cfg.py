from test import torchnet
import random
import numpy as np
import torch
import unittest
from src.nn.operations import op
from src.nn.config import Config
from src.nn.neural_net import ResidualBlock, NeuralNet
from test.unit.config import *


class TestNNConfig(unittest.TestCase):
    def test_learning_rate(self):
        filters = 4
        learning_rates = [10 ** (5 * random.uniform(-1, 0)) for _ in range(10)]
        for lr in learning_rates:
            cfg = Config(learning_rate=lr)
            torch_res = torchnet.TorchResBlock(filters)
            haida_res = ResidualBlock(filters, cfg)
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
        weight_decay = 0.1  # use a value large enough to have a notable effect
        cfg = Config(learning_rate=LEARNING_RATE, weight_decay=weight_decay)
        torch_res = torchnet.TorchResBlock(FILTERS)
        haida_res = ResidualBlock(FILTERS, cfg)
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
        optimizer = torch.optim.SGD(torch_res.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay*2)
        optimizer.zero_grad()
        loss(torch_results, torch.tensor(np_target, dtype=torch.float64)).backward()
        optimizer.step()
        haida_input_grads = haida_res.backprop(mse_derivative(haida_results, np_target))
        self.assertTrue(torch_res.compare_params(haida_res))

        # compare gradient with respect to input
        self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))

    def test_lr_sched(self):
        cfg = Config(lr_sched=[(0, 0.05), (5, 0.005), (10, 0.0005)])
        torch_net = torchnet.TorchNet(3, INPUT_CHANNELS, FILTERS, BOARD_SIZE)
        haida_net = NeuralNet(BOARD_SIZE, 3, INPUT_CHANNELS, FILTERS, cfg)
        torch_net.copy_trainable_params(haida_net)

        loss1 = torch.nn.CrossEntropyLoss()
        loss2 = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(torch_net.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
        for _ in range(15):
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
            optimizer.zero_grad()
            total_loss = loss1(torch_results[0], pi) + loss2(torch_results[1], z)
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            haida_input_grads = haida_net.backprop(pi.detach().numpy(), np.ndarray.flatten(z.detach().numpy()))
            self.assertTrue(torch_net.compare_params(haida_net))

            # compare gradient with respect to input
            self.assertTrue(np.allclose(torch_in.grad, haida_input_grads))
