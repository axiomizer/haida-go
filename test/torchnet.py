import torch
import numpy as np
from src.nn.neural_net import ConvolutionalBlock, ResidualBlock, PolicyHead, ValueHead, NeuralNet


class TorchConvBlock(torch.nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super().__init__(
            torch.nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False, dtype=torch.float64),
            torch.nn.BatchNorm2d(output_channels, dtype=torch.float64),
            torch.nn.ReLU()
        )

    def copy_to_haida(self):
        haida_conv = ConvolutionalBlock(self[0].in_channels, self[0].out_channels)
        haida_conv.kernels = np.copy(torch.transpose(self[0].weight, 0, 1).detach().numpy())
        return haida_conv

    def compare_params(self, haida_conv):
        weight_same = np.allclose(torch.transpose(self[0].weight, 0, 1).detach().numpy(), haida_conv.kernels)
        gamma_same = np.allclose(self[1].weight.detach().numpy(), haida_conv.bn.gamma)
        beta_same = np.allclose(self[1].bias.detach().numpy(), haida_conv.bn.beta)
        return weight_same and gamma_same and beta_same


class TorchResBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False, dtype=torch.float64)
        self.bn1 = torch.nn.BatchNorm2d(channels, dtype=torch.float64)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False, dtype=torch.float64)
        self.bn2 = torch.nn.BatchNorm2d(channels, dtype=torch.float64)

    def forward(self, x):
        skip = x
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        return torch.nn.functional.relu(self.bn2(self.conv2(x)) + skip)

    def copy_to_haida(self):
        haida_res = ResidualBlock(self.conv1.in_channels)
        haida_res.kernels1 = np.copy(torch.transpose(self.conv1.weight, 0, 1).detach().numpy())
        haida_res.kernels2 = np.copy(torch.transpose(self.conv2.weight, 0, 1).detach().numpy())
        return haida_res

    def compare_params(self, haida_res):
        w1_same = np.allclose(torch.transpose(self.conv1.weight, 0, 1).detach().numpy(), haida_res.kernels1)
        gamma1_same = np.allclose(self.bn1.weight.detach().numpy(), haida_res.bn1.gamma)
        beta1_same = np.allclose(self.bn1.bias.detach().numpy(), haida_res.bn1.beta)
        w2_same = np.allclose(torch.transpose(self.conv2.weight, 0, 1).detach().numpy(), haida_res.kernels2)
        gamma2_same = np.allclose(self.bn2.weight.detach().numpy(), haida_res.bn2.gamma)
        beta2_same = np.allclose(self.bn2.bias.detach().numpy(), haida_res.bn2.beta)
        return w1_same and gamma1_same and beta1_same and w2_same and gamma2_same and beta2_same


class TorchPolHead(torch.nn.Sequential):
    def __init__(self, input_channels, board_size):
        super().__init__(
            torch.nn.Conv2d(input_channels, 2, 1, bias=False, dtype=torch.float64),
            torch.nn.BatchNorm2d(2, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear((board_size ** 2) * 2, (board_size ** 2) + 1, dtype=torch.float64)
        )
        self.board_size = board_size

    def copy_to_haida(self):
        haida_pol = PolicyHead(in_filters=self[0].in_channels, board_size=self.board_size, raw=True)
        haida_pol.kernels = np.copy(torch.transpose(self[0].weight, 0, 1).detach().numpy())
        haida_pol.weights = np.copy(self[4].weight.detach().numpy())
        haida_pol.biases = np.copy(self[4].bias.detach().numpy())
        return haida_pol

    def compare_params(self, haida_pol):
        w1_same = np.allclose(torch.transpose(self[0].weight, 0, 1).detach().numpy(), haida_pol.kernels)
        gamma_same = np.allclose(self[1].weight.detach().numpy(), haida_pol.bn.gamma)
        beta_same = np.allclose(self[1].bias.detach().numpy(), haida_pol.bn.beta)
        w2_same = np.allclose(self[4].weight.detach().numpy(), haida_pol.weights)
        b2_same = np.allclose(self[4].bias.detach().numpy(), haida_pol.biases)
        return w1_same and gamma_same and beta_same and w2_same and b2_same


class TorchValHead(torch.nn.Sequential):
    def __init__(self, input_channels, board_size):
        super().__init__(
            torch.nn.Conv2d(input_channels, 1, 1, bias=False, dtype=torch.float64),
            torch.nn.BatchNorm2d(1, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(board_size ** 2, 256, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1, dtype=torch.float64),
            torch.nn.Tanh()
        )
        self.board_size = board_size

    def copy_to_haida(self):
        haida_val = ValueHead(in_filters=self[0].in_channels, board_size=self.board_size)
        haida_val.l1_kernels = np.ndarray.flatten(self[0].weight.detach().numpy())
        haida_val.l2_weights = np.copy(self[4].weight.detach().numpy())
        haida_val.l2_biases = np.copy(self[4].bias.detach().numpy())
        haida_val.l3_weights = np.ndarray.flatten(self[6].weight.detach().numpy())
        haida_val.l3_bias = self[6].bias.detach().numpy()[0]
        return haida_val

    def compare_params(self, haida_val):
        w1_same = np.allclose(np.ndarray.flatten(self[0].weight.detach().numpy()), haida_val.l1_kernels)
        gamma_same = np.allclose(self[1].weight.detach().numpy(), haida_val.bn.gamma)
        beta_same = np.allclose(self[1].bias.detach().numpy(), haida_val.bn.beta)
        w2_same = np.allclose(self[4].weight.detach().numpy(), haida_val.l2_weights)
        b2_same = np.allclose(self[4].bias.detach().numpy(), haida_val.l2_biases)
        w3_same = np.allclose(np.ndarray.flatten(self[6].weight.detach().numpy()), haida_val.l3_weights)
        b3_same = np.allclose(self[6].bias.detach().numpy()[0], haida_val.l3_bias)
        return w1_same and gamma_same and beta_same and w2_same and b2_same and w3_same and b3_same


class TorchNet(torch.nn.Module):
    def __init__(self, residual_blocks, input_channels, filters, board_size):
        super().__init__()
        self.conv_block = TorchConvBlock(input_channels, filters)
        self.res_blocks = torch.nn.ModuleList([TorchResBlock(filters) for _ in range(residual_blocks)])
        self.pol_head = TorchPolHead(filters, board_size)
        self.val_head = TorchValHead(filters, board_size)

    def forward(self, x):
        x = self.conv_block.forward(x)
        for res_block in self.res_blocks:
            x = res_block.forward(x)
        p = self.pol_head.forward(x)
        v = self.val_head.forward(x)
        return p, v

    def copy_to_haida(self):
        haida_net = NeuralNet()
        haida_net.conv = self.conv_block.copy_to_haida()
        haida_net.res = [r.copy_to_haida() for r in self.res_blocks]
        haida_net.pol = self.pol_head.copy_to_haida()
        haida_net.val = self.val_head.copy_to_haida()
        return haida_net

    def compare_params(self, haida_net):
        conv_same = self.conv_block.compare_params(haida_net.conv)
        res_count_same = len(self.res_blocks) == len(haida_net.res)
        res_same = [self.res_blocks[i].compare_params(haida_net.res[i]) for i in range(len(self.res_blocks))]
        pol_same = self.pol_head.compare_params(haida_net.pol)
        val_same = self.val_head.compare_params(haida_net.val)
        return conv_same and res_count_same and all(res_same) and pol_same and val_same
