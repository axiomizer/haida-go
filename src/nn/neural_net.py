from src.nn.conv_block import ConvolutionalBlock
from src.nn.res_block import ResidualBlock
from src.nn.policy_head import PolicyHead
from src.nn.value_head import ValueHead
from src.nn.config import Config


class NeuralNet:
    def __init__(self, board_size, residual_blocks=19, input_channels=17, filters=256, config=Config()):
        self.cfg = config

        self.conv = ConvolutionalBlock(input_channels, filters, config)
        self.res = [ResidualBlock(filters, config) for _ in range(residual_blocks)]
        self.pol = PolicyHead(filters, board_size, config)
        self.val = ValueHead(filters, board_size, config)

    def feedforward(self, in_activations):
        x = self.conv.feedforward(in_activations)
        for r in self.res:
            x = r.feedforward(x)
        return [self.pol.feedforward(x), self.val.feedforward(x)]

    def backprop(self, pi, z):
        pol_err = self.pol.backprop(pi)
        val_err = self.val.backprop(z)
        if len(pol_err) != len(val_err):
            raise ValueError("mismatched batch sizes?")
        err = [pol_err[i] + val_err[i] for i in range(len(pol_err))]
        for r in reversed(self.res):
            err = r.backprop(err)
        return self.conv.backprop(err)

    def create_checkpoint(self):
        return
