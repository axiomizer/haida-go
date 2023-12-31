from src.bot.nn.conv_block import ConvolutionalBlock
from src.bot.nn.res_block import ResidualBlock
from src.bot.nn.policy_head import PolicyHead
from src.bot.nn.value_head import ValueHead
from src.bot.nn.shared import AbstractNet


class HaidaNet(AbstractNet):
    def __init__(self, board_size, residual_blocks, input_channels, filters, config=None):
        super().__init__(config)

        self.conv = ConvolutionalBlock(input_channels, filters, self.cfg)
        self.res = [ResidualBlock(filters, self.cfg) for _ in range(residual_blocks)]
        self.pol = PolicyHead(filters, board_size, self.cfg)
        self.val = ValueHead(filters, board_size, self.cfg)

    def feedforward(self, in_activations):
        x = self.conv.feedforward(in_activations)
        for r in self.res:
            x = r.feedforward(x)
        return [self.pol.feedforward(x), self.val.feedforward(x)]

    def loss(self, target):
        return self.pol.loss(target[0]) + self.val.loss(target[1])

    # target is the list [pi, z]
    def error(self, target):
        return [self.pol.error(target[0]), self.val.error(target[1])]

    def backprop(self, err):
        if len(err[0]) != len(err[1]):
            raise ValueError("mismatched batch sizes?")
        pol_err = self.pol.backprop(err[0])
        val_err = self.val.backprop(err[1])
        err = [pe + ve for pe, ve in zip(pol_err, val_err)]
        for r in reversed(self.res):
            err = r.backprop(err)
        return self.conv.backprop(err)
