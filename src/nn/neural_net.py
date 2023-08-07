from src.nn.conv_block import ConvolutionalBlock
from src.nn.res_block import ResidualBlock
from src.nn.policy_head import PolicyHead
from src.nn.value_head import ValueHead
import src.nn.hyperparams as hp


class NeuralNet:
    conv = None
    res = []
    pol = None
    val = None

    # TODO: L2 regularization
    # TODO: Batch Norm (can get rid of biases once this is implemented)
    def __init__(self,
                 residual_blocks=hp.RESIDUAL_BLOCKS,
                 input_channels=hp.INPUT_PLANES,
                 filters=hp.FILTERS,
                 board_size=hp.BOARD_SIZE,
                 raw=False):
        self.conv = ConvolutionalBlock(in_filters=input_channels, out_filters=filters)
        self.res = [ResidualBlock(filters=filters) for _ in range(residual_blocks)]
        self.pol = PolicyHead(in_filters=filters, board_size=board_size, raw=raw)
        self.val = ValueHead(in_filters=filters, board_size=board_size)

    def feedforward(self, in_activations):
        x = self.conv.feedforward(in_activations)
        for r in self.res:
            x = r.feedforward(x)
        return [self.pol.feedforward(x), self.val.feedforward(x)]

    def backprop(self, pi, z):
        err = self.pol.backprop(pi) + self.val.backprop(z)
        for r in reversed(self.res):
            err = r.backprop(err)
        return self.conv.backprop(err)

    def create_checkpoint(self):
        return
