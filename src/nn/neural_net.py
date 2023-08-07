from src.nn.conv_block import ConvolutionalBlock
from src.nn.res_block import ResidualBlock
from src.nn.policy_head import PolicyHead
from src.nn.value_head import ValueHead
import src.nn.hyperparams as hp


class NeuralNet:
    head = None

    # TODO: L2 regularization
    # TODO: Batch Norm (can get rid of biases once this is implemented)
    def __init__(self,
                 residual_blocks=hp.RESIDUAL_BLOCKS,
                 input_channels=hp.INPUT_PLANES,
                 filters=hp.FILTERS,
                 board_size=hp.BOARD_SIZE,
                 raw=False):
        conv = ConvolutionalBlock(in_filters=input_channels, out_filters=filters)
        res = [ResidualBlock(filters=filters) for _ in range(residual_blocks)]
        conv.to = [res[0]]
        for i in range(residual_blocks - 1):
            res[i].to = [res[i+1]]
        pol = PolicyHead(in_filters=filters, board_size=board_size, raw=raw)
        val = ValueHead(in_filters=filters, board_size=board_size)
        res[-1].to = [pol, val]
        self.head = conv

    def feedforward(self, in_activations):
        return self.head.feedforward(in_activations)

    def sgd(self, examples):
        self.head.sgd(examples[0], examples[1], examples[2])

    def create_checkpoint(self):
        return
