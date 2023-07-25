from conv_block import ConvolutionalBlock
from res_block import ResidualBlock
from policy_head import PolicyHead
from value_head import ValueHead
import hyperparams as hp


class NeuralNet:
    head = None

    # TODO: L2 regularization
    # TODO: Batch Norm (can get rid of biases once this is implemented)
    def __init__(self, residual_blocks=hp.RESIDUAL_BLOCKS):
        conv = ConvolutionalBlock()
        res = [ResidualBlock() for _ in range(residual_blocks)]
        conv.to = res[0]
        for i in range(residual_blocks - 1):
            res[i].to = [res[i+1]]
        pol = PolicyHead()
        val = ValueHead()
        res[-1].to = [pol, val]
        self.head = conv

    def feedforward(self, in_activations):
        return self.head.feedforward(in_activations)

    def sgd(self, examples):
        self.head.sgd(examples[0], examples[1], examples[2])

    def create_checkpoint(self):
        return
