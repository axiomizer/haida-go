class MseStub:
    @staticmethod
    def sgd(a, pi, _):
        return [(-2 / (pi[i].size * len(a))) * (pi[i] - a[i]) for i in range(len(a))]

    @staticmethod
    def feedforward(in_activations):
        return in_activations
