class ProgressBar:
    total_steps = None
    steps = None
    length = None

    def __init__(self, total_steps, length=20):
        self.total_steps = total_steps
        self.steps = 0
        self.length = length

    def start(self):
        print('[' + ' '*self.length + '] 0%', end='', flush=True)

    def increment(self):
        if self.steps < self.total_steps:
            self.steps += 1
        progress = int(self.steps * self.length / self.total_steps)
        print('\r[' + '='*progress + ' '*(self.length-progress) + ']', end='', flush=True)
        print(' {}%'.format(int(self.steps*100/self.total_steps)), end='', flush=True)
        progress += 1

    def end(self):
        print('\r[' + '='*self.length + '] 100%', flush=True)
