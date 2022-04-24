
class WarmupLinearScheduler(object):
    def __init__(self, optimizer, max_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.lr = 0.0
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.warmup_increase = None
        self.lr_decay = None
        self._calculate()
        self._adapt_lr()

    def __call__(self, step):
        if step <= self.warmup_steps:
            self.lr += self.warmup_increase
        else:
            self.lr -= self.lr_decay
        assert not self.lr < 0
        self._adapt_lr()

    def _calculate(self):
        self.warmup_increase = self.max_lr / self.warmup_steps
        self.lr_decay = self.max_lr / (self.total_steps - self.warmup_steps)

    def _adapt_lr(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr