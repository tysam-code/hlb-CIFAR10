import torch


class OptSched:
    def __init__(self, batchsize, net, total_train_steps, num_low_lr_steps_for_ema) -> None:
        # One optimizer for the regular network, and one for the biases. This allows us to use the superconvergence onecycle training policy for our networks....
        self.opt = torch.optim.SGD(net.parameters(), lr=0.0005, weight_decay=0.02)
        self.lr_sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, max_lr=0.1, total_steps=total_train_steps, pct_start=0.2, anneal_strategy='cos', cycle_momentum=False)

    def lr_step(self):
        # We only want to step the lr_schedulers while we have training steps to consume. Otherwise we get a not-so-friendly error from PyTorch
        self.lr_sched.step()

    def step(self):
        self.opt.step()

    def zero_grad(self):
        # Using 'set_to_none' I believe is slightly faster (albeit riskier w/ funky gradient update workflows) than under the default 'set to zero' method
        self.opt.zero_grad()
