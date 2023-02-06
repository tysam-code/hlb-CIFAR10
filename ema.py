import copy

import torch
from torch import nn

hyp = {
    'ema': {
        'decay_base': .986
    },
}


class NetworkEMA(nn.Module):
    "Maintains a mirror network whoes weights are kept as moving average of network being trained"
    def __init__(self, net, ema_steps, decay=None):
        super().__init__()  # init the parent module so this module is registered properly
        self.net_ema = copy.deepcopy(net).eval(
        ).requires_grad_(False)  # copy the model

        # I believe this wasn't logged, but the EMA update power is adjusted by being raised to the power of the number of "every n" steps
        # to somewhat accomodate for whatever the expected information intake rate is. The tradeoff I believe, though, is that this is to some degree noisier as we
        # are intaking fewer samples of our distribution-over-time, with a higher individual weight each. This can be good or bad depending upon what we want.
        projected_ema_decay_val = hyp['ema']['decay_base'] ** ema_steps

        # you can update/hack this as necessary for update scheduling purposes :3
        self.decay = decay or projected_ema_decay_val

    def update(self, current_net):
        with torch.no_grad():
            # TODO: potential bug: assumes that the network architectures don't change during training (!!!!)
            for ema_net_parameter, incoming_net_parameter in zip(self.net_ema.state_dict().values(), current_net.state_dict().values()):
                if incoming_net_parameter.dtype in (torch.half, torch.float):
                    # update the ema values in place, similar to how optimizer momentum is coded
                    ema_net_parameter.mul_(self.decay).add_(
                        incoming_net_parameter.detach().mul(1. - self.decay))

    def forward(self, inputs):
        with torch.no_grad():
            return self.net_ema(inputs)