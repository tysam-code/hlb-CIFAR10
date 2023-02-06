import torch

hyp = {
    # to be filled by init
}

bias_scaler = 32

def init_split_parameter_dictionaries(network):
    params_non_bias = {'params': [], 'lr': hyp['opt']['non_bias_lr'], 'momentum': .85, 'nesterov': True, 'weight_decay': hyp['opt']['non_bias_decay']}
    params_bias     = {'params': [], 'lr': hyp['opt']['bias_lr'],     'momentum': .85, 'nesterov': True, 'weight_decay': hyp['opt']['bias_decay']}

    for name, p in network.named_parameters():
        if p.requires_grad:
            if 'bias' in name:
                params_bias['params'].append(p)
            else:
                params_non_bias['params'].append(p)
    return params_non_bias, params_bias

class OptSched:
    def __init__(self, batchsize, net, total_train_steps, num_low_lr_steps_for_ema) -> None:

        hyp['opt'] = {
            'bias_lr':        1.15 * 1.35 * 1. * bias_scaler/batchsize, # TODO: How we're expressing this information feels somewhat clunky, is there maybe a better way to do this? :'))))
            'non_bias_lr':    1.15 * 1.35 * 1. / batchsize,
            'bias_decay':     .85 * 4.8e-4 * batchsize/bias_scaler,
            'non_bias_decay': .85 * 4.8e-4 * batchsize,
            'percent_start': .2,
        }


        ## Stowing the creation of these into a helper function to make things a bit more readable....
        non_bias_params, bias_params = init_split_parameter_dictionaries(net)

        # One optimizer for the regular network, and one for the biases. This allows us to use the superconvergence onecycle training policy for our networks....
        self.opt = torch.optim.SGD(**non_bias_params)
        self.opt_bias = torch.optim.SGD(**bias_params)

        #opt = torch.optim.SGD(**non_bias_params)
        #opt_bias = torch.optim.SGD(**bias_params)

        # Adjust pct_start based upon how many epochs we need to finetune the ema at a low lr for
        pct_start = hyp['opt']['percent_start'] * (total_train_steps/(total_train_steps - num_low_lr_steps_for_ema))

        ## Not the most intuitive, but this basically takes us from ~0 to max_lr at the point pct_start, then down to .1 * max_lr at the end (since 1e16 * 1e-15 = .1 --
        ##   This quirk is because the final lr value is calculated from the starting lr value and not from the maximum lr value set during training)
        initial_div_factor = 1e16 # basically to make the initial lr ~0 or so :D
        final_lr_ratio = .135
        self.lr_sched      = torch.optim.lr_scheduler.OneCycleLR(self.opt,  max_lr=non_bias_params['lr'],
                                                            pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=total_train_steps-num_low_lr_steps_for_ema, anneal_strategy='linear', cycle_momentum=False)
        self.lr_sched_bias = torch.optim.lr_scheduler.OneCycleLR(self.opt_bias, max_lr=bias_params['lr'],
                                                            pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=total_train_steps-num_low_lr_steps_for_ema, anneal_strategy='linear', cycle_momentum=False)

    def lr_step(self):
        # We only want to step the lr_schedulers while we have training steps to consume. Otherwise we get a not-so-friendly error from PyTorch
        self.lr_sched.step()
        self.lr_sched_bias.step()

    def step(self):
        self.opt.step()
        self.opt_bias.step()

    def zero_grad(self):
        # Using 'set_to_none' I believe is slightly faster (albeit riskier w/ funky gradient update workflows) than under the default 'set to zero' method
        self.opt.zero_grad(set_to_none=True)
        self.opt_bias.zero_grad(set_to_none=True)
