# Note: The one change we need to make if we're in Colab is to uncomment this below block.
# If we are in an ipython session or a notebook, clear the state to avoid bugs
"""
try:
  _ = get_ipython().__class__.__name__
  ## we set -f below to avoid prompting the user before clearing the notebook state
  %reset -f
except NameError:
  pass ## we're still good
"""


import functools
from functools import partial
import os
import copy

import torch
torch.backends.cuda.matmul.allow_tf32 = True
from torch import nn

from network import make_net
from dataset import get_dataset, get_batches
from opt_sched import OptSched

# <-- teaching comments
# <-- functional comments
# You can run 'sed -i.bak '/\#\#/d' ./main.py' to remove the teaching comments if they are in the way of your work. <3

# This can go either way in terms of actually being helpful when it comes to execution speed.
# torch.backends.cudnn.benchmark = True

# This code was built from the ground up to be directly hackable and to support rapid experimentation, which is something you might see
# reflected in what would otherwise seem to be odd design decisions. It also means that maybe some cleaning up is required before moving
# to production if you're going to use this code as such (such as breaking different section into unique files, etc). That said, if there's
# ways this code could be improved and cleaned up, please do open a PR on the GitHub repo. Your support and help is much appreciated for this
# project! :)


# This is for testing that certain changes don't exceed X% portion of the reference GPU (here an A100)
# so we can help reduce a possibility that future releases don't take away the accessibility of this codebase.
# torch.cuda.set_per_process_memory_fraction(fraction=8./40., device=0) ## 40. GB is the maximum memory of the base A100 GPU


batchsize = 512

# To replicate the ~95.77% accuracy in 188 seconds runs, simply change the base_depth from 64->128 and the num_epochs from 10->80
hyp = {
    'ema': {
        'epochs': 2,
        'decay_base': .986,
        'every_n_steps': 2,
    },
    'scaling_factor': 1./10,
    'train_epochs': 10,
    'device': 'cuda',
    'data_cache_location': 'data.pt',
    'pad_amount': 3,
    'cutout_size': 0,
}

data = get_dataset(hyp['data_cache_location'],
                   hyp['device'],
                   hyp['pad_amount'])


class NetworkEMA(nn.Module):
    "Maintains a mirror network whoes weights are kept as moving average of network being trained"
    def __init__(self, net, decay):
        super().__init__()  # init the parent module so this module is registered properly
        self.net_ema = copy.deepcopy(net).eval(
        ).requires_grad_(False)  # copy the model
        # you can update/hack this as necessary for update scheduling purposes :3
        self.decay = decay

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


# Hey look, it's the soft-targets/label-smoothed loss! Native to PyTorch. Now, _that_ is pretty cool, and simplifies things a lot, to boot! :D :)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction='none')

logging_columns_list = ['epoch', 'train_loss', 'val_loss',
                        'train_acc', 'val_acc', 'ema_val_acc', 'total_time_seconds']
# define the printing function and print the column heads


def print_training_details(columns_list, separator_left='|  ', separator_right='  ', final="|", column_heads_only=False, is_final_entry=False):
    print_string = ""
    if column_heads_only:
        for column_head_name in columns_list:
            print_string += separator_left + column_head_name + separator_right
        print_string += final
        print('-'*(len(print_string)))  # print the top bar
        print(print_string)
        print('-'*(len(print_string)))  # print the bottom bar
    else:
        for column_value in columns_list:
            print_string += separator_left + column_value + separator_right
        print_string += final
        print(print_string)
    if is_final_entry:
        print('-'*(len(print_string)))  # print the final output bar


# print out the training column heads before we print the actual content for each run.
print_training_details(logging_columns_list, column_heads_only=True)

########################################
#           Train and Eval             #
########################################


def train_epoch(net, inputs, targets, epoch_step, opt_sched):
    train_acc, train_loss = None, None

    # Run everything through the network
    outputs = net(inputs)

    # Hardcoded for now, preserves some accuracy during the loss summing process, balancing out its regularization effects
    loss_scale_scaler = 1./16
    # If you want to add other losses or hack around with the loss, you can do that here.
    # Note, as noted in the original blog posts, the summing here does a kind of loss scaling
    loss = loss_fn(outputs, targets).mul(
        loss_scale_scaler).sum().div(loss_scale_scaler)
    # (and is thus batchsize dependent as a result). This can be somewhat good or bad, depending...

    # we only take the last-saved accs and losses from train
    if epoch_step % 50 == 0:
        train_acc = (outputs.detach().argmax(-1) ==
                     targets).float().mean().item()
        train_loss = loss.detach().cpu().item()/batchsize

    loss.backward()

    # Step for each optimizer, in turn.
    opt_sched.step()

    opt_sched.zero_grad()

    return train_acc, train_loss


def main():
    # Initializing constants for the whole run.
    net_ema = None  # Reset any existing network emas, we want to have _something_ to check for existence so we can initialize the EMA right from where the network is during training
    # (as opposed to initializing the network_ema from the randomly-initialized starter network, then forcing it to play catch-up all of a sudden in the last several epochs)

    total_time_seconds = 0.
    current_steps = 0.

    # TODO: Doesn't currently account for partial epochs really (since we're not doing "real" epochs across the whole batchsize)....
    num_steps_per_epoch = len(data['train']['images']) // batchsize
    total_train_steps = num_steps_per_epoch * hyp['train_epochs']
    ema_epoch_start = hyp['train_epochs'] - \
        hyp['ema']['epochs']
    num_cooldown_before_freeze_steps = 0
    num_low_lr_steps_for_ema = hyp['ema']['epochs'] * \
        num_steps_per_epoch

    # I believe this wasn't logged, but the EMA update power is adjusted by being raised to the power of the number of "every n" steps
    # to somewhat accomodate for whatever the expected information intake rate is. The tradeoff I believe, though, is that this is to some degree noisier as we
    # are intaking fewer samples of our distribution-over-time, with a higher individual weight each. This can be good or bad depending upon what we want.
    projected_ema_decay_val = hyp['ema']['decay_base'] ** hyp['ema']['every_n_steps']

    # Get network
    net = make_net(data,

                   hyp['scaling_factor'],
                   hyp['device'],
                   hyp['pad_amount'])

    opt_sched = OptSched(batchsize, net, total_train_steps, num_low_lr_steps_for_ema)

    # For accurately timing GPU code
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    # There's another repository that's mainly reorganized David's code while still maintaining some of the functional structure, and it
    # has a timing feature too, but there's no synchronizes so I suspect the times reported are much faster than they may be in actuality
    # due to some of the quirks of timing GPU operations.
    torch.cuda.synchronize()  # clean up any pre-net setup operations

    if True:  # Sometimes we need a conditional/for loop here, this is placed to save the trouble of needing to indent
        for epoch in range(hyp['train_epochs']):
            #################
            # Training Mode #
            #################
            torch.cuda.synchronize()
            starter.record()
            net.train()

            loss_train = None
            accuracy_train = None
            train_acc, train_loss = None, None

            # doesn't work with torch 2.0 yet
            # train_epoch_compiled = torch.compile(train_epoch)

            for epoch_step, (inputs, targets) in enumerate(
                    get_batches(data, key='train', batchsize=batchsize, cutout_size=hyp['cutout_size'])):
                train_acc_e, train_loss_e = train_epoch(
                    net, inputs, targets, epoch_step, opt_sched)
                train_acc, train_loss = train_acc_e or train_acc, train_loss_e or train_loss

                # the '-1' is because the lr scheduler tends to overshoot (even below 0 if the final lr is ~0) on the last step for some reason.
                if current_steps < total_train_steps - num_low_lr_steps_for_ema - 1:
                    opt_sched.lr_step()

                current_steps += 1

                if epoch >= ema_epoch_start and current_steps % hyp['ema']['every_n_steps'] == 0:
                    # Initialize the ema from the network at this point in time if it does not already exist.... :D
                    if net_ema is None or epoch_step < num_cooldown_before_freeze_steps:  # don't snapshot the network yet if so!
                        net_ema = NetworkEMA(
                            net, decay=projected_ema_decay_val)
                        continue
                    net_ema.update(net)

            ender.record()
            torch.cuda.synchronize()
            total_time_seconds += 1e-3 * starter.elapsed_time(ender)

            ####################
            # Evaluation  Mode #
            ####################
            net.eval()

            eval_batchsize = 1000
            assert data['eval']['images'].shape[
                0] % eval_batchsize == 0, "Error: The eval batchsize must evenly divide the eval dataset (for now, we don't have drop_remainder implemented yet)."
            loss_list_val, acc_list, acc_list_ema = [], [], []

            with torch.no_grad():
                for inputs, targets in get_batches(data, key='eval', batchsize=eval_batchsize, cutout_size=hyp['cutout_size']):
                    if epoch >= ema_epoch_start:
                        outputs = net_ema(inputs)
                        acc_list_ema.append(
                            (outputs.argmax(-1) == targets).float().mean())
                    outputs = net(inputs)
                    loss_list_val.append(
                        loss_fn(outputs, targets).float().mean())
                    acc_list.append(
                        (outputs.argmax(-1) == targets).float().mean())

                val_acc = torch.stack(acc_list).mean().item()
                ema_val_acc = None
                # TODO: We can fuse these two operations (just above and below) all-together like :D :))))
                if epoch >= ema_epoch_start:
                    ema_val_acc = torch.stack(acc_list_ema).mean().item()

                val_loss = torch.stack(loss_list_val).mean().item()
            # We basically need to look up local variables by name so we can have the names, so we can pad to the proper column width.
            # Printing stuff in the terminal can get tricky and this used to use an outside library, but some of the required stuff seemed even
            # more heinous than this, unfortunately. So we switched to the "more simple" version of this!

            def format_for_table(x, locals): return (f"{locals[x]}".rjust(len(x))) \
                if type(locals[x]) == int else "{:0.4f}".format(locals[x]).rjust(len(x)) \
                if locals[x] is not None \
                else " "*len(x)

            # Print out our training details (sorry for the complexity, the whole logging business here is a bit of a hot mess once the columns need to be aligned and such....)
            # We also check to see if we're in our final epoch so we can print the 'bottom' of the table for each round.
            print_training_details(list(map(partial(format_for_table, locals=locals(
            )), logging_columns_list)), is_final_entry=(epoch == hyp['train_epochs'] - 1))
    # Return the final ema accuracy achieved (not using the 'best accuracy' selection strategy, which I think is okay here....)
    return ema_val_acc


if __name__ == "__main__":
    acc_list = []
    for run_num in range(1):  # use 25 for final numbers
        acc_list.append(torch.tensor(main()))
    print("Mean and variance:", (torch.mean(torch.stack(acc_list)
                                            ).item(), torch.var(torch.stack(acc_list)).item()))
