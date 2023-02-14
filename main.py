from multiprocessing import freeze_support
from timeit import default_timer as timer

import torch
torch.backends.cuda.matmul.allow_tf32 = True
from torch import nn

from models.speedyresnet import make_net #speedyresnet
from dataset import get_dataset, get_batches
from opt_sched import OptSched
from ema import NetworkEMA
from logging_utils import print_headers, print_training_details, print_device_info
from evaluation import evaluate

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


# To replicate the ~95.77% accuracy in 188 seconds runs, simply change the base_depth from 64->128 and the num_epochs from 10->80
hyp = {
    'ema_start_before_epochs': 2,
    'ema_steps': 2,
    'scaling_factor': 1./10,
    'train_epochs': 10,
    'pad_amount': 3,
    'device': 'cuda',
    'dtype': torch.float16,
    'memory_format': torch.channels_last,
    'data_cache_location': 'data_cache.pt',
    'label_smoothing': 0.2,
    'batchsize': 512,
    'eval_batchsize': 1000, # eval set size should be divisible by this number
}

batchsize = hyp['batchsize']

data = get_dataset(hyp['data_cache_location'], hyp['device'], hyp['dtype'], hyp['pad_amount'])


# Hey look, it's the soft-targets/label-smoothed loss! Native to PyTorch. Now, _that_ is pretty cool, and simplifies things a lot, to boot! :D :)
loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['label_smoothing'], reduction='none')
loss_fn.to(hyp['device'])


def train_epoch(net, inputs, targets, epoch_step, opt_sched):
    train_acc, train_loss = None, None

    # Run everything through the network
    outputs = net(inputs)

    # Hardcoded for now, preserves some accuracy during the loss summing process, balancing out its regularization effects
    loss_scale_scaler = 1./(batchsize/32.0)
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

class RollingAverage:
    def __init__(self):
        self.mean = 0
        self.count = 0

    def add(self, num):
        self.count += 1
        self.mean += (num - self.mean) / self.count
        return self.mean

def main():
    freeze_support()

    # Initializing constants for the whole run.
    net_ema = None  # Reset any existing network emas, we want to have _something_ to check for existence so we can initialize the EMA right from where the network is during training
    # (as opposed to initializing the network_ema from the randomly-initialized starter network, then forcing it to play catch-up all of a sudden in the last several epochs)

    train_time, eval_time = 0., RollingAverage()
    current_steps = 0.

    # TODO: Doesn't currently account for partial epochs really (since we're not doing "real" epochs across the whole batchsize)....
    num_steps_per_epoch = len(data['train']['images']) // batchsize
    total_train_steps = num_steps_per_epoch * hyp['train_epochs']
    ema_epoch_start = hyp['train_epochs'] - hyp['ema_start_before_epochs']
    num_low_lr_steps_for_ema = hyp['ema_start_before_epochs'] * num_steps_per_epoch

    print('train size:', len(data['train']['images']))
    print('eval size:', len(data['eval']['images']))
    print('num_steps_per_epoch:', num_steps_per_epoch)


    # Get network
    net = make_net(data, hyp['scaling_factor'], hyp['device'], hyp['pad_amount'])
    net.to(device=hyp['device'], memory_format=hyp['memory_format'], dtype=hyp['dtype'])

    opt_sched = OptSched(batchsize, net, total_train_steps, num_low_lr_steps_for_ema)

    # For accurately timing GPU code
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    # There's another repository that's mainly reorganized David's code while still maintaining some of the functional structure, and it
    # has a timing feature too, but there's no synchronizes so I suspect the times reported are much faster than they may be in actuality
    # due to some of the quirks of timing GPU operations.
    torch.cuda.synchronize()  # clean up any pre-net setup operations

    print_headers()

    if True:  # Sometimes we need a conditional/for loop here, this is placed to save the trouble of needing to indent
        for epoch in range(hyp['train_epochs']):
            #################
            # Training Mode #
            #################
            torch.cuda.synchronize()
            starter.record() # type: ignore
            net.train()

            loss_train = None
            accuracy_train = None
            train_acc, train_loss = None, None

            # doesn't work with torch 2.0 yet
            # train_epoch_compiled = torch.compile(train_epoch)

            for epoch_step, (inputs, targets) in enumerate(
                    get_batches(data, key='train', batchsize=batchsize, memory_format=hyp['memory_format'])):
                train_acc_e, train_loss_e = train_epoch(
                    net, inputs, targets, epoch_step, opt_sched)
                train_acc, train_loss = train_acc_e or train_acc, train_loss_e or train_loss

                # the '-1' is because the lr scheduler tends to overshoot (even below 0 if the final lr is ~0) on the last step for some reason.
                if current_steps < total_train_steps - num_low_lr_steps_for_ema - 1:
                    opt_sched.lr_step()

                current_steps += 1

                if epoch >= ema_epoch_start and current_steps % hyp['ema_steps'] == 0:
                    # at each ema steps init or update moving average
                    if net_ema is None:
                        net_ema = NetworkEMA(net, hyp['ema_steps'])
                    else:
                        net_ema.update(net)

            ender.record() # type: ignore
            torch.cuda.synchronize()
            train_time += 1e-3 * starter.elapsed_time(ender)

            val_loss, val_acc, ema_val_acc, eval_time_s = \
                evaluate(net, net_ema, data, hyp['eval_batchsize'],
                         epoch, loss_fn, ema_epoch_start, memory_format=hyp['memory_format'])
            eval_time.add(eval_time_s)

            # We also check to see if we're in our final epoch so we can print the 'bottom' of the table for each round.
            print_training_details(vars=locals(), is_final_entry=(epoch == hyp['train_epochs'] - 1))

    # Return the final ema accuracy achieved (not using the 'best accuracy' selection strategy, which I think is okay here....)
    return ema_val_acc, train_time, eval_time.mean


if __name__ == "__main__":
    start_time = timer()

    acc_list, train_time_list = [], []
    for run_num in range(3):  # use 25 for final numbers
        print("Run:", run_num)
        ema_val_acc, train_time, eval_time_mean = main()
        acc_list.append(torch.tensor(ema_val_acc))
        train_time_list.append(torch.tensor(train_time))

    acc_list = torch.stack(acc_list)
    train_time_list = torch.stack(train_time_list)

    print_device_info(hyp['device'])

    print("Mean, StdDev, Min, Max Accuracy:", (torch.mean(acc_list).item(),
                                                torch.std(acc_list).item(),
                                                torch.min(acc_list).item(),
                                                torch.max(acc_list).item(),
                                                ))
    print("Mean, StdDev, Min, Max Train Time:", (torch.mean(train_time_list).item(),
                                                 torch.std(train_time_list).item(),
                                                 torch.min(train_time_list).item(),
                                                 torch.max(train_time_list).item(),
                                          ))

    print("Eval time/epoch mean(s):", eval_time_mean)

    print("Wall clock(s):", timer()-start_time)
