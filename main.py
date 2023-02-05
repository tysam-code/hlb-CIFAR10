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
import torch.nn.functional as F
from torch import nn

from dataset import get_dataset
from network import make_net

## <-- teaching comments
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
#torch.cuda.set_per_process_memory_fraction(fraction=8./40., device=0) ## 40. GB is the maximum memory of the base A100 GPU



batchsize = 512
bias_scaler = 32
# To replicate the ~95.77% accuracy in 188 seconds runs, simply change the base_depth from 64->128 and the num_epochs from 10->80
hyp = {
    'opt': {
        'bias_lr':        1.15 * 1.35 * 1. * bias_scaler/batchsize, # TODO: How we're expressing this information feels somewhat clunky, is there maybe a better way to do this? :'))))
        'non_bias_lr':    1.15 * 1.35 * 1. / batchsize,
        'bias_decay':     .85 * 4.8e-4 * batchsize/bias_scaler,
        'non_bias_decay': .85 * 4.8e-4 * batchsize,
        'scaling_factor': 1./10,
        'percent_start': .2,
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
            'num_examples': 50000,
        },
        'batch_norm_momentum': .8,
        'cutout_size': 0,
        'pad_amount': 3,
        'base_depth': 64 ## This should be a factor of 8 in some way to stay tensor core friendly
    },
    'misc': {
        'ema': {
            'epochs': 2,
            'decay_base': .986,
            'every_n_steps': 2,
        },
        'train_epochs': 10,
        'device': 'cuda',
        'data_location': 'data.pt',
    }
}

data = get_dataset(hyp['misc']['data_location'],
                   hyp['misc']['device'],
                   hyp['net']['pad_amount'])


## This is actually (I believe) a pretty clean implementation of how to do something like this, since shifted-square masks unique to each depth-channel can actually be rather
## tricky in practice. That said, if there's a better way, please do feel free to submit it! This can be one of the harder parts of the code to understand (though I personally get
## stuck on the fold/unfold process for the lower-level convolution calculations.
def make_random_square_masks(inputs, mask_size):
    ##### TODO: Double check that this properly covers the whole range of values. :'( :')
    if mask_size == 0:
        return None # no need to cutout or do anything like that since the patch_size is set to 0
    is_even = int(mask_size % 2 == 0)
    in_shape = inputs.shape

    # seed centers of squares to cutout boxes from, in one dimension each
    mask_center_y = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size//2-is_even, in_shape[-2]-mask_size//2-is_even)
    mask_center_x = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size//2-is_even, in_shape[-1]-mask_size//2-is_even)

    # measure distance, using the center as a reference point
    to_mask_y_dists = torch.arange(in_shape[-2], device=inputs.device).view(1, 1, in_shape[-2], 1) - mask_center_y.view(-1, 1, 1, 1)
    to_mask_x_dists = torch.arange(in_shape[-1], device=inputs.device).view(1, 1, 1, in_shape[-1]) - mask_center_x.view(-1, 1, 1, 1)

    to_mask_y = (to_mask_y_dists >= (-(mask_size // 2) + is_even)) * (to_mask_y_dists <= mask_size // 2)
    to_mask_x = (to_mask_x_dists >= (-(mask_size // 2) + is_even)) * (to_mask_x_dists <= mask_size // 2)

    final_mask = to_mask_y * to_mask_x ## Turn (y by 1) and (x by 1) boolean masks into (y by x) masks through multiplication. Their intersection is square, hurray! :D

    return final_mask

def batch_cutout(inputs, patch_size):
    with torch.no_grad():
        cutout_batch_mask = make_random_square_masks(inputs, patch_size)
        if cutout_batch_mask is None:
            return inputs # if the mask is None, then that's because the patch size was set to 0 and we will not be using cutout today.
        # TODO: Could be fused with the crop operation for sheer speeeeeds. :D <3 :))))
        cutout_batch = torch.where(cutout_batch_mask, torch.zeros_like(inputs), inputs)
        return cutout_batch

def batch_crop(inputs, crop_size):
    with torch.no_grad():
        crop_mask_batch = make_random_square_masks(inputs, crop_size)
        cropped_batch = torch.masked_select(inputs, crop_mask_batch).view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)
        return cropped_batch

def batch_flip_lr(batch_images, flip_chance=.5):
    with torch.no_grad():
        # TODO: Is there a more elegant way to do this? :') :'((((
        return torch.where(torch.rand_like(batch_images[:, 0, 0, 0].view(-1, 1, 1, 1)) < flip_chance, torch.flip(batch_images, (-1,)), batch_images)


########################################
#          Training Helpers            #
########################################

class NetworkEMA(nn.Module):
    def __init__(self, net, decay):
        super().__init__() # init the parent module so this module is registered properly
        self.net_ema = copy.deepcopy(net).eval().requires_grad_(False) # copy the model
        self.decay = decay ## you can update/hack this as necessary for update scheduling purposes :3

    def update(self, current_net):
        with torch.no_grad():
            for ema_net_parameter, incoming_net_parameter in zip(self.net_ema.state_dict().values(), current_net.state_dict().values()): # potential bug: assumes that the network architectures don't change during training (!!!!)
                if incoming_net_parameter.dtype in (torch.half, torch.float):
                    ema_net_parameter.mul_(self.decay).add_(incoming_net_parameter.detach().mul(1. - self.decay)) # update the ema values in place, similar to how optimizer momentum is coded

    def forward(self, inputs):
        with torch.no_grad():
            return self.net_ema(inputs)

# TODO: Could we jit this in the (more distant) future? :)
@torch.no_grad()
def get_batches(data_dict, key, batchsize):
    num_epoch_examples = len(data_dict[key]['images'])
    shuffled = torch.randperm(num_epoch_examples, device='cuda')
    crop_size = 32
    ## Here, we prep the dataset by applying all data augmentations in batches ahead of time before each epoch, then we return an iterator below
    ## that iterates in chunks over with a random derangement (i.e. shuffled indices) of the individual examples. So we get perfectly-shuffled
    ## batches (which skip the last batch if it's not a full batch), but everything seems to be (and hopefully is! :D) properly shuffled. :)
    if key == 'train':
        images = batch_crop(data_dict[key]['images'], crop_size) # TODO: hardcoded image size for now?
        images = batch_flip_lr(images)
        images = batch_cutout(images, patch_size=hyp['net']['cutout_size'])
    else:
        images = data_dict[key]['images']

    # Send the images to an (in beta) channels_last to help improve tensor core occupancy (and reduce NCHW <-> NHWC thrash) during training
    images = images.to(memory_format=torch.channels_last)
    for idx in range(num_epoch_examples // batchsize):
        if not (idx+1)*batchsize > num_epoch_examples: ## Use the shuffled randperm to assemble individual items into a minibatch
            yield images.index_select(0, shuffled[idx*batchsize:(idx+1)*batchsize]), \
                  data_dict[key]['targets'].index_select(0, shuffled[idx*batchsize:(idx+1)*batchsize]) ## Each item is only used/accessed by the network once per epoch. :D


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


## Hey look, it's the soft-targets/label-smoothed loss! Native to PyTorch. Now, _that_ is pretty cool, and simplifies things a lot, to boot! :D :)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction='none')

logging_columns_list = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'ema_val_acc', 'total_time_seconds']
# define the printing function and print the column heads
def print_training_details(columns_list, separator_left='|  ', separator_right='  ', final="|", column_heads_only=False, is_final_entry=False):
    print_string = ""
    if column_heads_only:
        for column_head_name in columns_list:
            print_string += separator_left + column_head_name + separator_right
        print_string += final
        print('-'*(len(print_string))) # print the top bar
        print(print_string)
        print('-'*(len(print_string))) # print the bottom bar
    else:
        for column_value in columns_list:
            print_string += separator_left + column_value + separator_right
        print_string += final
        print(print_string)
    if is_final_entry:
        print('-'*(len(print_string))) # print the final output bar

print_training_details(logging_columns_list, column_heads_only=True) ## print out the training column heads before we print the actual content for each run.

########################################
#           Train and Eval             #
########################################

def train_epoch(net, inputs, targets, epoch_step, opt, opt_bias):
    train_acc, train_loss = None, None

    ## Run everything through the network
    outputs = net(inputs)

    loss_scale_scaler = 1./16 # Hardcoded for now, preserves some accuracy during the loss summing process, balancing out its regularization effects
    ## If you want to add other losses or hack around with the loss, you can do that here.
    loss = loss_fn(outputs, targets).mul(loss_scale_scaler).sum().div(loss_scale_scaler) ## Note, as noted in the original blog posts, the summing here does a kind of loss scaling
                                            ## (and is thus batchsize dependent as a result). This can be somewhat good or bad, depending...

    # we only take the last-saved accs and losses from train
    if epoch_step % 50 == 0:
        train_acc = (outputs.detach().argmax(-1) == targets).float().mean().item()
        train_loss = loss.detach().cpu().item()/batchsize

    loss.backward()

    ## Step for each optimizer, in turn.
    opt.step()
    opt_bias.step()

    ## Using 'set_to_none' I believe is slightly faster (albeit riskier w/ funky gradient update workflows) than under the default 'set to zero' method
    opt.zero_grad(set_to_none=True)
    opt_bias.zero_grad(set_to_none=True)

    return train_acc, train_loss

def main():
    # Initializing constants for the whole run.
    net_ema = None ## Reset any existing network emas, we want to have _something_ to check for existence so we can initialize the EMA right from where the network is during training
                   ## (as opposed to initializing the network_ema from the randomly-initialized starter network, then forcing it to play catch-up all of a sudden in the last several epochs)

    total_time_seconds = 0.
    current_steps = 0.

    # TODO: Doesn't currently account for partial epochs really (since we're not doing "real" epochs across the whole batchsize)....
    num_steps_per_epoch      = len(data['train']['images']) // batchsize
    total_train_steps        = num_steps_per_epoch * hyp['misc']['train_epochs']
    ema_epoch_start          = hyp['misc']['train_epochs'] - hyp['misc']['ema']['epochs']
    num_cooldown_before_freeze_steps = 0
    num_low_lr_steps_for_ema = hyp['misc']['ema']['epochs'] * num_steps_per_epoch

    ## I believe this wasn't logged, but the EMA update power is adjusted by being raised to the power of the number of "every n" steps
    ## to somewhat accomodate for whatever the expected information intake rate is. The tradeoff I believe, though, is that this is to some degree noisier as we
    ## are intaking fewer samples of our distribution-over-time, with a higher individual weight each. This can be good or bad depending upon what we want.
    projected_ema_decay_val  = hyp['misc']['ema']['decay_base'] ** hyp['misc']['ema']['every_n_steps']

    # Adjust pct_start based upon how many epochs we need to finetune the ema at a low lr for
    pct_start = hyp['opt']['percent_start'] * (total_train_steps/(total_train_steps - num_low_lr_steps_for_ema))

    # Get network
    net = make_net(data,
                   hyp['net']['whitening']['kernel_size'],
                   hyp['opt']['scaling_factor'],
                   hyp['misc']['device'],
                   hyp['net']['whitening']['num_examples'],
                   hyp['net']['pad_amount'],
                   hyp['net']['base_depth'],
                   hyp['net']['batch_norm_momentum'])

    ## Stowing the creation of these into a helper function to make things a bit more readable....
    non_bias_params, bias_params = init_split_parameter_dictionaries(net)

    # One optimizer for the regular network, and one for the biases. This allows us to use the superconvergence onecycle training policy for our networks....
    opt = torch.optim.SGD(**non_bias_params)
    opt_bias = torch.optim.SGD(**bias_params)

    #opt = torch.optim.SGD(**non_bias_params)
    #opt_bias = torch.optim.SGD(**bias_params)

    ## Not the most intuitive, but this basically takes us from ~0 to max_lr at the point pct_start, then down to .1 * max_lr at the end (since 1e16 * 1e-15 = .1 --
    ##   This quirk is because the final lr value is calculated from the starting lr value and not from the maximum lr value set during training)
    initial_div_factor = 1e16 # basically to make the initial lr ~0 or so :D
    final_lr_ratio = .135
    lr_sched      = torch.optim.lr_scheduler.OneCycleLR(opt,  max_lr=non_bias_params['lr'], pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=total_train_steps-num_low_lr_steps_for_ema, anneal_strategy='linear', cycle_momentum=False)
    lr_sched_bias = torch.optim.lr_scheduler.OneCycleLR(opt_bias, max_lr=bias_params['lr'], pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=total_train_steps-num_low_lr_steps_for_ema, anneal_strategy='linear', cycle_momentum=False)

    ## For accurately timing GPU code
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ## There's another repository that's mainly reorganized David's code while still maintaining some of the functional structure, and it
    ## has a timing feature too, but there's no synchronizes so I suspect the times reported are much faster than they may be in actuality
    ## due to some of the quirks of timing GPU operations.
    torch.cuda.synchronize() ## clean up any pre-net setup operations


    if True: ## Sometimes we need a conditional/for loop here, this is placed to save the trouble of needing to indent
        for epoch in range(hyp['misc']['train_epochs']):
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
          #train_epoch_compiled = torch.compile(train_epoch)

          for epoch_step, (inputs, targets) in enumerate(get_batches(data, key='train', batchsize=batchsize)):
              train_acc_e, train_loss_e = train_epoch(net, inputs, targets, epoch_step, opt, opt_bias)
              train_acc, train_loss = train_acc_e or train_acc, train_loss_e or train_loss

              if current_steps < total_train_steps - num_low_lr_steps_for_ema - 1: # the '-1' is because the lr scheduler tends to overshoot (even below 0 if the final lr is ~0) on the last step for some reason.
                # We only want to step the lr_schedulers while we have training steps to consume. Otherwise we get a not-so-friendly error from PyTorch
                lr_sched.step()
                lr_sched_bias.step()

              current_steps += 1

              if epoch >= ema_epoch_start and current_steps % hyp['misc']['ema']['every_n_steps'] == 0:
                  ## Initialize the ema from the network at this point in time if it does not already exist.... :D
                  if net_ema is None or epoch_step < num_cooldown_before_freeze_steps: # don't snapshot the network yet if so!
                      net_ema = NetworkEMA(net, decay=projected_ema_decay_val)
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
          assert data['eval']['images'].shape[0] % eval_batchsize == 0, "Error: The eval batchsize must evenly divide the eval dataset (for now, we don't have drop_remainder implemented yet)."
          loss_list_val, acc_list, acc_list_ema = [], [], []

          with torch.no_grad():
              for inputs, targets in get_batches(data, key='eval', batchsize=eval_batchsize):
                  if epoch >= ema_epoch_start:
                      outputs = net_ema(inputs)
                      acc_list_ema.append((outputs.argmax(-1) == targets).float().mean())
                  outputs = net(inputs)
                  loss_list_val.append(loss_fn(outputs, targets).float().mean())
                  acc_list.append((outputs.argmax(-1) == targets).float().mean())

              val_acc = torch.stack(acc_list).mean().item()
              ema_val_acc = None
              # TODO: We can fuse these two operations (just above and below) all-together like :D :))))
              if epoch >= ema_epoch_start:
                  ema_val_acc = torch.stack(acc_list_ema).mean().item()

              val_loss = torch.stack(loss_list_val).mean().item()
          # We basically need to look up local variables by name so we can have the names, so we can pad to the proper column width.
          ## Printing stuff in the terminal can get tricky and this used to use an outside library, but some of the required stuff seemed even
          ## more heinous than this, unfortunately. So we switched to the "more simple" version of this!
          format_for_table = lambda x, locals: (f"{locals[x]}".rjust(len(x))) \
                                                    if type(locals[x]) == int else "{:0.4f}".format(locals[x]).rjust(len(x)) \
                                                if locals[x] is not None \
                                                else " "*len(x)

          # Print out our training details (sorry for the complexity, the whole logging business here is a bit of a hot mess once the columns need to be aligned and such....)
          ## We also check to see if we're in our final epoch so we can print the 'bottom' of the table for each round.
          print_training_details(list(map(partial(format_for_table, locals=locals()), logging_columns_list)), is_final_entry=(epoch == hyp['misc']['train_epochs'] - 1))
    return ema_val_acc # Return the final ema accuracy achieved (not using the 'best accuracy' selection strategy, which I think is okay here....)

if __name__ == "__main__":
    acc_list = []
    for run_num in range(1): # use 25 for final numbers
        acc_list.append(torch.tensor(main()))
    print("Mean and variance:", (torch.mean(torch.stack(acc_list)).item(), torch.var(torch.stack(acc_list)).item()))
