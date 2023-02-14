import os
import functools
from functools import partial
import platform

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F

hyp = {
    'data': {
        'cutout_size': 0,
    }
}

def is_windows()->bool:
    return platform.system()=='Windows'

def is_persistence_workers()->bool:
    return platform.system()!='Windows' or torch.__version__.split('.')[0]!='2'

def get_dataset(data_location, device, pad_amount):
    if not os.path.exists(data_location):
        cifar10_mean, cifar10_std = [
            torch.tensor([0.4913997551666284, 0.48215855929893703,
                        0.4465309133731618], device=device),
            torch.tensor([0.24703225141799082, 0.24348516474564,
                        0.26158783926049628],  device=device)
        ]

        transform = transforms.Compose([
            transforms.ToTensor()])

        cifar10 = torchvision.datasets.CIFAR10(
            '~/dataroot/', download=True,  train=True,  transform=transform)
        cifar10_eval = torchvision.datasets.CIFAR10(
            '~/dataroot/', download=False, train=False, transform=transform)

        # use the dataloader to get a single batch of all of the dataset items at once.
        train_dataset_gpu_loader = torch.utils.data.DataLoader(cifar10, batch_size=len(cifar10), drop_last=True,
                                                            shuffle=True, num_workers=4, pin_memory=True, pin_memory_device=device)
        eval_dataset_gpu_loader = torch.utils.data.DataLoader(cifar10_eval, batch_size=len(cifar10_eval), drop_last=True,
                                                            shuffle=False, num_workers=2, pin_memory=True, pin_memory_device=device)

        train_dataset_gpu = {}
        eval_dataset_gpu = {}

        train_dataset_gpu['images'], train_dataset_gpu['targets'] = [item.to(
            device=device, non_blocking=True) for item in next(iter(train_dataset_gpu_loader))]
        eval_dataset_gpu['images'],  eval_dataset_gpu['targets'] = [item.to(
            device=device, non_blocking=True) for item in next(iter(eval_dataset_gpu_loader))]

        def batch_normalize_images(input_images, mean, std):
            return (input_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

        # preload with our mean and std
        batch_normalize_images = partial(
            batch_normalize_images, mean=cifar10_mean, std=cifar10_std)

        # Batch normalize datasets, now. Wowie. We did it! We should take a break and make some tea now.
        train_dataset_gpu['images'] = batch_normalize_images(
            train_dataset_gpu['images'])
        eval_dataset_gpu['images'] = batch_normalize_images(
            eval_dataset_gpu['images'])

        data = {
            'train': train_dataset_gpu,
            'eval': eval_dataset_gpu
        }

        # Convert dataset to FP16 now for the rest of the process....
        data['train']['images'] = data['train']['images'].half()
        data['eval']['images'] = data['eval']['images'].half()

        torch.save(data, data_location)

    else:
        # This is effectively instantaneous, and takes us practically straight to where the dataloader-loaded dataset would be. :)
        # So as long as you run the above loading process once, and keep the file on the disc it's specified by default in the above
        # hyp dictionary, then we should be good. :)
        data = torch.load(data_location)


    # As you'll note above and below, one difference is that we don't count loading the raw data to GPU since it's such a variable operation, and can sort of get in the way
    # of measuring other things. That said, measuring the preprocessing (outside of the padding) is still important to us.

    # Pad the GPU training dataset
    if pad_amount > 0:
        # Uncomfortable shorthand, but basically we pad evenly on all _4_ sides with the pad_amount specified in the original dictionary
        data['train']['images'] = F.pad(
            data['train']['images'], (pad_amount,)*4, 'reflect')

    return data


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


# TODO: Could we jit this in the (more distant) future? :)
@torch.no_grad()
def get_batches(data_dict, key, batchsize, cutout_size=hyp['data']['cutout_size']):
    num_epoch_examples = len(data_dict[key]['images'])
    shuffled = torch.randperm(num_epoch_examples, device='cuda')
    crop_size = 32
    ## Here, we prep the dataset by applying all data augmentations in batches ahead of time before each epoch, then we return an iterator below
    ## that iterates in chunks over with a random derangement (i.e. shuffled indices) of the individual examples. So we get perfectly-shuffled
    ## batches (which skip the last batch if it's not a full batch), but everything seems to be (and hopefully is! :D) properly shuffled. :)
    if key == 'train':
        images = batch_crop(data_dict[key]['images'], crop_size) # TODO: hardcoded image size for now?
        images = batch_flip_lr(images)
        images = batch_cutout(images, patch_size=cutout_size)
    else:
        images = data_dict[key]['images']

    # Send the images to an (in beta) channels_last to help improve tensor core occupancy (and reduce NCHW <-> NHWC thrash) during training
    images = images.to(memory_format=torch.channels_last)
    for idx in range(num_epoch_examples // batchsize):
        if not (idx+1)*batchsize > num_epoch_examples: ## Use the shuffled randperm to assemble individual items into a minibatch
            yield images.index_select(0, shuffled[idx*batchsize:(idx+1)*batchsize]), \
                  data_dict[key]['targets'].index_select(0, shuffled[idx*batchsize:(idx+1)*batchsize]) ## Each item is only used/accessed by the network once per epoch. :D