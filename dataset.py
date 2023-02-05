import os
import functools
from functools import partial

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F


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
                                                            shuffle=True, num_workers=2, persistent_workers=False)
        eval_dataset_gpu_loader = torch.utils.data.DataLoader(cifar10_eval, batch_size=len(cifar10_eval), drop_last=True,
                                                            shuffle=False, num_workers=1, persistent_workers=False)

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
