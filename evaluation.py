from timeit import default_timer as timer

import torch

from dataset import get_batches

def evaluate(net, net_ema, data, eval_batchsize, cur_epoch, loss_fn, ema_epoch_start, memory_format):
    start_time, eval_cuda_time = timer(), 0.
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)

    net.eval()

    # check if eval batch size divides eval set size
    assert data['eval']['images'].shape[
        0] % eval_batchsize == 0, "Error: The eval batchsize must evenly divide the eval dataset (for now, we don't have drop_remainder implemented yet)."
    loss_list_val, acc_list, acc_list_ema = [], [], []

    with torch.no_grad():
        for inputs, targets in get_batches(data, key='eval', batchsize=eval_batchsize, memory_format=memory_format):
            # collect ema accuracy if this is last few epochs
            if cur_epoch >= ema_epoch_start:
                outputs = net_ema(inputs)
                acc_list_ema.append((outputs.argmax(-1) == targets).float().mean())

            # collect normal accuracies
            starter.record()
            outputs = net(inputs)
            ender.record()
            torch.cuda.synchronize()
            eval_cuda_time += 1e-3 * starter.elapsed_time(ender)

            loss_list_val.append(
                loss_fn(outputs, targets).float().mean())
            acc_list.append(
                (outputs.argmax(-1) == targets).float().mean())


        # get mean of normal accuracy
        val_acc = torch.stack(acc_list).mean().item()

        # get mean of ema accuracy
        ema_val_acc = None
        # TODO: We can fuse these two operations (just above and below) all-together like :D :))))
        if cur_epoch >= ema_epoch_start:
            ema_val_acc = torch.stack(acc_list_ema).mean().item()

        val_loss = torch.stack(loss_list_val).mean().item()

        return val_loss, val_acc, ema_val_acc, eval_cuda_time

