# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import contextlib
import os

import psutil
import ray
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.backends import cudnn
from torch.cuda.amp import GradScaler
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer

from archai.common import ml_utils, utils
from archai.common.config import Config
from archai.common.ordered_dict_logger import get_global_logger
from archai.supergraph.utils.multi_optim import MultiOptim

logger = get_global_logger()


class AmpUtils:
    def __init__(self, device, dtype, memory_format)->None:
        self.dtype = dtype
        self.device = device
        self.memory_format = memory_format

        # torch.amp has default 'O1' optimization level and cannot be configured further
        # torch.amp keeps BN in fp32
        # There is no loss_scale option in torch.amp

        self._scaler = None

        if self.is_mixed():
            # init enable mixed precision
            assert cudnn.enabled, "Amp requires cudnn backend to be enabled."
            self._scaler = GradScaler()

    def is_mixed(self)->bool:
        return self.dtype != torch.float32
    def sync_devices(self)->None:
        torch.cuda.synchronize(self.device)

    def backward(self, loss:torch.Tensor)->None:
        if self.is_mixed():
            self._scaler.scale(loss).backward()                 # pyright: ignore[reportGeneralTypeIssues, reportOptionalMemberAccess]
        else:
            loss.backward()

    def autocast(self):
        if self.is_mixed():
            return torch.cuda.amp.autocast(dtype=self.dtype)
        else:
            return contextlib.nullcontext()

    def step(self, opt_sched)->None:
        if self.is_mixed():
            #  self._scaler.unscale_ will be called automatically if it isn't called yet from grad clipping
            # https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.step
            self._scaler.step(opt_sched.opt)                            # pyright: ignore[reportOptionalMemberAccess]
            self._scaler.step(opt_sched.opt_bias)                            # pyright: ignore[reportOptionalMemberAccess]
            self._scaler.update()                                   # pyright: ignore[reportOptionalMemberAccess]
        else:
            opt_sched.step()

    def clip_grad(self, clip:float, model:nn.Module, opt_sched)->None:
        if clip > 0.0:
            if self.is_mixed():
                # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers
                self._scaler.unscale_(opt_sched.opt)            # pyright: ignore[reportOptionalMemberAccess]
                self._scaler.unscale_(opt_sched.opt_bias)            # pyright: ignore[reportOptionalMemberAccess]
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), clip)

    def state_dict(self):
        if self.is_mixed():
            return self._scaler.state_dict()            # pyright: ignore[reportOptionalMemberAccess]
        else:
            return None

    def load_state_dict(self, state_dict):
        if self.is_mixed():
            self._scaler.load_state_dict(state_dict)      # pyright: ignore[reportOptionalMemberAccess]
