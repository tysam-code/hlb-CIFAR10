from torch import nn
import torch.nn.functional as F
import torch
torch.backends.cuda.matmul.allow_tf32 = True


# set global defaults (in this particular file) for convolutions
default_conv_kwargs = {'kernel_size': 3, 'padding': 'same', 'bias': False}

hyp ={
    'net': {
        'whitening': {
            'kernel_size': 2,
            'num_examples': 50000,
        },
        'batch_norm_momentum': .8,
        'base_depth': 64 ## This should be a factor of 8 in some way to stay tensor core friendly
    },
}

# We might be able to fuse this weight and save some memory/runtime/etc, since the fast version of the network might be able to do without somehow....
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=None, weight=False, bias=True):
        assert momentum is not None
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

# Allows us to set default arguments for the whole convolution itself.


class Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kwargs = {**default_conv_kwargs, **kwargs}
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs

# can hack any changes to each residual group that you want directly in here


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, residual, short, pool, se, batch_norm_momentum):
        super().__init__()
        self.short = short
        self.pool = pool  # todo: we can condense this later
        self.se = se

        self.residual = residual
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = Conv(channels_in, channels_out)
        self.pool1 = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, momentum=batch_norm_momentum)
        self.activ = nn.GELU()

        # note: this has to be flat if we're jitting things.... we just might burn a bit of extra GPU mem if so
        if not short:
            self.conv2 = Conv(channels_out, channels_out)
            self.conv3 = Conv(channels_out, channels_out)
            self.norm2 = BatchNorm(channels_out, momentum=batch_norm_momentum)
            self.norm3 = BatchNorm(channels_out, momentum=batch_norm_momentum)

            self.se1 = nn.Linear(channels_out, channels_out//16)
            self.se2 = nn.Linear(channels_out//16, channels_out)

    def forward(self, x):
        x = self.conv1(x)
        if self.pool:
            x = self.pool1(x)
        x = self.norm1(x)
        x = self.activ(x)
        if self.short:  # layer 2 doesn't necessarily need the residual, so we just return it.
            return x
        residual = x
        if self.se:
            mult = torch.sigmoid(self.se2(self.activ(
                self.se1(torch.mean(residual, dim=(2, 3)))))).unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv3(x)
        if self.se:
            x = x * mult

        x = self.norm3(x)
        x = self.activ(x)
        x = x + residual  # haiku

        return x

# Set to 1 for now just to debug a few things....


class TemperatureScaler(nn.Module):
    def __init__(self, init_val):
        super().__init__()
        self.scaler = torch.tensor(init_val)

    def forward(self, x):
        x.float()  # save precision for the gradients in the backwards pass
        # I personally believe from experience that this is important
        # for a few reasons. I believe this is the main functional difference between
        # my implementation, and David's implementation...
        return x.mul(self.scaler)


class FastGlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Previously was chained torch.max calls.
        # requires less time than AdaptiveMax2dPooling -- about ~.3s for the entire run, in fact (which is pretty significant! :O :D :O :O <3 <3 <3 <3)
        return torch.amax(x, dim=(2, 3))  # Global maximum pooling


class SpeedyResNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict  # flexible, defined in the make_net function

    # This allows you to customize/change the execution order of the network as needed.
    def forward(self, x):
        if not self.training:
            x = torch.cat((x, torch.flip(x, (-1,))))
        x = self.net_dict['initial_block']['whiten'](x)
        x = self.net_dict['initial_block']['project'](x)
        x = self.net_dict['initial_block']['norm'](x)
        x = self.net_dict['initial_block']['activation'](x)
        x = self.net_dict['residual1'](x)
        x = self.net_dict['residual2'](x)
        x = self.net_dict['residual3'](x)
        x = self.net_dict['pooling'](x)
        x = self.net_dict['linear'](x)
        x = self.net_dict['temperature'](x)
        if not self.training:
            # Average the predictions from the lr-flipped inputs during eval
            orig, flipped = x.split(x.shape[0]//2, dim=0)
            x = .5 * orig + .5 * flipped
        return x


def get_patches(x, patch_shape=(3, 3), dtype=torch.float32):
    # TODO: Annotate
    c, (h, w) = x.shape[1], patch_shape
    # TODO: Annotate?
    return x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).to(dtype)


def get_whitening_parameters(patches):
    # TODO: Let's annotate this, please! :'D / D':
    n, c, h, w = patches.shape
    est_covariance = torch.cov(patches.view(n, c*h*w).t())
    # this is the same as saying we want our eigenvectors, with the specification that the matrix be an upper triangular matrix (instead of a lower-triangular matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(est_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.t().reshape(c*h*w, c, h, w).flip(0)

# Run this over the training set to calculate the patch statistics, then set the initial convolution as a non-learnable 'whitening' layer


def init_whitening_conv(layer, train_set=None, num_examples=None, previous_block_data=None, pad_amount=None, freeze=True, whiten_splits=None):
    if train_set is not None and previous_block_data is None:
        if pad_amount > 0:
            # if it's none, we're at the beginning of our network.
            previous_block_data = train_set[:num_examples, :,
                                            pad_amount:-pad_amount, pad_amount:-pad_amount]
        else:
            previous_block_data = train_set[:num_examples, :, :, :]
    if whiten_splits is None:
        # list of length 1 so we can reuse the splitting code down below
        previous_block_data_split = [previous_block_data]
    else:
        previous_block_data_split = previous_block_data.split(
            whiten_splits, dim=0)

    eigenvalue_list, eigenvector_list = [], []
    for data_split in previous_block_data_split:
        eigenvalues, eigenvectors = get_whitening_parameters(get_patches(
            data_split, patch_shape=layer.weight.data.shape[2:]))  # center crop to remove padding
        eigenvalue_list.append(eigenvalues)
        eigenvector_list.append(eigenvectors)

    eigenvalues = torch.stack(eigenvalue_list, dim=0).mean(0)
    eigenvectors = torch.stack(eigenvector_list, dim=0).mean(0)
    # for some reason, the eigenvalues and eigenvectors seem to come out all in float32 for this? ! ?! ?!?!?!? :'(((( </3
    set_whitening_conv(layer, eigenvalues.to(dtype=layer.weight.dtype),
                       eigenvectors.to(dtype=layer.weight.dtype), freeze=freeze)
    data = layer(previous_block_data.to(dtype=layer.weight.dtype))
    return data


def set_whitening_conv(conv_layer, eigenvalues, eigenvectors, eps=1e-2, freeze=True):
    shape = conv_layer.weight.data.shape
    conv_layer.weight.data[-eigenvectors.shape[0]:, :, :,
                           :] = (eigenvectors/torch.sqrt(eigenvalues+eps))[-shape[0]:, :, :, :]
    # We don't want to train this, since this is implicitly whitening over the whole dataset
    # For more info, see David Page's original blogposts (link in the README.md as of this commit.)
    if freeze:
        conv_layer.weight.requires_grad = False


def make_net(data, scaling_factor, device, pad_amount):
    # You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
    kernel_size = hyp['net']['whitening']['kernel_size']
    num_examples = hyp['net']['whitening']['num_examples']
    base_depth = hyp['net']['base_depth']
    batch_norm_momentum = hyp['net']['batch_norm_momentum']

    scaler = 2.
    depths = {
        # 64  w/ scaler at base value
        'init':   round(scaler**-1*base_depth),
        # 128 w/ scaler at base value
        'block1': round(scaler**1*base_depth),
        # 256 w/ scaler at base value
        'block2': round(scaler**2*base_depth),
        # 512 w/ scaler at base value
        'block3': round(scaler**3*base_depth),
        'num_classes': 10
    }
    # TODO: A way to make this cleaner??
    # Note, you have to specify any arguments overlapping with defaults (i.e. everything but in/out depths) as kwargs so that they are properly overridden (TODO cleanup somehow?)
    whiten_conv_depth = 3*kernel_size**2
    network_dict = nn.ModuleDict({
        'initial_block': nn.ModuleDict({
            'whiten': Conv(3, whiten_conv_depth, kernel_size=kernel_size, padding=0),
            'project': Conv(whiten_conv_depth, depths['init'], kernel_size=1),
            'norm': BatchNorm(depths['init'], momentum=batch_norm_momentum, weight=False),
            'activation': nn.GELU(),
        }),
        'residual1': ConvGroup(depths['init'], depths['block1'], residual=True,
                               short=False, pool=True, se=True, batch_norm_momentum=batch_norm_momentum),
        'residual2': ConvGroup(depths['block1'], depths['block2'], residual=True,
                               short=True, pool=True, se=True, batch_norm_momentum=batch_norm_momentum),
        'residual3': ConvGroup(depths['block2'], depths['block3'], residual=True,
                               short=False, pool=True, se=True, batch_norm_momentum=batch_norm_momentum),
        'pooling': FastGlobalMaxPooling(),
        'linear': nn.Linear(depths['block3'], depths['num_classes'], bias=False),
        'temperature': TemperatureScaler(scaling_factor)
    })

    net = SpeedyResNet(network_dict)
    net = net.to(device=device, dtype=data['train']['images'].dtype)

    # Initialize the whitening convolution
    with torch.no_grad():
        # Initialize the first layer to be fixed weights that whiten the expected input values of the network be on the unit hypersphere. (i.e. their...average vector length is 1.?, IIRC)
        init_whitening_conv(net.net_dict['initial_block']['whiten'],
                            data['train']['images'].index_select(0, torch.randperm(
                                data['train']['images'].shape[0], device=data['train']['images'].device)),
                            num_examples=num_examples,
                            pad_amount=pad_amount,
                            whiten_splits=5000)  # Hardcoded for now while we figure out the optimal whitening number
        # If you're running out of memory (OOM) feel free to decrease this, but
        # the index lookup in the dataloader may give you some trouble depending
        # upon exactly how memory-limited you are

    return net
