import toplayer
import torch
from torch import nn, Tensor
from typing import Union, Type, List, Optional, Callable

class Conv3x3(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        stride,
        input_size,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.input_size = input_size

        # Note that 3x3 kernels are replaced by 2x2 when stride > 1
        if stride == 1:
            kern_size = 3
        else:
            kern_size = 1
        self.topolayer = toplayer.TopLayer(in_channels * stride ** 2, out_channels, kern_size, 1, input_size // stride, bias=False)

    def forward(self, X):
        if self.stride != 1:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2] // self.stride, self.stride, X.shape[3]//self.stride, self.stride)
            X = X.permute(0, 1, 3, 5, 2, 4).reshape(X.shape[0], X.shape[1] * self.stride ** 2, X.shape[2], X.shape[4])
        
        return self.topolayer(X)


class Conv1x1(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        stride,
        input_size,
    ):
        super().__init__()
        assert stride > 1
        self.in_channels = in_channels
        self.stride = stride
        self.topolayer = toplayer.TopLayer(in_channels * stride ** 2, out_channels, 1, 1, input_size, bias=False)
        

    def forward(self, X):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2]//self.stride, 2, X.shape[3]//self.stride, 2)
        X = X.permute(0, 1, 3, 5, 2, 4).reshape(X.shape[0], X.shape[1] * self.stride ** 2, X.shape[2], X.shape[4])
        return self.topolayer(X)

class DownsampleUnstack(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, X):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2]//self.stride, 2, X.shape[3]//self.stride, 2)
        X = X.permute(0, 1, 3, 5, 2, 4).reshape(X.shape[0], X.shape[1] * self.stride ** 2, X.shape[2], X.shape[4])
        return X

    def extra_repr(self) -> str:
        return f"stride={self.stride}"

def conv1x1(
        in_channels,
        out_channels,
        stride,
        input_size,
    ):
    assert stride > 1
    return nn.Sequential(
        DownsampleUnstack(stride=stride),
        toplayer.TopLayer(in_channels * stride ** 2, out_channels, 1, 1, input_size, bias=False)
    )

def conv3x3(
        in_channels,
        out_channels,
        stride,
        input_size,
    ):
    if stride == 1:
        kern_size = 3
    else:
        kern_size = 1

    if stride == 1:
        return toplayer.TopLayer(in_channels * stride ** 2, out_channels, kern_size, 1, input_size // stride, bias=False)
    else:
        return nn.Sequential(
            DownsampleUnstack(stride=stride),
            toplayer.TopLayer(in_channels * stride ** 2, out_channels, kern_size, 1, input_size // stride, bias=False)
        )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_size: int = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes, planes, stride, input_size=input_size)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(planes, planes, stride=1, input_size=input_size // stride)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class Frontend(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        input_size
    ):
        # Note that the frontend necessarily has a stride of 2.
        super().__init__()
        self.stride = 2
        self.in_channels = in_channels
        self.topolayer = toplayer.TopLayer(in_channels * self.stride ** 2, out_channels, 3, 1, input_size // self.stride, bias=False)

    def forward(self, X):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2] // self.stride, self.stride, X.shape[3] // self.stride, self.stride)
        X = X.permute(0, 1, 3, 5, 2, 4).reshape(X.shape[0], X.shape[1] * self.stride ** 2, X.shape[2], X.shape[4])
        return self.topolayer(X)

def frontend(in_channels, out_channels, input_size):
    stride = 2
    return nn.Sequential(
        DownsampleUnstack(stride=stride),
        toplayer.TopLayer(in_channels * stride ** 2, out_channels, 3, 1, input_size // stride, bias=False)
    )

class TopNet(nn.Module):
    def __init__(
        self,
        block: BasicBlock,  # Only implemented BasicBlock, so can only implement ResNet18 or 34 analogues.
        layers: List[int],
        num_classes: int = 1000,
        input_size: int = 64,
        use_maxpool: bool = True,
        in_planes: int = 8,
    ) -> None:
        super().__init__()

        assert input_size % 32 == 0, "input_size must be divisible by 32"
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = 64
        self.in_planes = in_planes
        self.dilation = 1
        self.input_size = input_size
        groups = 1
        
        self.groups = groups
        
        self.conv1 = frontend(3, in_planes, input_size)

        self.bn1 = norm_layer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.use_maxpool = use_maxpool
        if self.use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, in_planes, layers[0], input_size=input_size//4)
        self.layer2 = self._make_layer(block, in_planes * 2, layers[1], stride=2, input_size=input_size//4)
        self.layer3 = self._make_layer(block, in_planes * 4, layers[2], stride=2, input_size=input_size//8)
        self.layer4 = self._make_layer(block, in_planes * 8, layers[3], stride=2, input_size=input_size//16)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8 * in_planes * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: BasicBlock,
        planes: int,
        blocks: int,
        stride: int = 1,
        input_size: int = None
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        input_size_2 = input_size
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride, input_size // stride),
                norm_layer(planes * block.expansion),
            )
            input_size_2 = input_size // stride

        layers = []
        layers.append(
            block(
                self.in_planes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, input_size
            )
        )


        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    input_size=input_size_2
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        assert x.shape[2] == self.input_size, f"Input images must be of prespecified input_size {self.input_size}"
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def TopNet18(input_size=64, in_planes=8):
    return TopNet(BasicBlock, [2, 2, 2, 2], input_size=input_size, in_planes=in_planes)


if __name__ == '__main__':
    # TODO: 
    # benchmark
    # Generalize wrt. parameters
    #          convolutions
    # Implement bias.
    # Base width.
    import time

    net = TopNet18(in_planes=16)
    net.to(device='cuda')
    X = torch.randn(64, 3, 64, 64, device='cuda')
    net(X)

    """
    timings = []

    for sz in [64, 128, 256, 512, 1024, 2048]:
        X = torch.randn(64, 3, 64, 64, device='cuda')
        t0 = time.time()
        Y = net(X)
        g = (Y.sum()).backward()
        timings.append({
            'time': (time.time() - t0),
            'sz': sz,
            'model': 'topnet'
        })
        del X

    del net

    from torchvision import models
    net = models.resnet18()
    net.to(device='cuda')

    for sz in [64, 128, 256, 512, 1024, 2048]:
        X = torch.randn(64, 3, 64, 64, device='cuda')
        t0 = time.time()
        Y = net(X)
        g = (Y.sum()).backward()
        timings.append({
            'time': (time.time() - t0),
            'sz': sz,
            'model': 'resnet'
        })
        del X

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(timings)

    f = sns.lineplot(data=df, x="sz", y="time", hue="model")
    f.set(xscale='log', yscale='log')

    f
    """

    

    