import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter as P


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        shape = (x.shape[0],) + self.shape
        return x.view(*shape)


class Flatten(Reshape):
    def __init__(self):
        super().__init__(-1)


class Conv(nn.Sequential):
    def __init__(self, in_c, out_c, face=3, pad=1):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, face, 1, pad),
            nn.BatchNorm2d(out_c),
        )


class ConvShrink(nn.Sequential):
    def __init__(self, in_c, out_c, face=3, pad=1):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, face, 2, pad),
            nn.BatchNorm2d(out_c),
        )


class Grow(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')


class ConvGrow(nn.Sequential):
    def __init__(self, in_c, out_c, face=3, pad=1):
        super().__init__(
            nn.ReLU(),
            Grow(2),
            nn.Conv2d(in_c, out_c, face, 1, pad),
            nn.BatchNorm2d(out_c),
        )


class Dense(nn.Sequential):
    def __init__(self, in_d, out_d):
        super().__init__(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_d, out_d),
            nn.BatchNorm1d(out_d),
        )


class Exp(nn.Module):
    def forward(self, x):
        return x.exp()


class Cat(nn.Module):
    def __init__(self, *paths):
        super().__init__()
        self.paths = nn.ModuleList(*paths)

    def forward(self, x):
        yy = []
        for path in self.paths:
            y = path(x)
            yy.append(y)
        return torch.cat(xx, 1)


class Append(nn.Module):
    def __init__(self, *path):
        super().__init__()
        self.path = nn.Sequential(*path)

    def forward(self, x):
        y = self.path(x)
        return torch.cat([x, y], 1)


class Skip(nn.Module):
    def __init__(self, *path):
        super().__init__()
        self.path = nn.Sequential(*path)
        self.gate = P(torch.Tensor([-2]))

    def forward(self, x):
        g = self.gate.sigmoid()
        y = self.path(x)
        return (1 - g) * x + g * y
