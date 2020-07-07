import torch
from torch import nn
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


class Conv2dBlock(nn.Sequential):
    def __init__(self, in_c, out_c, face=3, stride=1, pad=1):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, face, stride, pad),
            nn.BatchNorm2d(out_c),
        )


class LinearBlock(nn.Sequential):
    def __init__(self, in_d, out_d):
        super().__init__(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_d, out_d),
            nn.BatchNorm1d(out_d),
        )


class Skip(nn.Module):
    def __init__(self, *path):
        super().__init__()
        self.path = nn.Sequential(*path)
        self.gate = P(torch.Tensor([-2]))

    def forward(self, x):
        g = self.gate.sigmoid()
        y = self.path(x)
        return (1 - g) * x + g * y


class Encoder(nn.Sequential):
    def __init__(self, in_channels, body_channels, out_dim):
        c = body_channels
        d = c * 9
        super().__init__(
            # 64 to 32.
            nn.Conv2d(in_channels, c // 2, 3, 1, 1),
            nn.BatchNorm2d(c // 2),
            Conv2dBlock(c // 2, c, stride=2),

            # 32 to 16.
            Conv2dBlock(c, c, stride=2),

            # 16 to 8.
            Conv2dBlock(c, c, stride=2),

            # 8 to 4.
            Skip(
                Conv2dBlock(c, c, stride=1),
            ),
            Conv2dBlock(c, c, stride=2),

            # 4 to 2.
            #Skip(
            #    Conv2dBlock(c, c, stride=1),
            #),
            #Conv2dBlock(c, c, stride=2),

            # Dense.
            Flatten(),
            Skip(
                LinearBlock(d, d),
            ),
            nn.Linear(d, out_dim),
            nn.Sigmoid(),
        )
