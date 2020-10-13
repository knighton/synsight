from torch import nn

from .layer import Append, Conv, ConvGrow, ConvShrink, Dense


class ImageToVector(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_dim):
        c = mid_channels
        super().__init__(
            nn.Conv2d(in_channels, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            ConvShrink(c, c),  # 64 to 32.
            ConvShrink(c, c),  # 32 to 16.
            ConvShrink(c, c),  # 16 to 8.
            ConvShrink(c, c),  # 8 to 4.
            ConvShrink(c, c),  # 4 to 2.
            Flatten(),
            Dense(c * 4, c),
            nn.ReLU(),
            nn.Dropout(),
            nn.Lineaar(c, out_dim),
        )


class ImageToImage(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels):
        c = mid_channels
        super().__init__(
            Append(
                ConvShrink(in_channels, c),  # 64 to 32.
                Append(
                    ConvShrink(c, c),  # 32 to 16.
                    Append(
                        ConvShrink(c, c),  # 16 to 8.
                        Append(
                            ConvShrink(c, c),  # 8 to 4.
                            Append(
                                ConvShrink(c, c),  # 4 to 2.
                                Conv(c, c),
                                ConvGrow(c, c),  # 2 to 4.
                            ),
                            ConvGrow(c + c, c),  # 4 to 8.
                        ),
                        ConvGrow(c + c, c),  # 8 to 16.
                    ),
                    ConvGrow(c + c, c),  # 16 to 32.
                ),
                ConvGrow(c + c, c),  # 32 to 64.
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels + c, out_channels),
        )


class FaceRecognizer(ImageToVector):
    def __init__(self, img_channels, vec_dim, mid_channels=64):
        super().__init__(img_channels, mid_channels, vec_dim)


class FaceMasker(ImageToImage):
    def __init__(self, img_channels, mid_channels=64):
        super().__init__(img_channels, mid_channels, 1)


class FaceSwatcher(nn.Module):
    def __init__(self, img_channels, vec_dim, mid_channels=64):
        super().__init__()
        self.path = ImageToImage(img_channels + vec_dim, mid_channels,
                                 img_channels)

    def forward(self, vec, img):
        vec = vec.unsqueeze(2).unsqueeze(3)
        x = torch([img, vec], 1)
        return self.path(x)


class SwatchRecognizer(ImageToVector):
    def __init__(self, img_channels, vec_dim, mid_channels=64):
        super().__init__(img_channels, mid_channels, vec_dim)
