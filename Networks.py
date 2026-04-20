import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from torch import transpose

from src.models.gdn import GDN, IGDN



class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

        self.inner_chan = 128*3
        self.out_channel = 256
        self.conv = nn.Conv2d(in_channels=self.inner_chan, out_channels=self.out_channel, kernel_size=5, stride=1,
                               padding=2)
        self.relu = nn.PReLU()
        self.GDN = GDN(self.out_channel)

    def forward(self, x, high1=None, high2=None):

        x = torch.cat((x, high1, high2), dim=1)

        x = self.conv(x)
        x = self.GDN(x)
        x = self.relu(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResBlock, self).__init__()
        self.mid_dim = 128

        self.conv1 = nn.ConvTranspose2d(in_channels=input_dim, out_channels=self.mid_dim, kernel_size=5, stride=1, padding=2)
        self.GDN1 = GDN(self.mid_dim)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.mid_dim, out_channels=input_dim, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.PReLU()

    def forward(self, x):

        feature = self.conv1(x)
        cond = self.GDN1(feature)

        feature = self.conv2(cond)
        feature = self.relu2(feature)

        return x + feature


class Adapter1(nn.Module):
    def __init__(self, inner_dim=64):
        super(Adapter1, self).__init__()
        self.img_chan = 3
        self.inner_chan = 128

        self.conv1 = nn.Conv2d(in_channels=self.img_chan, out_channels=self.inner_chan, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.PReLU()
        self.GDN1 = GDN(self.inner_chan)

        self.conv2 = nn.Conv2d(in_channels=self.inner_chan, out_channels=self.inner_chan, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.PReLU()
        self.GDN2 = GDN(self.inner_chan)

        self.conv3 = nn.Conv2d(in_channels=self.inner_chan*2, out_channels=self.inner_chan, kernel_size=3, stride=1,
                               padding=1)
        self.relu3 = nn.PReLU()
        self.GDN3 = GDN(self.inner_chan)

    def forward(self, x, high=None):

        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        x = torch.cat((x, high), dim=1)

        x = self.conv3(x)
        x = self.GDN3(x)
        x = self.relu3(x)

        return x


class Adapter2(nn.Module):
    def __init__(self, inner_dim=64):
        super(Adapter2, self).__init__()
        self.img_chan = 3
        self.inner_chan = 128

        self.conv1 = nn.Conv2d(in_channels=self.img_chan, out_channels=self.inner_chan, kernel_size=9, stride=2, padding=4)
        self.relu1 = nn.PReLU()
        self.GDN1 = GDN(self.inner_chan)

        self.conv3 = nn.Conv2d(in_channels=self.inner_chan, out_channels=self.inner_chan, kernel_size=5, stride=1,
                               padding=2)
        self.relu3 = nn.PReLU()
        self.GDN3 = GDN(self.inner_chan)

        self.conv4 = nn.Conv2d(in_channels=self.inner_chan*2, out_channels=self.inner_chan, kernel_size=3, stride=1,
                               padding=1)
        self.relu4 = nn.PReLU()
        self.GDN4 = GDN(self.inner_chan)

    def forward(self, x, high=None):

        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.GDN3(x)
        x = self.relu3(x)

        x = torch.cat((x, high), dim=1)

        x = self.conv4(x)
        x = self.GDN4(x)
        x = self.relu4(x)

        return x

class Adapter3(nn.Module):
    def __init__(self, inner_dim=64):
        super(Adapter3, self).__init__()
        self.img_chan = 3
        self.inner_chan = 128

        self.conv1 = nn.Conv2d(in_channels=self.img_chan, out_channels=self.inner_chan, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.PReLU()
        self.GDN1 = GDN(self.inner_chan)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.inner_chan, out_channels=self.inner_chan, kernel_size=5, stride=2, padding=2,
                                        output_padding=1)
        self.relu2 = nn.PReLU()
        self.GDN2 = GDN(self.inner_chan)

        self.conv3 = nn.Conv2d(in_channels=self.inner_chan*2, out_channels=self.inner_chan, kernel_size=3, stride=1,
                               padding=1)
        self.relu3 = nn.PReLU()
        self.GDN3 = GDN(self.inner_chan)

    def forward(self, x, high=None):

        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        x = torch.cat((x, high), dim=1)

        x = self.conv3(x)
        x = self.GDN3(x)
        x = self.relu3(x)

        return x

class Encoder(nn.Module):
    def __init__(self, cfg=None):
        super(Encoder, self).__init__()
        self.default_cfg = [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        if cfg is None:
            cfg = self.default_cfg

        self.inner_channel = 256

        self.feature = self.make_layers(cfg, True)

    def make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = self.inner_channel
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        return x

class Hy_Enc1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Hy_Enc1, self).__init__()

        self.inn_ch = 64
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=self.inn_ch, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.PReLU()
        self.GDN1 = GDN(self.inn_ch)

        self.conv2 = nn.Conv2d(in_channels=self.inn_ch, out_channels=out_channel, kernel_size=5, stride=2, padding=2)
        self.GDN2 = GDN(out_channel)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        return x


class Hy_Dec1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Hy_Dec1, self).__init__()

        self.inn_ch = 128
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=self.inn_ch, kernel_size=5, stride=2, padding=2,
                                        output_padding=1)
        self.relu1 = nn.PReLU()
        self.GDN1 = GDN(self.inn_ch)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.inn_ch, out_channels=self.inn_ch, kernel_size=5, stride=2,
                                        padding=2,
                                        output_padding=1)
        self.relu2 = nn.PReLU()
        self.GDN2 = GDN(self.inn_ch)

        self.conv3 = nn.ConvTranspose2d(in_channels=self.inn_ch*2, out_channels=out_channel, kernel_size=5, stride=1, padding=2)
        self.GDN3 = GDN(out_channel)
        self.relu3 = nn.PReLU()

    def forward(self, x, high=None):

        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        x = torch.cat((x, high), dim=1)

        x = self.conv3(x)
        x = self.GDN3(x)
        x = self.relu3(x)

        return x


class Hy_Enc2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Hy_Enc2, self).__init__()

        self.inn_ch = 64
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=self.inn_ch, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.PReLU()
        self.GDN1 = GDN(self.inn_ch)

        self.conv2 = nn.Conv2d(in_channels=self.inn_ch, out_channels=out_channel, kernel_size=5, stride=2, padding=2)
        self.GDN2 = GDN(out_channel)
        self.relu2 = nn.PReLU()

        # self.Transformer = Transformer(1024, 2, 4, 256 // 4, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        # b, c, h, w = x.shape
        # x = x.reshape(b, c, h * w)
        # x = self.Transformer(x)
        # x = x.reshape(b, c, h, w)

        return x


class Hy_Dec2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Hy_Dec2, self).__init__()

        self.inn_ch = 128
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=self.inn_ch, kernel_size=5, stride=2, padding=2,
                                        output_padding=1)
        self.relu1 = nn.PReLU()
        self.GDN1 = GDN(self.inn_ch)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.inn_ch*2, out_channels=out_channel, kernel_size=5, stride=1, padding=2)
        self.GDN2 = GDN(out_channel)
        self.relu2 = nn.PReLU()

    def forward(self, x, high=None):

        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = torch.cat((x, high), dim=1)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        return x

class Hy_Enc3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Hy_Enc3, self).__init__()

        self.inn_ch = 64
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=self.inn_ch, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.PReLU()
        self.GDN1 = GDN(self.inn_ch)

        self.conv2 = nn.Conv2d(in_channels=self.inn_ch, out_channels=self.inn_ch, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.PReLU()
        self.GDN2 = GDN(self.inn_ch)

        self.conv3 = nn.Conv2d(in_channels=self.inn_ch, out_channels=out_channel, kernel_size=3, stride=2, padding=1)
        self.GDN3 = GDN(out_channel)
        self.relu3 = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.GDN3(x)
        x = self.relu3(x)

        return x


class Hy_Dec3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Hy_Dec3, self).__init__()

        self.inn_ch = 128
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=self.inn_ch, kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        self.relu1 = nn.PReLU()
        self.GDN1 = GDN(self.inn_ch)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.inn_ch, out_channels=self.inn_ch, kernel_size=3, stride=2,
                                        padding=1,
                                        output_padding=1)
        self.relu2 = nn.PReLU()
        self.GDN2 = GDN(self.inn_ch)

        self.conv3 = nn.ConvTranspose2d(in_channels=self.inn_ch, out_channels=self.inn_ch, kernel_size=5, stride=2,
                                        padding=2,
                                        output_padding=1)
        self.relu3 = nn.PReLU()
        self.GDN3 = GDN(self.inn_ch)

        self.conv4 = nn.ConvTranspose2d(in_channels=self.inn_ch*2, out_channels=out_channel, kernel_size=5, stride=1, padding=2)
        self.GDN4 = GDN(out_channel)
        self.relu4 = nn.PReLU()

    def forward(self, x, high=None):

        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.GDN3(x)
        x = self.relu3(x)

        x = torch.cat((x, high), dim=1)

        x = self.conv4(x)
        x = self.GDN4(x)
        x = self.relu4(x)

        return x

class HFM(nn.Module):
    def __init__(self):
        super(HFM, self).__init__()
        self.down = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.down(x)
        high = x - F.interpolate(x1, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return high


class HF_Enc(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HF_Enc, self).__init__()

        self.inn_ch = 128
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=self.inn_ch, kernel_size=5, stride=2, padding=2)
        self.GDN1 = GDN(self.inn_ch)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=self.inn_ch, out_channels=self.inn_ch, kernel_size=5, stride=2, padding=2)
        self.GDN2 = GDN(self.inn_ch)
        self.relu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(in_channels=self.inn_ch, out_channels=out_channel, kernel_size=5, stride=1, padding=2)
        self.GDN3 = GDN(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.GDN3(x)

        return x


class HF_Dec(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HF_Dec, self).__init__()

        self.inn_ch = 128
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=self.inn_ch, kernel_size=5,
                                        stride=1, padding=2)
        self.relu1 = nn.PReLU()
        self.IGDN1 = IGDN(self.inn_ch)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.inn_ch, out_channels=self.inn_ch, kernel_size=5,
                                        stride=2, padding=2, output_padding=1)
        self.IGDN2 = IGDN(self.inn_ch)
        self.relu2 = nn.PReLU()

        self.conv3 = nn.ConvTranspose2d(in_channels=self.inn_ch, out_channels=out_channel, kernel_size=5,
                                        stride=2, padding=2, output_padding=1)
        self.IGDN3 = IGDN(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.IGDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.IGDN2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.IGDN3(x)

        return x


class HF_Ref1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HF_Ref1, self).__init__()

        self.inn_ch = 128
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=self.inn_ch, kernel_size=5, stride=1, padding=2)
        self.GDN1 = GDN(self.inn_ch)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=self.inn_ch, out_channels=out_channel, kernel_size=5, stride=1, padding=2)
        self.GDN2 = GDN(out_channel)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        return x


class HF_Ref2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HF_Ref2, self).__init__()

        self.inn_ch = 128
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=self.inn_ch, kernel_size=5, stride=2, padding=2)
        self.GDN1 = GDN(self.inn_ch)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=self.inn_ch, out_channels=out_channel, kernel_size=5, stride=1, padding=2)
        self.GDN2 = GDN(out_channel)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        return x


class HF_Ref3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HF_Ref3, self).__init__()

        self.inn_ch = 128
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=self.inn_ch, kernel_size=5, stride=1, padding=2)
        self.GDN1 = GDN(self.inn_ch)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.ConvTranspose2d(in_channels=self.inn_ch, out_channels=out_channel, kernel_size=5, stride=2, padding=2,
                                        output_padding=1)
        self.GDN2 = GDN(out_channel)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.GDN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.GDN2(x)
        x = self.relu2(x)

        return x

class Decoder1(nn.Module):
    def __init__(self, cfg=None):
        super(Decoder1, self).__init__()
        self.default_cfg = [256, 256, 256, 256]
        if cfg is None:
            cfg = self.default_cfg

        self.inner_channel = 256

        self.feature = self.make_layers(cfg, True)

    def make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = self.inner_channel
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        return x


class Decoder2(nn.Module):
    def __init__(self, cfg=None):
        super(Decoder2, self).__init__()
        self.default_cfg = [256, 256, 256, 256]
        if cfg is None:
            cfg = self.default_cfg

        self.inner_channel = 256

        self.feature = self.make_layers(cfg, True)

    def make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = self.inner_channel
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        return x


class Decoder3(nn.Module):
    def __init__(self, cfg=None):
        super(Decoder3, self).__init__()
        self.default_cfg = [256, 256, 256, 256]
        if cfg is None:
            cfg = self.default_cfg

        self.inner_channel = 256

        self.feature = self.make_layers(cfg, True)

    def make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = self.inner_channel
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        return x

class Rev1(nn.Module):
    def __init__(self, in_chan):
        super(Rev1, self).__init__()
        self.img_chan = 3
        self.inner_chan = 128

        self.conv1 = nn.ConvTranspose2d(in_channels=in_chan, out_channels=self.inner_chan, kernel_size=5,
                                        stride=1, padding=2)
        self.IGDN1 = IGDN(self.inner_chan)
        self.relu1 = nn.PReLU()

        self.res = ResBlock(self.inner_chan)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.inner_chan, out_channels=self.img_chan, kernel_size=5,
                                        stride=1, padding=2)
        self.IGDN2 = IGDN(self.img_chan)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.IGDN1(x)
        x = self.relu1(x)

        x = self.res(x)

        x = self.conv2(x)
        x = self.IGDN2(x)
        x = self.sigmoid(x)

        return x


class Rev2(nn.Module):
    def __init__(self, in_chan):
        super(Rev2, self).__init__()
        self.img_chan = 3
        self.inner_chan = 128

        self.conv1 = nn.ConvTranspose2d(in_channels=in_chan, out_channels=self.inner_chan, kernel_size=5,
                                        stride=1, padding=2)
        self.IGDN1 = IGDN(self.inner_chan)
        self.relu1 = nn.PReLU()

        self.res = ResBlock(self.inner_chan)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.inner_chan, out_channels=self.img_chan, kernel_size=5,
                                        stride=2, padding=2, output_padding=1)
        self.IGDN2 = IGDN(self.img_chan)
        self.relu2 = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.IGDN1(x)
        x = self.relu1(x)

        x = self.res(x)

        x = self.conv2(x)
        x = self.IGDN2(x)
        x = self.sigmoid(x)

        return x


class Rev3(nn.Module):
    def __init__(self, in_chan):
        super(Rev3, self).__init__()
        self.img_chan = 3
        self.inner_chan = 128

        self.conv1 = nn.ConvTranspose2d(in_channels=in_chan, out_channels=self.inner_chan, kernel_size=5,
                                        stride=1, padding=2)
        self.IGDN1 = IGDN(self.inner_chan)
        self.relu1 = nn.PReLU()

        self.res = ResBlock(self.inner_chan)

        self.conv2 = nn.Conv2d(in_channels=self.inner_chan, out_channels=self.img_chan, kernel_size=5,
                                        stride=2, padding=2)
        self.IGDN2 = IGDN(self.img_chan)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.IGDN1(x)
        x = self.relu1(x)

        x = self.res(x)

        x = self.conv2(x)
        x = self.IGDN2(x)
        x = self.sigmoid(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.reshape = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5,
                                        stride=2, padding=2, output_padding=1),
                        nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5,
                                           stride=2, padding=2, output_padding=1)
                        )

        self.fusion = nn.Conv2d(64+256, 64, 3, stride=1, padding=1)

        self.down = nn.Sequential(
            DownSalmpe(64, 64, stride=2, padding=1),
            DownSalmpe(64, 128, stride=1, padding=1),
            DownSalmpe(128, 128, stride=2, padding=1),
            DownSalmpe(128, 256, stride=1, padding=1),
            DownSalmpe(256, 256, stride=2, padding=1),
            # DownSalmpe(256, 512, stride=1, padding=1),
            # DownSalmpe(512, 512, stride=2, padding=1),
        )
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 1024, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, high=None):
        x = self.conv1(x)

        high = self.reshape(high)
        x = torch.cat((x, high), dim=1)
        x = self.fusion(x)

        x = self.down(x)
        x = self.dense(x)
        return x


class DownSalmpe(nn.Module):
    def __init__(self, input_channel, output_channel,  stride, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer(x)
        return x
