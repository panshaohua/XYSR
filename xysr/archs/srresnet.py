'''
 FILENAME:      srresnet.py

 AUTHORS:       Pan Shaohua

 START DATE:    Friday September 9th 2022

 CONTACT:       shaohua.pan@quvideo.com

 INFO:          
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from xysr.archs.arch_util import ResidualBlockNoBN, make_layer, default_init_weights


class MSRResNet(nn.Module):
    '''Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.
    '''

    def __init__(self, in_channels=3, out_channels=3, num_feat=64, num_blocks=16, upscale_factor=4):
        super(MSRResNet, self).__init__()

        self.upscale=upscale_factor

        self.conv_head = nn.Conv2d(in_channels, num_feat, 3, 1, 1)

        self.body = make_layer(ResidualBlockNoBN, num_blocks, num_feat=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

        # activate function
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        #initialization
        default_init_weights([self.conv_head, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.act(self.conv_head(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.act(self.pixel_shuffle(self.upconv1(out)))
            out = self.act(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.act(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.act(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        out += base

        return out


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MSRResNet(in_channels=3, out_channels=3, num_feat=64, num_blocks=6, upscale_factor=2)

    model.eval()
    model = model.to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    img = torch.randn(1, 3, 360, 640)
    img = img.to(device)

    iterations = 10
    for i in range(iterations):
        start.record()
        out = model(img)
        end.record()
        torch.cuda.synchronize()

        print(start.elapsed_time(end))

    torch.onnx.export(
        model, 
        img,
        "onnx_files/srresnet.onnx",
        input_names=['lr'],
        output_names=['sr'],
        opset_version=11
    )
