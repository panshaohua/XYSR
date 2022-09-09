'''
 FILENAME:      arch_util.py

 AUTHORS:       Pan Shaohua

 START DATE:    Friday September 9th 2022

 CONTACT:       shaohua.pan@quvideo.com

 INFO:          
'''


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.modules.batchnorm as _BatchNorm


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    '''Initialize network weights.'''

    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_blocks, **kwarg):
    layers = []
    for _ in range(num_blocks):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    '''Residual block without BN.'''

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False) -> None:
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        residual = self.conv2(self.act(self.conv1(x)))

        return identity + residual * self.res_scale