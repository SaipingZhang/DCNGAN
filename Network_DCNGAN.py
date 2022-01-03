import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.dcn.deform_conv import ModulatedDeformConv
from ops.dcn.deform_conv import ModulatedDeformConvPack
# from ops.dcn.modulated_deformable_convolution import ModulatedDeformConv
# from ops.dcn.modulated_deformable_convolution import ModulatedDeformConvPack
import functools
from torch.autograd import Variable
import numpy as np
import torchvision



class FA(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):

        super(FA, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )

        self.offset = nn.Conv2d(
            nf, in_nc * 2 * self.size_dk, base_ks, padding=base_ks // 2
        )

        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc
        )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        out_lst = [self.in_conv(inputs)]
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        out = self.tr_conv(out_lst[-1])

        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )

        off = self.offset(self.out_conv(out))

        fused_feat = F.relu(self.deform_conv(inputs, off), inplace=True)
        return fused_feat


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def QE(input, input_nc=64, output_nc=1, ngf=64, n_downsample_global=3, n_blocks_global=9,
             norm='batch', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    QEnet = QEModule(input, input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        QEnet.cuda(gpu_ids[0])
    QEnet.apply(weights_init)
    return QEnet


class QEModule(nn.Module):
    def __init__(self, input, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(QEModule, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        self.fc = nn.Sequential(
            nn.Linear(4, 512),
            nn.Softplus()
        )
        self.n_blocks = n_blocks
        for i in range(0, n_blocks):
            setattr(self, 'resB' + str(i),
                    ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer))

        self.model = nn.Sequential(*model)

        model2 = [nn.ConvTranspose2d(ngf * 8, int(ngf * 8 / 2), kernel_size=3, stride=1, padding=1,
                                     output_padding=0),
                  norm_layer(int(ngf * 8 / 2)), activation]
        self.model2 = nn.Sequential(*model2)
        model3 = [nn.ConvTranspose2d(ngf * 4, int(ngf * 4 / 2), kernel_size=3, stride=1, padding=1,
                                     output_padding=0),
                  norm_layer(int(ngf * 4 / 2)), activation]
        self.model3 = nn.Sequential(*model3)
        model4 = [nn.ConvTranspose2d(ngf * 2, int(ngf * 2 / 2), kernel_size=3, stride=1, padding=1,
                                     output_padding=0),
                  norm_layer(int(ngf * 2 / 2)), activation]
        self.model4 = nn.Sequential(*model4)
        model5 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model5 = nn.Sequential(*model5)

    def forward(self, input, qp_num):
        b, c = qp_num.size()
        qp = F.one_hot(qp_num, 4)
        qp = qp.squeeze(1)
        qp = qp.to(torch.float32)
        qp = self.fc(qp)
        qp = qp.view(b, 512, 1, 1)

        out = self.model(input)
        for i in range(self.n_blocks):
            out = getattr(self, 'resB' + str(i))(out, qp)

        s1 = 2 * list(out.size())[2]
        s2 = 2 * list(out.size())[3]
        out = nn.functional.interpolate(input=out, size=(s1, s2), mode='bilinear', align_corners=False)
        out = self.model2(out)
        s1 = 2 * list(out.size())[2]
        s2 = 2 * list(out.size())[3]
        out = nn.functional.interpolate(input=out, size=(s1, s2), mode='bilinear', align_corners=False)
        out = self.model3(out)
        s1 = 2 * list(out.size())[2]
        s2 = 2 * list(out.size())[3]
        out = nn.functional.interpolate(input=out, size=(s1, s2), mode='bilinear', align_corners=False)
        out = self.model4(out)
        out = self.model5(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

        self.norm_layer = norm_layer(dim)
        self.activation = activation

        self.conv_block2 = self.build_conv_block2(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p)]
        return nn.Sequential(*conv_block)

    def build_conv_block2(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x, qp):
        conv_block_out = self.conv_block(x)
        conv_block_out = conv_block_out * qp
        conv_block_out = self.activation(self.norm_layer(conv_block_out))
        conv_block_out = self.conv_block2(conv_block_out)
        out = x + conv_block_out

        return out


class DCNGAN(nn.Module):
    def __init__(self, opts_dict):
        super(DCNGAN, self).__init__()

        self.input_len = 2 * radius + 1
        self.in_nc = 1 # for Y channel

        self.ffnet = FA(
            in_nc=self.in_nc * self.input_len,
            out_nc=64,
            nf=48,
            nb=8,
            deform_ks=3
        )

        self.QE = QE(input)

    def forward(self, radius, x, qp):
        out = self.FA(x)
        out = self.QE(out, qp)

        return out
