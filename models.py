from dataloader import *
import torch.nn.functional as F


# Defining the fnet model for image warping
def down_block(inputs, output_channel=64, stride=1):
    net = nn.Sequential(conv2(inputs, 3, output_channel, stride, use_bias=True), lrelu(0.2),
                        conv2(output_channel, 3, output_channel, stride, use_bias=True)
                        , lrelu(0.2), maxpool())

    return net


def up_block(inputs, output_channel=64, stride=1):
    net = nn.Sequential(conv2(inputs, 3, output_channel, stride, use_bias=True), lrelu(0.2),
                        conv2(output_channel, 3, output_channel, stride, use_bias=True)
                        , lrelu(0.2), nn.Upsample(scale_factor=2, mode="bilinear"))

    return net





# Defining the generator to upscale images
def residual_block(inputs, output_channel=64, stride=1):
    net = nn.Sequential(conv2(inputs, 3, output_channel, stride, use_bias=True), nn.ReLU(),
                        conv2(output_channel, 3, output_channel, stride, use_bias=False))

    return net


class generator(nn.Module):
    def __init__(self, gen_output_channels,args=None):
        super(generator, self).__init__()
        # create network model
        if args is None:
            raise ValueError("No args is provided for generator")

            # self.conv = nn.Sequential(conv2(17, 3, 64, 1), nn.ReLU()) #####this is bio init ,conv2(51, 3, 64, 1)is org##########
        self.conv = nn.Sequential(conv2(2, 3, 64, 1), nn.ReLU())

        self.num = args.num_resblock
        self.block_1_1 = None
        self.block_2_1 = None
        self.block_3_1 = None
        self.block_4_1 = None
        self.block_5 = None
        self.block_4_2 = None
        self.block_3_2 = None
        self.block_2_2 = None
        self.block_1_2 = None
        self.create_model()

    def forward(self, input):
        block_1_1_output = self.block_1_1(input)
        block_2_1_output = self.block_2_1(block_1_1_output)
        block_3_1_output = self.block_3_1(block_2_1_output)
        block_4_1_output = self.block_4_1(block_3_1_output)
        block_5_output = self.block_5(block_4_1_output)
        result = self.block_4_2(torch.cat((block_4_1_output, block_5_output), dim=1))
        result = self.block_3_2(torch.cat((block_3_1_output, result), dim=1))
        result = self.block_2_2(torch.cat((block_2_1_output, result), dim=1))
        result = self.block_1_2(torch.cat((block_1_1_output, result), dim=1))
        result = result + input[:, 0, :, :].unsqueeze(1)
        return result

    def create_model(self):
        kernel_size = 3
        padding = kernel_size // 2

        # block_1_1
        block_1_1 = []
        block_1_1.extend(self.add_block_conv(in_channels=2, out_channels=64, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_1.extend(self.add_block_conv(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_1.extend(self.add_block_conv(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_1_1 = nn.Sequential(*block_1_1)

        # block_2_1
        block_2_1 = [nn.MaxPool2d(kernel_size=2)]
        block_2_1.extend(self.add_block_conv(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_1.extend(self.add_block_conv(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_2_1 = nn.Sequential(*block_2_1)

        # block_3_1
        block_3_1 = [nn.MaxPool2d(kernel_size=2)]
        block_3_1.extend(self.add_block_conv(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_1.extend(self.add_block_conv(in_channels=256, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_3_1 = nn.Sequential(*block_3_1)

        # block_4_1
        block_4_1 = [nn.MaxPool2d(kernel_size=2)]
        block_4_1.extend(self.add_block_conv(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_1.extend(self.add_block_conv(in_channels=512, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_4_1 = nn.Sequential(*block_4_1)

        # block_5
        block_5 = [nn.MaxPool2d(kernel_size=2)]
        block_5.extend(self.add_block_conv(in_channels=512, out_channels=1024, kernel_size=kernel_size, stride=1,
                                           padding=padding, batchOn=True, ReluOn=True))
        block_5.extend(self.add_block_conv(in_channels=1024, out_channels=1024, kernel_size=kernel_size, stride=1,
                                           padding=padding, batchOn=True, ReluOn=True))
        block_5.extend(self.add_block_conv_transpose(in_channels=1024, out_channels=512, kernel_size=kernel_size, stride=2,
                                                     padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_5 = nn.Sequential(*block_5)

        # block_4_2
        block_4_2 = []
        block_4_2.extend(self.add_block_conv(in_channels=1024, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_2.extend(self.add_block_conv(in_channels=512, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_2.extend(
            self.add_block_conv_transpose(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_4_2 = nn.Sequential(*block_4_2)

        # block_3_2
        block_3_2 = []
        block_3_2.extend(self.add_block_conv(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_2.extend(self.add_block_conv(in_channels=256, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_2.extend(
            self.add_block_conv_transpose(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_3_2 = nn.Sequential(*block_3_2)

        # block_2_2
        block_2_2 = []
        block_2_2.extend(self.add_block_conv(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_2.extend(self.add_block_conv(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_2.extend(
            self.add_block_conv_transpose(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_2_2 = nn.Sequential(*block_2_2)

        # block_1_2
        block_1_2 = []
        block_1_2.extend(self.add_block_conv(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_2.extend(self.add_block_conv(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_2.extend(self.add_block_conv(in_channels=64, out_channels=1, kernel_size=1, stride=1,
                                             padding=0, batchOn=False, ReluOn=False))
        self.block_1_2 = nn.Sequential(*block_1_2)

    @staticmethod
    def add_block_conv(in_channels, out_channels, kernel_size, stride, padding, batchOn, ReluOn):
        seq = []
        # conv layer
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding)
        nn.init.normal_(conv.weight, 0, 0.01)
        nn.init.constant_(conv.bias, 0)
        seq.append(conv)

        # batch norm layer
        if batchOn:
            batch_norm = nn.BatchNorm2d(num_features=out_channels)
            nn.init.constant_(batch_norm.weight, 1)
            nn.init.constant_(batch_norm.bias, 0)
            seq.append(batch_norm)

        # relu layer
        if ReluOn:
            seq.append(nn.ReLU())
        return seq

    @staticmethod
    def add_block_conv_transpose(in_channels, out_channels, kernel_size, stride, padding, output_padding, batchOn, ReluOn):
        seq = []

        convt = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, output_padding=output_padding)
        nn.init.normal_(convt.weight, 0, 0.01)
        nn.init.constant_(convt.bias, 0)
        seq.append(convt)

        if batchOn:
            batch_norm = nn.BatchNorm2d(num_features=out_channels)
            nn.init.constant_(batch_norm.weight, 1)
            nn.init.constant_(batch_norm.bias, 0)
            seq.append(batch_norm)

        if ReluOn:
            seq.append(nn.ReLU())
        return seq





# Defining the discriminator for adversarial part
def discriminator_block(inputs, output_channel, kernel_size, stride):
    net = nn.Sequential(conv2(inputs, kernel_size, output_channel, stride, use_bias=False),
                        batchnorm(output_channel, is_training=True),
                        lrelu(0.2))
    return net



class discriminator_pic(nn.Module):
    def __init__(self, args=None):
        super(discriminator_pic, self).__init__()
        if args is None:
            raise ValueError("No args is provided for discriminator")
        self.conv = nn.Sequential(conv2(1, 3, 64, 1), lrelu(0.2))
        # block1
        self.block1 = discriminator_block(64, 64, 4, 2)
        self.resids1 = nn.ModuleList(
            [nn.Sequential(residual_block(64, 64, 1), batchnorm(64, True)) for i in range(int(args.discrim_resblocks))])

        # block2
        self.block2 = discriminator_block(64, args.discrim_channels, 4, 2)
        self.resids2 = nn.ModuleList([nn.Sequential(residual_block(args.discrim_channels, args.discrim_channels, 1),
                                                    batchnorm(args.discrim_channels, True)) for i in range(int(args.discrim_resblocks))])

        # block3
        self.block3 = discriminator_block(args.discrim_channels, args.discrim_channels, 4, 2)
        self.resids3 = nn.ModuleList([nn.Sequential(residual_block(args.discrim_channels, args.discrim_channels, 1),
                                                    batchnorm(args.discrim_channels, True)) for i in
                                      range(int(args.discrim_resblocks))])

        self.block4 = discriminator_block(args.discrim_channels, 64, 4, 2)

        self.block5 = discriminator_block(64, 9, 4, 2)

        # self.fc = denselayer(192, 1)##############this is for bio#############

        # self.fc = denselayer(36, 1)#############for crop-size=64 in mayor
        # self.fc = denselayer(48, 1)  #############for crop-size=128 in mayor
        self.fc = denselayer(2304, 1)  #############for crop-size=512 in mayor
    def forward(self, x):
        layer_list = []
        net = self.conv(x)
        net = self.block1(net)
        for block in self.resids1:
            net = block(net) + net
        layer_list.append(net)
        net = self.block2(net)
        for block in self.resids2:
            net = block(net) + net
        layer_list.append(net)
        net = self.block3(net)
        for block in self.resids3:
            net = block(net) + net
        layer_list.append(net)
        net = self.block4(net)
        layer_list.append(net)
        net = self.block5(net)
        net = net.view(net.shape[0], -1)
        net = self.fc(net)
        net = torch.sigmoid(net)
        return net, layer_list

class FNet(nn.Module):
    """ Optical flow estimation network
    """

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2*in_nc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True))

    def forward(self, x1, x2):
        """ Compute optical flow from x1 to x2
        """

        out = self.encoder1(torch.cat([x1, x2], dim=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = F.interpolate(
            self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False)
        # out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity
        out = torch.tanh(self.flow(out)) *24
        return out

def backward_warp(x, flow, mode='bilinear', padding_mode='border'):
    """ Backward warp `x` according to `flow`

        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat([
        flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
        flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], dim=1)

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default for PyTorch version < 1.4.0
    if int(''.join(torch.__version__.split('.')[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output


def space_to_depth(x, scale):
    """ Equivalent to tf.space_to_depth()
    """

    n, c, in_h, in_w = x.size()
    out_h, out_w = in_h // scale, in_w // scale

    x_reshaped = x.reshape(n, c, out_h, scale, out_w, scale)
    x_reshaped = x_reshaped.permute(0, 3, 5, 1, 2, 4)
    output = x_reshaped.reshape(n, scale * scale * c, out_h, out_w)

    return output

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_indexs=(8, 17, 26, 35)):
        super(VGGFeatureExtractor, self).__init__()

        # init feature layers
        self.features = torchvision.models.vgg19(pretrained=True).features
        for param in self.features.parameters():
            param.requires_grad = False
        # Notes:
        # 1. default feature layers are 8(conv2_2), 17(conv3_4), 26(conv4_4),
        #    35(conv5_4)
        # 2. features are extracted after ReLU activation
        self.feature_indexs = sorted(feature_indexs)

        # register normalization params
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # RGB
        std  = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        # assume input ranges in [0, 1]
        out = (x - self.mean) / self.std

        feature_list = []
        for i in range(len(self.features)):
            out = self.features[i](out)
            if i in self.feature_indexs:
                # clone to prevent overlapping by inplaced ReLU
                feature_list.append(out.clone())

        return feature_list