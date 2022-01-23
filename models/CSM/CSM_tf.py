import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Activation, ZeroPadding2D, AveragePooling2D
from tensorflow.keras import Sequential, Model
#from pairwise import Pairwise

class Bottleneck_TF(Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(out_channels, kernel_size=3,
                               strides=stride, use_bias=False)
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(out_channels * self.expansion,
                               kernel_size=1, use_bias=False)
        self.bn3 = BatchNormalization()
        self.relu = tf.keras.activations.relu
        self.downsample = downsample

        self.padding1 = ZeroPadding2D(1)

    def call(self, x, training=False):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out

class DeconvBottleneck_TF(Model):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super().__init__()
        self.expansion = expansion
        self.conv1 = Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = BatchNormalization()
        if stride == 1:
            self.conv2 = Conv2D(out_channels, kernel_size=3,
                                   strides=stride, use_bias=False)
        else:
            self.conv2 = Conv2DTranspose(out_channels,
                                            kernel_size=3,padding='same',
                                            strides=stride, use_bias=False,
                                            output_padding=1)
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(out_channels * self.expansion,
                               kernel_size=1, use_bias=False)
        self.bn3 =BatchNormalization()
        self.relu = tf.keras.activations.relu
        self.upsample = upsample

        self.stride = stride
        self.padding1 = ZeroPadding2D(1)

    def call(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.stride==1:
            out = self.padding1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)
            
        out += shortcut
        out = self.relu(out)

        return out

class CSM_TF(Model):
    def __init__(self, downblock, upblock, num_layers, n_classes, with_energy, input_edge, n_stages = -1):
        super().__init__()

        self.in_channels = 32
        self.n_classes = n_classes
        self.n_stages = n_stages
        self.with_energy = with_energy
        down_layer_size = 3
        up_layer_size = 3
        num_inchan = 2 if input_edge else 1
        self.relu = tf.keras.activations.relu
        self.sigmoid = tf.keras.activations.sigmoid # Activation('sigmoid')
        
        self.conv1 = Conv2D(32, kernel_size=7, strides=2, use_bias=False)
        self.padding1 = ZeroPadding2D(1)
        self.padding3 = ZeroPadding2D(3)
        self.bn1 = BatchNormalization()
        self.avgpool = AveragePooling2D(3, strides=2)

        self.dlayer1 = self._make_downlayer(downblock, 32, down_layer_size)
        self.dlayer2 = self._make_downlayer(downblock, 64, down_layer_size, stride=2)
        self.dlayer3 = self._make_downlayer(downblock, 128, down_layer_size, stride=2)
        self.dlayer4 = self._make_downlayer(downblock, 256, down_layer_size, stride=2)

        # stage1
        if self.n_stages >= 1 or self.n_stages == -1:
            self.uplayer1_1 = self._make_up_block(upblock, 256, up_layer_size, stride=2)
            self.uplayer2_1 = self._make_up_block(upblock, 128, up_layer_size, stride=2)
            upsample_1 = Sequential([
                Conv2DTranspose(32,
                                   kernel_size=1, strides=2,
                                   use_bias=False, output_padding=1),
                BatchNormalization()]
            )
            self.uplayer_stage_1 = DeconvBottleneck_TF(self.in_channels, 32, 1, 2, upsample_1)
            self.conv_seg_out_1 = Conv2D(n_classes, kernel_size=1, strides=1, use_bias=False)
            if self.with_energy:
                self.conv_e_out_1 = Conv2D(n_classes, kernel_size=1, strides=1, use_bias=False)

        # stage2
        if self.n_stages >= 2 or self.n_stages == -1:
            self.uplayer1_2 = self._make_up_block(upblock, 64, up_layer_size, stride=2)
            if self.with_energy:
                self.post_cat_2 = Conv2D(128, kernel_size=1, strides=1, use_bias=False)
            else:
                self.post_cat_2 = Conv2D(128, kernel_size=1, strides=1, use_bias=False)
            self.bn_2 = BatchNormalization()
            self.uplayer2_2 = self._make_up_block(upblock, 32, up_layer_size)
            upsample_2 = Sequential([
                Conv2DTranspose(32,
                                   kernel_size=1, strides=2,
                                   use_bias=False, output_padding=1),
                BatchNormalization()
            ]
            )
            self.uplayer_stage_2 = DeconvBottleneck_TF(64, 32, 1, 2, upsample_2)
            self.conv_seg_out_2 = Conv2D(n_classes, kernel_size=1, strides=1, use_bias=False)
            if self.with_energy:
                self.conv_e_out_2 = Conv2D(n_classes, kernel_size=1, strides=1, use_bias=False)
        
    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = Sequential([
                Conv2D(init_channels*block.expansion,
                          kernel_size=1, strides=stride, use_bias=False),
                BatchNormalization()
            ])
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return Sequential(layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        if stride != 1 or self.in_channels != init_channels * 2:
            if stride == 1:
                output_padding = 0
            else:
                output_padding = 1
            upsample = Sequential([
                Conv2DTranspose(init_channels*2,
                                   kernel_size=1, strides=stride,
                                   use_bias=False, output_padding=output_padding), #1),
                BatchNormalization(),
            ])
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return Sequential(layers)

    def call(self, x):
        # img = x

        x = self.padding3(x)
        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)
        x = self.padding1(x)
        x = self.avgpool(x)

        x = self.dlayer1(x)
        x = self.dlayer2(x)
        x = self.dlayer3(x)
        x = self.dlayer4(x)

        # Mid
        x = self.uplayer1_1(x)
        x_mid = self.uplayer2_1(x)

        # Stage 1
        x_stage1 = self.uplayer_stage_1(x_mid)
        x_seg_out1 = self.conv_seg_out_1(x_stage1)
        x_hands1 = x_seg_out1
        if self.with_energy:
            x_e_out1 = self.sigmoid(self.conv_e_out_1(x_stage1))
        
        if self.n_stages == 1:
            if self.with_energy:
                return x_hands1, x_e_out1
            else:
                return x_hands1
                
        # stage2
        x_mid2 = self.uplayer1_2(x_mid)
        if self.with_energy:
            x = tf.concat([x_mid2, x_seg_out1, x_e_out1], axis = -1)
        else:
            x = tf.concat([x_mid2, x_seg_out1], axis = -1)
        x = self.post_cat_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.uplayer2_2(x)
        x = self.uplayer_stage_2(x)
        
        x_seg_out2 = self.conv_seg_out_2(x)
        x_hands2 = x_seg_out2
        if self.with_energy:
            x_e_out2 = self.sigmoid(self.conv_e_out_2(x))
        
        if self.n_stages == 2:
            if self.with_energy:
                return x_hands2, x_e_out2
            else:
                return x_hands2
        else:
            if self.with_energy:
                return x_hands1, x_e_out1, x_hands2, x_e_out2
            else:
                return x_hands1, x_hands2

def CSM_baseline(**kwargs):
    return CSM_TF(Bottleneck_TF, DeconvBottleneck_TF, [3, 3, 3, 3], **kwargs)
