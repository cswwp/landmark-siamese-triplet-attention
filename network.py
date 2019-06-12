import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from backbone.shufflenetv2 import ShuffleNetV2
from layers.pooling import MAC, SPoC, GeM, RMAC, Rpool, AttentionPool
from layers.normalization import L2N, PowerLaw
from layers.spatialAttention import SpatialAttention2d

# for some models, we have imported features (convolutions) from caffe because the image retrieval performance is higher for them
FEATURES = {
    'vgg16'         : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50'      : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet152'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth',
}

# possible global pooling layers, each on of these can be made regional
POOLING = {
    'mac'  : MAC,
    'spoc' : SPoC,
    'gem'  : GeM,
    'rmac' : RMAC,
    'attention': AttentionPool,
}


# output dimensionality for supported architectures
OUTPUT_DIM = {
    'alexnet'       :  256,
    'vgg11'         :  512,
    'vgg13'         :  512,
    'vgg16'         :  512,
    'vgg19'         :  512,
    'resnet18'      :  512,
    'resnet34'      :  512,
    'resnet50'      : 2048,
    'resnet101'     : 2048,
    'resnet152'     : 2048,
    'densenet121'   : 1024,
    'densenet161'   : 2208,
    'densenet169'   : 1664,
    'densenet201'   : 1920,
    'squeezenet1_0' :  512,
    'squeezenet1_1' :  512,
    'shufflenet'  : 1024,
}


class ImageRetrievalNet(nn.Module):

    def __init__(self, features, pool, meta, mod=''):
        super(ImageRetrievalNet, self).__init__()
        if mod == 'shuffle':
            self.features = features
        else:
            self.features = nn.Sequential(*features)
        self.pool = pool
        self.norm = L2N()
        attention = meta['attention']
        if attention:
            self.spatialAttention = SpatialAttention2d(meta['outputdim'])
        else:
            self.spatialAttention = None

    def forward(self, x):
        o = self.features(x)
        if self.spatialAttention:
            score = self.spatialAttention(o)
            # print("shape:", score.shape)
            o = (o, score)
            o = self.pool(o)
            o = self.norm(o).squeeze(-1).squeeze(-1)
            # print('reqwrwqrqwr:', o.shape)
            # a = input()
        else:
            o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)
        return o

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     attention: {}\n'.format(self.meta['attention'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(params):
    architecture = params.get('architecture', 'resnet101')
    print("architecture:", architecture)
    pooling = params.get('pooling', 'gem')
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)
    # attention = params.get('attention', False)
    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    if pretrained:
        print("pretrained***")
        if architecture not in FEATURES:
            # initialize with network pretrained on imagenet in pytorch
            if architecture.startswith('shuffle'):
                net_in = ShuffleNetV2(n_class=1000, input_width=224, input_height=224, width_mult=0.5)
                shuffle_model_weight = '/home/wangwenpeng/work/cnnimageretrieval-pytorch/cirtorch/networks/shufflenetv2_imagenet_x0.5.pth'
                pretrained_dict = torch.load(shuffle_model_weight)
                model_dict = net_in.state_dict()
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                net_in.load_state_dict(model_dict)
                print('*******************************************')

            else:
                net_in = getattr(torchvision.models, architecture)(pretrained=True)
        else:
            net_in = getattr(torchvision.models, architecture)(pretrained=True)
    else:
        # initialize with random weights
        if architecture.startswith('shuffle'):
            net_in = ShuffleNetV2_nofc(n_class=1000, input_width=224, input_height=224, width_mult=0.5)
        else:
            net_in = getattr(torchvision.models, architecture)(pretrained=False)

    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    elif architecture.startswith('shuffle'):
        features = net_in
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    meta = {
        'architecture' : architecture,
        'pooling' : pooling,
        'mean' : mean,
        'std' : std,
        'outputdim' : dim,
        'attention' : True if pooling == 'attention' else False
    }
    pool = POOLING[pooling]()
    # create a generic image retrieval network
    if architecture.startswith('shuffle'):
        net = ImageRetrievalNet(features, pool, meta, 'shuffle')
    else:
        net = ImageRetrievalNet(features, pool, meta)
    return net


if __name__ == '__main__':
    input = torch.ones([1, 3, 224, 224])
    params = {}
    net = init_network(params)
    out = net.forward(input)
    print('result', out.shape)


