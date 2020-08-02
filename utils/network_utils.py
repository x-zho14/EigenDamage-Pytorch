from models import *


def get_network(network, depth, dataset):
    if network == 'vgg':
        return VGG(depth=depth, dataset=dataset)
    elif network == 'resnet':
        return resnet(depth=depth, dataset=dataset)
    elif network == 'presnet':
        return presnet(depth=depth, dataset=dataset)
    elif network == 'resnet32':
        if dataset == "cifar10":
            num_classes = 10
        else:
            num_classes = 100
        return resnet32(num_classes)
    else:
        raise NotImplementedError


def get_bottleneck_builder(network):
    if network == 'vgg':
        return BottleneckVGG
    elif network == 'resnet':
        return BottleneckResNet
    elif network == 'presnet':
        return BottleneckPResNet
    else:
        raise NotImplementedError


def stablize_bn(net, trainloader, device='cuda'):
    """Iterate over the dataset for stabilizing the
    BatchNorm statistics.
    """
    net = net.train()
    for batch, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        net(inputs)
