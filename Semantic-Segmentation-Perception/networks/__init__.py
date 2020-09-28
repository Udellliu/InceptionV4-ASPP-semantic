from .fcn import FCN8, FCN16, FCN32
from .erfnet import ERFNet
from .pspnet import PSPNet
from .segnet import SegNet
from .unet import UNet
from .inceptionv4 import InceptionV4
from .inceptionresnetv2 import InceptionResNetV2
from .utils import *

net_dic = {'erfnet' : ERFNet, 'fcn8' : FCN8, 'fcn16' : FCN16, 
                'fcn32' : FCN32, 'unet' : UNet, 'pspnet': PSPNet, 'segnet' : SegNet,'inceptionv4':InceptionV4,'inresv2':InceptionResNetV2}
                

def get_model(args):

    Net = net_dic[args.model]
    model = Net(args.num_classes)
    model.apply(weights_init)
    return model
