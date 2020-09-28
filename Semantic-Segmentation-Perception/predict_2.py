import os
import time
import torch
from options.test_options import TestOptions
from torch.autograd import Variable
import numpy as np
import time as t
from PIL import Image
from torch.utils.data import DataLoader
from utils.label2Img import label2rgb
from dataloader.transform import Transform_test
from dataloader.dataset import NeoData_test
from networks import get_model
from torchvision.utils import save_image
from eval import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main(args):
    despath = args.savedir
    if not os.path.exists(despath):
        os.mkdir(despath)
    c = -1
    for c, line in enumerate(open(r"/home/liuwenjie/ASPP_semantic/data/val/image.txt", 'rU')):
        c += 1
    print(c)
    imagedir = os.path.join(args.datadir, 'image.txt')
    labeldir = os.path.join(args.datadir, 'label.txt')

    transform = Transform_test(args.size)
    dataset_test = NeoData_test(imagedir, labeldir, transform)
    loader = DataLoader(dataset_test, num_workers=4, batch_size=1, shuffle=False)  # test data loader


    model = get_model(args)
    if args.cuda:
        model = model.cuda()
    chekhpoint = torch.load(args.model_dir)
    model.load_state_dict(chekhpoint['model_state_dict'])
    model.eval()
    count = 0
    time_start = time.time()
    for step, colign in enumerate(loader):

        img = colign[2].squeeze(0).numpy()  # image-numpy,original image
        images = colign[0]  # image-tensor
        label = colign[1]  # label-tensor

        if args.cuda:
            images = images.cuda()
            inputs = Variable(images, volatile=True)

        outputs = model(inputs)
        out = outputs[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()  # index of max-channel
        #print('out:', out)

        b, _, h, w = outputs.size()
        pred = outputs.permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes).max(1)[1].view(b, h, w)

        #add_to_confMatrix(outputs, label, confMatrix, perImageStats, nbPixels)  # add result to confusion matrix

        full_to_colour = {1: (255, 255, 255), 0: (0, 0, 0)}
        pred_remapped = pred.clone()
        pred = pred_remapped
        pred_colour = torch.zeros(b, 3, h, w)

        for k, v in full_to_colour.items():
            pred_r = torch.zeros(b, 1, h, w)
            pred = pred.reshape(1, 1, h, -1)
            pred_r[(pred == k)] = v[0]
            pred_g = torch.zeros(b, 1, h, w)
            pred_g[(pred == k)] = v[1]
            pred_b = torch.zeros(b, 1, h, w)
            pred_b[(pred == k)] = v[2]
            pred_colour.add_(torch.cat((pred_r, pred_g, pred_b), 1))
        #print(pred_colour[0].float())
        print('-----------------')
        pred = pred_colour[0].float().div(255)
        #print(pred)
        save_image(pred, despath + 'label2img_' + str(count) + '.png')
        #save_image(pred_colour, r'./predict_1_1.png')
        #label2img = label2rgb(out, img, n_labels=args.num_classes)  # merge segmented result with original picture
        #Image.fromarray(label2img).save(despath + 'label2img_' + str(count) + '.jpg')
        count += 1
        print("This is the {}th of image!".format(count))
    time_end=time.time()
    print('totally cost: ',time_end-time_start)
    alltime=time_end - time_start
    print('average time:',alltime/c)
if __name__ == '__main__':
    parser = TestOptions().parse()
    main(parser)

