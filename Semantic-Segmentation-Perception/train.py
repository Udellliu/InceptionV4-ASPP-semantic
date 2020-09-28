import os 
import time
import math
import utils
import torch
from eval import *
import torch.nn as nn
from tqdm import tqdm
from flopth import flopth

from thop import profile
from utils import evalIoU
from networks import get_model
from torch.autograd import Variable
from dataloader.dataset import NeoData
from torch.utils.data import DataLoader
from dataloader.transform import MyTransform
from torchvision.transforms import ToPILImage
from options.train_options import TrainOptions
from torch.optim import SGD, Adam, lr_scheduler
from criterion.criterion import CrossEntropyLoss2d
from metrics import runningScore, averageMeter
from evaluation import Evaluations
from criterion.PerceptualLoss import VggLoss
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
import time
NUM_CHANNELS = 3
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def get_loader(args):
    #add the weight of each class (1/ln(c+Pclass))
    #calculate the weights of each class

    #weight[0]=1.45
    ##weight[1]=54.38
    #weight[2] = 428.723
    imagepath_train = os.path.join(args.datadir, 'train/image.txt')
    labelpath_train = os.path.join(args.datadir, 'train/label.txt')
    imagepath_val = os.path.join(args.datadir, 'test/image.txt')
    labelpath_val = os.path.join(args.datadir, 'test/label.txt')

    train_transform = MyTransform(reshape_size=(299,299),crop_size=(299,299), augment=True)  # data transform for training set with data augmentation, including resize, crop, flip and so on
    val_transform = MyTransform(reshape_size=(299,299),crop_size=(299,299), augment=False)   #data transform for validation set without data augmentation

    dataset_train = NeoData(imagepath_train, labelpath_train, train_transform) #DataSet
    dataset_val = NeoData(imagepath_val, labelpath_val, val_transform)

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    return loader, loader_val

def train(args, model):

    NUM_CLASSES = args.num_classes #pascal=21, cityscapes=20
    savedir = args.savedir
    c=-1
    for c, line in enumerate(open(r"/home/liuwenjie/ASPP_semantic/data/test/image.txt", 'rU')):
        c += 1
    print(c)
    weight = torch.ones(NUM_CLASSES)
    start_epoch = 1
    loader, loader_val = get_loader(args)

    if args.cuda:
        #criterion = CrossEntropyLoss2d(weight).cuda()
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = CrossEntropyLoss2d(weight)
        #criterion = VggLoss(weight)
    #save log
    automated_log_path = savedir + "/automated_log.txt"
    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tPerceptual-loss\t\tNLLLoss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate\t\tOverAcc\t\tMeanAcc\t\tMeanIu")

    # Setup Metrics
    running_metrics_val = runningScore(NUM_CLASSES)
    val_loss_meter = averageMeter()

    optimizer = Adam(model.parameters(), args.lr, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)    #  learning rate changed every epoch

    if args.resume:
        print("resume from chechpoint...")
        #print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))


    start_epoch = start_epoch
    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)
        epoch_loss = []
        time_train = []

        #confmatrix for calculating IoU
        confMatrix = evalIoU.generateMatrixTrainId(evalIoU.args)
        perImageStats = {}
        nbPixels = 0
        usedLr = 0
        #for param_group in optimizer.param_groups:
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()

        print(flopth(model, in_size=[3, 299, 299]))

        count = 1
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        for step, (images, labels) in enumerate(loader):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)

            outputs = model(inputs)
            #loss = criterion(outputs, targets[:, 0])
            loss_nl = criterion(outputs, targets[:,0])

            VggModel = VggLoss()
            VggModel.cuda()

            Perceptual_Loss = VggModel(outputs, targets)
            loss = 0.1*Perceptual_Loss+0.9*loss_nl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # epoch_loss.append(loss.data[0])
            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            #Add outputs to confusion matrix    #CODE USING evalIoU.py remade from cityscapes/scripts/evaluation/evalPixelLevelSemanticLabeling.py
            if (args.iouTrain):
                add_to_confMatrix(outputs, labels, confMatrix, perImageStats, nbPixels)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print('loss: {} (epoch: {}, step: {})'.format(average,epoch,step),
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        iouAvgStr, iouTrain, classScoreList = cal_iou(evalIoU, confMatrix)
        print("EPOCH IoU on TRAIN set: ", iouAvgStr)

        # calculate eval-loss and eval-IoU
        time_start = time.time()
        average_epoch_loss_val, iouVal = eval(args, model, loader_val, criterion, epoch)
        model.eval()
        with torch.no_grad():
            for i_val, (images_val, labels_val) in tqdm(enumerate(loader_val)):
                if args.cuda:
                    images_val = images.cuda()
                    labels_val = labels.cuda()
                outputs = model(images_val)
                #val_loss = loss_fn(input=outputs, target=labels_val)
                val_loss_1 = criterion(outputs, targets[:, 0])
                #print("test_val_loss_1:" , val_loss_1)
                #targets.cuda()
                #VggModel = VggLoss()
                #VggModel.cuda()
                #Perceptual_Loss = VggModel(outputs, targets)
                #print("test_Perceptual_Loss:" , Perceptual_Loss)
                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()

                running_metrics_val.update(gt, pred)
                val_loss = val_loss_1
                val_loss_meter.update(val_loss.item())

        pred_np = np.array(pred)
        gt_np = np.array(gt)
        # print(gt_np.shape)
        gt_np = np.reshape(gt_np, (-1, 299))
        gt_np = gt_np.tolist()

        pred_np = np.reshape(pred_np, (-1, 299))
        pred_np = pred_np.tolist()
        # np.savetxt('gt.txt', gt_np, fmt='%d')
        # np.savetxt('pred.txt', pred_np, fmt='%d')

        pred_np = [i[0] for i in pred_np]
        gt_np = [j[0] for j in gt_np]
        f1 = f1_score(gt_np, pred_np, average='weighted')
        precision = precision_score(gt_np, pred_np, average='weighted')
        recall = recall_score(gt_np, pred_np, average='weighted')
        print("f1: ", f1)
        print("precision: ", precision)
        print("recall: ", recall)
        allf1 = f1_score(gt_np, pred_np, average=None)
        np.savetxt('f1', allf1)


        OverallAcc,MeanAcc,MeanIoU = running_metrics_val.get_scores()
        print("OverallAcc: ", OverallAcc)
        time_end = time.time()
        testtime=(time_end-time_start)/c
        print("each image test time:",testtime)
        with open(automated_log_path, "a") as myfile:
            # myfile.write("\n%d\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.8f\t\t%.5f\t\t%.5f\t\t%.5f\t\tprecision:%s\t\trecall:%s\t\tf1:%s\t\tmean:%s" %(epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain,iouVal, usedLr,OverallAcc,MeanAcc,MeanIoU,precision,recall,f1,mean ))
            myfile.write("\n%d\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.8f\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f" % (
                epoch, average_epoch_loss_train,Perceptual_Loss,loss_nl, average_epoch_loss_val, iouTrain, iouVal, usedLr, OverallAcc, f1, precision, recall))

        if epoch % args.epoch_save == 0:
            #if iouVal >= best_iou or OverallAcc >= best_acc:
                #best_iou = iouVal
                #best_acc = OverallAcc
            n = 0
            for root, dirs, files in os.walk('./save_models/'):
                for name in files:
                    if (name.endswith(".pth")):
                        n += 1
                        os.remove(os.path.join(root, name))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, '{}_{}.pth'.format(os.path.join(args.savedir, args.model), str(epoch)))

    return(model)

def main(args):
    '''
        Train the model and record training options.
    '''
    savedir = '{}'.format(args.savedir)
    modeltxtpath = os.path.join(savedir,'model.txt')

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(savedir + '/opts.txt', "w") as myfile: #record options
        myfile.write(str(args))

    model = get_model(args)     #load model

    with open(modeltxtpath, "w") as myfile:  #record model
        myfile.write(str(model))

    if args.cuda:
        model = model.cuda()
    print("========== TRAINING ===========")
    model = train(args,model)
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':

    parser = TrainOptions().parse()
    main(parser)
