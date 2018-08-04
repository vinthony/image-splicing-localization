from __future__ import print_function, absolute_import

import os
import argparse
import time
# import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
import torchvision.datasets as datasets

import sys

from tools.blockwise_view import blockwise_view
import scripts.utils
from scripts.utils.logger import Logger, savefig
from scripts.utils.evaluation import accuracy, AverageMeter, final_preds
from scripts.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from scripts.utils.osutils import mkdir_p, isfile, isdir, join
from scripts.utils.imutils import batch_with_heatmap
from scripts.utils.transforms import fliplr, flip_back
import scripts.models as models
import scripts.datasets as datasets
from tensorboardX import SummaryWriter

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

best_acc = 0


def main(args):
    global best_acc

    title = '_'+args.data+'_' + args.arch
    
    args.checkpoint = args.checkpoint + title
    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    writer = SummaryWriter(args.checkpoint+'/'+'ckpt')

    # create model
    print("==> creating model Splicing ")
    model = models.__dict__[args.arch]()

    print(model)

    wgt = torch.Tensor([1,50]);
    
    if 'columbia' in args.data:
        wgt = torch.Tensor([1,5]);
    elif 'sample' in args.data :
        wgt = torch.Tensor([1,8]);

    # define loss function (criterion) and optimizer
    criterion = torch.nn.NLLLoss(wgt)
    criterion2d = torch.nn.NLLLoss(wgt)
    criterionL1 = torch.nn.L1Loss()

    if args.gpu:    
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model).cuda()
        criterion.cuda()
        criterion2d.cuda()
        criterionL1.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=args.lr,
                                betas=(0.9,0.999),
                                weight_decay=args.weight_decay)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.finetune:
        if isfile(args.finetune):
            print("=> loading checkpoint '{}'".format(args.finetune))
            checkpoint = torch.load(args.finetune)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.finetune, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:        
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss Label','Train Loss Mask', 'Val Loss Label','Val Loss Mask', 'Train Acc Label','Train Acc Mask', 'Val Acc Label', 'Val Acc Mask', 'Loss Smooth'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    BASE_DIR = args.base_dir
    splicing_dataset_loader = datasets.Splicing

    train_loader = torch.utils.data.DataLoader(
        splicing_dataset_loader(BASE_DIR,args.data+'/train.txt',arch=args.arch),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    
    val_loader = torch.utils.data.DataLoader(
        splicing_dataset_loader(BASE_DIR,args.data+'/val.txt',arch=args.arch),
        batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    vis_loader = torch.utils.data.DataLoader(
        datasets.SplicingFull(BASE_DIR, args.data +
                                '/val.txt', arch=args.arch),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    if args.evaluate:
        print('\nEvaluation only') 
        test_loader = torch.utils.data.DataLoader(
        splicing_dataset_loader(BASE_DIR,args.data+'/test.txt',arch=args.arch),
        batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=False)

        val_loss_label, val_acc_label, val_loss_mask, val_acc_mask = validate(
            test_loader, model, [criterion, criterion2d], args)

        print('val_loss_label:', val_loss_label, 'val_acc_label:',
              val_acc_label, 'val_loss_mask:', val_loss_mask, 'val_acc_mask:', val_acc_mask)

        return

    lr = args.lr

    for epoch in range(args.start_epoch, args.epochs):

        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)

        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *=  args.sigma_decay
            val_loader.dataset.sigma *=  args.sigma_decay

        # train for one epoch
        train_loss_label, train_acc_label, train_loss_mask, train_acc_mask = train(
            train_loader, model, [criterion, criterion2d], optimizer,args)

        print('train_loss_label:', train_loss_label, 'train_acc_label:',
              train_acc_label, 'train_loss_mask:', train_loss_mask, 'train_acc_mask:', train_acc_mask)

        # evaluate on validation set
        val_loss_label, val_acc_label, val_loss_mask, val_acc_mask = validate(
            val_loader, model, [criterion, criterion2d], args)

        print('val_loss_label:', val_loss_label, 'val_acc_label:',
              val_acc_label, 'val_loss_mask:', val_loss_mask, 'val_acc_mask:', val_acc_mask)

        # Visualization train
        writer.add_scalar('train/loss/label', train_loss_label, epoch)
        writer.add_scalar('train/loss/mask', train_loss_mask, epoch)
        writer.add_scalar('train/acc/label', train_acc_label, epoch)
        writer.add_scalar('train/acc/mask', train_acc_mask, epoch)
        # Visualization val
        writer.add_scalar('val/loss/label', val_loss_label, epoch)
        writer.add_scalar('val/loss/mask', val_loss_mask, epoch)
        writer.add_scalar('val/acc/label', val_acc_label, epoch)
        writer.add_scalar('val/acc/mask', val_acc_mask, epoch)

        # visualization learning rate
        writer.add_scalar('lr', lr, epoch)

        tmp_acc = 0

        for i, (inputs, labels, target, imfull) in enumerate(vis_loader):
            # measure data loading time

            with torch.no_grad():
                inputs_var = torch.autograd.Variable(inputs.view(-1, 3, 64, 64))
                imfull_var = torch.autograd.Variable(imfull.view(-1, 3, 224, 224))

            if args.gpu:
                inputs_var = inputs_var.cuda()
                imfull_var = imfull_var.cuda()
                
            pred_label, pred_mask = model(inputs_var, imfull_var)

            _, max_cls_channel = torch.max(pred_label.cpu().data,dim=1)  
            _, max_seg_channel = torch.max(pred_mask.cpu().data,dim=1)

#           x,y            
            pred_class = max_cls_channel.view(-1,1,1,1).repeat(1,1,64,64)
            pred_class = pred_class.contiguous().view(target.size(1)//64,target.size(2)//64,64,64).permute(0,2,1,3).contiguous().view(target.size(1), target.size(2))
            pred_mask = max_seg_channel.contiguous().view(target.size(1)//64, target.size(2) //
                                                          64, 64, 64).permute(0, 2, 1, 3).contiguous().view(target.size(1), target.size(2))


            writer.add_scalar('vis/acc/label/'+str(i),(max_cls_channel == labels[0].long()).sum(), epoch)
            writer.add_scalar('vis/acc/mask/'+str(i),
                              (pred_mask == target.long()).sum(), epoch)
            
            writer.add_image('vis/label/'+str(i),pred_class.float(),epoch)
            writer.add_image('vis/seg/'+str(i), pred_mask.float(), epoch)
            writer.add_image('vis/seg_gt/'+str(i), target, epoch)

            tmp_acc = tmp_acc + (target.long() == pred_mask).sum()

        valid_acc = tmp_acc/len(vis_loader)

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    writer.close()


def train(train_loader, model, criterions, optimizer,args):

    losses_label = AverageMeter()
    acces_label = AverageMeter()
    losses_mask = AverageMeter()
    acces_mask = AverageMeter()

    criterion_classification = criterions[0]
    criterion_segmentation = criterions[1]

    # switch to train mode
    model.train()


    for i, (inputs, target, label, full_image) in enumerate(train_loader):
        # measure data loading time

        if args.gpu:
            inputs = inputs.cuda()
            full_image = full_image.cuda()
            target = target.cuda()
            label = label.cuda()
        
        input_var = torch.autograd.Variable(inputs)
        full_image_var = torch.autograd.Variable(full_image)
        target_var = torch.autograd.Variable(target.long())
        label_var = torch.autograd.Variable(label.long())
	
        # compute output
        output_label,output_mask = model(input_var,full_image_var)
        loss_label = criterion_classification(output_label, label_var)
        loss_mask = criterion_segmentation(output_mask, target_var)

        loss = loss_label + 10*loss_mask

        acc_label = accuracy(output_label.data, label)
        acc_mask = accuracy(output_mask.data, target)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses_label.update(loss_label.item(), inputs.size(0))
        losses_mask.update(loss_mask.item(), inputs.size(0))
        acces_label.update(acc_label, inputs.size(0))
        acces_mask.update(acc_mask, inputs.size(0))

    return losses_label.avg, acces_label.avg, losses_mask.avg, acces_mask.avg


def validate(val_loader, model, criterions, args):
    criterion_classification = criterions[0]
    criterion_segmentation = criterions[1]

    losses_label = AverageMeter()
    acces_label = AverageMeter()
    losses_mask = AverageMeter()
    acces_mask = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (inputs, target, label, full_image) in enumerate(val_loader):
        # measure data loading time
        if args.gpu:
            inputs = inputs.cuda()
            full_image = full_image.cuda()
            target = target.cuda()
            label = label.cuda()

        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs)
            full_image_var = torch.autograd.Variable(full_image)
            target_var = torch.autograd.Variable(target.long())
            label_var = torch.autograd.Variable(label.long())

        # compute output
        output_label,output_mask = model(input_var,full_image_var)
        
        loss_label = criterion_classification(output_label, label_var)
        loss_mask = criterion_segmentation(output_mask, target_var)

        acc_label = accuracy(output_label.data.cpu(), label.cpu())
        acc_mask = accuracy(output_mask.data.cpu(), target.cpu())

        # measure accuracy and record loss
        losses_label.update(loss_label.item(), inputs.size(0))
        losses_mask.update(loss_mask.item(), inputs.size(0))
        acces_label.update(acc_label, inputs.size(0))
        acces_mask.update(acc_mask, inputs.size(0))

    return losses_label.avg, acces_label.avg, losses_mask.avg, acces_mask.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Splicing Training')
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='iccv2017_full',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=2, type=int, metavar='N',
                        help='Number of keypoints')
    # Training strategy
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[20, 40],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('--base-dir', default='/home/mb55411/dataset/splicing/NC2016_Test/', type=str, metavar='PATH',help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--data', default='dataX', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--finetune', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')

    #modify
    parser.add_argument('--update', default='', type=str,
                        help='the impovement of hybrid method')
    parser.add_argument('--gpu',default=False,type=bool)

    main(parser.parse_args())
