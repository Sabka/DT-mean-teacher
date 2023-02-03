import os
import shutil

import numpy as np
import torch

from Datasets.data import NO_LABEL
from misc.utils import *
# from tensorboardX import SummaryWriter
import datetime
from parameters import get_parameters
import models

from misc import ramps
from Datasets import data
from models import losses

import torchvision.transforms as transforms


import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets

np.random.seed(5)
torch.manual_seed(5)

args = None

best_prec1 = 0
global_step = 0

def main(args):

    global global_step
    global best_prec1

    train_transform = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness= 0.4, contrast = 0.4, saturation = 0.4, hue = 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470,  0.2435,  0.2616))]))

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470,  0.2435,  0.2616))
    ])

    traindir = os.path.join(args.datadir, args.train_subdir)
    evaldir = os.path.join(args.datadir, args.eval_subdir)

    dataset = torchvision.datasets.ImageFolder(traindir, train_transform)


    # TODO refactor, just trying
    total_labels = 50000
    num_classes = 10
    labeled_portion = 4000 # max 35 000
    per_class = labeled_portion/num_classes # max 3 500

    anim = {'bird', 'frog', 'cat', 'horse', 'dog', 'deer'}
    inanim = {'ship', 'truck', 'automobile', 'airplane'}

    labels = {}

    for group in [anim, inanim]:
        for cls in group:
            with open('data-local/labels/custom/'+ cls +".txt", "r") as f:
                labels_tmp = {}
                for line in f:
                    img, lab = line.strip().split(' ')
                    labels_tmp[img] = lab
                    if len(labels_tmp) == per_class: break
                labels.update(labels_tmp)

    labeled_idxs, unlabeled_idxs, label_frequencies = data.relabel_dataset(dataset, labels)

    print(f'==> Labeled: {len(labeled_idxs)}, ratio [A/INA]: {label_frequencies[dataset.class_to_idx["animate"]]}/{label_frequencies[dataset.class_to_idx["inanimate"]]}, unlabeled: {len(unlabeled_idxs)}, total: {len(labeled_idxs) + len(unlabeled_idxs)}')

    if args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    # vytvorenie iteratorov cez data
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)


    # Intializing the models
    # student
    student_model = models.__dict__[args.model](args, data=None).to(args.device)#.cuda()
    #teacher
    teacher_ema_model = models.__dict__[args.model](args,nograd = True, data=None).to(args.device)#.cuda()

    optimizer = torch.optim.SGD(student_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    if args.evaluate:
        print('Evaluating the primary model')
        acc1 = validate(eval_loader, student_model)
        print('Accuracy of the Student network on the 10000 test images: %d %%' % (
                acc1))
        print('Evaluating the Teacher model')
        acc2 = validate(eval_loader, teacher_ema_model)
        print('Accuracy of the Teacher network on the 10000 test images: %d %%' % (
                acc2))
        return

    if args.saveX == True:
        save_path = '{},{},{}epochs,b{},lr{}'.format(
            args.model,
            args.optim,
            args.epochs,
            args.batch_size,
            args.lr)
        time_stamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(time_stamp, save_path)
        save_path = os.path.join(args.dataName, save_path)
        save_path = os.path.join(args.save_path, save_path)
        print(f'==> Will save Everything to {save_path}')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # trenovanie mean teachera
    for epoch in range(args.start_epoch, args.epochs):

        # 1 trenovanie
        train(train_loader, student_model, teacher_ema_model, optimizer, epoch)

        # evaluovanie po niekolkych trenovaniach
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            prec1 = validate(eval_loader, student_model)

            print('==> Accuracy of the Student network on the 10000 test images: %d %%' % (
                prec1))

            ema_prec1 = validate(eval_loader, teacher_ema_model)

            print('==> Accuracy of the Teacher network on the 10000 test images: %d %%' % (
                ema_prec1))

            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        # zapisanie po niekolkych trenovaniach
        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.model,
                'state_dict': student_model.state_dict(),
                'ema_state_dict': teacher_ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_path, epoch + 1)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha)
        ema_param.data = torch.add(ema_param.data, param.data, alpha=(1 - alpha))


def train(train_loader, student_model, teacher_ema_model, optimizer, epoch):

    global global_step
    lossess = AverageMeter()
    running_loss = 0.0

    # Binary MT change
    class_supervised_criterion = nn.BCELoss(reduction='mean').to(args.device) #.cuda()

    # Binary MT change
    consistency_criterion = nn.MSELoss(reduction='sum').to(args.device) #.cuda()

    # trenovaci mod, nastavujeme kvoli spravaniu niektorych vrstiev
    student_model.train()
    teacher_ema_model.train()

    
    # pocitanie lossy
    for i, ((input, ema_input), target) in enumerate(train_loader): # iterujeme cez treningove batche

        # if i > 0: break
        # print(torch.Tensor.tolist(target).count(-1))

        if (input.size(0) != args.batch_size):
            continue

        # prekonvertovanie na tenzory
        input_var = torch.autograd.Variable(input).to(args.device) # .cuda()
        target_var = torch.autograd.Variable(target).to(args.device) # .cuda()) #async=True))


        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0


        # trenovanie
        student_model_out,student_model_h = student_model(input_var)

        # cross entrophy loss - average supervised loss S
        # updated to BCELoss for mean teacher
        student_model_out = student_model_out.view(256).to(torch.float32)
        #student_model_out -= student_model_out.min(0, keepdim=True)[0]
        #student_model_out /= student_model_out.max(0, keepdim=True)[0]
        student_model_out_labels = (student_model_out>torch.tensor([0.5]).to(args.device)).float()*1
        class_loss = class_supervised_criterion(student_model_out[194:], target_var.to(torch.float32)[194:]) / minibatch_size


        # urcenie celkovej loss podla toho, ci super alebo semisuper
        with torch.no_grad():
            ema_input_var = torch.autograd.Variable(ema_input)
            ema_input_var = ema_input_var.to(args.device).to(args.device) #.cuda()

        teacher_ema_model_out,teacher_ema_h = teacher_ema_model(ema_input_var)

        #ema_logit = teacher_ema_model_out
        #ema_logit = Variable(ema_logit.detach().data, requires_grad=False)


                                         # postupne sa zvysuje dolezitost konzistencie
        consistency_weight = get_current_consistency_weight(epoch)
                                                        # mse teachera a studenta
                # update pre binary MT

        consistency_loss = consistency_weight * consistency_criterion(student_model_h, teacher_ema_h) / minibatch_size

        loss = class_loss + consistency_loss
        # print(loss, class_loss, consistency_loss)

        # uprava vah studenta
        optimizer.zero_grad() # Sets the gradients of all optimized torch.Tensor s to zero.
        loss.backward()
        #for param in student_model.parameters():
        #    print(param.grad.data.sum())
        #for param in student_model.parameters():
        #    print(param.data)
        #    break
        optimizer.step()
        global_step += 1
        #for param in student_model.parameters():
        #    print(param.data)
        #    break
        

        # uprava vah teachera
        update_ema_variables(student_model, teacher_ema_model, args.ema_decay, global_step)

        # print statistics
        running_loss += loss.item()

        pr_freq = 20
        if i % pr_freq == pr_freq-1:    # print every <pr_freq> mini-batches
            print(f'Epoch: {epoch + 1}/{args.epochs}, Iteration: {i + 1}/{len(train_loader)}, Train loss: {round(running_loss / pr_freq, 5)}') #, Acc: {None}, Time: {None}')
            running_loss = 0.0

        lossess.update(loss.item(), input.size(0))

    return lossess,running_loss

def validate(eval_loader, model):

    print("===> Validating")

    model.eval()
    total = 0
    correct = 0
    for i, (input, target) in enumerate(eval_loader):

        with torch.no_grad():
            input_var = input.to(args.device) # .cuda()
            target_var = target.to(args.device) # .cuda() # async=True)

            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0

            # compute output
            output1, output_h = model(input_var)

            output1 = (output1.view(target_var.size(0)).to(torch.float32) > torch.tensor([0.5]).to(args.device)).float() * 1

            # _, predicted = torch.max(output1.data, 1)
            total += target_var.size(0)
            correct += (output1 == target_var).sum().item()

    return 100 * correct / total

def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        print('Best Model Saved: ');print(best_path)

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == '__main__':
    args = get_parameters()

    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")
    print(f"==> Using device {args.device}")

    main(args)
