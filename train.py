from __future__ import print_function
import os
from data import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
#from utils import Visualizer, view_model
import torch
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *
import pickle


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':

    opt = Config()
    #if opt.display:
        #visualizer = Visualizer()
    device = torch.device("cuda")

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    #identity_list = get_lfw_list(opt.lfw_test_list)
    #img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        import pickle

        # ✅ torchvision의 정식 ResNet50 사용 (3채널, avgpool+fc 포함)
        model = resnet50()

        with open('models/resnet50_ft_weight.pkl', 'rb') as f:
            state_dict = pickle.load(f, encoding='latin1')

        # ✅ numpy.ndarray → torch.Tensor 변환
        state_dict = {
            k.replace('module.', ''): torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in state_dict.items()
        }

        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f'Missing keys:    {missing}')
        print(f'Unexpected keys: {unexpected}')

        # ✅ fc 레이어를 512-dim feature extractor로 교체
        model.fc = nn.Linear(2048, 512)


    if opt.backbone != 'resnet50' :
        print(f'Loading pretrained weights from: {opt.load_model_path}')
        state_dict = torch.load(opt.load_model_path, map_location='cpu')
        # DataParallel로 저장된 경우 'module.' prefix 제거
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

    #파라미터 프리즈
    for param in model.parameters():
        param.requires_grad = False

    if opt.backbone == 'resnet50' :
        for param in model.fc.parameters():
            param.requires_grad = True
    print('Backbone frozen: all parameters requires_grad = False')

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.1, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)
    #classifier = OccupationClassifier(num_class = opt.num_classes)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    for i in range(opt.max_epoch):
        scheduler.step()

        model.eval()
        metric_fc.train()

        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()

            with torch.no_grad():
                feature = model(data_input)

            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii
            print('total iteration : {}'.format(iters))

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                print(output)
                print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                #if opt.display:
                #    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                #    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, model+"trained", i)
            save_model(metric_fc, opt.checkpoints_path, "classifier_metrix+"+opt.backbone, i)

        #acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        #if opt.display:
        #    visualizer.display_current_results(iters, acc, name='test_acc')
