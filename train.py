import argparse
import os

import torch
import torch.nn.functional as F
import datetime
from SAM2UNet import SAM2UNet
from utils.dataset_rgb_strategy2 import SalObjDataset

from utils.utils import adjust_lr, AvgMeter
import torch.nn as nn
import torch.utils.data as data
import math
from data_cod import test_dataset
import cv2
import warnings
import torch.distributed as dist
warnings.filterwarnings('ignore')

import numpy as np


def get_loader(image_root, gt_root,depth_root, batchsize, trainsize,distributed=False):
    dataset = SalObjDataset(image_root, gt_root,depth_root, trainsize)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=8,
                                  pin_memory=True, drop_last=True,sampler=sampler)
    return data_loader, sampler

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank
        )
        torch.cuda.set_device(args.gpu)
        dist.barrier()
        distributed = True

        print(f"Distributed training initialized: rank {args.rank}/{args.world_size}, gpu {args.gpu}")

    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        distributed = True

        print(f"SLURM distributed training: rank {args.rank}, gpu {args.gpu}")

    else:
        print("Not using distributed mode")
        distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    return distributed, args.gpu



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr_gen', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
    parser.add_argument('--task_all', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
    parser.add_argument('-beta1_gen', type=float, default=0.5, help='beta of Adam for generator')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--world_size', type=int, default=4, help='weight_decay')
    parser.add_argument('--rank', type=int, default=0, help='weight_decay')
    parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')
    parser.add_argument('--gpu', type=str, default="0123", help='reduced channel of saliency feat')
    return parser.parse_args()
opt = get_args()
def train():

    torch.autograd.set_detect_anomaly(True)  # 必须放在所有模型初始化之前
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    distributed, local_rank = init_distributed_mode(args=opt)


    ## load data
    image_sod_root = "./Data_all/DUTS/Train/Images/"
    image_cod_root = "./Data_all/COD-D/Train_depth/Images/"
    image_shadow_root = "./Data_all/SBU/Train/Images/"
    image_orsi_root = "./Data_all/ORSI/Train/EORSSD/Images/"
    image_polyp_root = "./Data_all/xirouData/train/Images/"
    image_covid_root = "./Data_all/COVID/Train/Images/"
    image_breast_root = "./Data_all/Breast/Train/Images/"
    image_skin_root = "./Data_all/ISIC2018/Train/Images/"

    gt_sod_root = "./Data_all/DUTS/Train/GT/"
    gt_cod_root = "./Data_all/COD-D/Train_depth/GT/"
    gt_shadow_root = "./Data_all/SBU/Train/GT/"
    gt_orsi_root = "./Data_all/ORSI/Train/EORSSD/GT/"
    gt_polyp_root = "./Data_all/xirouData/train/GT/"
    gt_covid_root = "./Data_all/COVID/Train/GT/"
    gt_breast_root = "./Data_all/Breast/Train/GT/"
    gt_skin_root = "./Data_all/ISIC2018/Train/GT/"

    depth_sod_root = "./Data_all/DUTS/Train/depth/"
    depth_cod_root = "./Data_all/COD-D/Train_depth/depth/"
    depth_shadow_root = "./Data_all/SBU/Train/depth/"
    depth_orsi_root = "./Data_all/ORSI/Train/EORSSD/depth/"
    depth_polyp_root = "./Data_all/xirouData/train/depth/"
    depth_covid_root = "./Data_all/COVID/Train/depth/"
    depth_breast_root = "./Data_all/Breast/Train/depth/"
    depth_skin_root = "./Data_all/ISIC2018/Train/depth/"




    train_sod_loader, sod_sampler = get_loader(image_sod_root, gt_sod_root, depth_sod_root, batchsize=opt.batchsize, trainsize=opt.trainsize,distributed=distributed)
    train_cod_loader, cod_sampler = get_loader(image_cod_root, gt_cod_root, depth_cod_root, batchsize=opt.batchsize, trainsize=opt.trainsize,distributed=distributed)
    train_shadow_loader, shadow_sampler = get_loader(image_shadow_root, gt_shadow_root, depth_shadow_root, batchsize=opt.batchsize, trainsize=opt.trainsize,distributed=distributed)
    train_orsi_loader, orsi_sampler = get_loader(image_orsi_root, gt_orsi_root, depth_orsi_root, batchsize=opt.batchsize, trainsize=opt.trainsize,distributed=distributed)
    train_polyp_loader, polyp_sampler = get_loader(image_polyp_root, gt_polyp_root, depth_polyp_root, batchsize=opt.batchsize, trainsize=opt.trainsize,distributed=distributed)
    train_covid_loader, covid_sampler = get_loader(image_covid_root, gt_covid_root, depth_covid_root, batchsize=opt.batchsize, trainsize=opt.trainsize,distributed=distributed)
    train_breast_loader, breast_sampler = get_loader(image_breast_root, gt_breast_root,depth_breast_root, batchsize=opt.batchsize, trainsize=opt.trainsize,distributed=distributed)
    train_skin_loader, skin_sampler = get_loader(image_skin_root, gt_skin_root,depth_skin_root, batchsize=opt.batchsize, trainsize=opt.trainsize,distributed=distributed)
    total_step = len(train_cod_loader)
    iter_num = math.ceil(max(len(train_cod_loader),len(train_sod_loader),len(train_polyp_loader),len(train_shadow_loader),len(train_orsi_loader),len(train_covid_loader), len(train_breast_loader), len(train_skin_loader)))

    print(iter_num)
    print(total_step)
    save_path = './cpts/'

    print("开始初始化模型，优化器...")
    generator = SAM2UNet(bs=opt.batchsize)

    generator.cuda()
    generator_optimizer = torch.optim.Adam(generator.parameters(), opt.lr_gen)

    generator = nn.parallel.DistributedDataParallel(
        generator,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    print("Start Training...")
    for epoch in range(1, opt.epoch + 1):
        sod_sampler.set_epoch(epoch)
        cod_sampler.set_epoch(epoch)
        shadow_sampler.set_epoch(epoch)
        orsi_sampler.set_epoch(epoch)
        polyp_sampler.set_epoch(epoch)
        covid_sampler.set_epoch(epoch)
        breast_sampler.set_epoch(epoch)
        skin_sampler.set_epoch(epoch)

        generator.train()
        loss_record = AvgMeter()
        print('Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

        train_sod_loader_iter = iter(train_sod_loader)
        train_cod_loader_iter = iter(train_cod_loader)
        train_shadow_loader_iter = iter(train_shadow_loader)
        train_orsi_loader_iter = iter(train_orsi_loader)
        train_polyp_loader_iter = iter(train_polyp_loader)
        train_covid_loader_iter = iter(train_covid_loader)
        train_breast_loader_iter = iter(train_breast_loader)
        train_skin_loader_iter = iter(train_skin_loader)
        for i, (image_orsi, gt_orsi,depth_orsi) in enumerate(train_orsi_loader_iter):
            if (i + 1) > iter_num: break

            try:
                image_sod, gt_sod ,depth_sod= next(train_sod_loader_iter)
            except StopIteration:
                train_sod_loader_iter = iter(train_sod_loader)
                image_sod, gt_sod ,depth_sod= next(train_sod_loader_iter)

            try:
                image_cod, gt_cod ,depth_cod= next(train_cod_loader_iter)
            except StopIteration:
                train_cod_loader_iter = iter(train_cod_loader)
                image_cod, gt_cod ,depth_cod= next(train_cod_loader_iter)

            try:
                image_shadow, gt_shadow,depth_shadow = next(train_shadow_loader_iter)
            except StopIteration:
                train_shadow_loader_iter = iter(train_shadow_loader)
                image_shadow, gt_shadow ,depth_shadow= next(train_shadow_loader_iter)

            try:
                image_polyp, gt_polyp,depth_polyp = next(train_polyp_loader_iter)
            except StopIteration:
                train_polyp_loader_iter = iter(train_polyp_loader)
                image_polyp, gt_polyp ,depth_polyp= next(train_polyp_loader_iter)

            try:
                image_covid, gt_covid,depth_covid = next(train_covid_loader_iter)
            except StopIteration:
                train_covid_loader_iter = iter(train_covid_loader)
                image_covid, gt_covid ,depth_covid= next(train_covid_loader_iter)

            try:
                image_breast, gt_breast,depth_breast = next(train_breast_loader_iter)
            except StopIteration:
                train_breast_loader_iter = iter(train_breast_loader)
                image_breast, gt_breast ,depth_breast= next(train_breast_loader_iter)

            try:
                image_skin, gt_skin,depth_skin = next(train_skin_loader_iter)
            except StopIteration:
                train_skin_loader_iter = iter(train_skin_loader)
                image_skin, gt_skin ,depth_skin= next(train_skin_loader_iter)

            images = torch.cat([image_sod, image_cod, image_shadow, image_orsi, image_polyp, image_covid, image_breast, image_skin], dim=0)
            gts = torch.cat([gt_sod, gt_cod, gt_shadow, gt_orsi, gt_polyp, gt_covid, gt_breast, gt_skin], dim=0)
            depth_cod = depth_cod.repeat(1, 3, 1, 1)
            depth_sod = depth_sod.repeat(1, 3, 1, 1)
            depth_polyp = depth_polyp.repeat(1, 3, 1, 1)
            depth_shadow = depth_shadow.repeat(1, 3, 1, 1)
            depth_orsi = depth_orsi.repeat(1, 3, 1, 1)
            depth_covid = depth_covid.repeat(1, 3, 1, 1)
            depth_breast = depth_breast.repeat(1, 3, 1, 1)
            depth_skin = depth_skin.repeat(1, 3, 1, 1)


            depths = torch.cat([depth_sod, depth_cod, depth_shadow, depth_orsi, depth_polyp, depth_covid, depth_breast, depth_skin], dim=0)

            images = images.cuda()
            gts = gts.cuda()
            depths= depths.cuda()

            masks = generator(images,depths)
            loss1 = structure_loss(masks, gts)

            loss = loss1

            loss.backward()
            generator_optimizer.step()
            generator_optimizer.zero_grad()

            loss_record.update(loss.data, opt.batchsize)
            if i % 500 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Pre Loss: {:.4f}, Pre Loss1: {:.4f}'.
                          format(datetime.datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(),loss.data))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if epoch >= 50:
            torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
            w_path = save_path + 'Model_' + str(epoch) + '_gen.pth'
            if epoch>=50:
                test_sod(w_path, task_num=0)
                test_cod(w_path, task_num=1)
                test_shadow(w_path, task_num=2)
                test_orsi(w_path, task_num=3)
                test_polyp(w_path, task_num=4)
                val_covid_IOUDICE(w_path, task_num=5)
                val_Brrast_IOUDICE(w_path, task_num=6)
                val_Skin_IOUDICE(w_path, task_num=7)

best_mae = 10000
best_epoch=0

def test_sod(w_path,task_num=None):
    global best_mae, best_epoch

    test_path='./Data_all/DUTS/'
    test_datasets = ['Test']
    save_path = './cpts/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    generator = SAM2UNet(task_num=task_num)
    data = torch.load(w_path)
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)

    generator.cuda()
    generator.eval()
    for dataset in test_datasets:
        save_path = './test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = test_path + dataset + '/Images/'
        gt_root = test_path + dataset + '/GT/'
        d_root = test_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, d_root, opt.trainsize)
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()

            device = next(generator.parameters()).device
            imgs = image.permute(0, 2, 3, 1).cpu().numpy()
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = (torch.as_tensor((imgs[b_i]).astype(dtype=np.uint8), device=device)
                               .permute(2, 0, 1).contiguous())
                dict_input['image'] = input_image
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            res = generator(image,depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().detach().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])


        mae = mae_sum / test_loader.size

        print('DUTS Res mae is : ',mae)

def test_cod(w_path,task_num=None):
    opt = get_args()
    global best_mae, best_epoch
    test_path='./Data_all/COD-D/Test_depth/'
    test_datasets = ['COD10K']

    save_path = './cpts/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # print("Test : 开始初始化模型，优化器...")
    generator = SAM2UNet(task_num=task_num)
    data = torch.load(w_path)
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)
    generator.cuda()
    generator.eval()
    for dataset in test_datasets:
        save_path = './test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = test_path + dataset + '/Images/'
        gt_root = test_path + dataset + '/GT/'
        d_root = test_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, d_root, opt.trainsize)
        mae_sum = 0


        for i in range(test_loader.size):  # 250
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()
            res = generator(image,depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().detach().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])


        mae = mae_sum / test_loader.size
        print(dataset, 'Res mae is : ', mae)

def test_shadow(w_path,task_num=None):
    test_path='./Data_all/SBU/'
    test_datasets = ['Test']

    save_path = './cpts/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    generator = SAM2UNet(task_num=task_num)
    data = torch.load(w_path)
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)
    generator.cuda()
    generator.eval()
    for dataset in test_datasets:
        save_path = './test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = test_path + dataset + '/Images/'
        gt_root = test_path + dataset + '/GT/'
        d_root = test_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, d_root, opt.trainsize)
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()
            res = generator(image,depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().detach().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        print(dataset,'SBU Res mae is : ',mae)

def test_orsi(w_path,task_num=None):
    test_path='./Data_all/ORSI/Test/'
    test_datasets = ['EORSSD']

    save_path = './cpts/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    generator = SAM2UNet(task_num=task_num)
    data = torch.load(w_path)
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)
    generator.cuda()
    generator.eval()
    for dataset in test_datasets:
        save_path = './test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = test_path + dataset + '/Images/'
        gt_root = test_path + dataset + '/GT/'
        d_root = test_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, d_root, opt.trainsize)
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()
            res = generator(image,depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().detach().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        print(dataset,'Res mae is : ',mae)

def test_polyp(w_path,task_num=None):
    test_path='./Data_all/xirouData/Val/'
    test_datasets = ['CVC-300','CVC-ClinicDB','CVC-ColonDB', 'ETIS-LaribPolypDB','Kvasir']

    save_path = './cpts/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    generator = SAM2UNet(task_num=task_num)
    data = torch.load(w_path)
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)
    generator.cuda()
    generator.eval()
    for dataset in test_datasets:
        save_path = './test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = test_path + dataset + '/Images/'
        gt_root = test_path + dataset + '/GT/'
        d_root = test_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, d_root, opt.trainsize)
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()
            res = generator(image,depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().detach().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        print(dataset,'Res mae is : ',mae)

def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou

def val_covid_IOUDICE(w_path,task_num=None):
    """
    validation function
    """
    global best_metric_dict, best_score, best_epoch
    dataset_path = './Data_all/COVID/'
    test_datasets = ['Test']
    print("开始初始化模型，优化器...")
    generator = SAM2UNet(task_num=task_num)
    data = torch.load(w_path)
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)
    generator.cuda()
    generator.eval()
    for dataset in test_datasets:
        save_path = 'test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/Images/'
        gt_root = dataset_path + dataset + '/GT/'
        g_root = dataset_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root,g_root, opt.trainsize)
        dice_bank = []
        iou_bank = []
        acc_bank = []
        generator.eval()
        with torch.no_grad():
            for i in range(test_loader.size):
                image, gt, depth,name, image_for_post = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                depth = depth.repeat(1, 3, 1, 1).cuda()
                res = generator(image,depth)
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().detach().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                cv2.imwrite(save_path + name, res * 255)
                res = 1 * (res > 0.5)
                dice = mean_dice_np(gt, res)
                iou = mean_iou_np(gt, res)
                acc = np.sum((res == gt)) / (res.shape[0] * res.shape[1])

                acc_bank.append(acc)
                dice_bank.append(dice)
                iou_bank.append(iou)

            print(dataset,'covid Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
                  format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))

def val_Brrast_IOUDICE(w_path,task_num=None):
    """
    validation function
    """
    global best_metric_dict, best_score, best_epoch
    dataset_path = './Data_all/ISIC2018/'
    test_datasets = ['Bre']

    print("开始初始化模型，优化器...")
    generator = SAM2UNet(task_num=task_num)
    data = torch.load(w_path)
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)
    generator.cuda()
    generator.eval()
    for dataset in test_datasets:
        save_path = 'test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/Images/'
        gt_root = dataset_path + dataset + '/GT/'
        g_root = dataset_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root,g_root, opt.trainsize)
        dice_bank = []
        iou_bank = []
        acc_bank = []
        generator.eval()
        with torch.no_grad():
            for i in range(test_loader.size):
                image, gt, depth,name, image_for_post = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                depth = depth.repeat(1, 3, 1, 1).cuda()

                res = generator(image,depth)
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().detach().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = 1 * (res > 0.5)
                dice = mean_dice_np(gt, res)
                iou = mean_iou_np(gt, res)
                acc = np.sum((res == gt)) / (res.shape[0] * res.shape[1])

                acc_bank.append(acc)
                dice_bank.append(dice)
                iou_bank.append(iou)

            print(dataset,'Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
                  format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))

def val_Skin_IOUDICE(w_path,task_num=None):
    """
    validation function
    """
    global best_metric_dict, best_score, best_epoch
    dataset_path = './Data_all/ISIC2018/'
    test_datasets = ['ISIC']

    print("开始初始化模型，优化器...")
    generator = SAM2UNet(task_num=task_num)
    data = torch.load(w_path)
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)
    generator.cuda()
    generator.eval()
    for dataset in test_datasets:
        save_path = 'test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/Images/'
        gt_root = dataset_path + dataset + '/GT/'
        g_root = dataset_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root,g_root, opt.trainsize)
        dice_bank = []
        iou_bank = []
        acc_bank = []
        generator.eval()
        with torch.no_grad():
            for i in range(test_loader.size):
                image, gt, depth,name, image_for_post = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                depth = depth.repeat(1, 3, 1, 1).cuda()

                res = generator(image,depth)
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().detach().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = 1 * (res > 0.5)
                dice = mean_dice_np(gt, res)
                iou = mean_iou_np(gt, res)
                acc = np.sum((res == gt)) / (res.shape[0] * res.shape[1])

                acc_bank.append(acc)
                dice_bank.append(dice)
                iou_bank.append(iou)

            print(dataset,'Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
                  format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))


if __name__ == '__main__':
    train()

