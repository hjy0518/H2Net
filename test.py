import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import cv2
from SAM2UNet import SAM2UNet
from data_cod import test_dataset


def get_loader(image_root, gt_root, trainsize):
    dataset = test_dataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True, )
    return data_loader

def test(nums_ID):

    task_dict = [
        {"sod": 0},
        {"cod":1},
        {"shadow":2},
        {"orsi":3},
        {"polyp":4},
        {"covid":5},
        {"breast":6},
        {"skin":7}
                 ]
    k, v = list(task_dict[nums_ID].items())[0]
    print("k is :",k)
    print("v is :",v)
    if v == 0:
        dataset_path = './Data_all/DUTS/'
        test_datasets = ['Test']
    elif v == 1:
        dataset_path = './Data_all/COD-D/Test_depth/'
        test_datasets = ['COD10K']
    elif v == 2:
        dataset_path = './Data_all/SBU/'
        test_datasets = ['Test']
    elif v == 3:
        dataset_path = './Data_all/ORSI/Test/'
        test_datasets = ['EORSSD']
    elif v == 4:
        dataset_path = './Data_all/xirouData/Val/'
        test_datasets = ['CVC-300','CVC-ClinicDB','CVC-ColonDB', 'ETIS-LaribPolypDB','Kvasir']
    elif v == 5:
        dataset_path = './Data_all/COVID/'
        test_datasets = ['Test']
    elif v == 6:
        dataset_path = './Data_all/Breast/'
        test_datasets = ['test']
    elif v == 7:
        dataset_path = './Data_all/ISIC2018/'
        test_datasets = ['ISIC']
    else:
        print("输入有误！！！")

    print("开始初始化模型，优化器...")
    generator = SAM2UNet(task_num=v)
    try:
        state_dict = torch.load('', map_location='cpu')

        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        generator.load_state_dict(state_dict)
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Loading failed: {str(e)}")

    generator.cuda()
    generator.eval()
    for dataset in test_datasets:
        save_path = 'TESTMAPS/' + k + '/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/Images/'
        gt_root = dataset_path + dataset + '/GT/'
        g_root = dataset_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root,g_root, 384)

        with torch.no_grad():
            mae_sum = 0
            for i in range(test_loader.size):
                image, gt,depth, name, img_for_post = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                depths = depth.repeat(1, 3, 1, 1).cuda()
                res = generator(image,depths)
                res= F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

                cv2.imwrite(save_path + name, res * 255)
            mae = mae_sum / test_loader.size
            print('mae is : ',mae)


if __name__ == '__main__':
    for i in range(8):
        test(i)
