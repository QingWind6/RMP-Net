import os

import torch
from PIL import Image
from sklearn import metrics
# from unet import Unet
from torch.utils.data import DataLoader
from os.path import join

from tqdm import tqdm

# from nets.unet import Unet
from unet import Unet
import  numpy as np
import copy

# from utils.dataloader import UnetDataset
from utils.dataloader_medical import UnetDataset, unet_dataset_collate
from utils.utils import preprocess_input, cvtColor
from utils.utils_metrics import fast_hist, per_class_iu, per_Accuracy, per_class_PA_Recall


def cal_cm(y_true,y_pred):
    y_true=y_true.reshape(1,-1).squeeze()
    y_pred=y_pred.reshape(1,-1).squeeze()
    cm=metrics.confusion_matrix(y_true,y_pred)
    return cm

def Intersection_over_Union(confusion_matrix):
    intersection = np.diag(confusion_matrix)#交集
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)#并集
    IoU = intersection / union #交并比，即IoU
    return IoU

def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):
    print('Num classes', num_classes)
    # -----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    # ------------------------------------------------#
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    # ------------------------------------------------#
    #   读取每一个（图片-标签）对
    # ------------------------------------------------#
    for ind in range(len(gt_imgs)):
        # ------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        # ------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))
        # ------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        # ------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        # ------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        # ------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            )
            )
    # ------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    # ------------------------------------------------#
    IoUs = per_class_iu(hist)

    return IoUs
    # print(IoUs)
    # PA_Recall = per_class_PA_Recall(hist)
    # Precision = per_class_Precision(hist)
    # ------------------------------------------------#
    #   逐类别输出一下mIoU值
    # ------------------------------------------------#
    # if name_classes is not None:
    #     for ind_class in range(num_classes):
    #         print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
    #               + '; Recall (equal to the PA)-' + str(
    #             round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

            # -----------------------------------------------------------------#
            #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
            # -----------------------------------------------------------------#
            # print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(
            #     round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
            # return np.array(hist, np.int), IoUs, PA_Recall, Precision


if __name__ == '__main__':
    # model = Unet(num_classes=2, pretrained=False, backbone='vgg').cuda()

    # image = r"DeepFish/VOC2007/JPEGImages/7117_Chaetodon_vagabundus_3_f000020.png"
    # image = Image.open(image)
    # image = cvtColor(image)
    # # ---------------------------------------------------#
    # #   对输入图像进行一个备份，后面用于绘图
    # # ---------------------------------------------------#
    # old_img = copy.deepcopy(image)
    # orininal_h = np.array(image).shape[0]
    # orininal_w = np.array(image).shape[1]
    # # ---------------------------------------------------------#
    # #   给图像增加灰条，实现不失真的resize
    # #   也可以直接resize进行识别
    # # ---------------------------------------------------------#
    # image_data, nw, nh = resize_image(image, (512, 512))
    # # ---------------------------------------------------------#
    # #   添加上batch_size维度
    # # ---------------------------------------------------------#
    # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    #
    # with torch.no_grad():
    #     images = torch.from_numpy(image_data)
    #     images = images.cuda()
    #
    #     # ---------------------------------------------------#
    #     #   图片传入网络进行预测
    #     # ---------------------------------------------------#
    #     pr = model(images)[0]
    #
    # # cm = cal_cm()
    # print(pr.shape)

    input_shape = [512, 512]
    num_classes = 2
    name_classes = ["background", "fish"]
    miou_mode = 0
    # VOCdevkit_path  = 'DeepFish/VOC2007'
    # with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"), "r") as f:
    #     train_lines = f.readlines()

    VOCdevkit_path = 'DeepFish/'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        unet = Unet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".png")
            image = Image.open(image_path)
            image = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        IoUs= compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数


    print(IoUs)



