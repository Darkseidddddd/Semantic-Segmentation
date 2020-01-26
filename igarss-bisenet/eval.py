from potsdam import potsdam
import torch
import argparse
import os
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou, compute_cm_cks_cr
import tqdm
import time
from cm_plot import plot_cm, plot_confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
def eval(model,dataloader, args, csv_path):
    print('start test!')
    with torch.no_grad():
        total_pred = np.array([0])
        total_label = np.array([0])
        total_cm = np.zeros((6,6))
        model.eval()
        precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')
        hist = np.zeros((args.num_classes, args.num_classes))
        total_time = 0
        total_cks, total_f1 = 0.0, 0.0
        length = len(dataloader)
        print('length: %d' %length)
        for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            start = time.clock()
            predict = model(data).squeeze()
            end = time.clock()
            # 转为类别矩阵
            predict = reverse_one_hot(predict)
            predict = np.array(predict)
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # end = time.clock()
            # 测试花费时间
            total_time += (end-start)
            label = label.squeeze()
            # 转换为类别矩阵
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label)

            # 计算cm
            # total_pred = np.append(total_pred, predict.flatten())
            # total_label = np.append(total_label, label.flatten())
            # if (i+1) % 8 == 0:
            #     total_cm += confusion_matrix(total_label[1:], total_pred[1:])
            #     total_label = np.array([0])
            #     total_pred = np.array([0])

            # 计算kappa，总的算平均
            cks = cohen_kappa_score(label.flatten(), predict.flatten())
            total_cks += cks
            f1 = f1_score(label.flatten(), predict.flatten(), average='macro')
            total_f1 += f1
            # label = colour_code_segmentation(np.array(label), label_info)
            # 计算oa
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            # 记录总的精度
            precision_record.append(precision)
        # 保存cm
        # np.savetxt('cm.txt', total_cm)
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou_dict, miou = cal_miou(miou_list, csv_path)
        print('IoU for each class:')
        for key in miou_dict:
            print('{}:{},'.format(key, miou_dict[key]))
        tq.close()
        print('oa for test: %.3f' % precision)
        print('mIoU for test: %.3f' % miou)
        
        # 计算cm, kappa, cr //作废
        cm, cks, cr = compute_cm_cks_cr(predict, label)
        # print('cm for test:\n', cm)
        total_cks /= length
        print('kappa for test: %.4f' %total_cks)
        total_f1 /= length
        print('f1 for test: %.4f' %total_f1)
        fps = length/total_time
        print('fps: %.2f' %fps)
        return precision, cm, total_cks, cr

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    args = parser.parse_args(params)

    # create dataset and dataloader
    test_path = os.path.join(args.data, 'test')
    # test_path = os.path.join(args.data, 'train')
    test_label_path = os.path.join(args.data, 'test_label')
    # test_label_path = os.path.join(args.data, 'train_labels')
    print(test_path, test_label_path)
    csv_path = os.path.join(args.data, 'class_dict.csv')
    csv_path = 'potsdam_512_IRRG/class_dict_potsdam.csv'
    dataset = potsdam(test_path, test_label_path, csv_path, scale=(args.crop_height, args.crop_width), mode='test')
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # get label info
    # label_info = get_label_info(csv_path)
    # test
    eval(model, dataloader, args, csv_path)


if __name__ == '__main__':
    params = [
        '--checkpoint_path', 'test/best_dice_loss.pth',
        '--data', 'potsdam_512_IRRG',
        '--cuda', '0,1',
        # '--use_gpu', 'False',
        '--context_path', 'resnet50',
        '--num_classes', '6'
    ]
    main(params)
