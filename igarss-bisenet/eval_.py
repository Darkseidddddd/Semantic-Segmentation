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
def eval(model, args):
    print('start test!')
    model.eval()

    length = 2016
    t = []
    print('length: %d' %length)
    tq = tqdm.tqdm(length)
    tq.set_description('test')
    data = torch.Tensor(1, 3, 512, 512).cuda()
    print(data.item())
    for i in range(length):
        tq.update(1)
        start = time.clock()
        predict = model(data).squeeze()
        end = time.clock()
        t.append(end-start)
        total_time += (end-start)

    np.savetxt('t.txt', np.array(t))
    fps = length/total_time
    print('fps: %.2f' %fps)

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
    eval(model, args)


if __name__ == '__main__':
    params = [
        '--checkpoint_path', 'test/best_dice_loss.pth',
        '--data', 'potsdam_512_IRRG',
        '--cuda', '0',
        # '--use_gpu', 'False',
        '--context_path', 'resnet50',
        '--num_classes', '6'
    ]
    main(params)
