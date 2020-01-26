import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from potsdam import potsdam
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu, compute_cm_cks_cr
from loss import DiceLoss
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

def val(args, model, dataloader, data_name):
    print('start val!')
    # label_info = get_label_info(csv_path)
    total_cks, total_f1 = 0.0, 0.0
    total_pred = np.array([0])
    total_label = np.array([0])
    length = len(dataloader)
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            # print('label size: ', label.size())
            # print('data size: ', data.size())
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            # print('label_cuda size: ', label.size())
            # print('data_cuda size: ', data.size())
            predict = model(data).squeeze()
            # print('predict size: ', predict.size())
            
            predict = reverse_one_hot(predict)
            predict = np.array(predict)

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label)
            
            total_pred = np.append(total_pred, predict.flatten())
            total_label = np.append(total_label, label.flatten())
            if (i+1) % 8 == 0:
                 # total_cm += confusion_matrix(total_label[1:], total_pred[1:])
                 cks = cohen_kappa_score(total_label[1:], total_pred[1:])
                 total_label = np.array([0])
                 total_pred = np.array([0])
                 total_cks += cks
            # cks = cohen_kappa_score(label.flatten(), predict.flatten())
            # total_cks += cks
            f1 = f1_score(label.flatten(), predict.flatten(), average='macro')
            total_f1 += f1

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        # print('precision per pixel for test: %.3f' % precision)
        print('oa for %s: %.3f' %(data_name, precision))
        # print('mIoU for validation: %.3f' % miou)
        print('mIoU for %s: %.3f' %(data_name, miou))
        cm, cks, cr = compute_cm_cks_cr(predict, label)
        total_f1 /= length
        total_cks = total_cks / (length // 8)
        # print('cm:\n', cm)
        print('kappa for %s: %.4f' %(data_name, total_cks))
        print('f1 for {}:\n'.format(data_name), total_f1)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou, cm, total_cks, total_f1


def train(args, model, optimizer, dataloader_train, dataloader_val_train, dataloader_test):
    writer = SummaryWriter(log_dir='runs_50_adadelta',comment=''.format(args.optimizer, args.context_path))
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss()
    max_miou = 0
    step = 0
    for epoch in range(args.epoch_start_i, args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            output, output_sup1, output_sup2 = model(data)
            loss1 = loss_func(output, label)
            loss2 = loss_func(output_sup1, label)
            loss3 = loss_func(output_sup2, label)
            loss = loss1 + loss2 + loss3
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, 'latest_dice_loss.pth'))

        if epoch % args.validation_step == 0:
            #precision, miou = val(args, model, dataloader_val)
            oa, miou, cm, cks, f1 = val(args, model, dataloader_val_train, 'train')
            oa_test, miou_test, cm_test, cks_test, f1_test = val(args, model, dataloader_test, 'test')
            if miou > max_miou:
                max_miou = miou
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
            #writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/oa_train', oa, epoch)
            writer.add_scalar('epoch/oa_test', oa_test, epoch)
            #writer.add_scalar('epoch/miou val', miou, epoch)
            writer.add_scalar('epoch/miou_train', miou, epoch)
            writer.add_scalar('epoch/miou_test', miou_test, epoch)
            writer.add_scalar('epoch/cks_train', cks, epoch)
            writer.add_scalar('epoch/cks_test', cks_test, epoch)
            writer.add_scalar('epoch/f1_train', f1, epoch)
            writer.add_scalar('epoch/f1_test', f1_test, epoch)
            with open(os.path.join(args.save_model_path, 'classification_results.txt'), mode='a') as f:
                f.write('epoch: '+str(epoch)+'\n')
                # f.write('train time:\t' + str(train_time))
                # f.write('\ntest time:\t' + str(test_time))
                f.write('\nmiou:\t' + str(miou))
                f.write('\noverall accuracy:\t' + str(oa))
                f.write('\ncohen kappa:\t' + str(cks))
                f.write('\nconfusion matrix:\n')
                f.write(str(cm))
                f.write('\nf1:\t' + str(f1))
                f.write('\n\n')


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="potsdam", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')

    args = parser.parse_args(params)

    # create dataset and dataloader
    # train_path = [os.path.join(args.data, 'train'), os.path.join(args.data, 'val')]
    train_path = [os.path.join(args.data, 'train')]
    # train_label_path = [os.path.join(args.data, 'train_labels'), os.path.join(args.data, 'val_labels')]
    train_label_path = [os.path.join(args.data, 'train_label')]
    # test_path = os.path.join(args.data, 'test')
    # test_label_path = os.path.join(args.data, 'test_labels')
    # test_label_path = os.path.join(args.data, 'test_label')
    csv_path = os.path.join(args.data, 'class_dict_potsdam.csv')

    # create dataset and dataloader
    test_path = os.path.join(args.data, 'test')
    # test_path = os.path.join(args.data, 'train')
    test_label_path = os.path.join(args.data, 'test_label')
    # test_label_path = os.path.join(args.data, 'train_labels')
    # print(test_path, test_label_path)
    # csv_path = os.path.join(args.data, 'class_dict.csv')
    # csv_path = 'potsdam_512_IRRG/class_dict_potsdam.csv'
    
    dataset_train = potsdam(train_path, train_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                           loss=args.loss, mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    # dataset_val = potsdam(test_path, test_label_path, csv_path, scale=(args.crop_height, args.crop_width),
    #                      loss=args.loss, mode='test')
    
    dataset_val_train = potsdam(train_path, train_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                           loss=args.loss, mode='test')
    dataloader_val_train = DataLoader(
        dataset_val_train,
        # this has to be 1
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    dataset_test = potsdam(test_path, test_label_path, csv_path, scale=(args.crop_height, args.crop_width), mode='test')
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    elif args.optimizer == 'Adadelta':
	    optimizer = torch.optim.Adadelta(model.parameters(), args.learning_rate,rho=0.9,weight_decay=1e-4) 
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    # train(args, model, optimizer, dataloader_train, dataloader_val)
    train(args, model, optimizer, dataloader_train, dataloader_val_train, dataloader_test)

    # val(args, model, dataloader_val, csv_path)


if __name__ == '__main__':
    params = [
        '--num_epochs', '1000',
        '--learning_rate', '2.5e-2',
        '--data', 'potsdam_512_IRRG/',
        '--num_workers', '8',
        '--num_classes', '6',
        '--cuda', '0,1,2', # 2 GPU
        '--batch_size', '8',  # 6 for resnet101, 12 for resnet18
        '--save_model_path', './checkpoints_50_adadelta',
        # '--pretrained_model_path', './checkpoints_18_sgd/latest_dice_loss.pth',
        '--context_path', 'resnet50',  # only support resnet18 and resnet101 and resnet50
        '--optimizer', 'Adadelta',
        # '--epoch_start_i', '313', 

    ]
    main(params)

