import argparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names.append('shufflenet')
pool_names = ['mac', 'spoc', 'gem', 'attention']
loss_names = ['OnlineContrastive', 'OnlineTriplet']
optimizer_names = ['sgd', 'adam']

def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')
    # init network parameters
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer_test'], required=True)
    parser.add_argument('--root', type=str, default='/data5/wwp/landmark/DATAV2/train/train')
    parser.add_argument('--training-dataset', '-d', metavar='DATASET', default='landmark-120k')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101', choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet101)')
    parser.add_argument('--pool', '-p', metavar='POOL', default='mac', choices=pool_names,
                        help='pooling options: ' +
                             ' | '.join(pool_names) +
                             ' (default: gem)')
    parser.add_argument('--log_dir', type=str, default='log')

    parser.add_argument('--loss', '-l', metavar='LOSS', default='contrastive',
                        choices=loss_names,
                        help='training loss options: ' +
                             ' | '.join(loss_names) +
                             ' (default: contrastive)')
    parser.add_argument('--loss-margin', '-lm', metavar='LM', default=1., type=float,
                        help='loss margin: (default: 0.7)')

    # train/val options specific for image retrieval learning
    parser.add_argument('--image-size', default=224, type=int, metavar='N',
                        help='maximum size of longer image side used for training (default: 1024)')
    parser.add_argument('--n_classes', type=int, default=16, required=True)
    parser.add_argument('--n_samples', type=int, default=16, required=True)
    # parser.add_argument('--attention', type=bool, default=True, required=True)
    # standard train/val options
    parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                        help='gpu id used for training (default: 0)')
    parser.add_argument('--workers', '-j', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adam',
                        choices=optimizer_names,
                        help='optimizer options: ' +
                             ' | '.join(optimizer_names) +
                             ' (default: adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-6)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--log_interval', default=100, type=int,
                        metavar='N', help='log writer epoch frequency (default: 1)')
    parser.add_argument('--train_show_pairs', type=int, default=5)
    parser.add_argument('--resume', default='', type=str, metavar='FILENAME',
                        help='name of the latest checkpoint (default: None)')

    config = parser.parse_args()
    return config










