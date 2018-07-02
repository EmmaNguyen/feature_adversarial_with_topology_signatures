import os
import math
import argparse

import torch
import torch.nn as nn
from torch import optim

from architecture.neural_topo_nets import MyModel
from utils.topo_utils import Provider, train_test_from_dataset, Trainer, LearningRateScheduler
from utils.topo_utils import ConsoleBatchProgress, PredictionMonitor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='an absolute path to dgm (h5).')
    parser.add_argument('--num_epochs', type=int, default=300, help='a number of epochs, e.g. number of times running entire of data.')
    parser.add_argument('--momentum', type=float, default=0.7, help='a number indicating momentum.')
    parser.add_argument('--lr_start', type=float, default=0.1, help='a starting value of learning rate (lr) to be tried.')
    parser.add_argument('--lr_step', type=float, default=20, help='a number of incremental steps in learning rate (lr).')
    parser.add_argument('--lr_adaption', type=float, default=0.5, help='a coefficient to jump between values of learning rate (lr).')
    parser.add_argument('--test_ratio', type=float, default=0.5, help='a ratio between test set and training set.')
    parser.add_argument('--batch_size', type=int, default=128, help='a number indicating a batch size.')
    parser.add_argument('--use_cuda', type=bool, default=False, help='True if use cuda, otherwise False')
    return parser.parse_args()

print('Starting experiment...')
accuracies = []
n_runs = 5

def _create_trainer(model, opt, data_train, data_test):
    optimizer = optim.SGD(model.parameters(), lr=opt.lr_start, momentum=opt.momentum)
    loss = nn.CrossEntropyLoss()
    trainer = Trainer(model=model,
                         optimizer=optimizer,
                         loss=loss,
                         train_data=data_train,
                         n_epochs=opt.num_epochs,
                         cuda=opt.use_cuda,
                         variable_created_by_model=True)

    def determine_lr(self, **kwargs):
        """
        Todo: check reference to find learning_rate
        """
        epoch = kwargs['epoch_count']
        if epoch % opt.lr_step == 0:
            return params.lr_start / 2 ** (epoch / op.lr_step)

    lr_scheduler = LearningRateScheduler(determine_lr, verbose=True)
    lr_scheduler.register(trainer)
    progress = ConsoleBatchProgress()
    progress.register(trainer)
    prediction_monitor_test = PredictionMonitor(data_test,
                                                verbose=True,
                                                eval_every_n_epochs=1,
                                                variable_created_by_model=True)
    prediction_monitor_test.register(trainer)
    trainer.prediction_monitor = prediction_monitor_test
    return trainer

def _data_setup(opt):
    view_name_template = 'dim_0_dir_{}'
    subscripted_views = sorted([view_name_template.format(i) for i in range(32)])
    assert (str(len(subscripted_views)) in opt.data_path)

    print('Loading provider...')
    dataset = Provider()
    dataset.read_from_h5(opt.data_path)

    assert all(view_name in dataset.view_names for view_name in subscripted_views)

    print('Create data loader...')
    data_train, data_test = train_test_from_dataset(dataset,
                                                    test_size=opt.test_ratio,
                                                    batch_size=opt.batch_size)

    return data_train, data_test, subscripted_views

def experiment():
    opt = parse_arguments()
    opt.use_cuda = True if torch.cuda.is_available() else False
    data_train, data_test, subscripted_views = _data_setup(opt)
    model = MyModel(subscripted_views)   #subscripted_views is a number of directions to reconstruct a image
    trainer = _create_trainer(model, opt, data_train, data_test)
    trainer.run()
    last_10_accuracies = list(trainer.prediction_monitor.accuracies.values())[-10:]
    mean = np.mean(last_10_accuracies)
    return mean

for i in range(1, n_runs + 1):
    print('Start run {}'.format(i))
    result = experiment()
    accuracies.append(result)

with open(os.path.join(os.path.dirname(__file__), 'result_animal.txt'), 'w') as f:
    for i, r in enumerate(accuracies):
        f.write('Run {}: {}\n'.format(i, r))
    f.write('\n')
    f.write('mean: {}\n'.format(np.mean(accuracies)))
