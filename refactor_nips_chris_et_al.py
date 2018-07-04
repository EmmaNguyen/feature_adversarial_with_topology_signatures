import os
import math
import argparse

import torch
import torch.nn as nn
from torch import optim

from architecture.neural_topo_nets import PHConvNet
from utils.topo_utils import Provider, train_test_from_dataset, Trainer, LearningRateScheduler
from utils.topo_utils import ConsoleBatchProgress, PredictionMonitor
from utils.utils import export_result

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='an absolute path to dgm (h5).')
    parser.add_argument('--output_file', type=str, default="test_deployment.txt", help='filename of results from all experiments.')
    parser.add_argument('--writing_mode', type=bool, default=False, help="False if write a new file unless True")
    parser.add_argument('--num_epochs', type=int, default=3, help='a number of epochs, e.g. number of times running entire of data.')
    parser.add_argument('--momentum', type=float, default=0.7, help='a number indicating momentum.')
    parser.add_argument('--lr_start', type=float, default=0.1, help='a starting value of learning rate (lr) to be tried.')
    parser.add_argument('--lr_step', type=float, default=20, help='a number of incremental steps in learning rate (lr).')
    parser.add_argument('--lr_adaption', type=float, default=0.5, help='a coefficient to jump between values of learning rate (lr).')
    parser.add_argument('--test_ratio', type=float, default=0.5, help='a ratio between test set and training set.')
    parser.add_argument('--batch_size', type=int, default=128, help='a number indicating a batch size.')
    parser.add_argument('--num_experiment', type=int, default=5, help='a number of experiments repeated.')
    return parser.parse_args()

def _create_trainer(model, opt, data_train, data_test):
    optimizer = optim.SGD(model.parameters(), lr=opt.lr_start, momentum=opt.momentum)
    loss = nn.CrossEntropyLoss()   #Todo: Change to binary cross entropy
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss=loss,
                      train_data=data_train,
                      n_epochs=opt.num_epochs,
                      cuda=torch.cuda.is_available(),
                      variable_created_by_model=True)

    def determine_lr(self, **kwargs):
        """
        Todo: check reference to find learning_rate
        """
        epoch = kwargs['epoch_count']
        if epoch % opt.lr_step == 0:
            return opt.lr_start / 2 ** (epoch / opt.lr_step)

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

def load_data(opt):
    view_name_template = 'dim_0_dir_{}'
    subscripted_views = sorted([view_name_template.format(i) for i in range(32)])
    assert (str(len(subscripted_views)) in opt.data_path)
    print("[ Load provider data ]")
    dataset = Provider()
    dataset.read_from_h5(opt.data_path)
    assert all(view_name in dataset.view_names for view_name in subscripted_views)
    print("[ Create data loader]")
    data_train, data_test = train_test_from_dataset(dataset,
                                                    test_size=opt.test_ratio,
                                                    batch_size=opt.batch_size)
    return data_train, data_test, subscripted_views

def run_experiment(opt):
    print("[ Start experimenting ]")
    data_train, data_test, subscripted_views = load_data(opt)
    for i in range(1, opt.num_experiment + 1):
        print('[ Run experiment {} ]'.format(i))
        model = PHConvNet(subscripted_views)   #subscripted_views is a number of directions to reconstruct a image
        trainer = _create_trainer(model, opt, data_train, data_test)
        trainer.run()
        last_ten_accuracies = list(trainer.prediction_monitor.accuracies.values())[-10:]
        average_accuracy = sum(last_ten_accuracies)/ 10.0
        export_result(opt.output_file, average_accuracy, opt.writing_mode)
    print("[ End experimenting ]")

if __name__=='__main__':
    opt = parse_arguments()
    run_experiment(opt)
