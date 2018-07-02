import os

import torch
import torch.nn as nn
from torch import optim

from architecture.neural_topo_nets import MyModel
from utils.topo_utils import Provider, train_test_from_dataset, Trainer, LearningRateScheduler
from utils.topo_utils import ConsoleBatchProgress, PredictionMonitor

parent = "/home/emma/Research/GAN/nips2017/"
provider_path = os.path.join(parent, 'data/dgm_provider/npht_small_train_32dirs.h5')
raw_data_path = os.path.join(parent, 'data/raw_data/small_train/')
print('Starting experiment...')
accuracies = []
n_runs = 5


def _create_trainer(model, params, data_train, data_test):
    optimizer = optim.SGD(model.parameters(), lr=params['lr_start'],momentum=params['momentum'])
    loss = nn.CrossEntropyLoss()
    trainer = Trainer(model=model,
                         optimizer=optimizer,
                         loss=loss,
                         train_data=data_train,
                         n_epochs=params['epochs'],
                         cuda=params['cuda'],
                         variable_created_by_model=True)

    def determine_lr(self, **kwargs):
        """
        """
        epoch = kwargs['epoch_count']
        if epoch % params['lr_ep_step'] == 0:
            return params['lr_start'] / 2 ** (epoch / params['lr_ep_step'])

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


def _parameters():
    return {'data_path': None,
        'epochs': 300,
        'momentum': 0.7,
        'lr_start': 0.1,
        'lr_ep_step': 20,
        'lr_adaption': 0.5,
        'test_ratio': 0.5,
        'batch_size': 128,
        'cuda': False}

def _data_setup(params):
    view_name_template = 'dim_0_dir_{}'
    subscripted_views = sorted([view_name_template.format(i) for i in range(32)])
    assert (str(len(subscripted_views)) in params['data_path'])

    print('Loading provider...')
    dataset = Provider()
    dataset.read_from_h5(params['data_path'])

    assert all(view_name in dataset.view_names for view_name in subscripted_views)

    print('Create data loader...')
    data_train, data_test = train_test_from_dataset(dataset,
                                                    test_size=params['test_ratio'],
                                                    batch_size=params['batch_size'])

    return data_train, data_test, subscripted_views

def experiment(data_path):
    params = _parameters()
    params['data_path'] = data_path
    if torch.cuda.is_available():
        params['cuda'] = True
    data_train, data_test, subscripted_views = _data_setup(params)
    model = MyModel(subscripted_views)   #subscripted_views is a number of directions to reconstruct a image
    trainer = _create_trainer(model, params, data_train, data_test)
    trainer.run()
    last_10_accuracies = list(trainer.prediction_monitor.accuracies.values())[-10:]
    mean = np.mean(last_10_accuracies)
    return mean

for i in range(1, n_runs + 1):
    print('Start run {}'.format(i))
    result = experiment(provider_path)
    accuracies.append(result)

with open(os.path.join(os.path.dirname(__file__), 'result_animal.txt'), 'w') as f:
    for i, r in enumerate(accuracies):
        f.write('Run {}: {}\n'.format(i, r))
    f.write('\n')
    f.write('mean: {}\n'.format(np.mean(accuracies)))
