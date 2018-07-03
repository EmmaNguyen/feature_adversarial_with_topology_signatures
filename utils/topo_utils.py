import os
import sys
import time
import h5py
import shutil
import json
import datetime
import multiprocessing

import imageio
import skimage
import numpy as np
import skimage.morphology
import scipy.misc
import scipy.ndimage
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing.label import LabelEncoder
from collections import defaultdict
from torch.nn import Module
from collections import Iterable

from torch import Tensor, LongTensor
from torch.tensor import _TensorBase
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from pershombox import calculate_discrete_NPHT_2d
# import chofer_torchex.utils.trainer as tr
from chofer_torchex.utils.trainer.plugins import *


DGM_MIN_PERSISTENCE_THRESHOLD = 0.01

def reduce_to_largest_connected_component(img):
    label_map, n = skimage.morphology.label(img, neighbors=4, background=0, return_num=True)
    volumes = []
    for i in range(n):
        volumes.append(np.count_nonzero(label_map == (i + 1)))
    arg_max = np.argmax(volumes)
    img = (label_map == (arg_max + 2))
    return img

def get_npht(img, number_of_directions):
    img = np.ndarray.astype(img, bool)
    npht = calculate_discrete_NPHT_2d(img, number_of_directions)
    return npht

def threhold_dgm(dgm):
    return list(p for p in dgm if p[1]-p[0] > DGM_MIN_PERSISTENCE_THRESHOLD)


# In[3]:


class ProviderError(Exception):
    pass

class NameSpace:
    pass

class Provider:
    _serial_str_keys = NameSpace()
    _serial_str_keys.data_views = 'data_views'
    _serial_str_keys.str_2_int_label_map = 'str_2_int_label_map'
    _serial_str_keys.meta_data = 'meta_data'

    def __init__(self, data_views={}, str_2_int_label_map=None, meta_data={}):
        self.data_views = data_views
        self.str_2_int_label_map = str_2_int_label_map
        self.meta_data = meta_data
        self._cache = NameSpace()

    def add_view(self, name_of_view, view):
        assert type(name_of_view) is str
        assert isinstance(view, dict)
        assert all([type(label) is str for label in view.keys()])
        assert name_of_view not in self.data_views

        self.data_views[name_of_view] = view

    def add_str_2_int_label_map(self, label_map):
        assert isinstance(label_map, dict)
        assert all([type(str_label) is str for str_label in label_map.keys()])
        assert all([type(int_label) is int for int_label in label_map.values()])
        self.str_2_int_label_map = label_map

    def add_meta_data(self, meta_data):
        assert isinstance(meta_data, dict)
        self.meta_data = meta_data

    def _check_views_are_consistent(self):
        if len(self.data_views) > 0:
            first_view = next(iter(self.data_views.values()))
            # Check if every view has the same number of labels.
            lenghts_same = [len(first_view) == len(view) for view in self.data_views.values()]
            if not all(lenghts_same):
                raise ProviderError('Not all views have same amount of label groups.')
            # Check if every view has the same labels.
            labels_same = [set(first_view.keys()) == set(view.keys()) for view in self.data_views.values()]
            if not all(labels_same):
                raise ProviderError('Not all views have the same labels in their label groups.')
            # Check if every label group has the same number of subjects in each view.
            labels = first_view.keys()
            for k in labels:
                label_groups_cons = [set(first_view[k].keys()) == set(view[k].keys()) for view in
                                     self.data_views.values()]
                if not all(label_groups_cons):
                    raise ProviderError('There is some inconsistency in the labelgroups.'
                                        + ' Not the same subject ids in each view for label {}'.format(k))

    def _check_str_2_int_labelmap(self):
        """
        assumption: _check_views_are_consistent allready called.
        """
        first_view = list(self.data_views.values())[0]
        # Check if the labels are the same.
        if not set(self.str_2_int_label_map.keys()) == set(first_view.keys()):
            raise ProviderError('self.str_2_int_label_map has not the same labels as the data views.')
        # Check if int labels are int.
        if not all([type(v) is int for v in self.str_2_int_label_map.values()]):
            raise ProviderError('Labels in self.str_2_int_label have to be of type int.')

    def _check_state_for_serialization(self):
        if len(self.data_views) == 0:
            raise ProviderError('Provider must have at least one view.')
        self._check_views_are_consistent()
        if self.str_2_int_label_map is not None:
            self._check_str_2_int_labelmap()

    def _prepare_state_for_serialization(self):
        self._check_state_for_serialization()
        if self.str_2_int_label_map is None:
            self.str_2_int_label_map = {}
            first_view = list(self.data_views.values())[0]
            for i, label in enumerate(first_view):
                self.str_2_int_label_map[label] = i + 1

    def dump_as_h5(self, file_path):
        self._prepare_state_for_serialization()
        with h5py.File(file_path, 'w') as file:
            data_views_grp = file.create_group(self._serial_str_keys.data_views)
            for view_name, view in self.data_views.items():
                view_grp = data_views_grp.create_group(view_name)
                for label, label_subjects in view.items():
                    label_grp = view_grp.create_group(label)
                    for subject_id, subject_values in label_subjects.items():
                        label_grp.create_dataset(subject_id, data=subject_values)
            label_map_grp = file.create_group(self._serial_str_keys.str_2_int_label_map)
            for k, v in self.str_2_int_label_map.items():
                # since the lua hdf5 implementation seems to have issues reading scalar values we
                # dump the label as 1 dimensional tuple.
                label_map_grp.create_dataset(k, data=(v,))
            meta_data_group = file.create_group(self._serial_str_keys.meta_data)
            for k, v in self.meta_data.items():
                if type(v) is str:
                    v = np.string_(v)
                    dset = meta_data_group.create_dataset(k, data=v)
                else:
                    meta_data_group.create_dataset(k, data=v)

    def read_from_h5(self, file_path):
        with h5py.File(file_path, 'r') as file:
            # load data_views
            data_views = dict(file[self._serial_str_keys.data_views])
            for view_name, view in data_views.items():
                view = dict(view)
                data_views[view_name] = view

                for label, label_group in view.items():
                    label_group = dict(label_group)
                    view[label] = label_group

                    for subject_id, value in label_group.items():
                        label_group[subject_id] = file[self._serial_str_keys.data_views][view_name][label][subject_id][
                            ()]

            self.data_views = data_views

            # load str_2_int_label_map
            str_2_int_label_map = dict(file[self._serial_str_keys.str_2_int_label_map])
            for str_label, str_to_int in str_2_int_label_map.items():
                str_2_int_label_map[str_label] = str_to_int[()]

            self.str_2_int_label_map = str_2_int_label_map
            for k, v in self.str_2_int_label_map.items():
                self.str_2_int_label_map[k] = int(v[0])

            # load meta_data
            meta_data = dict(file[self._serial_str_keys.meta_data])
            for k, v in meta_data.items():
                meta_data[k] = v[()]
            self.meta_data = meta_data
        return self

    def select_views(self, views: [str]):
        data_views = {}
        for view in views:
            data_views[view] = self.data_views[view]
        return Provider(data_views=data_views, str_2_int_label_map=self.str_2_int_label_map, meta_data=self.meta_data)

    @property
    def sample_id_to_label_map(self):
        if not hasattr(self._cache, 'sample_id_to_label_map'):
            self._cache.sample_id_to_label_map = {}
            for label, label_data in self.data_views[self.view_names[0]].items():
                for sample_id in label_data:
                    self._cache.sample_id_to_label_map[sample_id] = label

        return self._cache.sample_id_to_label_map

    @property
    def view_names(self):
        return list(self.data_views.keys())

    @property
    def labels(self):
        return list(self.data_views[self.view_names[0]].keys())

    @property
    def sample_labels(self):
        for i in range(len(self)):
            _, label = self[i]
            yield label

    @property
    def sample_ids(self):
        if not hasattr(self._cache, 'sample_ids'):
            first_view = self.data_views[self.view_names[0]]
            self._cache.sample_ids = sum([list(label_group.keys()) for label_group in first_view.values()], [])

        return self._cache.sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        sample_label = self.sample_id_to_label_map[sample_id]
        x = {}
        for view_name, view_data in self.data_views.items():
            x[view_name] = view_data[sample_label][sample_id]
        return x, sample_label


# In[4]:


def explode_home_symbol(path: str):
    assert isinstance(path, str)
    if path[0] == "~":
        return os.path.expanduser("~") + path[1::]  # Maybe cross platform issue here
    else:
        return path


class FileSystemObject(object):
    """
    Class representing a generic file system object.
    """
    def __init__(self, name, path):
        self.name = name
        self.path = path

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

class FileSystemObjectCollection(object):
    """
    Generic collection of FileSystemObjects.
    """
    def __init__(self, path: str=None):
        self._content = None
        path = explode_home_symbol(path)

        if path is not None:
            if not os.path.isdir(path):
                raise ValueError("Parameter path has to be a valid directory!")
            self._get_content_by_path(path)
        if self._content is None:
            self._content = []
    def _get_content_by_path(self, path: str)->Iterable:
        raise NotImplementedError()
    def __iter__(self):
        return iter(self._content)

class File(FileSystemObject):
    """
    Representation of a file in the filesystem.
    """
    def open(self, mode: str):
        """
        Get a file descriptor for the file.
        :param mode: mode to open file descriptor with (modes like in open(...))
        :return: (str)
        """
        return open(self.path, mode)

class Folder(FileSystemObject):
    """
    Representation of a folder in the filesystem.
    """
    def __init__(self, path, create=False):
        path = explode_home_symbol(path)
        if path is None:
            raise ValueError("path is None!")
        if not os.path.isdir(path):
            if not create:
                raise ValueError("{} path has to be a valid existing directory!".format(path))
            else:
                os.mkdir(path)
        super().__init__(os.path.basename(path), path)

    def _get_all_direct_sub_files(self)->[File]:
        return_value = []
        for file_name, file_path in [(name, os.path.join(self.path, name)) for name in os.listdir(self.path)]:
            if os.path.isfile(file_path):
                return_value.append(File(file_name, file_path))
        return return_value

    def files(self, recursive=False, name_pred=None):
        """
        Gives a list of all files in the folder.
        :param recursive: If true, also the files in all direct and indirect sub folders are returned.
        :param name_pred: A predicate which is applied to the name property of each file. If it is false the file
        is filtered out.
        :return: [File]
        """
        return_value = self._get_all_direct_sub_files()
        if recursive:
            sub_folders = self.folders(recursive=True)
            return_value += sum([folder.files(name_pred) for folder in sub_folders], [])
        if name_pred and len(return_value) > 0:
            return_value = [f for f in return_value if name_pred(f.name)]
        return return_value

    def _get_all_direct_sub_folders(self):
        return_value = []
        for file_name, dir_path in [(name, os.path.join(self.path, name)) for name in os.listdir(self.path)]:
            if os.path.isdir(dir_path):
                return_value.append(Folder(dir_path))
        return return_value

    def folders(self, recursive=False, name_pred=None)->[FileSystemObject]:
        """
        Get a list of all folders in folder.
        :param recursive: If true, also the folders in all direct and indirect sub folders are returned.
        :param name_pred: A predicate which is applied to the name property of each folder. If it is false the
        folder is filtered out.
        """
        return_value = self._get_all_direct_sub_folders()
        if recursive:
            return_value += sum([folder.folders(recursive=True, name_pred=name_pred) for folder in return_value], [])
        if name_pred and len(return_value > 0):
            return_value = [f for f in return_value if name_pred(f.name)]
        return return_value

    def content(self, recursive=False, name_filter=None):
        """
        Get a list of the folder's content.
        :param recursive: If true, the recursive content of the folder is returned.
        :param name_filter: A predicate which is applied to the name property of each folder or file. If it is false the
        object is filtered out.
        :return: :rtype:(FileSystemObject)
        """
        return self.folders(recursive=recursive, name_pred=name_filter) + self.files(recursive=recursive, name_pred=name_filter)

class FileCollection(FileSystemObjectCollection):
    """
    Collection which consists only of files.
    """
    def _get_content_by_path(self, path: str):
        self._content = []

        for file_name, file_path in [(name, os.path.join(path, name)) for name in os.listdir(path)]:
            if os.path.isfile(file_path):
                self._content.append(File(file_name, file_path))

class FolderCollection(FileSystemObjectCollection):
    """
    Collection which consists only of folders.
    """
    def _get_content_by_path(self, path: str) -> Iterable:
        self._content = []
        for file_name, file_path in [(name, os.path.join(path, name)) for name in os.listdir(path)]:
            if os.path.isdir(file_path):
                self._content.append(File(file_name, file_path))

# In[6]:


class PersistenceDiagramProviderCollate:
    def __init__(self, provider, wanted_views: [str] = None,
                 label_map: callable = lambda x: x,
                 output_type=torch.FloatTensor,
                 target_type=torch.LongTensor):
        provided_views = provider.view_names
        if wanted_views is None:
            self.wanted_views = provided_views
        else:
            for wv in wanted_views:
                if wv not in provided_views:
                    raise ValueError('{} is not provided by {} which provides {}'.format(wv, provider, provided_views))
            self.wanted_views = wanted_views
        if not callable(label_map):
            raise ValueError('label_map is expected to be callable.')
        self.label_map = label_map
        self.output_type = output_type
        self.target_type = target_type

    def __call__(self, sample_target_iter):
        batch_views_unprepared, batch_views_prepared, targets = defaultdict(list), {}, []
        for dgm_dict, label in sample_target_iter:
            for view_name in self.wanted_views:
                dgm = list(dgm_dict[view_name])
                dgm = self.output_type(dgm)
                batch_views_unprepared[view_name].append(dgm)
            targets.append(self.label_map(label))
        targets = self.target_type(targets)
        return batch_views_unprepared, targets

class SubsetRandomSampler:
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def train_test_from_dataset(dataset,
                            test_size=0.2,
                            batch_size=64,
                            wanted_views=None):
    sample_labels = list(dataset.sample_labels)
    label_encoder = LabelEncoder().fit(sample_labels)
    sample_labels = label_encoder.transform(sample_labels)
    label_map = lambda l: int(label_encoder.transform([l])[0])
    collate_fn = PersistenceDiagramProviderCollate(dataset, label_map=label_map, wanted_views=wanted_views)
    sp = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    train_i, test_i = list(sp.split([0]*len(sample_labels), sample_labels))[0]
    data_train = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=False,
                            sampler=SubsetRandomSampler(train_i.tolist()))
    data_test = DataLoader(dataset,
                           batch_size=batch_size,
                           collate_fn=collate_fn,
                           shuffle=False,
                           sampler=SubsetRandomSampler(test_i.tolist()))
    return data_train, data_test

class UpperDiagonalThresholdedLogTransform(object):
    def __init__(self, nu):
        self.b_1 = (torch.Tensor([1, 1]) / np.sqrt(2))
        self.b_2 = (torch.Tensor([-1, 1]) / np.sqrt(2))
        self.nu = nu

    def __call__(self, dgm):
        if dgm.ndimension() == 0:
            return dgm
        if dgm.is_cuda:
            self.b_1 = self.b_1.cuda()
            self.b_2 = self.b_2.cuda()
        x = torch.mul(dgm, self.b_1.repeat(dgm.size(0), 1))
        x = torch.sum(x, 1).squeeze()
        y = torch.mul(dgm, self.b_2.repeat( dgm.size(0), 1))
        y = torch.sum(y, 1).squeeze()
        i = (y <= self.nu)
        y[i] = torch.log(y[i] / self.nu) + self.nu
        ret = torch.stack([x, y], 1)
        return ret

def pers_dgm_center_init(n_elements):
    centers = []
    while len(centers) < n_elements:
        x = np.random.rand(2)
        if x[1] > x[0]:
            centers.append(x.tolist())
    return torch.Tensor(centers)

def train_PHConvNet(model, opt, data_train, data_test):
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

from torch.optim import Optimizer
from typing import Callable

class Event:
    def __init__(self):
        self._callbacks = []

    def __call__(self, default_kwargs: {}, **kwargs):
        kwargs.update(default_kwargs)
        for callback in self._callbacks:
            callback(**kwargs)

    def append(self, callback):
        if not callable(callback):
            raise ValueError('Expected callable.')
        self._callbacks.append(callback)

class TrainerEvents:
    def __init__(self):
        self.pre_run = Event()
        self.pre_epoch = Event()
        self.post_epoch = Event()
        self.post_batch_backward = Event()
        self.pre_batch = Event()
        self.post_batch = Event()
        self.post_epoch_gui = Event()
        self.post_run = Event()

class Trainer(object):
    def __init__(self, model: Module, loss: Callable, optimizer: Optimizer, train_data: DataLoader, n_epochs,
                 cuda=False, cuda_device_id=None,variable_created_by_model=False):
        self.n_epochs = n_epochs
        self.model = model
        self.criterion = loss
        self.optimizer = optimizer
        self.train_data = train_data
        self.epoch_count = 1
        self.cuda = cuda
        self.cuda_device_id = cuda_device_id
        self.variable_created_by_model = variable_created_by_model
        self.return_value = {}
        self.events = TrainerEvents()

    def _get_default_event_kwargs(self):
        return {'trainer': self,
                'model': self.model,
                'epoch_count': self.epoch_count,
                'cuda': self.cuda}

    @property
    def iteration_count(self):
        return self.batch_count * self.epoch_count

    def register_plugin(self, plugin):
        plugin.register(self)

    def run(self):
        self.events.pre_run(self._get_default_event_kwargs())
        if self.cuda:
            self.model.cuda(self.cuda_device_id)
        for i in range(1, self.n_epochs + 1):
            self.epoch_count = i
            self.events.pre_epoch(self._get_default_event_kwargs(),
                                  optimizer=self.optimizer,
                                  train_data=self.train_data,
                                  max_epochs=self.n_epochs,
                                  current_epoch_number=i)
            self._train_epoch()
            self.events.post_epoch(self._get_default_event_kwargs(), trainer=self)
            self.events.post_epoch_gui(self._get_default_event_kwargs())
        self.events.post_run(self._get_default_event_kwargs())
        return self.return_value

    def _train_epoch(self):
        self.model.train()
        for i, (batch_input, batch_target) in enumerate(self.train_data, start=1):
            self.events.pre_batch(self._get_default_event_kwargs(),
                                  batch_input=batch_input,
                                  batch_target=batch_target)
            batch_input, batch_target = self.data_typing(batch_input, batch_target)
            target_var = Variable(batch_target)
            if not self.variable_created_by_model:
                batch_input = Variable(batch_input)
            def closure():
                batch_output = self.model(batch_input)
                loss = self.criterion(batch_output, target_var)
                loss.backward()
                assert len(loss.data) == 1
                self.events.post_batch_backward(self._get_default_event_kwargs(),
                                                batch_output=batch_output,
                                                loss=float(loss.data[0]))
                return loss
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.events.post_batch(self._get_default_event_kwargs(),
                                   batch_input=batch_input,
                                   batch_target=batch_target,
                                   current_batch_number=i)

    @staticmethod
    def before_data_typing_hook(batch_input, batch_targets):
        return batch_input, batch_targets

    @staticmethod
    def after_data_typing_hook(batch_input, batch_targets):
        return batch_input, batch_targets

    def data_typing(self, batch_input, batch_targets):
        batch_input, batch_targets = self.before_data_typing_hook(batch_input, batch_targets)
        tensor_cast = Tensor.cuda if self.cuda else Tensor.cpu
        def cast(x):
            if isinstance(x, torch.tensor._TensorBase):
                return tensor_cast(x)
            elif isinstance(x, list):
                return [cast(v) for v in x]
            elif isinstance(x, dict):
                return {k: cast(v) for k, v in x.items()}
            elif isinstance(x, tuple):
                return tuple(cast(v) for v in x)
            else:
                return x
        batch_input = cast(batch_input)
        batch_targets = cast(batch_targets)
        batch_input, batch_targets = self.after_data_typing_hook(batch_input, batch_targets)
        return batch_input, batch_targets
#
#
# import chofer_torchex.utils.trainer as tr
# from chofer_torchex.utils.trainer.plugins import *
