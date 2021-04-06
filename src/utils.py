# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Colombine Verzat <colombine.verzat@idiap.ch>

# This file is part of als-classification.
#
# als-classification is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# als-classification is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with als-classification. If not, see <http://www.gnu.org/licenses/>.

from src.dataset import CustomDataset
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import time
import numpy as np
import gc
import yaml
import os
from typing import Dict, List


def get_project_root():
    """Get absolute path of the project"""
    return str(Path(__file__).resolve().parent.parent)


def get_config(name):
    """Read config file"""
    fname = f'config_{name}.yml'
    with open(os.path.join(get_project_root(), f'config/{fname}')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


# functions to train/test models

def create_train_dataloader(config, database, train_indices, classification, protocol, channels, fold):
    """Return Pytorch loader over train dataset"""
    basic_data_transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])])

    training_samples = len(train_indices) * 16 * 6
    if training_samples == 0:
        raise ValueError('0 indices to select in protocol file')
    print(f'Number of training samples : {training_samples}')

    # dataset augmentation
    data_transforms = dict()
    data_transforms[0] = basic_data_transform  # original
    data_transforms[1] = transforms.Compose(
        [transforms.RandomRotation((90, 90)), basic_data_transform])  # 90 degrees rotation
    data_transforms[2] = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=1), basic_data_transform])  # horizontal mirror
    data_transforms[3] = transforms.Compose(
        [transforms.RandomVerticalFlip(p=1), basic_data_transform])  # vertical mirror
    data_transforms[4] = transforms.Compose([transforms.RandomRotation((90, 90)), transforms.RandomHorizontalFlip(p=1),
                                             basic_data_transform])  # 90 degrees rotation of horizontal mirror
    data_transforms[5] = transforms.Compose([transforms.RandomRotation((90, 90)), transforms.RandomVerticalFlip(p=1),
                                             basic_data_transform])  # 90 degrees rotation of vertical mirror
    train_datasets = []

    for t in data_transforms.keys():
        train_dataset = CustomDataset(config, database, train_indices, data_transforms[t], classification, protocol,
                                      channels, fold)
        train_datasets.append(train_dataset)
    train_datasets = torch.utils.data.ConcatDataset(train_datasets)
    # special collate_fn so that images equal to None don't cause trouble in the batch
    train_loader = DataLoader(dataset=train_datasets, batch_size=config['batch_size'], shuffle=True, num_workers=0,
                              collate_fn=collate_fn)
    return train_loader


def create_test_dataloader(config, database, test_indices, classification, protocol, channels, fold):
    """Return Pytorch loader over test dataset"""
    basic_data_transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])])

    test_samples = len(test_indices) * 16
    if test_samples == 0:
        raise ValueError('0 indices to select in protocol file')
    print(f'Number of test samples : {test_samples}')

    test_dataset = CustomDataset(config, database, test_indices, basic_data_transform, classification, protocol,
                                 channels, fold)
    # special collate_fn so that images equal to None dont cause trouble in the batch
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0,
                             collate_fn=collate_fn)
    return test_loader, test_dataset


def collate_fn(batch):
    """Custom function to remove empty images from batch
    taken from: https://www.programmersought.com/article/86531627156/"""
    import re
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    _use_shared_memory = False
    r"""Whether to use shared memory in default_collate"""
    from torch._six import container_abcs, string_classes, int_classes
    error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

    numpy_type_map = {
        'float64': torch.DoubleTensor,
        'float32': torch.FloatTensor,
        'float16': torch.HalfTensor,
        'int64': torch.LongTensor,
        'int32': torch.IntTensor,
        'int16': torch.ShortTensor,
        'int8': torch.CharTensor,
        'uint8': torch.ByteTensor,
    }

    if isinstance(batch, list):
        batch = [(image, image_id, label) for (image, image_id, label) in batch if image is not None]
    if batch == []:
        return (None, None, None)
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return collate_fn([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)  # ok
        return [collate_fn(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))


def init_model(model, device, learning_rate):
    """Put model, criterion and optimizer on device (CPU/GPU)"""
    model.to(device)
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=learning_rate,
                                momentum=momentum)
    criterion.to(device)
    return criterion, optimizer


def train(net, loader, criterion, optimizer, device):
    """Perform one epoch over the input set"""
    net.train()
    total_loss = 0

    for i, (batch, target, index) in enumerate(loader):
        batch, target = batch.to(device), target.to(device)
        output = net(batch)
        loss = criterion.forward(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def count_nb_correct(output, target):
    """Count the number of correct labels compared to the ground truth"""
    pred = output.argmax(dim=1)
    return pred.eq(target).sum().item()


def roc_auc(targets, preds):
    """Compute roc_auc score"""
    y_true = targets.numpy()
    y_pred = preds.numpy()
    return roc_auc_score(y_true, y_pred)


def compute_metrics(targets, predictions, scores, debug=False):
    """Compute some measures of performance"""
    if debug:
        print(f'Number of samples: {len(targets)}')
        positives_indices = np.argwhere(targets == 1)
        negatives_indices = np.argwhere(targets == 0)
        positives_predictions = np.argwhere(predictions == 1)
        negatives_predictions = np.argwhere(predictions == 0)
        print(f'- positives : {len(positives_indices)}, detected: {len(positives_predictions)}')
        print(f'- negatives : {len(negatives_indices)}, detected: {len(negatives_predictions)}')
    try:
        auroc = roc_auc_score(targets, scores)
    except ValueError as e:
        print(e.args[0])
        auroc = np.nan
    return auroc


def validate(net, loader, device, save_metrics=False):
    """Evaluate over one epoch"""
    net.eval()
    n_correct = 0
    if save_metrics:
        targets = []
        predictions = []
        scores = []
        probabilities = []
        indices = []
    with torch.no_grad():
        for i, (batch, target, index) in enumerate(loader):
            if batch == None:  # if batch is empty (can happen if all images are black and were excluded)
                continue  # go to next batch
            batch, target = batch.to(device), target.to(device)
            output = net(batch)
            pred = output.argmax(dim=1)
            n_correct += pred.eq(target).sum().item()
            if save_metrics:
                targets.append(target.detach().cpu().numpy())
                predictions.append(pred.detach().cpu().numpy())
                scores.append(output[:, 1].detach().cpu().numpy())
                probabilities.append(torch.sigmoid(output[:, 1]).detach().cpu().numpy())
                indices.append(index.detach().cpu().numpy())
    if save_metrics:
        y_targets = np.concatenate(targets)
        y_predictions = np.concatenate(predictions)
        y_scores = np.concatenate(scores)
        y_probabilities = np.concatenate(probabilities)
        y_indices = np.concatenate(indices)

    acc = float(n_correct) / float(len(loader.dataset))
    if save_metrics:
        return acc, y_targets, y_predictions, y_scores, y_probabilities, y_indices
    else:
        return acc


def train_and_evaluate(n_epochs, model, train_loader, test_loader, criterion, optimizer, device, title, save_state_dict,
                       debug):
    """Train and evaluate over n_epochs"""
    print(f"The training is done on the {'GPU' if next(model.parameters()).is_cuda else 'CPU'}")
    beg = time.perf_counter()
    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        train_loss = train(model, train_loader, criterion, optimizer, device)  # Perform training
        if debug:  # Test on train set to get training accuracy
            train_accuracy = validate(model, train_loader, device, save_metrics=False)
            print(
                f"epoch {epoch} loss {train_loss:.3f} - accuracy {100 * train_accuracy:.2f} - time {time.perf_counter() - t0:.2f} seconds")
        else:
            print(f"epoch {epoch} loss {train_loss:.3f} - time {time.perf_counter() - t0:.2f} seconds")
    print(f"Training took {time.perf_counter() - beg:.2f} seconds")
    test_acc, y_targets, y_predictions, y_scores, y_probabilities, y_indices = validate(model, test_loader, device,
                                                                                        save_metrics=True)
    auroc = compute_metrics(y_targets, y_predictions, y_scores)
    print(f'Test AUROC is : {auroc:.4f}')
    print()
    if save_state_dict:
        directory = os.path.join(get_project_root(), "models")
        state_dict = f'{directory}/state_dict_{title}.pt'
        torch.save(model.state_dict(), state_dict)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return auroc
