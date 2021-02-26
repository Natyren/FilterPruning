import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as accuracy
import time
import random
import os
import numpy as np
from sklearn.metrics import precision_score, accuracy_score as accuracy
from sklearn.cluster import KMeans


def train_val(CFG, model, dataloaders, criterion, optimizer, scheduler, device):
    for epoch in range(CFG.epochs):

        print('Epoch {}'.format(epoch))
        time.sleep(0.3)
        for phase in ['train', 'val']:

            y_preds, y_trues = [], []
            tr_loss = 0
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()

            tk = tqdm(dataloaders[phase], total=len(dataloaders[phase]), position=0, leave=True)
            l = len(dataloaders[phase])
            for step, batch in enumerate(tk):

                optimizer.zero_grad()

                labels = batch[1].to(device)
                imgs = batch[0].to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(imgs)
                    loss = criterion(preds, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                y_preds.extend(torch.argmax(preds.cpu(), axis=1))
                y_trues.extend(labels.cpu())
                tr_loss += loss.item()
            epoch_loss = tr_loss / l
            acc = accuracy(y_preds, y_trues)
            print('Phase {}'.format(phase))
            print('Accuracy {}'.format(acc))
            print('Loss {}'.format(epoch_loss))
            print()
            time.sleep(0.3)

        scheduler.step()


def validate(model, dataloaders, verbose=True):
    preds, trues = [], []
    model.eval()
    model.cpu()
    if verbose:
        tk = tqdm(dataloaders['val'], total=len(dataloaders['val']), position=0, leave=True)
    else:
        tk = dataloaders['val']
    for _, batch in enumerate(tk):
        img = batch[0]
        lbls = batch[1]

        pred = model(img)
        preds.extend(pred.argmax(1))
        trues.extend(lbls)
    if verbose:
        print("Аккуратность на тестовом датасете равна {}".format(accuracy(trues, preds)))
        print("Точность на тестовом датасете равна {}".format(precision_score(trues, preds, average='macro')))
    else:
        return accuracy(trues, preds)


def seed_everything(seed):
    import imgaug
    imgaug.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prune(model, regime='reduce', k=1):
    for name, param in model.named_parameters():
        if 'conv' in name:
            shape = param.shape
            if regime == 'reduce':
                alg = KMeans(n_clusters=shape[0] - k)
            elif regime == 'union':
                alg = KMeans(n_clusters=k)
            elif regime == 'custom':
                if 'layer2' in name:
                    alg = KMeans(n_clusters=shape[0] - 2 * k)
                else:
                    alg = KMeans(n_clusters=shape[0] - k)
            else:
                raise ValueError('regime is not correct')
            param_arr = param.view(shape[0], -1).detach().cpu()
            pos = alg.fit_predict(param_arr)
            centers = alg.cluster_centers_
            for idx in range(shape[0]):
                param[idx] = torch.tensor(centers[pos[idx]], requires_grad=True).view(shape[1], shape[2], shape[3])
