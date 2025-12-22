import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import h5py
import torch.nn.functional as F

NEW_SIZE = 128

# To get the images and labels from file
with h5py.File(r"C:\Users\nicol\Desktop\Universita\ProgettoCM\Galaxy10_DECals.h5", 'r') as F_h5:
    images = np.array(F_h5['images'])
    labels = np.array(F_h5['ans'])

print("Elaborazione immagini in corso...")


images = torch.tensor(images, dtype=torch.float32) / 255.0 
labels = torch.tensor(labels, dtype=torch.float32)

# Pytorch vuole il formato (N, 3, H, W) ma noi lo diamo così (N, H, W, 3)
if images.shape[-1] == 3:
    images = images.permute(0, 3, 1, 2)
    channels = 3

# Riscaliamo le immagini
images = F.interpolate(images, size=(NEW_SIZE, NEW_SIZE), mode='bilinear', align_corners=False)

train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
X_train, Y_train, X_test, Y_test = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

#  Selecting the right galaxy classes

target_classes = [2, 6, 9]

mask = np.isin(Y_train, target_classes)
X_train = X_train[mask]
Y_train = Y_train[mask]
mask = np.isin(Y_test, target_classes)
X_test = X_test[mask]
Y_test = Y_test[mask]

save_path = r"C:\Users\nicol\Desktop\Universita\ProgettoCM\Galaxy10_128x128_Processed.h5"
print(f"Salvataggio del dataset ridotto in {save_path}...")

with h5py.File(save_path, 'w') as f:
    # Salviamo i dati di training
    # .cpu().numpy() serve a portare i dati fuori da PyTorch
    f.create_dataset('X_train', data=X_train.cpu().numpy().astype('float16'))
    f.create_dataset('Y_train', data=Y_train.cpu().numpy())
    
    # Salviamo i dati di test
    f.create_dataset('X_test', data=X_test.cpu().numpy().astype('float16'))
    f.create_dataset('Y_test', data=Y_test.cpu().numpy())

print("Salvataggio completato! Ora hai un file leggero pronto all'uso.")