import torch
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import torch.nn.functional as F

NEW_SIZE = 64

with h5py.File(r"C:\Users\franc\OneDrive\Desktop\Università\CM\ProgettoCM\Galaxy10_DECals.h5", 'r') as F_h5:
    images = np.array(F_h5['images'])
    labels = np.array(F_h5['ans'])

print("Elaborazione immagini in corso")


images = torch.tensor(images, dtype=torch.float32) / 255.0
labels = torch.tensor(labels, dtype=torch.long)

# Pytorch vuole il formato (N, 3, H, W) ma noi lo diamo così (N, H, W, 3)
if images.shape[-1] == 3:
    images = images.permute(0, 3, 1, 2)
    channels = 3

# Merging dei pixel
images = F.interpolate(images, size=(NEW_SIZE, NEW_SIZE), mode='bilinear', align_corners=False)

train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
X_train, Y_train, X_test, Y_test = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

# Definiamo le classi target
target_classes = [2, 6, 9]

def balance_and_filter(X, Y, classes):
    # Convertiamo le etichette in numpy per lavorarci meglio
    Y_np = Y.numpy() # Vettore con numeri da 0 a 9 corrispondenti alle classi
    
    # Filtriamo solo le immagini che appartengono alle classi target
    # Creiamo una maschera booleana iniziale, associa i label dei dati alle target classes, true se appartiene alle classi selezionate
    mask = np.isin(Y_np, classes)
    X_filtered = X[mask]
    Y_filtered = Y_np[mask] 
    
    # Troviamo il numero minimo di campioni tra le classi scelte
    min_count = float('inf')
    for cls in classes:
        count = np.sum(Y_filtered == cls)
        print(f"Classe {cls}: trovate {count} immagini.")
        if count < min_count:
            min_count = count
    
    print(f"-> Bilanciamento il dataset a {min_count} immagini per ogni classe")
    
    # Selezioniamo casualmente gli indici per ottenere lo stesso numero
    balanced_indices = []
    for cls in classes:
        # Trova gli indici di questa specifica classe nel dataset filtrato
        cls_indices = np.where(Y_filtered == cls)[0] # Lista delle posizoni delle galassie appartenti ad una data classe
        
        selected_indices = np.random.choice(cls_indices, min_count, replace=False) # Prende dalla lista una posizione un numero pari
                                                                                   # a min_count di volte, senza ripetizioni
        balanced_indices.extend(selected_indices) # Attacca gli indici alla lista finale 
    
    # Convertiamo la lista in array e mescoliamo per non avere le classi ordinate
    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)

    # Y_filtered era numpy, lo riconvertiamo in Tensor
    return X_filtered[balanced_indices], torch.tensor(Y_filtered[balanced_indices], dtype=torch.long)

print("=== Bilanciamento Training Set ===")
X_train, Y_train = balance_and_filter(X_train, Y_train, target_classes)

print("\n=== Bilanciamento Test Set ===")
X_test, Y_test = balance_and_filter(X_test, Y_test, target_classes)

# Salviamo le immagini riscalate in un nuovo file
save_path = r"C:\Users\franc\OneDrive\Desktop\Università\CM\ProgettoCM\Galaxy10_64x64_169.h5"
print(f"\nSalvataggio del dataset ridotto in {save_path}")

with h5py.File(save_path, 'w') as f:
    f.create_dataset('X_train', data=X_train.cpu().numpy().astype('float16'))
    f.create_dataset('Y_train', data=Y_train.cpu().numpy())
    f.create_dataset('X_test', data=X_test.cpu().numpy().astype('float16'))
    f.create_dataset('Y_test', data=Y_test.cpu().numpy())

print("Salvataggio completato!")