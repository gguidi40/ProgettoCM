import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import h5py
import itertools
import sys
import os
import shutil
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import random
from Log import Logger
from Shape import ShapeDataset
from Modello import ConvVAE
from Train import Trainer


## 1. Scrittura della shell su file txt

cartella_attuale = r"C:\Users\franc\OneDrive\Desktop\Università\CM\ProgettoCM"
percorso_log = os.path.join(cartella_attuale, "grid_search_convvae.txt") # Unisce il percorso cartella attuale con il file txt

sys.stdout = Logger()

print("\n" + "="*30)
print(f"Logger attivato correttamente")
print(f"Il file si trova qui: {percorso_log}")
print("="*30 + "\n")


## 2. Utilizzo della GPU

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Utilizzo device: {device}")


## 3. Caricamento e organizzazione dati

with h5py.File(r"C:\Users\franc\OneDrive\Desktop\Università\CM\ProgettoCM\Galaxy10_64x64_Balanced.h5", 'r') as F_h5:
    # Nel file i dati sono già splittati in train e test
    X_train = np.array(F_h5['X_train'])
    Y_train = np.array(F_h5['Y_train'])
    X_test = np.array(F_h5['X_test'])
    Y_test = np.array(F_h5['Y_test'])

# Trasformiamo i dati in tensori di torch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_test  = torch.tensor(Y_test, dtype=torch.long)

# Correzione Dimensioni: Da (N, 128, 128, 3) a (N, 3, 128, 128)
if X_test.shape[-1] == 3:
    X_train = X_train.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)
elif X_test.shape[2] == 3:
    X_train = X_train.permute(0, 2, 1, 3)
    X_test = X_test.permute(0, 2, 1, 3)

# Split Train/Val
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.1, shuffle=True, random_state=42
)

## 7. Grid search

# Definzione dei parametri

param_grid = {                     # Dizionario dei parametri
    'batch_size': [32, 64],
    'learning_rate': [1e-3, 5e-4, 1e-4],
    'base_channels': [16, 32],
    'latent_dim': [128, 256, 512],
    'num_layers': [3, 4, 5],
    'beta': [0.01, 0.001]
}

keys = param_grid.keys() # Estraiamo i nomi dei parametri e li mettiamo in una lista
values = param_grid.values()
combinations = list(itertools.product(*values)) # Creaiamo tutte le combinazioni possibili

best_config = None
global_best_loss = float('inf')
results = []

print(f"Inizio Grid Search su {len(combinations)} combinazioni")

trainer = Trainer()  # Creiamo un'istanza della classe Trainer
for combo in combinations:
    # Crea un dizionario per la configurazione corrente
    config = dict(zip(keys, combo))

    try:  # Utilizziamo try ed except per non fermare il ciclo in caso di errori
        val_loss = trainer.train_evaluate_model(config, epochs=15, X_train, Y_train, X_val, Y_val, device)

        results.append((config, val_loss))

        if val_loss < global_best_loss:
            global_best_loss = val_loss
            best_config = config
            # Salviamo il file con i pesi del modello migliore
            percorso_sorgente = os.path.join(cartella_attuale, "migliori_pesi_galassie.pth")
            percorso_destinazione = os.path.join(cartella_attuale, "pesi_finali.pth")
            if os.path.exists(percorso_sorgente):
                shutil.copy(percorso_sorgente, percorso_destinazione)
                print(f"--> Nuova configurazione migliore! Salvata in: {percorso_destinazione}")
            else:
                print(f"ERRORE: Non trovo il file {percorso_sorgente}. Controlla il nome in torch.save()")

    except Exception as e:
        print(f"Errore con configurazione {config}: {e}")

print("\n========================================")
print(f"Grid Search Completata.")
print(f"Migliore Configurazione: {best_config}")
print(f"Migliore Validation Loss: {global_best_loss}")
print("========================================")


## 8. Test del modello

print("\n" + "="*30)
print("Inizio del test")
print("="*30)

test_loader = DataLoader(ShapeDataset(X_test, Y_test), batch_size=best_config['batch_size'], shuffle=False) # Creiamo il test set

# Istanziamo il modello con la configurazione migliore
final_model = ConvVAE(
    img_size=64,
    base_channels=best_config['base_channels'],
    latent_dim=best_config['latent_dim'],
    num_layers=best_config['num_layers']
).to(device)

# Caricamento dei pesi
percorso_caricamento = os.path.join(cartella_attuale, "pesi_finali.pth")

final_model.load_state_dict(torch.load(percorso_caricamento, weights_only=True))

print("Pesi caricati correttamente")

## SSIM

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

test_ssim = 0.0

print(f"Calcolo metriche su {len(X_test)} immagini")

batch_casuale_idx = random.randint(0, len(test_loader) - 1)

with torch.no_grad():
    for i, (X_batch, _) in enumerate(test_loader):
        X_batch = X_batch.to(device)
        recon, _, _ = final_model(X_batch)  # Ricostruzione delle immagini del modello finale

        # SSIM
        test_ssim += ssim_metric(recon, X_batch).item() # Confronta la ricostruzione con l'originale usando la metrica SSIM

# Calcolo finale SSIM
final_ssim_score = test_ssim / len(test_loader)


## FID

# Utilizziamo feature=2048 per avere una valutazione più precisa delle immagini
fid_metric = FrechetInceptionDistance(feature=2048).to(device)

# Funzione che genera galassie con il nostro vae
def genera_per_fid(model, num_immagini, latent_dim, device):
    model.eval()
    with torch.no_grad():
        # Generiamo vettori casuali nello spazio latente
        z = torch.randn(num_immagini, latent_dim).to(device)

        # Passiamo i vettori al decoder
        fake_images = model.decode(z)

        # Trasformazione in uint8 per la FID
        fake_uint8 = (fake_images * 255).clamp(0, 255).to(torch.uint8)

        return fake_uint8

# Carichiamo le immagini reali
for X_batch, _ in test_loader:
    real_uint8 = (X_batch * 255).clamp(0, 255).to(torch.uint8).to(device)
    fid_metric.update(real_uint8, real=True)

# Generiamo e carichiamo le immagini fake
num_totale_immagini = len(test_loader.dataset)
batch_size_gen = best_config['batch_size']

for _ in range(0, num_totale_immagini, batch_size_gen):
    # Generiamo un batch di galassie sintetiche
    fake_uint8 = genera_per_fid(final_model, batch_size_gen, best_config['latent_dim'], device)

    # Carichiamo le immagini generate
    fid_metric.update(fake_uint8, real=False)

# Calcolo finale FID
fid_generazione = fid_metric.compute().item()
print(f"FID (Generazione Pura): {fid_generazione:.2f}")

print("\n=== RISULTATI TEST ===")
print(f"Configurazione usata: {best_config}")
print(f"SSIM Finale: {final_ssim_score:.4f} (più vicino a 1 = migliore)")
print(f"FID Finale:  {fid_generazione:.2f} (più basso = migliore)")
print("======================\n")


## 9. Confronto tra immagini random

def visualizza_confronto(model, test_loader):
    model.eval()
    with torch.no_grad():
        # Trasformiamo il loader in una lista di batch per poterne scegliere uno a caso
        batch_totali = list(test_loader)
        batch_scelto, _ = random.choice(batch_totali) # Scegliamo un batch random

        # Portiamo il batch sul device
        batch_scelto = batch_scelto.to(device)

        # Generiamo le ricostruzioni per tutto il batch
        recon_batch, _, _ = model(batch_scelto)

        # Scegliamo 8 indici casuali all'interno del batch
        batch_size_attuale = batch_scelto.size(0)
        indici = random.sample(range(batch_size_attuale), min(8, batch_size_attuale))

        fig, axes = plt.subplots(2, 8, figsize=(15, 4))
        for i, idx in enumerate(indici):
            # Originali
            img_originale = batch_scelto[idx].cpu().permute(1, 2, 0)
            axes[0, i].imshow(img_originale)
            axes[0, i].set_title("Originali")
            axes[0, i].axis('off')

            # Ricostruzioni
            img_ricostruita = recon_batch[idx].cpu().permute(1, 2, 0)
            axes[1, i].imshow(img_ricostruita)
            axes[1, i].set_title("Ricostruzioni")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

visualizza_confronto(final_model, test_loader)
