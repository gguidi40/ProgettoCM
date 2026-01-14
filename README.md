# ProgettoCM
Repositorio per il progetto di computing methods 
# Generazione di Immagini di Galassie con CVAE 

Progetto universitario che implementa un **Convolutional Varietional Autoencoder (CVAE)** in **PyTorch** per la ricostruzione e generazione di immagini di galassie tramite Deep Learning.

L’obiettivo è apprendere una rappresentazione latente compatta delle immagini astronomiche e utilizzarla per generare nuove immagini simili a quelle del dataset.

---

##  Descrizione del Progetto

Un **Convolutional Varietional Autoencoder** è una rete neurale composta da:
- **Encoder**: riduce la dimensionalità dell’immagine estraendo le feature principali
- **Decoder**: ricostruisce l’immagine originale a partire dalla rappresentazione latente

In questo progetto il modello viene addestrato su un dataset di immagini di galassie per impararne le strutture morfologiche (bracci, nucleo, forma) e generare immagini coerenti.

---

##  Funzionalità

- Caricamento e preprocessing del dataset
- Definizione dell’architettura Encoder/Decoder
- Addestramento del modello
- Visualizzazione immagini originali vs ricostruite
- Generazione di nuove immagini
- Salvataggio e caricamento del modello addestrato

---

##  Librerie Utilizzate

     
- **PyTorch**
- NumPy
- Matplotlib
- torchvision

---

## Struttura del Progetto



├── data/                   # Dataset delle immagini di galassie
├── Project/               # codice del progetto
├── Plots/                 # Modelli salvati
├── Output_imagis/                # Output e immagini generate
               
