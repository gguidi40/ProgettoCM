# Generazione di Immagini di Galassie con CVAE 

Progetto universitario che implementa un **Convolutional Varietional Autoencoder (CVAE)** in **PyTorch** per la ricostruzione e generazione di immagini di galassie tramite Deep Learning.

L’obiettivo è apprendere una rappresentazione latente compatta delle immagini astronomiche e utilizzarla per generare nuove immagini simili a quelle del dataset.

Il progetto si basa sul dataset Galaxy10 DECals (https://astronn.readthedocs.io/en/latest/galaxy10.html), che comprende 17.736 immagini 256x256 a colori suddivise in 10 classi morfologiche, tra queste ne sono state scelte tre: Round Smooth, Unbarred Tight Spiral ed Edge-on with Bulge.


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

     
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- h5py
- Scikit-Learn
- Torchmetrics
- os / sys
- shutil
- itertools
- random

---

## Documentazione
[Relazione del Progetto](./Relazione/RelazioneprogettoCM.pdf)

---

## Struttura del Progetto

```text

├── Output_images/          # Immagini di output generate dal modello
├── Plots/                  # Grafici delle curve di loss e metriche
├── Project/                # Codice sorgente del progetto
│   ├── Data_Clipping.py    # Preprocessing e ridimensionamento immagini
│   ├── Generate.py         # Generazione di nuove immagini sintetiche
│   ├── Log.py              # Utility per il logging degli esperimenti
│   ├── Loss.py             # Definizione della Loss function
│   ├── Main.py             # Script principale (Grid Search)
│   ├── Model.py            # Architettura della rete (ConvVAE)
│   ├── Plot.py             # Funzioni per la creazione dei grafici
│   ├── Shape.py            # Dataset loader custom
│   ├── Train.py            # Loop di training e validazione
│   └── Visualize.py        # Visualizzazione qualitativa ricostruzioni
├── Relazione/              # Cartella contente la relazione del progetto
├── data/                   # Dataset immagini (32x32, 64x64)
├── README.md               # Documentazione generale
├── grid_search_convvae.txt # Risultati testuali della Grid Search
└── pesi_finali.pth         # Pesi salvati del modello addestrato
               
