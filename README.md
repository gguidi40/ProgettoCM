# ProgettoCM
Repositorio per il progetto di computing methods 
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

## Struttura del Progetto

```text

├── data/                   # Dataset delle immagini di galassie
├── Project/                # Codice sorgente del progetto
│   ├── Data_Clipping.py    # Modulo di preprocessing per il ridimensionamento delle immagini
│   ├── Generate.py         # Script per la generazione di immagini sintetiche
│   ├── Log.py              # Utility di logging per il salvataggio degli output
│   ├── Loss.py             # Definizione della Loss function
│   ├── Main.py             # Script per l'esecuzione della Grid Search
│   ├── Model.py            # Architettura della rete neurale
│   ├── Plot.py             # Visualizzazione delle metriche e della Loss
│   ├── Shape.py            # Classe Dataset per il caricamento dati in PyTorch
│   ├── Train.py            # Implementazione del training loop e della validazione
│   └── Visualize.py        # Strumenti per la valutazione qualitativa di immagini ricostruite e generate
├── Plots/                  # Modelli salvati (.h5, .pth)
├── Output_images/          # Output e immagini generate
└── README.md
               
