import matplotlib.pyplot as plt
import torch
import random

class visualizer:
    def __init__(self):
        pass
    
    def visualizza_galassie_fake(self, immagini, n=16):
        
        cols = 4  # Fissiamo la larghezza a 4
        rows = 4
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        fig.suptitle("Galassie generate artificialmente", fontsize=16)

        for i, ax in enumerate(axes.flat):
            # Portiamo su CPU 
            img = immagini[i].cpu().permute(1, 2, 0).numpy()

            # Se sono uint8 (0-255), le mostriamo così come sono
            # Se fossero float (0-1), imshow capirebbe lo stesso
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Serve per settare il layout dell'immagine, lascia 3% sotto il 5% sopra e usa tutto lo spazio ai lati
        plt.show()

    def visualizza_confronto(self, model, test_loader, device, n_esempi=16):
        model.eval()
        with torch.no_grad():
            batch_totali = list(test_loader)
            # Cerchiamo solo i batch che hanno almeno 'n_esempi' immagini
            # b[0] è il tensore delle immagini X, b[0].size(0) è la dimensione del batch
            batches_validi = [b for b in batch_totali if b[0].size(0) >= n_esempi]

            if len(batches_validi) > 0:
                batch_scelto, _ = random.choice(batches_validi)
            else:
                # se tutti i batch sono piccoli
                # Allora prendiamo quello che c'è e avvisiamo
                print(f"ATTENZIONE: Il DataLoader ha batch_size < {n_esempi}. Visualizzo il massimo possibile.")
                batch_scelto, _ = random.choice(batch_totali)
            
            batch_scelto = batch_scelto.to(device)

            # Generiamo le ricostruzioni
            recon_batch, _, _ = model(batch_scelto)

            # Ricalcoliamo n_cols per sicurezza nel caso fossimo finiti nell'else di sopra
            batch_size_attuale = batch_scelto.size(0)
            n_cols = min(16, batch_size_attuale) 
            indici = random.sample(range(batch_size_attuale), n_cols)

            # Creiamo la figura
            fig, axes = plt.subplots(2, n_cols, figsize=(25, 6), squeeze=False) 
            # squeeze=False forza 'axes' a essere sempre una matrice [righe, colonne]
            
            for i, idx in enumerate(indici):
                # Originali (Riga 0)
                img_originale = batch_scelto[idx].cpu().permute(1, 2, 0).numpy()
                axes[0, i].imshow(img_originale.clip(0, 1))
                axes[0, i].axis('off')
                if i == 0: 
                    axes[0, i].set_title("Originali", loc='left', fontsize=14, fontweight='bold')

                # Ricostruzioni (Riga 1)
                img_ricostruita = recon_batch[idx].cpu().permute(1, 2, 0).numpy()
                axes[1, i].imshow(img_ricostruita.clip(0, 1))
                axes[1, i].axis('off')
                if i == 0: 
                    axes[1, i].set_title("Ricostruzioni", loc='left', fontsize=14, fontweight='bold')

            plt.subplots_adjust(wspace=0.05, hspace=0.1) # Riduce lo spazio tra le immagini per farle sembrare una griglia
            plt.show()

    def visualizza_metamorfosi_classi(self, model, X_data, Y_data, device, classe_A=0, classe_B=1, steps=10):
        model.eval()

        # Troviamo gli indici
        indici_A = (Y_data == classe_A).nonzero(as_tuple=True)[0]
        indici_B = (Y_data == classe_B).nonzero(as_tuple=True)[0]

        # CONTROLLO DI SICUREZZA
        if len(indici_A) == 0 or len(indici_B) == 0:
            classi_disponibili = torch.unique(Y_data).tolist()
            print(f"ERRORE: Una delle classi ({classe_A} o {classe_B}) non è presente nel Test Set.")
            print(f"Classi effettivamente presenti in Y_test: {classi_disponibili}")
            return # Esce dalla funzione senza crashare

        # Indice random
        idx_A = indici_A[random.randint(0, len(indici_A) - 1)]
        idx_B = indici_B[random.randint(0, len(indici_B) - 1)]

        # Estraiamo le immagini
        img_start = X_data[idx_A].unsqueeze(0).to(device)
        img_end = X_data[idx_B].unsqueeze(0).to(device)

        with torch.no_grad():
            mu_start, _ = model.encode(img_start)
            mu_end, _ = model.encode(img_end)

            alphas = torch.linspace(0, 1, steps).to(device)
            z_interp = torch.cat([(1 - a) * mu_start + a * mu_end for a in alphas])
            immagini_generate = model.decode(z_interp).cpu()

        # Visualizzazione
        fig, axes = plt.subplots(1, steps, figsize=(20, 4))
        fig.suptitle(f"Metamorfosi: Classe {classe_A} ➔ Classe {classe_B}", fontsize=16)
        for i in range(steps):
            img = immagini_generate[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img.clip(0, 1))
            axes[i].axis('off')
        plt.show()