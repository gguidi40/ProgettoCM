import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from Shape import ShapeDataset
from Model import ConvVAE
from Loss import LossCalculator

class Trainer:
    def __init__(self):
        pass
    
    def train_evaluate_model(self, config, img_size, epochs, X_train, Y_train, X_val, Y_val, device):
        print(f"Training con configurazione: {config}")

        train_loader = DataLoader(ShapeDataset(X_train, Y_train), batch_size=config['batch_size'], shuffle=True)
        val_loader   = DataLoader(ShapeDataset(X_val, Y_val), batch_size=config['batch_size'])

        current_beta = config.get('beta')
        print(current_beta)
        # Inizializza il modello e spostalo sul DEVICE passato come argomento
        model = ConvVAE(
            img_size=img_size,
            base_channels=config['base_channels'],
            latent_dim=config['latent_dim'],
            num_layers=config['num_layers']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Istanziamo la Loss passando il device
        loss_calc = LossCalculator(device)

        best_val_loss = float('inf')
        
        # Recuperiamo path corrente per salvare i pesi
        cartella_attuale = r"C:\Users\franc\OneDrive\Desktop\Università\CM\ProgettoCM"

        # Ciclo di training
        ssim_list = []
        KLD_list = []
        avg_tr_loss_list = []
        avg_val_loss_list = []

        # Warm-up di beta
        annealing_epochs = 10 
        target_beta = config.get('beta', 0.05)
        start_beta = 0.0001 

        for epoch in range(epochs):
            if epoch < annealing_epochs:
                current_beta = start_beta + (target_beta - start_beta) * (epoch / annealing_epochs)
            else:
                current_beta = target_beta

            model.train()
            tr_loss = 0
            ssim_count = 0
            KLD_count = 0

            for X_batch, _ in train_loader:
                # Sposta i dati sul device
                X_batch = X_batch.to(device)
                
                optimizer.zero_grad()              # Elimina i gradienti dell'errore precedente
                recon, mu, logvar = model(X_batch) # L'immagine passa attraverso il modello
                batch_loss, loss_ssim, KLD = loss_calc.vae_loss_ssim(recon, X_batch, mu, logvar, beta=current_beta) # Calcola la loss usando l'istanza loss_calc
                ssim_count += loss_ssim.item()
                KLD_count += KLD.item()
                batch_loss.backward()
                optimizer.step()
                tr_loss += batch_loss.item()

            avg_tr_loss = tr_loss / len(train_loader)
            avg_tr_loss_list.append(avg_tr_loss)
            avg_ssim_count = ssim_count / len(train_loader)
            avg_KLD_count = KLD_count / len(train_loader)
            ssim_list.append(avg_ssim_count) 
            KLD_list.append(avg_KLD_count) 

            # Validation
            model.eval()
            v_loss = 0
            with torch.no_grad():  # Non calcoliamo il gradiente nella validation
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(device)
                    recon_x, mu, logvar = model(X_batch)
                    
                    val_batch_loss, _, _ = loss_calc.vae_loss_ssim(recon_x, X_batch, mu, logvar, beta=current_beta)
                    v_loss += val_batch_loss.item()

            avg_val_loss = v_loss / len(val_loader)  # Validation loss divisa per la lunghezza di un singolo batch
            avg_val_loss_list.append(avg_val_loss)

            # Salviamo il modello
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                nome_file_pesi = "migliori_pesi_galassie.pth"
                percorso_completo = os.path.join(cartella_attuale, nome_file_pesi)
                torch.save(model.state_dict(), percorso_completo)


            print(f"Ep {epoch+1}, Tr loss: {avg_tr_loss:.2f} | Val Loss: {avg_val_loss:.2f} | Beta attuale: {current_beta:.5f}")

        return best_val_loss, ssim_list, KLD_list, avg_tr_loss_list, avg_val_loss_list