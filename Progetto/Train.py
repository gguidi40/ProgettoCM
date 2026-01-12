import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import shutil
from Shape import ShapeDataset
from Modello import ConvVAE
from Loss import LossCalculator

class Trainer:
    
    def train_evaluate_model(self, config, epochs, X_train, Y_train, X_val, Y_val, device):
        print(f"Training con configurazione: {config}")

        train_loader = DataLoader(ShapeDataset(X_train, Y_train), batch_size=config['batch_size'], shuffle=True)
        val_loader   = DataLoader(ShapeDataset(X_val, Y_val), batch_size=config['batch_size'])

        current_beta = config.get('beta', 0.01)
        img_size = config.get('img_size', 64)
        # Inizializza il modello e spostalo sul DEVICE passato come argomento
        model = ConvVAE(
            img_size=img_size,
            base_channels=config['base_channels'],
            latent_dim=config['latent_dim'],
            num_layers=config['num_layers']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Istanziamo la Loss passando il device!
        loss_calc = LossCalculator(device)

        best_val_loss = float('inf')
        
        # Recuperiamo path corrente per salvare i pesi
        cartella_attuale = r"C:\Users\franc\OneDrive\Desktop\Università\CM\ProgettoCM"

        # Ciclo di training
        for epoch in range(epochs):
            model.train()
            tr_loss = 0
            
            for X_batch, _ in train_loader:
                # Sposta i dati sul device
                X_batch = X_batch.to(device)
                
                optimizer.zero_grad()              # Elimina i gradienti dell'errore precedente
                recon, mu, logvar = model(X_batch) # L'immagine passa attraverso il modello
                batch_loss = loss_calc.vae_loss_ssim(recon, X_batch, mu, logvar, beta=current_beta) # Calcola la loss usando l'istanza loss_calc
                batch_loss.backward()
                optimizer.step()
                
                tr_loss += batch_loss.item()

            # Validation
            model.eval()
            v_loss = 0
            with torch.no_grad():  # Non calcoliamo il gradiente nella validation
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(device)
                    recon_x, mu, logvar = model(X_batch)
                    
                    val_batch_loss = loss_calc.vae_loss_ssim(recon_x, X_batch, mu, logvar, beta=current_beta)
                    v_loss += val_batch_loss.item()

            avg_val_loss = v_loss / len(val_loader)  # Validation loss divisa per la lunghezza di un singolo batch

            # Salvataggio pesi migliori
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                nome_file_pesi = "migliori_pesi_galassie.pth"
                percorso_completo = os.path.join(cartella_attuale, nome_file_pesi)
                torch.save(model.state_dict(), percorso_completo)

            print(f"Ep {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f}")

        return best_val_loss