import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

class LossCalculator:
    def __init__(self, device):
        self.device = device
        # Spostiamo la metrica sul device passato
        self.ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def vae_loss_ssim(self, recon, x, mu, logvar, beta=0.001):
        '''la ssim ha la funzione di separare le
        galassie nello spazio latente;
        la KLD invece raggruppa i cluster di
        galassie simili;
        il termine beta serve come bilanciamento
        tra le due loss
        '''
        ssim_val = self.ssim_module(recon, x)  # ssim_val calcola quanto sono uguali due immagini
        loss_ssim = 1 - ssim_val               # loss_ssim calcola la differenza tra le immagini
        
        # Calcolo KLD
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # La KLD misura la distanza tra la distribuzione generata e quella a media 0 e varianza 1
        
        return loss_ssim + beta * KLD, loss_ssim, KLD