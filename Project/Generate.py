from Model import ConvVAE
from Visualize import visualizer
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cartella_attuale = r"C:\Users\franc\OneDrive\Desktop\Università\CM\ProgettoCM"

# Istanziamo il modello con la configurazione migliore
final_model = ConvVAE(
    img_size=64,
    base_channels=64,
    latent_dim=48,
    num_layers=4
).to(device)

# Caricamento dei pesi
percorso_caricamento = os.path.join(cartella_attuale, "pesi_finali.pth")

final_model.load_state_dict(torch.load(percorso_caricamento, weights_only=True))

print("Pesi caricati correttamente")

# Funzione per generare le galassie.
def image_gen(model, num_immagini, latent_dim, device):
    model.eval()
    with torch.no_grad():
        # Generiamo vettori casuali nello spazio latente
        z = torch.randn(num_immagini, latent_dim).to(device)

        # Passiamo i vettori al decoder
        fake_images = model.decode(z)

        fake_uint8 = (fake_images * 255).clamp(0, 255).to(torch.uint8)

        return fake_uint8

visuals = visualizer()

galassie_sintetiche = image_gen(
    model=final_model,
    latent_dim=48,
    num_immagini=16,
    device=device)

visuals.visualizza_galassie_fake(galassie_sintetiche)