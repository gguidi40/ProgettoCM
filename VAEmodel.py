import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import h5py
import torch.nn.functional as F



# To get the images and labels from file
with h5py.File(r"C:\Users\nicol\Desktop\Universita\ProgettoCM\Galaxy10_DECals.h5", 'r') as F_h5:
    images = np.array(F_h5['images'])
    labels = np.array(F_h5['ans'])

print("Elaborazione immagini in corso...")


images = torch.tensor(images, dtype=torch.float32) / 255.0 
labels = torch.tensor(labels, dtype=torch.float32)

# Pytorch vuole il formato (N, 3, H, W) ma noi lo diamo così (N, H, W, 3)
if images.shape[-1] == 3:
    images = images.permute(0, 3, 1, 2)
    channels = 3

train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
X_train, Y_train, X_test, Y_test = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

#  Selecting the right galaxy classes

target_classes = [2, 6, 9]

mask = np.isin(Y_train, target_classes)
X_train = X_train[mask]
Y_train = Y_train[mask]
mask = np.isin(Y_test, target_classes)
X_test = X_test[mask]
Y_test = Y_test[mask]


def plot_random_samples(x_data, y_data, num_samples=9):
    # 1. Seleziona indici casuali (per non vedere sempre le stesse prime immagini)
    indices = np.random.choice(len(x_data), num_samples, replace=False)
    
    # Crea una griglia 3x3 (o adatta in base a num_samples)
    plt.figure(figsize=(10, 10))
    
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i + 1)
        
        # Prende l'immagine e l'etichetta
        img = x_data[idx]
        label = y_data[idx]

        img_to_show = img.permute(1, 2, 0)
        
        # 2. Gestione visualizzazione:
        # Se l'immagine ha shape (69, 69, 1) o (1, 69, 69), usiamo squeeze() per togliere la dimensione 1
        img_to_show = np.squeeze(img_to_show)
        
        # Mostra l'immagine
        # Usa cmap='gray' se sono in bianco e nero, toglilo se sono a colori
        plt.imshow(img_to_show) 
        
        # 3. Metti l'etichetta come titolo: QUI devi vedere solo 2, 6 o 9
        plt.title(f"Label: {label}")
        plt.axis('off') # Nasconde gli assi per pulizia
    
    plt.tight_layout()
    plt.show()
#plot_random_samples(X_train, Y_train)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(3)  # add channel dim
X_test  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(3)
Y_train = torch.tensor(Y_train, dtype=torch.long) #torch.long serve a settare i valori come interi
Y_test  = torch.tensor(Y_test, dtype=torch.long)



class ShapeDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
batch_size=32
train_loader = DataLoader(ShapeDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(ShapeDataset(X_test, Y_test), batch_size=batch_size)

class NNVAE(nn.Module):
    def __init__(self, img_size=10, hidden_dim=64, latent_dim=20):
        super().__init__()
        self.img_size = img_size
        self.in_dim = img_size * img_size  # flatten image

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.in_dim),
            nn.Sigmoid()  # output in [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        out = self.decoder(z)
        return out.view(-1, 1, self.img_size, self.img_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
def vae_loss(recon, x, mu, logvar):
    # Flatten for BCE
    recon_flat = recon.view(recon.size(0), -1)
    x_flat = x.view(x.size(0), -1)
    BCE = nn.functional.cross_entropy(recon_flat, x_flat, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #print("bce",BCE/2.,"kld",KLD)
    return BCE/2. + KLD

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NNVAE(img_size=16, hidden_dim=348, latent_dim=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 50

losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(X_batch)
        loss = vae_loss(recon, X_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

plt.figure(figsize=(6,4))
plt.plot(range(1, epochs+1), losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("VAE Training Loss")
plt.grid(True)
plt.show()