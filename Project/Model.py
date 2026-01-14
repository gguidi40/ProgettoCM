import torch
import torch.nn as nn

class ConvVAE(nn.Module):
    def __init__(self, img_size, base_channels, latent_dim, num_layers):
        super(ConvVAE, self).__init__() # Prende il costruttore della classe nn.Module
        self.img_size = img_size
        self.num_layers = num_layers

        # ENCODER
        enc_modules = [] # Lista contenente i layers
        in_channels = 3  # RGB
        current_channels = base_channels # Filtri del primo strato
        current_res = img_size

        for i in range(num_layers):
            enc_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, current_channels, kernel_size=3, stride=2, padding=1), # Il padding serve ad aggiungere una cornice di pixel
                    nn.BatchNorm2d(current_channels), # Calcola media e varianza dei pixel e riscala i dati
                    nn.LeakyReLU(0.2) # Funzione di attivazione
                )
            )
            in_channels = current_channels # L'input del layer successivo sarà l'output del layer precedente
            current_channels *= 2          # Raddoppiamo i filtri a ogni layer
            current_res //= 2              # La risoluzione si dimezza (stride=2)

        self.encoder = nn.Sequential(*enc_modules)

        # Dimensione finale dell'immagine
        self.flatten_dim = in_channels * current_res * current_res

        # Media e varianza
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # DECODER
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        dec_modules = [] # Lista contenente i layers

        dec_in_channels = in_channels

        for i in range(num_layers):
            out_channels = dec_in_channels // 2 if i < num_layers - 1 else 3 # Si dimezzano i canali ad ogni layer e imponiamo che l'ultimo ne abbia 3

            dec_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dec_in_channels, out_channels,            # Transpose ingrandisce l'immagine
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    # Se non è l'ultimo layer, mettiamo BatchNorm e ReLU
                    nn.BatchNorm2d(out_channels) if i < num_layers - 1 else nn.Identity(), # Nei layers intermedi si ha la normalizzazione dei pixel mentre
                                                                                           # in quello finale vogliamo che non vengano modificati
                    nn.LeakyReLU(0.2) if i < num_layers - 1 else nn.Sigmoid()
                )
            )
            dec_in_channels = out_channels

        self.decoder = nn.Sequential(*dec_modules)
        self.final_res = current_res

    def encode(self, x):
        '''self.encoder fa passare l'input
        attraverso l'encoder;
        torch.flatten trasforma i tensori
        in una "lista" di numeri
        '''
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        '''creiamo una variabile gaussiana di
        media mu e varianza std
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        '''self.decoder_input espande il vettore
        di input;
        h.view trasforma h in un oggetto che
        possa essere letto dalla rete convoluzionale
        '''
        h = self.decoder_input(z)
        h = h.view(-1, self.flatten_dim // (self.final_res**2), self.final_res, self.final_res)
        return self.decoder(h)

    def forward(self, x):
        '''mette insieme le funzioni precedenti
        '''
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar