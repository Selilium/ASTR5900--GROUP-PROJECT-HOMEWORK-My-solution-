
"""
Lily Erickson
ASTR 5900 - Computational Physics & Astrophysics
Homework Project: Denoising Autoencoder for Sine Curves

In this script I generate synthetic sine wave data, add noise, and train
a neural network autoencoder to recover the clean signal.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time

# -------------------------------------------------------
# STEP 1: Generate Sine Wave Dataset
# -------------------------------------------------------

print("Generating sine wave dataset...")

N_CURVES = 10000
N_POINTS = 100

t = np.linspace(0, 2*np.pi, N_POINTS)

clean_data = []

for i in range(N_CURVES):

    # Random parameters
    A = np.random.uniform(0.5, 1.5)
    w = np.random.uniform(0.5, 2)
    phi = np.random.uniform(0, 2*np.pi)

    sine = A * np.sin(w*t + phi)

    clean_data.append(sine)

clean_data = np.array(clean_data)

# -------------------------------------------------------
# STEP 2: Add Gaussian Noise
# -------------------------------------------------------

print("Adding Gaussian noise...")

noise_scale = 0.2
noise = np.random.normal(0, noise_scale, clean_data.shape)

noisy_data = clean_data + noise

# -------------------------------------------------------
# STEP 3: Plot Example Signals
# -------------------------------------------------------

plt.figure(figsize=(10,6))

for i in range(4):

    plt.subplot(2,2,i+1)
    plt.plot(t, clean_data[i], label="Clean")
    plt.plot(t, noisy_data[i], label="Noisy", linestyle="dashed")

    plt.title("Example Sine Curve")
    plt.legend()

plt.tight_layout()
plt.savefig("example_noise.png")
plt.show()

# -------------------------------------------------------
# STEP 4: Train / Validation Split
# -------------------------------------------------------

xtrain, xvalid, ytrain, yvalid = train_test_split(
    noisy_data,
    clean_data,
    test_size=0.1
)

print("Training set size:", xtrain.shape)
print("Validation set size:", xvalid.shape)

# Convert to torch tensors

xtrain = torch.tensor(xtrain, dtype=torch.float32)
ytrain = torch.tensor(ytrain, dtype=torch.float32)

xvalid = torch.tensor(xvalid, dtype=torch.float32)
yvalid = torch.tensor(yvalid, dtype=torch.float32)

# -------------------------------------------------------
# STEP 5: Define Autoencoder
# -------------------------------------------------------

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(100,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,100)
        )

    def forward(self,x):

        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        return reconstructed


model = Autoencoder()

# -------------------------------------------------------
# STEP 6: Training Setup
# -------------------------------------------------------

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50

train_losses = []
valid_losses = []

print("Training autoencoder...")

start_time = time.time()

for epoch in range(EPOCHS):

    # Training
    model.train()
    output = model(xtrain)
    loss = loss_function(output, ytrain)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():

        valid_output = model(xvalid)
        valid_loss = loss_function(valid_output, yvalid)

    valid_losses.append(valid_loss.item())

    if epoch % 5 == 0:
        print(f"Epoch {epoch} Train Loss {loss.item():.5f}")

end_time = time.time()

print("Training time:", end_time-start_time)

# -------------------------------------------------------
# STEP 7: Plot Loss Curves
# -------------------------------------------------------

plt.figure()

plt.plot(train_losses,label="Training Loss")
plt.plot(valid_losses,label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("loss_curve.png")
plt.show()

# -------------------------------------------------------
# STEP 8: Reconstruction Example
# -------------------------------------------------------

model.eval()

with torch.no_grad():

    reconstructed = model(xvalid).numpy()

plt.figure(figsize=(10,6))

for i in range(4):

    plt.subplot(2,2,i+1)

    plt.plot(t,yvalid[i],label="Clean")
    plt.plot(t,xvalid[i],label="Noisy",linestyle="dashed")
    plt.plot(t,reconstructed[i],label="Reconstructed")

    plt.legend()

plt.tight_layout()

plt.savefig("reconstruction.png")

plt.show()