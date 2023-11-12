import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

# Daten laden
data = np.loadtxt('XFEL_KW0_Results_2.csv', delimiter=',')
data = data[:int(data.shape[0]/100), :]  # weniger Daten damit es schneller ist, nur provisorisch!!!
# Aufteilen in Features und Outcome
features = data[:, :8]
outcome = data[:, 8:]

# Normalisieren der Daten
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Aufteilen in Train und Test-Daten
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, test_size=0.2, random_state=42)

# Konvertieren in PyTorch Tensoren und erstellen von DataLoadern
train_data = TensorDataset(torch.from_numpy(features_train).float(), torch.from_numpy(outcome_train).float())
test_data = TensorDataset(torch.from_numpy(features_test).float(), torch.from_numpy(outcome_test).float())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Definieren des neuronalen Netzwerks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        return self.fc(x)

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training des Modells
for epoch in range(10000):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Ausgabe des Losses alle 10 Epochen
    if epoch % 10 == 0:
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        print(f'Epoch: {epoch}, Train Loss: {loss.item()}, Test Loss: {test_loss/len(test_loader)}')
        model.train()

