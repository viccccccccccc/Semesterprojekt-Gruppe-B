import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
import matplotlib.pyplot as plt

# Daten laden und löschen konstanter Spalten
daten = np.delete(np.loadtxt('data.csv', delimiter=','), [6, 8], 1)

anz_input = 7
anz_output = daten.shape[1] - anz_input

epochs = 1500
batch_size = 128
test_size = 1. / 3

init_lr = 0.001
last_lr = 0.0001
gamma = np.power((last_lr / init_lr), (1 / epochs))  # Gamma wird so gewählt dass init -> last in Trainingszeit


# Definieren des neuronalen Netzwerks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(anz_input, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, anz_output)
        )

    def forward(self, x):
        return self.fc(x)


def prepare_data(reduction):
    # ggf reduzieren die Datenmenge
    data = daten[:int(daten.shape[0] * reduction), :]
    # Aufteilen in Features und Outcome
    features = data[:, :anz_input]
    outcome = data[:, anz_input:]
    # Normalisieren der Daten
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # Aufteilen in Train und Test-Daten
    features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome,
                                                                                  test_size=test_size, random_state=42)
    # Konvertieren in PyTorch Tensoren und erstellen von DataLoadern
    train_data = TensorDataset(torch.from_numpy(features_train).float(), torch.from_numpy(outcome_train).float())
    test_data = TensorDataset(torch.from_numpy(features_test).float(), torch.from_numpy(outcome_test).float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Training des Modells
def train(train_loader, test_loader):
    train_losses = []
    test_losses = []
    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma, verbose=False)
    for epoch in range(epochs):
        loss_sum = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Ausgabe des Losses alle 10 Epochen
        if (epoch + 1) % 5 == 0:
            test_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    loss_for_print = criterion(outputs, labels)
                    test_loss += loss_for_print.item()
            avg_train_loss = loss_sum / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            print(
                f'Epoch: {epoch + 1}/{epochs}, Train Loss: {avg_train_loss},'
                f' Test Loss: {avg_test_loss}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
            model.train()
    return train_losses, test_losses


def loss_curves(train_loss, test_loss):
    ax = plt.gca()
    ax.set_ylim([0, 0.01])
    plt.plot(train_loss, '-b')
    plt.plot(test_loss, '-g')
    plt.show()


train_dl, test_dl = prepare_data(0.05)
train_l, test_l = train(train_dl, test_dl)
loss_curves(train_l, test_l)
