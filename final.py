import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import matplotlib.pyplot as plt
import datetime

# Daten laden und löschen konstanter Spalten
daten = np.delete(np.loadtxt('data.csv', delimiter=','), [6, 8], 1)

anz_input = 7
anz_output = daten.shape[1] - anz_input

output_every_k = 1
inference_points = 20

epochs = 30
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
    data = daten[:int(daten.shape[0] * reduction), :]
    features = data[:, :anz_input]
    outcome = data[:, anz_input:]

    # Skalierung der Daten mit MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    outcome = scaler.fit_transform(outcome)

    features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome,
                                                                                  test_size=test_size, random_state=42)

    train_data = TensorDataset(torch.from_numpy(features_train).float(), torch.from_numpy(outcome_train).float())
    test_data = TensorDataset(torch.from_numpy(features_test).float(), torch.from_numpy(outcome_test).float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    idx = np.random.choice(np.arange(len(features_test)), inference_points, replace=False)
    test_in = features_test[idx]
    test_out = outcome_test[idx]

    return train_loader, test_loader, test_in, test_out


# Training des Modells
def train(train_loader, test_loader):
    train_losses = []
    test_losses = []
    best_model_loss = 1e10
    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma, verbose=False)
    last_time = datetime.datetime.now()
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
        if (epoch + 1) % output_every_k == 0:
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
            now = last_time
            last_time = datetime.datetime.now()
            timediff = last_time - now
            seconds = timediff.total_seconds()
            remainig_seconds = seconds * (epochs - epoch) / output_every_k

            if avg_test_loss < best_model_loss:
                best_model_loss = avg_test_loss
                torch.save(model, 'model_save.tar')

            print(
                f'Epoch: {epoch + 1}/{epochs}, Train Loss: {avg_train_loss},'
                f' Test Loss: {avg_test_loss}, Learning Rate: {optimizer.param_groups[0]["lr"]},'
                f' Approx. time left: {int(remainig_seconds / 60)} min')
            model.train()
    return train_losses, test_losses


def plotte_krasse_sachen(train_loss, test_loss, model, test_input, gold):
    model.eval()
    inputs_torch = torch.from_numpy(test_input).float()
    with torch.no_grad():
        predictions = model(inputs_torch)
    prediction = predictions.numpy()

    zeilen, spalten = gold.shape

    ax = plt.gca()
    ax.set_ylim([0, 10 * min([min(train_loss), min(test_loss)])])
    plt.plot(train_loss, '-b')
    plt.plot(test_loss, '-g')

    fig, axs = plt.subplots(spalten, figsize=(6, 20))
    for i in range(spalten):
        # Zeichnen Sie die Punkte für jede Zeile in der i-ten Spalte
        axs[i].scatter(range(zeilen), gold[:, i], color='green', label='gold', s=100)
        axs[i].scatter(range(zeilen), prediction[:, i], color='blue', label='prediction', s=10)
        axs[i].legend()
        axs[i].set_title(f'Label {i + 1}')
    plt.tight_layout()

    plt.show()


train_dl, test_dl, tin, tout = prepare_data(1)
train_l, test_l = train(train_dl, test_dl)
plotte_krasse_sachen(train_l, test_l, torch.load('model_save.tar'), tin, tout)
