import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import  matplotlib.pyplot as plt
import matplotlib
import datetime
matplotlib.use("Agg")


# Daten laden und löschen konstanter Spalten
daten = np.delete(np.loadtxt('XFEL_KW0_Results_2.csv', delimiter=','), [6, 8], 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #init gpu training
print(device)

anz_input = 7
anz_output = daten.shape[1] - anz_input

output_every_k = 1
inference_points = 20

epochs = 450
batch_size = 128
test_size = 1. / 3

init_lr = 0.001
last_lr = 0.00005
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
    model.to(device)                #move model to gpu
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma, verbose=False)
    last_time = datetime.datetime.now()
    for epoch in range(epochs):
        loss_sum = 0
        for inputs, labels in train_loader:
            inputs,labels =inputs.to(device), labels.to(device)#move data to gpu
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
                    inputs,labels =inputs.to(device), labels.to(device)#move data to gpu
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
                torch.save(model, f'models/model_save_{epochs}.tar')

            print(
                f'Epoch: {epoch + 1}/{epochs}, Train Loss: {format(avg_train_loss,".8f")},'
                f' Test Loss: {format(avg_test_loss,".8f")}, Learning Rate: {format(optimizer.param_groups[0]["lr"],".8f")},'
                f' Approx. time left: {int(remainig_seconds / 60)} min')
            model.train()
            np.savez(f'losses/losses_{epochs}.npz',name1=train_losses,name2=test_losses)
    return train_losses, test_losses


def plotte_krasse_sachen(losses, model, test_input, gold):
    model.eval()
    model= model.to(torch.device('cpu'))        #make sure the model is on the cpu
    inputs_torch = torch.from_numpy(test_input).float()
    with torch.no_grad():
        predictions = model(inputs_torch)
    prediction = predictions.numpy()

    zeilen, spalten = gold.shape

    fig, ax = plt.subplots()
    ax.set_ylim([0, max(losses['name1'])])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f'Loss Graphs ({epochs} Epochs)')
    ax.plot(losses['name1'], '-b',label="Train Losses")
    ax.plot(losses['name2'], '-g',label="Test Losses")
    ax.legend()

    plt.savefig(f'plots/final_losses_{epochs}.png')

    plt.figure(1)
    plt.figure(figsize=(10, 10))
    colors = ['b', 'g', 'r', 'c', 'm']  # Farben für die Labels
    label_names = ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5']  # Namen der Labels


    for j in range(spalten):
        plt.scatter(gold[:, j], prediction[:, j], c=colors[j], s=10, label=label_names[j])

    # Zeichnen Sie die Diagonale x=y
    plt.plot([0, 1], [0, 1], 'k-', alpha=0.75, zorder=0)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Gold')
    plt.ylabel('Prediction')
    plt.title('Scatter plot of Gold vs Prediction')
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.savefig(f'plots/final_inference_{epochs}.png')

    plt.figure(1)
    plt.figure(figsize=(10, 10))
    colors = ['b', 'g', 'r', 'c', 'm']  # Farben für die Labels
    label_names = ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5']  # Namen der Labels


    diff = gold-prediction
    for j in range(spalten):

        plt.scatter(np.full((1,zeilen),j+1), diff[:, j], c=colors[j], s=10, label=label_names[j])


    plt.xlabel('Param')
    plt.ylabel('Gold-Prediction')
    plt.title('Scatter plot of Gold Prediction Difference')
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.savefig(f'plots/final_inference_difference_{epochs}.png')





train_dl, test_dl, tin, tout = prepare_data(1)
#train_l, test_l = train(train_dl, test_dl)
plotte_krasse_sachen(np.load(f'losses/losses_{epochs}.npz'),torch.load(f'models/model_save_{epochs}.tar',map_location = torch.device('cpu')), tin, tout)
