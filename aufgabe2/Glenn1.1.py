import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ExponentialLR

epochs=10
output_every_k =1
init_lr = 0.01
last_lr = init_lr


gamma = np.power((last_lr / init_lr), (1 / epochs))  # Gamma wird so gewählt dass init -> last in Trainingszeit
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #init gpu training
print(device)

# Liest die Daten aus und speichert sie in 2 (hdf5-Dataset)-npArrays X und Y

X = np.load("X.npy")
Y = np.load("Y.npy")


# Erstellen Sie DataLoaders




X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print("split complete")
class MyDataset(Dataset):
    def __init__(self, x_in, y_out):
        self.X = x_in
        self.Y = y_out

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Erstellen Sie Instanzen Ihres Datasets


train_dataset = MyDataset(X_train, Y_train)
test_dataset = MyDataset(X_test, Y_test)
print("dataset complete")
# Erstellen Sie DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("dataloader complete")
class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()
        self.fc = nn.Linear(7, 256)
        self.deconv1 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0), 16, 4, 4)  # reshape to match the shape needed for the first deconvolution layer
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        return x


def create_data_loader(X, Y, batch_size):
    tensor_x = torch.Tensor(X) 
    tensor_y = torch.Tensor(Y)
    dataset = TensorDataset(tensor_x,tensor_y) 
    return DataLoader(dataset, batch_size=batch_size)

# Definieren Sie eine Trainingsmethode
def train(model,train_loader, criterion, optimizer, test_loader):
    train_losses = []
    test_losses = []
    best_model_loss = 1e10
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

        


# Ausführung

print("init complete")
model = DeconvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
train_l,test_l = train(model, train_dataloader, criterion, optimizer, test_dataloader)
