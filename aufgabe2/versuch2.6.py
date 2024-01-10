import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import h5py
import random
import os

### Hyperparamaters ###
batch_size = 256
anteil_test = 0.2
output_size = 64*64
num_epochs = 50
save_every_k = 2
test_train_split = 1./5
learning_rate = 0.001
#######################



class CustomDataset(Dataset):
    def __init__(self, my_reader, key_list):
        self.reader = my_reader
        self.key_list = key_list

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        return self.reader[self.key_list[idx]]


class H5Reader:
    def __init__(self, hdf5_path):
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        self.active_normalize = False
        self.x_max = []
        self.y_max = []

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            datapoint = self.file[idx]
        else:
            datapoint = self.file[self.key_list[idx]]

        x = datapoint["X"][:7]
        y = (datapoint["Y"][:][:]).flatten()
        x_div = x
        y_div = y
        if self.active_normalize:
            x_div = np.divide(x, self.x_max)
            y_div = np.divide(y, self.y_max)
        return x_div, y_div

    def normalize(self):
        x_max, y_max = self.find_max()
        self.active_normalize = True
        for i in range(len(x_max)):
            if x_max[i] == 0:
                x_max[i] = 1
        for j in range(len(y_max)):
            if y_max[j] == 0:
                y_max[j] = 1
        self.x_max = x_max
        self.y_max = y_max

    def split(self, anteil_test):
        keys = self.key_list.copy()
        random.shuffle(keys)
        split_index = int(len(keys) * anteil_test)
        test = keys[:split_index]
        train = keys[split_index:]

        return CustomDataset(self, train), CustomDataset(self, test)

    def find_max(self):
        if not os.path.exists('max_values'):
            os.makedirs('max_values')
        if os.path.exists('max_values/x.npz') and os.path.exists('max_values/y.npz'):
            x_max = np.load(f'max_values/x.npz')
            y_max = np.load(f'max_values/y.npz')
            x_max = x_max['name1']
            y_max = y_max['name1']
        else:
            x_max, y_max = self[0]
            for i in range(len(self)):
                x, y = self[i]
                x_max = [max(ai, bi) for ai, bi in zip(x_max, x)]
                y_max = [max(ai, bi) for ai, bi in zip(y_max, y)]
            np.savez(f'max_values/x.npz', name1=x_max)
            np.savez(f'max_values/y.npz', name1=y_max)
        return x_max, y_max






"""class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()
        self.fc = nn.Linear(7, 256)  # Fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1) # Deconvolution layer
        self.deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # Deconvolution layer
        self.deconv3 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # Deconvolution layer
        self.deconv4 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # Deconvolution layer

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0), 1, 16, 16)  # reshape to match the shape needed for the first deconvolution layer
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.deconv4(x)
        #for element in x: element.flatten()
        x=x.view(x.size(0),output_size)
        return x"""


class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()
        self.initial_shape = (8, 8)  # Example shape
        self.initial_channels = 1
        self.fc_input_size = self.initial_channels * self.initial_shape[0] * self.initial_shape[1]
        self.fc = nn.Linear(7, self.fc_input_size)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=2, padding=1),    # Output: [batch_size, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # Output: [batch_size, 16, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),    # Output: [batch_size, 8, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),     # Output: [batch_size, 1, 256, 256]
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.initial_channels, self.initial_shape[0], self.initial_shape[1])
        x = self.deconv_layers(x)
        x = x.view(x.size(0), -1)
        return x
    
def calculate_accuracy(outputs, labels, threshold=0.05):
    correct = (torch.abs(outputs - labels) <= threshold).float()  
    accuracy = correct.mean().item()
    return accuracy

def train():
    model = DeconvNet()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.1)
    os.mkdir(run_directory)
    train_losses = []
    test_losses = []

    print("anfang")
    last_time = datetime.datetime.now()
    run_directory = last_time.strftime("%d.%m.%y, %H:%M:%S")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, labels in train_dataloader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            labels = labels.view(-1, 4096)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= len(train_dataloader)
        train_losses.append(epoch_train_loss)

        now = last_time
        last_time = datetime.datetime.now()
        timediff = last_time - now
        minutes = timediff.total_seconds()/60
        remainig_minutes = minutes * (num_epochs - epoch)

        model.eval()
        epoch_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                labels = labels.view(-1, 4096)
                outputs = model(inputs)
                test_loss = criterion(outputs, labels)
                epoch_test_loss += test_loss.item()
                accuracy = calculate_accuracy(outputs, labels)
                total_accuracy += accuracy
        
        epoch_test_loss /= len(test_dataloader)
        test_losses.append(epoch_test_loss)
        total_accuracy /= len(test_dataloader)

        scheduler.step()

        print(f'Epoch {epoch+1} from {num_epochs}, Train Loss: {epoch_train_loss}, Test Loss: {epoch_test_loss}, ')
        print(f"Accuracy after Epoch {epoch + 1}: {total_accuracy * 100:.2f}%, Aprox. Time left: {remainig_minutes} minutes")

        if (epoch + 1) % 10 == 0:  # Every 10 epochs, save the loss graph
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
            plt.plot(range(1, epoch + 2), test_losses, label='Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Loss up to Epoch {epoch + 1}')
            plt.savefig(f'{run_directory}/LossEpoch{epoch + 1}.png')
            plt.close()

    np.savez(f'{run_directory}/losses.npz', train_losses=train_losses, test_losses=test_losses)


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(device)
reader = H5Reader("data.h5")
reader.normalize()
train_dataset, test_dataset = reader.split(test_train_split)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=72, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=72, shuffle=False)
print("datensatz geladen und gesplittet!")

train()
