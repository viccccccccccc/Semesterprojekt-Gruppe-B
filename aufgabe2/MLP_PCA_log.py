import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import h5py
import random
import os
import joblib

### Hyperparamaters ###
batch_size = 256
anteil_test = 0.2
output_size = 256
num_epochs = 500
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
    
class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, true):
        return torch.mean((torch.log1p(pred) - torch.log1p(true)) ** 2)

class H5Reader:
    def __init__(self, hdf5_path):
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        self.active_normalize = False
        self.x_max = []
        self.y_max = []
        self.x_min = []
        self.y_min = []

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            datapoint = self.file[idx]
        else:
            datapoint = self.file[self.key_list[idx]]

        x = datapoint["X"][:7]
        y = (datapoint["Y"][:][:]).flatten()

        ### resize NUR für den verkleinerten 2m datensatz ###
        #if y.shape[0] == 65536:
         #   y = y.reshape(256, 256)[::4, ::4].flatten()

        """x_div = x
        y_div = y
        if self.active_normalize:
            x_div = np.divide(x, self.x_max)
            y_div = np.divide(y, self.y_max)
        return x_div, y_div"""

        if self.active_normalize:
            x = (x - self.x_min) / (self.x_max - self.x_min)
            y = (y - self.y_min) / (self.y_max - self.y_min)
        return x, y

    def normalize(self):
        """x_max, y_max = self.find_max()
        self.active_normalize = True
        for i in range(len(x_max)):
            if x_max[i] == 0:
                x_max[i] = 1
        for j in range(len(y_max)):
            if y_max[j] == 0:
                y_max[j] = 1
        self.x_max = x_max
        self.y_max = y_max"""

        x_max, y_max = self.find_max()
        x_min, y_min = self.find_min()
        self.active_normalize = True

        # Adjust for zeros to avoid division by zero
        self.x_max = np.where(x_max == 0, 1, x_max)
        self.y_max = np.where(y_max == 0, 1, y_max)
        self.x_min = x_min
        self.y_min = y_min

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
        if os.path.exists('max_values/x.npz') and os.path.exists('max_values/y_pca.npz'):  # Adjusted file name
            x_max = np.load('max_values/x.npz')['name1']
            y_max = np.load('max_values/y_pca.npz')['name1']  # Adjusted file name
        else:
            first_entry = self[0]  # Get the first entry to initialize x_max and y_max
            x_max, y_max = first_entry[0], first_entry[1]  # Initialize x_max and y_max
            for i in range(1, len(self)):  # Start from the second entry
                x, y_pca = self[i]  # Get current entry
                x_max = np.maximum(x_max, x)  # Use np.maximum for element-wise max
                y_max = np.maximum(y_max, y_pca)  # Use np.maximum
            np.savez('max_values/x.npz', name1=x_max)
            np.savez('max_values/y_pca.npz', name1=y_max)  # Adjusted file name
        return x_max, y_max
    
    def find_min(self):
        if os.path.exists('max_values/x_min.npz') and os.path.exists('max_values/y_min.npz'):
            x_min = np.load('max_values/x_min.npz')['name1']
            y_min = np.load('max_values/y_min.npz')['name1']
        else:
            first_entry = self[0]
            x_min, y_min = first_entry[0], first_entry[1]
            for i in range(1, len(self)):
                x, y = self[i]
                x_min = np.minimum(x_min, x)
                y_min = np.minimum(y_min, y)
            np.savez('max_values/x_min.npz', name1=x_min)
            np.savez('max_values/y_min.npz', name1=y_min)
        return x_min, y_min
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(7, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
def calculate_accuracy(outputs, labels, threshold=0.05):
    correct = (torch.abs(outputs - labels) <= threshold).float()  
    accuracy = correct.mean().item()
    return accuracy

def load_pca_model(pca_model_path):
        pca_model = joblib.load(pca_model_path)
        return pca_model

def predictions_to_images(predictions, hdf5_original_path, pca_model_path):
    pca_model = load_pca_model(pca_model_path)
    
    with h5py.File(hdf5_original_path, "r") as original_file:
        keys = list(original_file.keys())
        np.random.shuffle(keys)
        
        for i in range(3):  # anzahl bilder
            original_key = keys[i]
            original_grp = original_file[original_key]
            
            predicted_output = predictions[i]
            print("predicted output in der images schleife: ", predicted_output)
            y_reversed = pca_model.inverse_transform(predicted_output)
            
            predicted_image = y_reversed.reshape(256, 256)
            
            original_image = original_grp["Y"][:]
            
            assert original_image.shape == (256, 256), "Original image has incorrect shape!"
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(predicted_image, cmap='viridis')
            axes[0].set_title(f"Predicted Image - Index: {i}")
            axes[0].axis('off')  
            
            axes[1].imshow(original_image, cmap='viridis')
            axes[1].set_title(f"Original Image - Key: {original_key}")
            axes[1].axis('off')  
            
            plt.colorbar(axes[1].imshow(original_image, cmap='viridis'), ax=axes, location='bottom')
            plt.show()

lambda_l1 = 0.001
def train():
    model = MLP()
    model.to(device)
    criterion = MSLELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    train_losses = []
    test_losses = []

    hdf5_path2 = "/vol/tmp/feuforsp/rzp-1_sphere1mm_train_2million.h5"
    pca_model_path = "/vol/tmp/feuforsp/gruppe_b/pca256.pkl"

    print("anfang")
    last_time = datetime.datetime.now()
    run_directory = last_time.strftime("%d-%m-%y_%H-%M-%S")
    os.makedirs(run_directory, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, labels in train_dataloader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            total_loss = loss + lambda_l1 * l1_penalty

            epoch_train_loss += total_loss.item()
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
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                outputs = model(inputs)
                test_loss = criterion(outputs, labels)
                epoch_test_loss += test_loss.item()
                accuracy = calculate_accuracy(outputs, labels)
                total_accuracy += accuracy
                all_predictions.extend(outputs.cpu().numpy())

        all_predictions_array = np.array(all_predictions)
        print("array von allen predictions: ", all_predictions_array)
        print(f"all_predictions_array Shape: {all_predictions_array.shape}")
    
        epoch_test_loss /= len(test_dataloader)
        test_losses.append(epoch_test_loss)
        total_accuracy /= len(test_dataloader)
        scheduler.step(epoch_test_loss)

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


            predictions_to_images(all_predictions_array, hdf5_path2, pca_model_path)

    np.savez(f'{run_directory}/losses.npz', train_losses=train_losses, test_losses=test_losses)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
reader = H5Reader("/vol/tmp/feuforsp/gruppe_b/pca256.h5")
reader.normalize()
train_dataset, test_dataset = reader.split(test_train_split)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=72, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=72, shuffle=False)
print("datensatz geladen und gesplittet!")

train()
