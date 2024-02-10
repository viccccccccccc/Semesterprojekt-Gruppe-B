import tkinter as tk
from tkinter import ttk
import numpy as np
import torch.nn as nn
import torch
import joblib as jl
import matplotlib.pyplot as plt



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 256),

        )

    def forward(self, x):
        return self.fc(x)


model = torch.load("64x64/model_best.tar", map_location=torch.device('cpu'))
model = model.cpu()
pca = jl.load("64x64/pca256.pkl")
scaler_y = jl.load("64x64/scaler_yPCA.joblib")
scaler_x = jl.load("64x64/scaler_xPCA.joblib")
def generate(input):
    for i in range(len(input)):
        input[i] = scaler_x[i].transform(input[i].reshape(-1, 1)).flatten()

    torch_data= torch.from_numpy(input)
    with torch.no_grad():
        torch_data = torch_data.float()
        torch_data = torch_data.cpu()
        outputs = model(torch_data)
        outputs = outputs.numpy()

        outputs = scaler_y.inverse_transform(outputs.reshape(-1, 1)).flatten()

        result = pca.inverse_transform(outputs)
        result[result<1]=0
        result = result.reshape(64,64)
        test= np.max(result)-np.min(result)
        if(test<200):
            result = np.zeros((64,64))
        return result

c = 0
r5value = 0

def update(*args):
    global c
    global r5value
    c = (c + 1) % 3

    # Stellen Sie sicher, dass r6 immer kleiner als r0 ist
    if r6.get() >= r1.get():
        r6.set(r1.get())

    # Aktualisieren Sie die Labels mit den aktuellen Werten
    r0_label.config(text=f"r0: {r0.get():.2f}")
    r1_label.config(text=f"r1: {r1.get():.2f}")
    r2_label.config(text=f"r2: {r2.get():.2f}")
    r3_label.config(text=f"r3: {r3.get():.2f}")
    r4_label.config(text=f"r4: {r4.get():.2f}")
    r6_label.config(text=f"r6: {r6.get():.2f}")

    # Generieren Sie ein Bild mit der bereitgestellten Methode
    if c==0 or r5value != int(r5.get()):
        r5value = int(r5.get())
        input = np.array([float(r0.get()), float(r1.get()), float(r2.get()), float(r3.get()), float(r4.get()), float(r5.get()), float(r6.get())])
        result = generate(input)
        plt.imshow(result, cmap="turbo")
        plt.draw()
        plt.pause(0.0)

root = tk.Tk()
root.title("GUI mit Schiebereglern")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

# Erstellen Sie Schieberegler und Labels für die aktuellen Werte
r0 = tk.DoubleVar()
ttk.Label(mainframe, text="r0").grid(column=1, row=1, sticky=tk.W)
r0_label = ttk.Label(mainframe)
r0_label.grid(column=2, row=1, sticky=tk.W)
ttk.Scale(mainframe, from_=70, to=150, length=400, variable=r0, command=update).grid(column=2, row=2, sticky=(tk.W, tk.E))

r1 = tk.DoubleVar()
ttk.Label(mainframe, text="r1").grid(column=1, row=3, sticky=tk.W)
r1_label = ttk.Label(mainframe)
r1_label.grid(column=2, row=3, sticky=tk.W)
ttk.Scale(mainframe, from_=1.5, to=3.0, length=400, variable=r1, command=update).grid(column=2, row=4, sticky=(tk.W, tk.E))

r2 = tk.DoubleVar()
ttk.Label(mainframe, text="r2").grid(column=1, row=5, sticky=tk.W)
r2_label = ttk.Label(mainframe)
r2_label.grid(column=2, row=5, sticky=tk.W)
ttk.Scale(mainframe, from_=300, to=800, length=400, variable=r2, command=update).grid(column=2, row=6, sticky=(tk.W, tk.E))

r3 = tk.DoubleVar()
ttk.Label(mainframe, text="r3").grid(column=1, row=7, sticky=tk.W)
r3_label = ttk.Label(mainframe)
r3_label.grid(column=2, row=7, sticky=tk.W)
ttk.Scale(mainframe, from_=0.0, to=100.0, length=400, variable=r3, command=update).grid(column=2, row=8, sticky=(tk.W, tk.E))

r4 = tk.DoubleVar()
ttk.Label(mainframe, text="r4").grid(column=1, row=9, sticky=tk.W)
r4_label = ttk.Label(mainframe)
r4_label.grid(column=2, row=9, sticky=tk.W)
ttk.Scale(mainframe, from_=0.0, to=100.0, length=400, variable=r4, command=update).grid(column=2, row=10, sticky=(tk.W, tk.E))

# Erstellen Sie Dropdown-Menü für r5
ttk.Label(mainframe, text="r5").grid(column=1, row=11, sticky=tk.W)
r5 = ttk.Combobox(mainframe, values=[3000, 5000, 10000], state="readonly")
r5.grid(column=2, row=11, sticky=(tk.W, tk.E))
r5.bind("<<ComboboxSelected>>", update)

r6 = tk.DoubleVar()
ttk.Label(mainframe, text="r6").grid(column=1, row=13, sticky=tk.W)
r6_label = ttk.Label(mainframe)
r6_label.grid(column=2, row=13, sticky=tk.W)
ttk.Scale(mainframe, from_=0.5, to=3.0, length=400, variable=r6, command=update).grid(column=2, row=14, sticky=(tk.W, tk.E))

# Erstellen Sie ein Label für das Bild
image_label = ttk.Label(mainframe)
image_label.grid(column=3, row=1, rowspan=14)

r0.set(70)
r1.set(1.5)
r2.set(300)
r3.set(0)
r4.set(0)
r5.set("3000")
r6.set(0.5)

plt.imshow(np.zeros((64, 64)), cmap="turbo")
plt.draw()
plt.pause(0.00001)

root.mainloop()

plt.close()
