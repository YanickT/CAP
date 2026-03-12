from src.datagen import DataGen
from network import SpinNetwork
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

path = "./test"


cat = lambda value, cats: (np.abs(cats - value)).argmin()



print("Create Dataloader")
test_files = [file for file in os.listdir(path) if file[-4:] == ".npz"]
random.shuffle(test_files)
print(f"\t{len(test_files)} test data found")
spin_test_loader = DataGen(test_files, "a", path=path, bs=1)

print("Create Networks")
spin_net = SpinNetwork()

print("Load Networks")
device = torch.device('cpu')
spin_net.model.load_state_dict(torch.load(f"./saves/Spin_network.net", map_location=device))

print("Save some test data")
if not os.path.exists("results"):
    os.mkdir("results")

diffs = []
spin_cats = np.array([round(i / 10 - 0.9, 2) for i in range(19)])
spin_cats_n = len(spin_cats)
inc_cats = np.array(list(range(10, 180, 10)))
inc_cats_n = len(inc_cats)
print(spin_cats, spin_cats_n)
print(inc_cats, inc_cats_n)

spin_cat_n = np.zeros((spin_cats_n,))
spin_con_matrix = np.zeros((spin_cats_n, spin_cats_n))
spins_is = []
ests = []

for i, (spin_img, spin_target) in enumerate(spin_test_loader):
    print(f"\r{i}th batch", end="")
    for j, (img, target_spin) in enumerate(zip(spin_img, spin_target)):
        # spins
        is_spin = round(target_spin.item(), 1)
        if is_spin != -0.5:
            continue

        spins_is.append(is_spin)
        estimate_spin = spin_net.predict(img)
        ests.append(estimate_spin)

print(np.median(ests))
plt.hist(ests, bins=15)
plt.title("Distribution of estimates for true spin $a=-0.5$")
plt.xlabel("Estimated spin")
plt.ylabel("Number of predictions")
plt.show()
