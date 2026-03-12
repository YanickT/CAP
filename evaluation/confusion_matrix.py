from src.datagen import DataGen
from network import SpinNetwork, IncNetwork
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
inc_test_loader = DataGen(test_files, "inc", path=path, bs=1)

print("Create Networks")
spin_net = SpinNetwork()
inc_net = IncNetwork()

print("Load Networks")
device = torch.device('cpu')
spin_net.model.load_state_dict(torch.load(f"./saves/Spin_network.net", map_location=device))
inc_net.model.load_state_dict(torch.load(f"./saves/Inc_network.net", map_location=device))

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

inc_cat_n = np.zeros((inc_cats_n,))
inc_con_matrix = np.zeros((inc_cats_n, inc_cats_n))

for i, ((spin_img, spin_target), (inc_img, inc_target)) in enumerate(zip(spin_test_loader, inc_test_loader)):
    print(f"\r{i}th batch", end="")
    for j, (img, target_spin, target_inc) in enumerate(zip(spin_img, spin_target, inc_target)):
        # spins
        is_spin = round(target_spin.item(), 1)
        spins_is.append(is_spin)
        estimate_spin = spin_net.predict(img)

        is_cat = cat(is_spin, spin_cats)
        est_cat = cat(estimate_spin, spin_cats)

        spin_cat_n[is_cat] += 1
        spin_con_matrix[est_cat, is_cat] += 1

        # inclination
        is_inc = target_inc.item()
        estimate_inc = inc_net.predict(img)

        is_cat = cat(is_inc, inc_cats)
        est_cat = cat(estimate_inc, inc_cats)

        inc_cat_n[is_cat] += 1
        inc_con_matrix[est_cat, is_cat] += 1

print("\n===========Spins===========")
print(spin_cat_n)
print(f"Control: {sum(spin_cat_n)}")
print(spin_con_matrix)

spin_con_matrix = spin_con_matrix / spin_cat_n[None, :] * 100
plt.imshow(spin_con_matrix, cmap="hot")
plt.title("Confusion matrix for Spins (values in %)")
for (j, i), label in np.ndenumerate(spin_con_matrix):
    plt.text(i, j, int(round(label, 0)), ha='center', va='center')

plt.xticks(ticks=list(range(len(spin_cats))), labels=[str(cat) if i % 2 == 0 else "" for i, cat in enumerate(spin_cats)])
plt.yticks(ticks=list(range(len(spin_cats))), labels=[str(cat) if i % 2 == 0 else "" for i, cat in enumerate(spin_cats)])
plt.xlabel("Spin (true)")
plt.ylabel("Spin (estimate)")
plt.show()


print("\n===========Ink===========")
print(inc_cat_n)
print(f"Control: {sum(inc_cat_n)}")
print(inc_con_matrix)

inc_con_matrix = inc_con_matrix / inc_cat_n[None, :] * 100
plt.imshow(inc_con_matrix, cmap="hot")
plt.title("Confusion matrix for Inclination (values in %)")
for (j, i), label in np.ndenumerate(inc_con_matrix):
    plt.text(i, j, int(round(label, 0)), ha='center', va='center')

plt.xticks(ticks=list(range(len(inc_cats))), labels=[str(cat) if i % 2 == 0 else "" for i, cat in enumerate(inc_cats)])
plt.yticks(ticks=list(range(len(inc_cats))), labels=[str(cat) if i % 2 == 0 else "" for i, cat in enumerate(inc_cats)])
plt.xlabel("Inc (true)")
plt.ylabel("Inc (estimate)")
plt.show()
