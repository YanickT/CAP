from src.datagen import DataGen
from network import SpinNetwork, IncNetwork
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

path = "./test"

print("Create Dataloader")
test_files = [file for file in os.listdir(path) if file[-4:] == ".npz"]
random.shuffle(test_files)
print(f"\t{len(test_files)} test data found")
spin_test_loader = DataGen(test_files, "a", path=path)
inc_test_loader = DataGen(test_files, "inc", path=path)

print("Create Networks")
spin_net = SpinNetwork()
inc_net = IncNetwork()

print("Load Networks")
device = torch.device('cpu')
spin_net.model.load_state_dict(torch.load("./saves/Spin_network.net", map_location=device))
inc_net.model.load_state_dict(torch.load("./saves/Inc_network.net", map_location=device))

spin_diffs = {}  # spin_diffs[spin] = List[diffs]
inc_diffs = {}  # inc_diffs[spin] = List[diffs]

for (spin_img, spin_target), (inc_img, inc_target) in zip(spin_test_loader, inc_test_loader):
    for i in range(spin_img.shape[0]):
        spin_target_ = round(spin_target[i].item(), 2)
        inc_target_ = int(inc_target[i].item())
        spin_guess = spin_net.predict(spin_img[i])
        inc_guess = inc_net.predict(inc_img[i])
        if spin_target_ in spin_diffs:
            spin_diffs[spin_target_].append(abs(spin_target_ - spin_guess))
        else:
            spin_diffs[spin_target_] = [abs(spin_target_ - spin_guess)]

        if inc_target_ in inc_diffs:
            inc_diffs[inc_target_].append(abs(inc_target_ - inc_guess))
        else:
            inc_diffs[inc_target_] = [abs(inc_target_ - inc_guess)]

print(spin_diffs)
print(inc_diffs)

spin_diffs = {spin: (np.mean(np.array(spin_diffs[spin])), np.std(np.array(spin_diffs[spin]), ddof=1)) for spin in
              spin_diffs}
inc_diffs = {inc: (np.mean(np.array(inc_diffs[inc])), np.std(np.array(inc_diffs[inc]), ddof=1)) for inc in inc_diffs}

sorted = list(spin_diffs.items())
sorted.sort(key=lambda x: x[0])
x, y = tuple(zip(*sorted))
y, yerr = tuple(zip(*y))
plt.bar(x, y, yerr=yerr, width=0.1, color="#00000000", edgecolor="#0000FF", capsize=8)
plt.xticks(x[::2])
plt.grid()
plt.ylabel("Mean absolute error")
plt.xlabel("True spin")
plt.title("Mean absolute error over true spin")
plt.show()

fig, ax = plt.subplots()
sorted = list(inc_diffs.items())
sorted.sort(key=lambda x: x[0])
x, y = tuple(zip(*sorted))
y, yerr = tuple(zip(*y))
ax.bar(x, y, yerr=yerr, width=10, color="#00000000", edgecolor="#0000FF", ecolor="#0000AA", capsize=8)
ax2 = ax.twinx()
ax2.bar(x, [y_ / x_ for y_, x_ in zip(y, x)], yerr=[y_ / x_ for y_, x_ in zip(yerr, x)], width=10, color="#00000000",
        edgecolor="#FE9A2E", capsize=8, ecolor="#E98519")
ax.set_xticks(x[::2])
plt.grid()
ax.set_ylabel("Mean absolute error (MAE) [°]", color="#0000FF")
ax2.set_ylabel("$\\frac{MAE}{inclination}$", color="#FE9A2E")
ax.set_xlabel("True inclination")
plt.title("Error over true inclination")
plt.show()
