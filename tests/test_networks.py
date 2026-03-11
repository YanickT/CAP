from datagen import DataGen
from network import SpinNetwork, IncNetwork
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


ROOT = "C:/Users/yanic/Documents/GitHub/CAP/"
path = r"C:\Users\yanic\Documents\GitHub\CAP\test"


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
spin_net.model.load_state_dict(torch.load(f"{ROOT}/Spin_network.net", map_location=device))
inc_net.model.load_state_dict(torch.load(f"{ROOT}/Inc_network.net", map_location=device))

print("Save some test data")
if not os.path.exists(f"{ROOT}/results"):
    os.mkdir(f"{ROOT}/results")

diffs = []

for i, ((spin_img, spin_target), (inc_img, inc_target)) in enumerate(zip(spin_test_loader, inc_test_loader)):
    if i == 0:
        for j, (img, target_spin, target_inc) in enumerate(zip(spin_img, spin_target, inc_target)):
            plt.imshow(img.reshape(128, 128))
            is_spin = target_spin.item()
            is_inc = target_inc.item()

            estimate_spin = spin_net.predict(img)
            estimate_inc = inc_net.predict(img)
            diffs.append((abs(is_spin - estimate_spin), abs(is_inc - estimate_inc)))
            plt.title(
                f"Spin: {is_spin: .2f}  Estimation: {estimate_spin: .2f} | Inc: {is_inc: .2f} Estimation: {estimate_inc: .2f}")
            plt.savefig(f"{ROOT}/results/Test_{j}.png")
    break

spin_diffs, inc_diffs = tuple(zip(*diffs))
spin_diff = sum(spin_diffs) / len(spin_diffs)
inc_diff = sum(inc_diffs) / len(inc_diffs)
print(f"\tMean absolute difference over results:\n\tSpin: {spin_diff}\n\tInc : {inc_diff}")
spin_total_diff = np.array(spin_net.eval(spin_test_loader))
inc_total_diff = np.array(inc_net.eval(inc_test_loader))

with open("results/summery.txt", "w", encoding="utf-8") as doc:
    doc.write(f"""\
    Summery: 
    Spin (results) : {spin_diff}
    Inc  (results) : {inc_diff}
    
    Spin (total)   : {np.mean(spin_total_diff)}±{np.std(spin_total_diff, ddof=1)}
    Inc  (total)   : {np.mean(inc_total_diff)}±{np.std(inc_total_diff, ddof=1)}
    """)
