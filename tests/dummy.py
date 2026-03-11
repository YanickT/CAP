import os
import re

functions = {
    "a": (lambda x: float(re.findall("a-?\d+.?\d*", x)[0].replace("a", ""))),
    "inc": (lambda x: float(re.findall("inc\d+", x)[0].replace("inc", ""))),
    "kv": (lambda x: float(re.findall("kv\d+.?\d*", x)[0].replace("kv", ""))),
    "h": (lambda x: float(re.findall("h\d+.?\d*", x)[0].replace("h", "")))

}

path = r"C:\Users\yanic\Documents\GitHub\CAP\test"
test_files = [file for file in os.listdir(path) if file[-4:] == ".npz"]

hs = []
as_ = []
incs = []
kvs = []
for file in test_files:
    hs.append(functions["h"](file))
    as_.append(functions["a"](file))
    incs.append(functions["inc"](file))
    kvs.append(functions["kv"](file))


as_ = list(set(as_))
hs = list(set(hs))
incs = list(set(incs))
kvs = list(set(kvs))
hs.sort()
as_.sort()
incs.sort()
kvs.sort()
print(as_, len(as_))
print(incs, len(incs))
print(kvs, len(kvs))
print(hs, len(hs))
