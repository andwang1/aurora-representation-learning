import pickle as pk

with open("data.txt", "r") as f:
    inputs = [[float(i.strip()) for i in line.split(",")] for line in f.readlines()[::3]]
    f.seek(0, 0)
    trajectories = [[float(i.strip()) for i in line.split(",")] for line in f.readlines()[1::3]]
    f.seek(0, 0)
    # skip the empty lines that occur when there are no impacts at all
    impacts = [[float(i.strip()) for i in line.split(",")] for line in f.readlines()[2::3] if line != "\n"]

with open("data.pk", "wb") as f:
    pk.dump({"inputs": inputs, "trajectories": trajectories, "impacts": impacts}, f)