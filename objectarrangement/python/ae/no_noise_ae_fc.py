import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
import pickle as pk
from ae import VAE_FC
from dataset import ActionTrajectory

# Read the data from pickle
with open("data5.pk", "rb") as f:
    data = pk.load(f)
    inputs = data["inputs"]
    trajectories = data["trajectories"]
    impacts = data["impacts"]

# Hyper Parameters
VERBOSE = False
FILE_LOG = True
NUM_BATCH_TO_FILE = 1
FULL_LOSS = True

batch_size = 64
learning_rate = 1e-3
num_epochs = 3000

beta = 1

input_dim = len(inputs[0])
output_dim = len(trajectories[0])
en_hid_dim1 = 10
latent_dim = 2
de_hid_dim1 = 30
de_hid_dim2 = 50

# Create train and test sets
train_pct = 0.75
val_pct = 0.15
test_pct = 0.1

# Device selection
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

# Create datasets
train_len = int(train_pct * len(inputs))
val_len = int(val_pct * len(inputs))
test_len = int(test_pct * len(inputs))

train_len = int(train_pct * 2000)
val_len = int(val_pct * 2000)
test_len = int(test_pct * 2000)
trajectory_dataset = ActionTrajectory(inputs[:2000], trajectories[:2000])

# trajectory_dataset = ActionTrajectory(inputs, trajectories)
traj_train, traj_val, traj_test = torch.utils.data.random_split(trajectory_dataset, [train_len, val_len, test_len])
loader_train = torch.utils.data.DataLoader(traj_train, batch_size=batch_size, shuffle=True)
loader_val = torch.utils.data.DataLoader(traj_val, batch_size=batch_size, shuffle=False)
loader_test = torch.utils.data.DataLoader(traj_test, batch_size=batch_size, shuffle=False)

num_train_samples = len(loader_train.dataset)
num_val_samples = len(loader_val.dataset)
num_test_samples = len(loader_test.dataset)

# Create model
model = VAE_FC(input_dim, en_hid_dim1, latent_dim, de_hid_dim1, de_hid_dim2, output_dim, beta, VERBOSE, device).to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Track losses for plotting
train_loss_per_epoch = []
L2_train_loss_per_epoch = []
KL_train_loss_per_epoch = []
out_var_train_per_epoch = []

val_loss_per_epoch = []
L2_val_loss_per_epoch = []
KL_val_loss_per_epoch = []
out_var_val_per_epoch = []

for epoch in range(num_epochs):
    # Tracking
    train_loss = 0
    L2_train_loss_epoch = 0
    KL_train_loss_epoch = 0
    out_var_train_epoch = 0

    # Training
    model.train()
    for batch_idx, (actions, label_trajectories) in enumerate(loader_train):
        actions = actions.to(device)
        optimizer.zero_grad()
        pred_trajectories, out_var, en_mu, en_logvar = model(actions)

        # Losses summed over the batch
        total_loss, KL_loss, L2_loss = model.loss_function(en_mu, en_logvar, out_var, pred_trajectories, label_trajectories.to(device), FULL_LOSS)

        # Recording losses summed over batch
        train_loss += total_loss.item()
        L2_train_loss_epoch += L2_loss.item()
        KL_train_loss_epoch += KL_loss.item()
        out_var_train_epoch += torch.sum(out_var).item()

        # Averaging total loss over batch size
        total_loss /= actions.size(0)
        total_loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch + 1}/{num_epochs}, \tTLoss: {round(train_loss / num_train_samples, 4)}, \tL2: {round(L2_train_loss_epoch / num_train_samples, 4)}, \tVar: {round(out_var_train_epoch / num_train_samples, 4)}")

    train_loss_per_epoch.append(train_loss / num_train_samples)
    L2_train_loss_per_epoch.append(L2_train_loss_epoch / num_train_samples)
    KL_train_loss_per_epoch.append(KL_train_loss_epoch / num_train_samples)
    out_var_train_per_epoch.append(out_var_train_epoch / num_train_samples)

    # Tracking
    val_loss = 0
    L2_val_loss_epoch = 0
    KL_val_loss_epoch = 0
    out_var_val_epoch = 0

    # Validation
    model.eval()
    with torch.no_grad():
        for batch_idx, (actions, label_trajectories) in enumerate(loader_val):
            actions = actions.to(device)
            optimizer.zero_grad()
            pred_trajectories, out_var, en_mu, en_logvar = model(actions)

            # Losses summed over the batch
            total_loss, KL_loss, L2_loss = model.loss_function(en_mu, en_logvar, out_var, pred_trajectories,
                                                               label_trajectories.to(device), FULL_LOSS)

            # Recording aggregate losses for plotting
            val_loss += total_loss.item()
            L2_val_loss_epoch += L2_loss.item()
            KL_val_loss_epoch += KL_loss.item()
            out_var_val_epoch += torch.sum(out_var).item()

        print(f"Epoch: {epoch + 1}/{num_epochs}, \tVLoss: {round(val_loss / num_val_samples, 4)}, \tL2: {round(L2_val_loss_epoch / num_val_samples, 4)}, \tVar: {round(out_var_val_epoch / num_val_samples, 4)}")

        val_loss_per_epoch.append(val_loss / num_val_samples)
        L2_val_loss_per_epoch.append(L2_val_loss_epoch / num_val_samples)
        KL_val_loss_per_epoch.append(KL_val_loss_epoch / num_val_samples)


# Final Testing
# Tracking
test_loss = 0
L2_test_loss_epoch = 0
KL_test_loss_epoch = 0
out_var_test_epoch = 0

# Logging
predictions = np.empty((NUM_BATCH_TO_FILE, batch_size * output_dim))
labels = np.empty((NUM_BATCH_TO_FILE, batch_size * output_dim))

model.eval()
with torch.no_grad():
    for batch_idx, (actions, label_trajectories) in enumerate(loader_test):
        actions = actions.to(device)
        optimizer.zero_grad()
        pred_trajectories, out_var, en_mu, en_logvar = model(actions)

        # Losses summed over the batch
        total_loss, KL_loss, L2_loss = model.loss_function(en_mu, en_logvar, out_var, pred_trajectories,
                                                           label_trajectories.to(device), FULL_LOSS)

        # Recording aggregate losses for plotting
        test_loss += total_loss.item()
        L2_test_loss_epoch += L2_loss.item()
        KL_test_loss_epoch += KL_loss.item()
        out_var_test_epoch += torch.sum(out_var).item()

        # Save for logging to files
        if FILE_LOG and batch_idx < NUM_BATCH_TO_FILE:
            predictions[batch_idx, :] = torch.flatten(pred_trajectories.cpu()).numpy()
            labels[batch_idx, :] = torch.flatten(label_trajectories).numpy()

    print(f"Final Test, \tTLoss: {round(test_loss / num_test_samples, 4)}, \tL2: {round(L2_test_loss_epoch / num_test_samples, 4)}, \tVar: {round(out_var_test_epoch / num_test_samples, 4)}")

if FILE_LOG:
    f_predictions = open("pred.txt", "w")
    f_labels = open("labels.txt", "w")
    for pred, label in zip(predictions, labels):
        f_predictions.write(np.array2string(pred, separator=",") + "\n")
        f_labels.write(np.array2string(label, separator=",") + "\n")

    f_predictions.close()
    f_labels.close()

test_sample = inputs[123]
test_label = trajectories[123]
test_pred = model(torch.tensor(test_sample, device=device))

print("TEST SAMPLE DIFF MAP")
print(test_pred[0].cpu() - torch.tensor(test_label))
print("SQerror")
print(F.mse_loss(test_pred[0].cpu(), torch.tensor(test_label), reduction="sum").item())

# Plot reconstruction and var, original vs random
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10))
ax1.plot(range(len(L2_train_loss_per_epoch)), L2_train_loss_per_epoch, label="Original - Train")
ax1.plot(range(len(L2_val_loss_per_epoch)), L2_val_loss_per_epoch, label="Original - Val")
# ax1.plot(range(len(rolling_means_random)), rolling_means_random, color="red", label="Random")
ax1.set_title("Mean Reconstruction Error")
ax2.plot(range(len(out_var_train_per_epoch)), out_var_train_per_epoch, label="Original - Train")
ax2.plot(range(len(out_var_val_per_epoch)), out_var_val_per_epoch, label="Original - Val")
# ax2.plot(range(len(rolling_var_random)), rolling_var_random, color="red", label="Random")
ax2.set_title("Mean Variance")
plt.legend()
plt.savefig("ae.png")
