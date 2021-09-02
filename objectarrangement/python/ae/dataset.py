import torch


class ActionTrajectory(torch.utils.data.Dataset):
    def __init__(self, inputs, trajectories):
        super(ActionTrajectory, self).__init__()
        self.inputs = torch.tensor(inputs)
        self.trajectories = torch.tensor(trajectories)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        inputs, trajectories = self.inputs[idx], self.trajectories[idx]
        return inputs, trajectories

