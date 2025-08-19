import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mpi4py import MPI

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(rank, size, dataset, model, criterion, optimizer, epochs=10):
    # Split dataset into chunks based on rank
    chunk_size = len(dataset) // size
    subset = torch.utils.data.Subset(dataset, range(rank * chunk_size, (rank + 1) * chunk_size))
    dataloader = DataLoader(subset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Gather the loss from all nodes
        local_loss = torch.tensor([loss.item()], dtype=torch.float32)
        global_loss = torch.zeros(1, dtype=torch.float32)
        MPI.COMM_WORLD.Allreduce(local_loss, global_loss, op=MPI.SUM)
        if rank == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {global_loss.item() / size}')

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Generate some random data
    inputs = torch.randn(1000, 10)
    targets = torch.randn(1000, 1)
    dataset = TensorDataset(inputs, targets)

    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train(rank, size, dataset, model, criterion, optimizer)

if __name__ == "__main__":
    main()
