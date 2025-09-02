# torch_mnist_ddp.py - PyTorch Distributed Data Parallel MNIST training
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import socket

def init_distributed():
    """Initialize distributed training"""
    # Initialize the process group
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    
    # Set the device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    return device

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Rank {dist.get_rank()}: Train Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Calculate average metrics
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total_samples
    
    print(f'Rank {dist.get_rank()}: Train Epoch {epoch} - Average Loss: {avg_loss:.6f}, '
          f'Accuracy: {accuracy:.2f}%')

def test_epoch(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
    
    # All-reduce for distributed metrics
    test_loss_tensor = torch.tensor(test_loss).to(device)
    correct_tensor = torch.tensor(correct).to(device)
    total_tensor = torch.tensor(total_samples).to(device)
    
    dist.all_reduce(test_loss_tensor)
    dist.all_reduce(correct_tensor)
    dist.all_reduce(total_tensor)
    
    if dist.get_rank() == 0:
        avg_loss = test_loss_tensor.item() / total_tensor.item()
        accuracy = 100. * correct_tensor.item() / total_tensor.item()
        print(f'Test Epoch {epoch} - Average Loss: {avg_loss:.6f}, '
              f'Accuracy: {accuracy:.2f}%')

def main():
    # Print system info
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f'Starting training on rank {rank} of {world_size}')
    print(f'Hostname: {socket.gethostname()}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
    
    # Initialize distributed training
    device = init_distributed()
    
    print(f'Rank {dist.get_rank()}: Using device: {device}')
    
    # Create datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Use a temp directory that should exist
    data_dir = '/tmp/mnist_data'
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), 
                                     rank=dist.get_rank(), shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), 
                                    rank=dist.get_rank(), shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler)
    
    # Create model and move to device
    model = Net().to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f'Rank {dist.get_rank()}: Starting training for 3 epochs...')
    for epoch in range(1, 4):  # Train for 3 epochs
        train_sampler.set_epoch(epoch)  # Ensure different shuffling per epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        test_epoch(model, device, test_loader, epoch)
    
    print(f'Rank {dist.get_rank()}: Training completed!')
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
