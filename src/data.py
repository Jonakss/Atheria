import torch.utils.data as data
import torchvision as tv

def get_dataloaders(batch_size):
    """
    Downloads and prepares MNIST dataset, returning training and validation DataLoaders.
    """
    dataset = tv.datasets.MNIST("data", download=True, transform=tv.transforms.ToTensor())
    train_dataset, val_dataset = data.random_split(dataset, [55000, 5000])

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader