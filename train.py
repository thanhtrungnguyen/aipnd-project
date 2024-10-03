# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Tuple, Dict, List

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# Argument parsing for flexibility
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a new network on a dataset of images")
    parser.add_argument("data_dir", type=str, help="Directory of dataset")
    parser.add_argument("--save_dir", type=str, default="./checkpoint.pth", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="densenet121",
                        help="Model architecture (e.g., densenet121 or vgg16)")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, nargs="+", default=[512], help="Hidden units for the classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    return parser.parse_args()


# Function to get the device based on user preference and availability
def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        print("GPU is available. Using GPU for computation.")
        return torch.device('cuda')
    else:
        print("Using CPU for computation.")
        return torch.device('cpu')


# Load data using transforms and dataloaders
def load_data(data_dir: str) -> Tuple[Dict[str, datasets.ImageFolder], Dict[str, DataLoader]]:
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    data_transforms: Dict[str, transforms.Compose] = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            norm
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            norm
        ])
    }

    image_datasets: Dict[str, datasets.ImageFolder] = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in
        ['train', 'valid', 'test']
    }
    dataloaders: Dict[str, DataLoader] = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in
        ['train', 'valid', 'test']
    }

    return image_datasets, dataloaders


# Function to build and set up the model
def build_model(arch: str, hidden_units: List[int]) -> nn.Module:
    # Load a pre-trained model
    if arch == "densenet121":
        model = models.densenet121(pretrained=True)
        input_features = model.classifier.in_features
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_features = model.classifier[0].in_features
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Freeze the feature parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units[0]),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units[0], 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    return model


# Prepare the model for training
def prepare_model(arch: str, hidden_units: List[int], learning_rate: float) -> Tuple[nn.Module, nn.NLLLoss, Optimizer]:
    model = build_model(arch, hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return model, criterion, optimizer


# Training function
def train_model(
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        device: torch.device,
        criterion: nn.Module,
        optimizer: Optimizer,
        epochs: int = 5
) -> None:
    model.to(device)
    steps = 0
    print_every = 40

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0.0
                accuracy = 0.0

                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {validation_loss / len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy / len(dataloaders['valid']):.3f}")

                running_loss = 0
                model.train()

def main():
    args = parse_args()

    # Get the computation device
    device = get_device(args.gpu)

    # Load datasets and dataloaders
    image_datasets, dataloaders = load_data(args.data_dir)

    # Prepare the model, criterion, and optimizer
    model, criterion, optimizer = prepare_model(args.arch, args.hidden_units, args.learning_rate)

    # Train the model
    train_model(model, dataloaders, device, criterion, optimizer, args.epochs)

    # Save the model checkpoint
    checkpoint = {
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'epochs': args.epochs
    }

    torch.save(checkpoint, args.save_dir)
    print(f"Model checkpoint saved to {args.save_dir}")

# Main script
if __name__ == "__main__":
    main()
