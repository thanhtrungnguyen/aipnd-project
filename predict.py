import torch
from torchvision import models
from torch import nn
import json
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the class for an input image")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to category names JSON file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    return parser.parse_args()

# Function to load the model checkpoint
def load_checkpoint(filepath: str) -> nn.Module:
    # Load the checkpoint
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']

    # Load model architecture
    if arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Adjust the classifier to match checkpoint structure
    if 'classifier.0.weight' in checkpoint['state_dict']:
        # Assume a Sequential classifier was used
        input_size = model.classifier.in_features
        hidden_units = 512  # Adjust this if needed
        output_size = len(checkpoint['class_to_idx'])

        model.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, output_size),
            nn.LogSoftmax(dim=1)
        )

    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


# Function to process an image to be suitable for model input
def process_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)

    # Resize and crop the image
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))

    # Convert image to numpy array and normalize
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    # Convert to tensor
    return torch.from_numpy(np_image).float().unsqueeze(0)


# Function to predict the class of an image using the model
def predict(
        image_tensor: torch.Tensor,
        model: nn.Module,
        topk: int = 5,
        device: torch.device = torch.device('cpu')
) -> Tuple[List[float], List[int]]:
    model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model.forward(image_tensor)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)

    return top_p.squeeze().tolist(), top_class.squeeze().tolist()


# Function to load category to name mapping from a JSON file
def load_category_names(category_names_path: str) -> Dict[str, str]:
    with open(category_names_path, 'r') as f:
        category_names = json.load(f)
    return category_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the class for an input image")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to category names JSON file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    return parser.parse_args()


# Main function for making predictions
def main():
    args = parse_args()

    # Load the model from checkpoint
    model = load_checkpoint(args.checkpoint)

    # Set device based on user preference
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Process the input image
    image_tensor = process_image(args.input)

    # Predict the class (or classes) of the image
    top_p, top_class = predict(image_tensor, model, topk=args.top_k, device=device)

    # Map classes to category names if provided
    if args.category_names:
        cat_to_name = load_category_names(args.category_names)
        top_class_names = [cat_to_name[str(i)] for i in top_class]
    else:
        top_class_names = [str(cls) for cls in top_class]

    # Print the predictions
    print(f"Top {args.top_k} Predictions:")
    for i in range(len(top_class_names)):
        print(f"{i + 1}: {top_class_names[i]} with probability {top_p[i]:.3f}")


if __name__ == "__main__":
    main()
