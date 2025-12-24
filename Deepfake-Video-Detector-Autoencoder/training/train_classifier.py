import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.classifier import DeepfakeClassifier

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_classifier():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['use_gpu'] else 'cpu')
    print(f"Using device: {device}")

    # Paths
    # Ideally should point to split folders: train/real, train/fake
    # Config keys might need adjustment if they only point to 'processed_data'
    # Assuming standard structure: data/train/Real and data/train/Fake
    
    # We'll try to deduce robust paths from config or default to standard kaggle paths if needed
    base_data_path = "data/kaggle_140k/real_vs_fake/real-vs-fake"
    train_dir = os.path.join(base_data_path, "train")
    val_dir = os.path.join(base_data_path, "test") # Use test as val for simplicity or split train
    
    # Check if paths exist
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return

    # Hyperparameters
    batch_size = 32
    num_epochs = 1  # Fast training for now
    learning_rate = 1e-4
    img_size = 224 # Standard for EfficientNet

    # Datasets
    print("Loading Datasets...")
    transform = get_transforms(img_size)
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    print(f"Classes: {train_dataset.classes}") # Should be ['fake', 'real'] usually
    # If classes are ['fake', 'real'], label 0 is fake, 1 is real?
    # We want 1=Fake, 0=Real generally for "Probability of Fake"
    # But ImageFolder sorts alphabetically: 'fake'=0, 'real'=1 maybe?
    # Let's check:
    class_to_idx = train_dataset.class_to_idx
    print(f"Class mapping: {class_to_idx}")
    
    # We need to ensure the target matches our logic.
    # If model outputs p(Fake), then Fake should be 1.
    # We might need a target transform if 'fake' is 0.
    
    fake_label = class_to_idx.get('fake') if 'fake' in class_to_idx else class_to_idx.get('FAKE')
    real_label = class_to_idx.get('real') if 'real' in class_to_idx else class_to_idx.get('REAL')
    
    # Custom collate or transform to ensure Fake=1, Real=0
    # Or just interpret the output: if output is p(class 1) and class 1 is Real, then p(Real).
    # Let's stick thereto: Output is p(Class 1).
    # If Class 1 is Real, then output is p(Real).
    # We want p(Fake).
    # So if Class 1 is Real, p(Fake) = 1 - p(Real).
    # Let's keep it standard training and handle interpretation in inference.
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    print("Initializing Model...")
    model = DeepfakeClassifier(pretrained=True).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    print("Starting Training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/((total/batch_size)+1), 'acc': 100*correct/total})
            
        print(f"Epoch {epoch+1} Train Accuracy: {100*correct/total:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1} Val Accuracy: {100*val_correct/val_total:.2f}%")

    # Save
    save_dir = "models/trained"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "classifier.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Save Class Mapping for inference
    mapping_path = os.path.join(save_dir, "classifier_classes.yaml")
    with open(mapping_path, 'w') as f:
        yaml.dump(class_to_idx, f)
    print(f"Class mapping saved to {mapping_path}")

if __name__ == "__main__":
    train_classifier()
