import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
import scipy.io
from scipy.ndimage import gaussian_filter
import time
from datetime import datetime, timedelta

class ShanghaiTechDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        """
        Args:
            root_dir (string): Directory with all the images and ground truth.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): If True, creates dataset from training set, else from test set.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # Get all image files from the images directory
        self.image_dir = os.path.join(self.root_dir, 'train_data' if train else 'test_data', 'images')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        
        # Ground truth directory
        self.gt_dir = os.path.join(self.root_dir, 'train_data' if train else 'test_data', 'ground-truth')
        
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
        print(f"GT directory: {self.gt_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def create_density_map(self, points, image_shape, sigma=15):
        """Create density map from point annotations"""
        h, w = image_shape[:2]
        density_map = np.zeros((h, w), dtype=np.float32)
        
        if len(points) == 0:
            return density_map
            
        # Add gaussian at each point location
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < w and 0 <= y < h:
                # Create a small gaussian around the point
                y_start = max(0, y - 3*sigma)
                y_end = min(h, y + 3*sigma + 1)
                x_start = max(0, x - 3*sigma)
                x_end = min(w, x + 3*sigma + 1)
                
                for yi in range(y_start, y_end):
                    for xi in range(x_start, x_end):
                        distance_sq = (xi - x)**2 + (yi - y)**2
                        density_map[yi, xi] += np.exp(-distance_sq / (2 * sigma**2))
        
        return density_map
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Load ground truth .mat file
        gt_name = 'GT_' + os.path.splitext(img_name)[0] + '.mat'
        gt_path = os.path.join(self.gt_dir, gt_name)
        
        try:
            gt_data = scipy.io.loadmat(gt_path)
            # The annotation points are usually stored in 'image_info' -> 'location' or similar
            if 'image_info' in gt_data:
                points = gt_data['image_info'][0][0][0][0][0]  # Navigate the nested structure
            elif 'annPoints' in gt_data:
                points = gt_data['annPoints']
            else:
                # Try to find any array-like data
                for key in gt_data.keys():
                    if not key.startswith('__') and isinstance(gt_data[key], np.ndarray):
                        points = gt_data[key]
                        break
                else:
                    points = np.array([])
            
            if len(points.shape) > 1 and points.shape[1] >= 2:
                points = points[:, :2]  # Take only x, y coordinates
            else:
                points = np.array([])
                
        except Exception as e:
            print(f"Error loading {gt_path}: {e}")
            points = np.array([])
        
        # Create density map
        density_map = self.create_density_map(points, (original_size[1], original_size[0]))
        count = len(points) if len(points.shape) > 1 else 0
        
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
            
        # Resize density map to match the transformed image size (224x224)
        if self.transform:
            density_map = Image.fromarray(density_map)
            density_map = density_map.resize((224, 224), Image.BICUBIC)
            density_map = np.array(density_map)
            # Scale the density values proportionally
            scale_factor = (224 * 224) / (original_size[0] * original_size[1])
            density_map = density_map * scale_factor
            
        # Convert density map to tensor
        density_map = torch.from_numpy(density_map).float().unsqueeze(0)  # Add channel dimension
        count = torch.tensor(count, dtype=torch.float32)
        
        return image, density_map, count

def print_training_info():
    """Print comprehensive training background information"""
    print("=" * 80)
    print("🚀 CROWD COUNTING WITH DEEP LEARNING - TRAINING SESSION")
    print("=" * 80)
    print("📊 DATASET: ShanghaiTech Crowd Counting Dataset")
    print("   • Part A: Dense crowds (avg ~500 people per image)")
    print("   • Part B: Sparse crowds (avg ~120 people per image)")
    print("   • Total: ~1,198 training images + 316 test images")
    print("   • Ground Truth: Point annotations converted to density maps")
    print()
    print("🧠 MODEL ARCHITECTURE: ResNet50 + Custom Density Head")
    print("   • Backbone: Pre-trained ResNet50 (ImageNet)")
    print("   • Custom Head: 4-layer CNN for density map regression")
    print("   • Input: 224×224 RGB images")
    print("   • Output: 224×224 density maps")
    print("   • Loss Function: Mean Squared Error (MSE)")
    print()
    print("🎯 TRAINING STRATEGY:")
    print("   • Phase 1 (Epochs 1-5): Train only density head")
    print("   • Phase 2 (Epochs 6+): Fine-tune entire model")
    print("   • Evaluation Metric: Mean Absolute Error (MAE)")
    print("   • Data Augmentation: Random flips + color jitter")
    print()
    print("💻 SYSTEM INFO:")
    if torch.backends.mps.is_available():
        print("   • Accelerator: Apple Silicon GPU (MPS)")
        print("   • Memory: Shared GPU/CPU memory")
    elif torch.cuda.is_available():
        print("   • Accelerator: NVIDIA GPU (CUDA)")
        print(f"   • GPU: {torch.cuda.get_device_name()}")
        print(f"   • VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   • Accelerator: CPU only")
    print(f"   • PyTorch Version: {torch.__version__}")
    print(f"   • Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

def create_data_loaders(batch_size, data_dir):
    """Create training and validation data loaders for ShanghaiTech dataset"""
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Fixed size for ResNet
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets for both parts (A and B)
    train_dataset_a = ShanghaiTechDataset(
        os.path.join(data_dir, 'part_A'),
        transform=train_transform,
        train=True
    )
    train_dataset_b = ShanghaiTechDataset(
        os.path.join(data_dir, 'part_B'),
        transform=train_transform,
        train=True
    )
    
    val_dataset_a = ShanghaiTechDataset(
        os.path.join(data_dir, 'part_A'),
        transform=val_transform,
        train=False
    )
    val_dataset_b = ShanghaiTechDataset(
        os.path.join(data_dir, 'part_B'),
        transform=val_transform,
        train=False
    )
    
    # Combine datasets
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_a, train_dataset_b])
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_a, val_dataset_b])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if torch.backends.mps.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if torch.backends.mps.is_available() else False
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device, epoch_num):
    """Train for one epoch with detailed timing information"""
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    
    total_batches = len(train_loader)
    epoch_start_time = time.time()
    batch_times = []
    
    print(f"🏃‍♂️ Training Epoch {epoch_num} - {total_batches} batches to process")
    print("-" * 60)
    
    for batch_idx, (data, density_maps, counts) in enumerate(train_loader):
        batch_start_time = time.time()
        
        data, density_maps, counts = data.to(device), density_maps.to(device), counts.to(device)
        
        optimizer.zero_grad()
        output = model(data)  # This should output density maps
        
        # Calculate loss
        loss = criterion(output, density_maps)
        
        # Calculate MAE for counts
        pred_counts = torch.sum(output, dim=(2,3))  # Sum over H,W dimensions
        pred_counts = torch.sum(pred_counts, dim=1)  # Sum over channel dimension
        mae = torch.mean(torch.abs(pred_counts - counts))
        
        loss.backward()
        optimizer.step()
        
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        running_loss += loss.item()
        running_mae += mae.item()
        
        # Calculate timing statistics
        avg_batch_time = np.mean(batch_times)
        remaining_batches = total_batches - (batch_idx + 1)
        eta_epoch = remaining_batches * avg_batch_time
        
        # Progress indicator
        progress = (batch_idx + 1) / total_batches * 100
        
        if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
            print(f"📦 Batch {batch_idx+1:3d}/{total_batches} [{progress:5.1f}%] | "
                  f"Loss: {loss.item():.4f} | MAE: {mae.item():.2f} | "
                  f"Time: {batch_time:.2f}s | ETA: {eta_epoch/60:.1f}min")
    
    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / len(train_loader)
    epoch_mae = running_mae / len(train_loader)
    
    print(f"✅ Epoch {epoch_num} Training Complete!")
    print(f"   Total Time: {epoch_time/60:.2f} minutes")
    print(f"   Avg Batch Time: {np.mean(batch_times):.2f}s")
    print(f"   Throughput: {len(train_loader.dataset)/epoch_time:.1f} samples/sec")
    
    return epoch_loss, epoch_mae

def validate(model, val_loader, criterion, device, epoch_num):
    """Validate the model with timing information"""
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    
    val_start_time = time.time()
    total_batches = len(val_loader)
    
    print(f"🔍 Validating Epoch {epoch_num}...")
    
    with torch.no_grad():
        for batch_idx, (data, density_maps, counts) in enumerate(val_loader):
            data, density_maps, counts = data.to(device), density_maps.to(device), counts.to(device)
            output = model(data)
            
            val_loss += criterion(output, density_maps).item()
            
            pred_counts = torch.sum(output, dim=(2,3))  # Sum over H,W dimensions
            pred_counts = torch.sum(pred_counts, dim=1)  # Sum over channel dimension
            val_mae += torch.mean(torch.abs(pred_counts - counts)).item()
            
            # Progress indicator for validation
            if batch_idx % 10 == 0:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"   Validation Progress: {progress:5.1f}% ({batch_idx+1}/{total_batches})")
    
    val_time = time.time() - val_start_time
    val_loss /= len(val_loader)
    val_mae /= len(val_loader)
    
    print(f"✅ Validation Complete! Time: {val_time:.2f}s")
    
    return val_loss, val_mae

class DensityResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet50
        self.base_model = resnet50(weights='IMAGENET1K_V2')
        
        # Remove the original fully connected layer and average pooling
        self.base_model.fc = nn.Identity()
        self.base_model.avgpool = nn.Identity()
        
        # Add custom head for density map prediction
        self.density_head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),  # Output single channel density map
            nn.ReLU()  # Ensure non-negative outputs
        )
        
    def forward(self, x):
        # Get features from ResNet backbone
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        # Pass through density head
        x = self.density_head(x)
        
        # Upsample to match input size (224x224)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return x

def main():
    """Main training function"""
    # Print comprehensive training information
    print_training_info()
    
    # Configuration
    config = {
        'batch_size': 4,  # Reduced for memory efficiency
        'learning_rate': 1e-5,  # Lower learning rate for fine-tuning
        'weight_decay': 1e-4,
        'epochs': 100,
        'data_dir': '/Users/yesvinv/SUMMER_INTERN_PROJ/try2/archive/ShanghaiTech',
        'save_path': 'resnet50_density_model.pth'
    }
    
    print("⚙️  TRAINING CONFIGURATION:")
    print(f"   • Batch Size: {config['batch_size']}")
    print(f"   • Learning Rate: {config['learning_rate']}")
    print(f"   • Weight Decay: {config['weight_decay']}")
    print(f"   • Total Epochs: {config['epochs']}")
    print(f"   • Data Directory: {config['data_dir']}")
    print()
    
    # Device setup - prioritize MPS for Mac GPU acceleration
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('🚀 Using device: MPS (Mac GPU acceleration)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('🚀 Using device: CUDA')
    else:
        device = torch.device('cpu')
        print('⚠️  Using device: CPU (Consider using GPU for faster training)')
    
    training_start_time = time.time()
    
    try:
        # Data loaders
        print("\n📁 Loading datasets...")
        data_load_start = time.time()
        train_loader, val_loader = create_data_loaders(
            batch_size=config['batch_size'], 
            data_dir=config['data_dir']
        )
        data_load_time = time.time() - data_load_start
        print(f"✅ Datasets loaded in {data_load_time:.2f}s")
        
        # Model
        print("\n🧠 Initializing model...")
        model_init_start = time.time()
        model = DensityResNet().to(device)
        model_init_time = time.time() - model_init_start
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model initialized in {model_init_time:.2f}s")
        print(f"   • Total Parameters: {total_params:,}")
        print(f"   • Trainable Parameters: {trainable_params:,}")
        print(f"   • Model Size: ~{total_params * 4 / 1e6:.1f} MB")
        
        # Loss function - MSE for density map regression
        criterion = nn.MSELoss()
        
        # Optimizer - only train the density head first
        optimizer = optim.Adam(
            model.density_head.parameters(),  # Only train the new head initially
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        print(f"\n🎯 Phase 1: Training density head only (Epochs 1-5)")
        print(f"🎯 Phase 2: Fine-tuning entire model (Epochs 6+)")
        print("\n" + "="*80)
        
        # Training loop
        best_val_mae = float('inf')
        epoch_times = []
        
        for epoch in range(config['epochs']):
            epoch_overall_start = time.time()
            
            print(f'\n📅 EPOCH {epoch+1}/{config["epochs"]} - {datetime.now().strftime("%H:%M:%S")}')
            if epoch < 5:
                print("🎯 Phase 1: Training density head only")
            else:
                print("🎯 Phase 2: Fine-tuning entire model")
            
            # Train
            train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
            
            # Validate
            val_loss, val_mae = validate(model, val_loader, criterion, device, epoch+1)
            
            epoch_time = time.time() - epoch_overall_start
            epoch_times.append(epoch_time)
            
            # Calculate ETA
            avg_epoch_time = np.mean(epoch_times)
            remaining_epochs = config['epochs'] - (epoch + 1)
            eta_total = remaining_epochs * avg_epoch_time
            eta_formatted = str(timedelta(seconds=int(eta_total)))
            
            print(f"\n📊 EPOCH {epoch+1} SUMMARY:")
            print(f"   🏃 Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f}")
            print(f"   🔍 Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")
            print(f"   ⏱️  Epoch Time: {epoch_time/60:.2f} min | Avg: {avg_epoch_time/60:.2f} min")
            print(f"   ⏳ ETA: {eta_formatted} ({remaining_epochs} epochs remaining)")
            
            # After 5 epochs, start fine-tuning the entire model
            if epoch == 4:  # 0-indexed, so epoch 4 is the 5th epoch
                optimizer = optim.Adam(
                    model.parameters(),  # Now train all parameters
                    lr=config['learning_rate']/10,  # Lower learning rate
                    weight_decay=config['weight_decay']
                )
                print(f"\n🔄 SWITCHING TO PHASE 2: Fine-tuning entire model with LR={config['learning_rate']/10}")
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                improvement = "🎉 NEW BEST!"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                    'config': config
                }, config['save_path'])
            else:
                improvement = f"📈 Best: {best_val_mae:.2f}"
            
            print(f"   💾 {improvement} Current: {val_mae:.2f}")
            print("=" * 80)
        
        total_training_time = time.time() - training_start_time
        
        print(f"\n🎊 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   • Total Training Time: {str(timedelta(seconds=int(total_training_time)))}")
        print(f"   • Best Validation MAE: {best_val_mae:.2f}")
        print(f"   • Average Epoch Time: {np.mean(epoch_times)/60:.2f} minutes")
        print(f"   • Model saved to: {config['save_path']}")
        print(f"   • Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ ERROR DURING TRAINING: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()