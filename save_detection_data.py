import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append('SIBA')
import models
from util import *

if __name__ == '__main__':
    print("Saving datasets for SIBA detection experiments...\n")
    
    # Create directory for detection data
    os.makedirs('detection_data', exist_ok=True)
    
    # Load the trigger
    print("Loading trigger...")
    uap = np.load('save_trigger/uap.npy')
    mask = np.load('save_trigger/mask.npy')
    uap_torch = torch.from_numpy(uap)
    mask_torch = torch.from_numpy(mask)
    
    # ==========================================
    # a) Save Clean Test Data
    # ==========================================
    print("\n[1/3] Saving clean test data...")
    test_dataset = datasets.CIFAR10(root='data', 
                                    train=False, 
                                    transform=transforms.ToTensor(), 
                                    download=True)
    
    # Convert to arrays
    test_images = []
    test_labels = []
    for img, label in test_dataset:
        test_images.append(img.numpy())
        test_labels.append(label)
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    # Save clean test data
    np.save('detection_data/clean_test_images.npy', test_images)
    np.save('detection_data/clean_test_labels.npy', test_labels)
    print(f"✓ Saved clean test data: {test_images.shape} images")
    print(f"  - clean_test_images.npy: shape {test_images.shape}")
    print(f"  - clean_test_labels.npy: shape {test_labels.shape}")
    
    # ==========================================
    # b) Save Triggered Test Data
    # ==========================================
    print("\n[2/3] Saving triggered test data...")
    
    # Apply trigger to all test images
    triggered_images = []
    trigger_pattern = uap_torch * mask_torch.repeat(3, 1, 1)
    
    for img in test_images:
        img_tensor = torch.from_numpy(img)
        triggered_img = img_tensor + trigger_pattern
        triggered_img = torch.clamp(triggered_img, 0, 1)
        triggered_images.append(triggered_img.numpy())
    
    triggered_images = np.array(triggered_images)
    
    # Save triggered test data
    np.save('detection_data/triggered_test_images.npy', triggered_images)
    np.save('detection_data/triggered_test_labels.npy', test_labels)  # Original labels
    np.save('detection_data/trigger_target_class.npy', np.array([0]))  # Target class
    print(f"✓ Saved triggered test data: {triggered_images.shape} images")
    print(f"  - triggered_test_images.npy: shape {triggered_images.shape}")
    print(f"  - triggered_test_labels.npy: original labels shape {test_labels.shape}")
    print(f"  - trigger_target_class.npy: target class = 0")
    
    # ==========================================
    # c) Save Training Data with Poison Labels
    # ==========================================
    print("\n[3/3] Saving training data with poison information...")
    
    train_dataset = datasets.CIFAR10(root='data', 
                                     train=True, 
                                     transform=transforms.ToTensor(), 
                                     download=True)
    
    # Recreate the poisoning process from train_poison_cifar.py
    poison_rate = 0.01  # Same as used in training
    y_target = 0  # Same as used in training
    
    train_images = []
    train_labels = []
    poison_flags = []  # 1 if poisoned, 0 if clean
    
    set_random_seed(2)  # Same seed as in training
    num_samples = len(train_dataset)
    num_poison = int(poison_rate * num_samples)
    
    # Randomly select indices to poison
    all_indices = np.arange(num_samples)
    np.random.shuffle(all_indices)
    poison_indices = set(all_indices[:num_poison])
    
    print(f"Total training samples: {num_samples}")
    print(f"Poison rate: {poison_rate} ({num_poison} samples)")
    
    for idx in range(num_samples):
        img, label = train_dataset[idx]
        
        if idx in poison_indices:
            # Apply trigger
            img_triggered = img + trigger_pattern
            img_triggered = torch.clamp(img_triggered, 0, 1)
            train_images.append(img_triggered.numpy())
            train_labels.append(y_target)  # Changed to target class
            poison_flags.append(1)
        else:
            # Keep clean
            train_images.append(img.numpy())
            train_labels.append(label)
            poison_flags.append(0)
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    poison_flags = np.array(poison_flags)
    
    # Save training data with poison information
    np.save('detection_data/train_images.npy', train_images)
    np.save('detection_data/train_labels.npy', train_labels)
    np.save('detection_data/poison_flags.npy', poison_flags)
    np.save('detection_data/poison_indices.npy', np.array(list(poison_indices)))
    
    print(f"✓ Saved training data with poison information:")
    print(f"  - train_images.npy: shape {train_images.shape}")
    print(f"  - train_labels.npy: shape {train_labels.shape}")
    print(f"  - poison_flags.npy: shape {poison_flags.shape} ({poison_flags.sum()} poisoned samples)")
    print(f"  - poison_indices.npy: {len(poison_indices)} poisoned indices")
    
    # Also save trigger pattern separately for convenience
    np.save('detection_data/trigger_pattern.npy', trigger_pattern.numpy())
    np.save('detection_data/trigger_mask.npy', mask)
    np.save('detection_data/trigger_uap.npy', uap)
    
    print(f"\n✓ Saved trigger components:")
    print(f"  - trigger_pattern.npy: full trigger pattern")
    print(f"  - trigger_mask.npy: trigger mask")
    print(f"  - trigger_uap.npy: UAP component")
    
    # Save metadata
    metadata = {
        'poison_rate': poison_rate,
        'target_class': y_target,
        'num_train_samples': num_samples,
        'num_poison_samples': num_poison,
        'num_test_samples': len(test_dataset),
        'image_shape': list(test_images.shape[1:]),
        'num_classes': 10,
        'dataset': 'CIFAR-10'
    }
    
    import json
    with open('detection_data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✓ Saved metadata to metadata.json")
    
    print("\n" + "="*60)
    print("All detection data saved successfully!")
    print("="*60)
    print("\nSummary of saved files in 'detection_data/':")
    print("\nFor measuring benign accuracy & clean activations:")
    print("  - clean_test_images.npy")
    print("  - clean_test_labels.npy")
    print("\nFor measuring attack success rate & backdoor activations:")
    print("  - triggered_test_images.npy")
    print("  - triggered_test_labels.npy")
    print("  - trigger_target_class.npy")
    print("\nFor input-based detection methods:")
    print("  - train_images.npy")
    print("  - train_labels.npy")
    print("  - poison_flags.npy (ground truth)")
    print("  - poison_indices.npy")
    print("\nTrigger information:")
    print("  - trigger_pattern.npy")
    print("  - trigger_mask.npy")
    print("  - trigger_uap.npy")
    print("\nMetadata:")
    print("  - metadata.json")
