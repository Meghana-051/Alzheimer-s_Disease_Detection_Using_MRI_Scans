import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

# 1. Custom Dataset Wrapper to simulate raw MRI tensor injection
class BrainMRIDataset(Dataset):
    def __init__(self, num_samples=100, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        # Simulating 128x128 single-channel MRI slices
        self.data = np.random.randint(0, 255, (num_samples, 128, 128, 3), dtype=np.uint8)
        self.labels = np.random.randint(0, 4, num_samples) # 4 classes of Dementia

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# 2. Advanced Transfer Learning Architecture (ResNet50 Backbone)
class AlzheimerResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerResNet, self).__init__()
        # Load pre-trained ResNet50 weights
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze initial feature-extraction convolutional blocks
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer with a specialized clinical classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# 3. Explainable AI Implementation: Grad-CAM Engine Layer
# Replace the GradCAM class in advanced_engine/train_and_explain.py with this:

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register clean module-level hooks instead of manual tensor hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # Save the forward pass feature map activations cleanly
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # Capture the raw gradient flow values entering the layer
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        
        # 1. TEMPORARILY UNFREEZE TARGET LAYER: This lets PyTorch build the gradient math graph 
        # for this specific layer during forward/backward passes without modifying model weights.
        for param in self.target_layer.parameters():
            param.requires_grad = True
            
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        class_loss = output[0, class_idx]
        class_loss.backward()
        
        # 2. RE-FREEZE TARGET LAYER: Lock parameters right back down to preserve transfer learning stability
        for param in self.target_layer.parameters():
            param.requires_grad = False
        
        # Safety check to make sure module hooks captured everything cleanly
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradient or Activation module hook execution failed. Confirm model paths.")
        
        # 3. Compute spatial feature importance metrics safely
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2)) # Global Average Pooling over channel gradients
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        heatmap = np.maximum(heatmap, 0) # Apply mathematical ReLU transformation
        heatmap = cv2.resize(heatmap, (128, 128))
        
        # Avoid division-by-zero errors if the image matrix is empty
        denom = heatmap.max() - heatmap.min()
        if denom == 0: 
            denom = 1e-10
        
        heatmap = (heatmap - heatmap.min()) / denom # Min-Max Normalization
        return heatmap