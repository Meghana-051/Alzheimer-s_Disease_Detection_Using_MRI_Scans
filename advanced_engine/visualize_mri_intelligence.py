import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from train_and_explain import AlzheimerResNet, GradCAM, BrainMRIDataset

def generate_clinical_dashboard():
    print("[*] Initializing Diagnostic Visualization Engine...")
    
    # 1. Instantiate our upgraded network architecture
    model = AlzheimerResNet(num_classes=4)
    model.eval() # Set model to evaluation mode
    
    # Target the absolute deepest convolutional layer
    target_layer = model.backbone.layer4[-1]
    grad_cam_processor = GradCAM(model, target_layer)
    
    # 2. Fetch a mock MRI slice from our dataset module
    transform_pipeline = transforms.Compose([transforms.ToTensor()])
    dataset = BrainMRIDataset(num_samples=1, transform=transform_pipeline)
    input_tensor, mock_label = dataset[0]
    
    # Add batch dimension required by PyTorch (1, 3, 128, 128)
    input_batch = input_tensor.unsqueeze(0)
    
    # 3. Generate predictions and the corresponding Grad-CAM activation matrix
    logits = model(input_batch)
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    
    # Run backpropagation through the Grad-CAM hook layers
    heatmap = grad_cam_processor.generate_heatmap(input_batch, class_idx=predicted_class_idx)
    
    # 4. Convert original input tensor back to a displayable NumPy image
    original_img = input_tensor.permute(1, 2, 0).numpy()
    original_img = (original_img * 255).astype(np.uint8)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY) # Keep medical look gray
    original_img_3ch = cv2.merge([original_img, original_img, original_img])
    
    # 5. Resize heatmap and recolor into an explicit Jet color map overlay
    heatmap_resized = cv2.resize(heatmap, (128, 128))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Blending the raw MRI canvas with the Grad-CAM overlay map
    diagnostic_overlay = cv2.addWeighted(original_img_3ch, 0.6, heatmap_color, 0.4, 0)
    
    # 6. Plot the side-by-side diagnostic dashboard
    classes_map = {0: "Non-Demented", 1: "Very Mild", 2: "Mild", 3: "Moderate"}
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f"Input Structural MRI Slice\n(Ground Truth: {classes_map[mock_label]})", fontsize=10)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(diagnostic_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Grad-CAM Heatmap Interpretation\n(Model Output Class: {classes_map[predicted_class_idx]})", fontsize=10)
    axes[1].axis('off')
    
    # Save the output visualization directly to the engine workspace directory
    output_path = 'advanced_engine/diagnostic_report.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"[+] Diagnostic report image successfully generated and saved to: {output_path}")

if __name__ == "__main__":
    generate_clinical_dashboard()