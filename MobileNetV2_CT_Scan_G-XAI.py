# Import required libraries
import torch
from torchvision import models, transforms, datasets
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
from captum.attr import FeatureAblation

# Define the CT MobileNetV2 class
class CT_MobileNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super(CT_MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# Creates a dataLoader for the full dataset with specified batch size of 32
def create_full_dataloader(dataset, batch_size=32):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

# Processes a batch of data for SHAP value computation
def process_shap_batch(e, batch, device):
    batch = batch.to(device)
    batch_shap_values = e.shap_values(batch)
    
    if isinstance(batch_shap_values, list):
        return [sv.cpu().numpy() if isinstance(sv, torch.Tensor) else sv 
                for sv in batch_shap_values]
    
    return batch_shap_values.cpu().numpy() if isinstance(batch_shap_values, torch.Tensor) else batch_shap_values

# Collects background data for SHAP analysis from the dataloader
def collect_background_data(dataloader, background_size, device):
    background_data = []
    background_collected = 0
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            if background_collected >= background_size:
                break
            remaining = background_size - background_collected
            if inputs.size(0) > remaining:
                inputs = inputs[:remaining]
            background_data.append(inputs)
            background_collected += inputs.size(0)
    
    return torch.cat(background_data).to(device)

# Computes SHAP values for the entire dataset using the saved MobileNetV2 weights 
def compute_shap_values(model, dataloader, device, class_names, output_dir='./SHAP_Plots_CT_Scan'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    background = collect_background_data(dataloader, background_size=50, device=device)
    
    e = shap.GradientExplainer(model, background)
    
    all_inputs = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Collecting data"):
            all_inputs.append(inputs)
    
    all_inputs = torch.cat(all_inputs)
    
    batch_size = 32
    n_samples = len(all_inputs)
    all_shap_values = []
    
    for i in tqdm(range(0, n_samples, batch_size), desc="Computing SHAP values"):
        batch_end = min(i + batch_size, n_samples)
        batch = all_inputs[i:batch_end]
        
        batch_shap_values = process_shap_batch(e, batch, device)
        all_shap_values.append(batch_shap_values)
        
        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    if isinstance(all_shap_values[0], list):
        shap_values = [
            np.concatenate([batch[i] for batch in all_shap_values], axis=0)
            for i in range(len(all_shap_values[0]))
        ]
    else:
        shap_values = np.concatenate(all_shap_values, axis=0)
    
    processed_shap_values = ([sv.mean(axis=(2, 3)).mean(axis=2) for sv in shap_values] 
                           if isinstance(shap_values, list) 
                           else shap_values.mean(axis=(2, 3)).mean(axis=2))
    
    X = all_inputs.cpu().numpy().mean(axis=(2, 3))
    
    num_features = X.shape[1]
    feature_names = [f'CT Scan Feature {chr(65+i)}' for i in range(num_features)]  # Using A, B, C
    
    generate_shap_plots(processed_shap_values, X, feature_names, class_names, output_dir)
    
    return processed_shap_values, X, feature_names

# Generates SHAP visualization plots for the computed SHAP values
def generate_shap_plots(processed_shap_values, X, feature_names, class_names, output_dir):
    
    plt.figure(figsize=(20, 15))
    shap.summary_plot(
        processed_shap_values,
        X,
        feature_names=feature_names,
        class_names=class_names,
        show=False,
        max_display=len(feature_names),
        plot_type="violin"
    )
    
    plt.title("Impact of CT Scan Features on Lung Cancer Classification", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'SHAP_CT_Scan.png'), bbox_inches='tight', dpi=300)
    plt.close()

# Generates Partial Dependence Plots (PDP) using Captum for feature analysis
def generate_pdp_captum(model, X, feature_names, device, class_names, output_dir='./PDP_Plots_CT_Scan'):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    model.to(device)

    X_tensor = torch.from_numpy(X).float()
    X_reshaped = X_tensor.reshape(-1, 3, 1, 1)
    X_reshaped = X_reshaped.repeat(1, 1, 128, 128)

    for class_idx, class_name in enumerate(class_names):
        ablator = FeatureAblation(model)
        feature_attributions = []

        for feature_idx in range(X.shape[1]):
            input_tensor = X_reshaped.clone().to(device)
            feature_mask = torch.zeros_like(input_tensor, dtype=torch.long)
            feature_mask[:, feature_idx] = 1

            attributions = ablator.attribute(
                inputs=input_tensor,
                target=class_idx,
                feature_mask=feature_mask,
                perturbation_type='gaussian_noise',
                n_samples=10
            )
            feature_attributions.append(attributions.mean().item())

        # Sort features by importance and assign colors
        importance_indices = np.argsort(feature_attributions)
        num_features = len(feature_attributions)
        
        # Divide features into three groups
        low_importance = importance_indices[:num_features//3]
        mid_importance = importance_indices[num_features//3:2*num_features//3]
        high_importance = importance_indices[2*num_features//3:]
        
        # Create color map
        colors = ['#FF4444'] * num_features  # Red
        for idx in mid_importance:
            colors[idx] = '#FFCC00'  # Yellow
        for idx in high_importance:
            colors[idx] = '#4CAF50'  # Green

        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(feature_names)), feature_attributions, align='center')
        

        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Feature Importance')
        plt.title(f'Top 3 Most Important Features in CT-Scan Diagnosis for {class_name}\n' + 
                     'Green: High Impact | Yellow: Moderate Impact | Red: Low Impact', 
                     fontsize=14, pad=20)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'PDP_{class_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    
    data_dir = './CT_scan_processed_128x128'
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading dataset...")
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Create full dataset loader
    full_loader = create_full_dataloader(dataset)
    
    # Class names go from 0, 1, 2, 3
    class_names = ['Adenocarcinoma', 'Large_Cell_Carcinoma', 'Normal', 'Squamous_Cell_Carcinoma']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CT_MobileNetV2()
    model.to(device)
    
    weights_path = './MobileNetV2_CT_Scan/MobileNetV2_fold_1_best.pth'
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    
    adjusted_state_dict = { # Adjust state_dict keys to prevent model loading failure
        f"model.{key}" if not key.startswith('classifier.1.1') else key.replace('classifier.1.1', 'model.classifier.1'): value 
        for key, value in state_dict.items() 
    }
    
    model.load_state_dict(adjusted_state_dict)
    model.eval()
    
    
    shap_values, X, feature_names = compute_shap_values(
        model, full_loader, device, class_names
    )
    
    if shap_values is not None:
        generate_pdp_captum(
            model, X, feature_names, device, class_names
        )
    
    print("XAI FINISHED")

