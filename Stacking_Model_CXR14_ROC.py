# Import required libraries
import torch
from torchvision import models, transforms, datasets 
from PIL import Image
import os
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = {
    "ResNet18": "./ResNet18_CXR14/epoch_21_model.pth",
    "ConvNeXtTiny": "./ConvNeXtTiny_CXR14/epoch_9_model.pth",
    "EfficientNetV2S": "./EfficientNetV2S_CXR14/epoch_8_model.pth", 
    "DenseNet121": "./DenseNet121_CXR14/epoch_10_model.pth"
}

meta_model_paths = {
    'LR': "./Hyperparameters_CXR14/LR_Model.joblib",
    'GB': "./Hyperparameters_CXR14/GB_Model.joblib", 
    'RF': "./Hyperparameters_CXR14/RF_Model.joblib",
    'AB': "./Hyperparameters_CXR14/AB_Model.joblib",
    'ET': "./Hyperparameters_CXR14/ET_Model.joblib",
    'LGB': "./Hyperparameters_CXR14/LGB_Model.joblib"
}

with open("./Meta-Ensemble_CXR14/F1_Threshold.json", 'r') as f:
    F1_threshold = json.load(f)['best_threshold']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to load model based on name
def load_model(model_name):
    print(f"Loading model: {model_name}")
    
    if model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif model_name == "ConvNeXtTiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
    elif model_name == "EfficientNetV2S":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    else:
        raise ValueError("FATAL ERROR. MODEL WEIGHTS NOT PRESENT.")
    
    model.load_state_dict(torch.load(model_paths[model_name], map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# Load models
cnn_models = [load_model(name) for name in model_paths.keys()]
base_models = {name: joblib.load(path) for name, path in meta_model_paths.items()}
meta_model = joblib.load("./Meta-Ensemble_CXR14/Stacked_Model_GB.joblib")

# Load dataset
data_dir = './CXR14_processed_224x224'
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Store predictions
y_true = []
y_scores = []

# Get predictions
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    y_true.append(labels.item())
    
    cnn_features = []
    with torch.no_grad():
        for model in cnn_models:
            output = model(inputs)
            probability = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            cnn_features.append(probability[0])
    cnn_features = np.array(cnn_features).reshape(1, -1)
    
    meta_features = np.zeros((1, len(meta_model_paths)))
    for idx, (_, model) in enumerate(base_models.items()):
        meta_features[:, idx] = model.predict_proba(cnn_features)[:, 1]
    
    final_prob = meta_model.predict_proba(meta_features)[0, 1]
    y_scores.append(final_prob)

# Convert to numpy arrays
y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Set the style
plt.style.use('bmh')

plt.figure(figsize=(8, 8))

# Plot ROC curve
plt.plot(fpr, tpr, lw=2.5, label=f'ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')

# Set axes labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])

plt.legend(loc='lower right')

plt.savefig('ROC_Curve_CXR14.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()