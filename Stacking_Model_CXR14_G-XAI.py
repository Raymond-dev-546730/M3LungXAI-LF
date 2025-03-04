# Import required libraries
import os
import json
import torch
import numpy as np
import joblib
import random
from torchvision import datasets, transforms, models
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import PartialDependenceDisplay
import tqdm

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the best saved epoch states (hand picked based on validation AUC and Loss)
model_paths = {
    "ResNet18": "./ResNet18_CXR14/epoch_21_model.pth",
    "ConvNeXtTiny": "./ConvNeXtTiny_CXR14/epoch_9_model.pth",
    "EfficientNetV2S": "./EfficientNetV2S_CXR14/epoch_8_model.pth", 
    "DenseNet121": "./DenseNet121_CXR14/epoch_10_model.pth" 
}

# Meta-model paths
meta_model_paths = {
    'LR': "./Hyperparameters_CXR14/LR_Model.joblib",
    'GB': "./Hyperparameters_CXR14/GB_Model.joblib",
    'RF': "./Hyperparameters_CXR14/RF_Model.joblib",
    'AB': "./Hyperparameters_CXR14/AB_Model.joblib",
    'ET': "./Hyperparameters_CXR14/ET_Model.joblib",
    'LGB': "./Hyperparameters_CXR14/LGB_Model.joblib",
}

# Path for the final ensemble (Meta-Ensemble) model
meta_ensemble_model_path = "./Meta-Ensemble_CXR14/Stacked_Model_GB.joblib"
F1_threshold_path = "./Meta-Ensemble_CXR14/F1_Threshold.json"

# Load the best threshold
with open(F1_threshold_path, 'r') as f:
    F1_threshold = json.load(f)['best_threshold']

# Base directories for plots
shap_plot_dir = "./SHAP_Plots_CXR14"
pdp_plot_dir = "./PDP_Plots_CXR14"

# Create separate subfolders for Meta-Model XAI and Meta-Ensemble XAI
shap_plot_dir_meta_model = os.path.join(shap_plot_dir, "Meta-Model")
shap_plot_dir_meta_ensemble = os.path.join(shap_plot_dir, "Meta-Ensemble")
pdp_plot_dir_meta_model = os.path.join(pdp_plot_dir, "Meta-Model")
pdp_plot_dir_meta_ensemble = os.path.join(pdp_plot_dir, "Meta-Ensemble")
for d in [shap_plot_dir, pdp_plot_dir,
          shap_plot_dir_meta_model, shap_plot_dir_meta_ensemble,
          pdp_plot_dir_meta_model, pdp_plot_dir_meta_ensemble]:
    os.makedirs(d, exist_ok=True)

# Define image transformations
val_transforms = transforms.Compose([
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

def get_predictions(model, dataloader):
    all_preds = []
    with torch.no_grad():
        for inputs, _ in tqdm.tqdm(dataloader, desc=f"Predicting with {model.__class__.__name__}"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability for class 1 (Normal)
            all_preds.extend(probabilities.cpu().numpy())
    return np.array(all_preds)

def batch_predict(X, model):
    batch_size = 32
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
        with torch.no_grad():
            pred = model.predict_proba(batch)[:, 1]
        predictions.extend(pred)
    return np.array(predictions)

# Prepare full dataset
data_dir = "./CXR14_processed_224x224"
full_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)

# Create a dataloader for the full dataset
loader = torch.utils.data.DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load CNN models and generate predictions for the full dataset
print("Generating predictions for the full dataset")
resnet18 = load_model("ResNet18")
convnexttiny = load_model("ConvNeXtTiny")
efficientnetv2s = load_model("EfficientNetV2S")
densenet121 = load_model("DenseNet121")

resnet18_preds = get_predictions(resnet18, loader)
convnexttiny_preds = get_predictions(convnexttiny, loader)
efficientnetv2s_preds = get_predictions(efficientnetv2s, loader)
densenet121_preds = get_predictions(densenet121, loader)

# Build the base CNN features matrix
X_cnn = np.column_stack((resnet18_preds, convnexttiny_preds, efficientnetv2s_preds, densenet121_preds))

# Get labels from the full dataset
labels = np.array([img[1] for img in full_dataset.imgs])

# Build the meta-features matrix
meta_features = np.zeros((len(X_cnn), len(meta_model_paths)))
calibrated_models = {}
for idx, (model_name, path) in enumerate(meta_model_paths.items()):
    print(f"Loading and predicting with meta-model {model_name}")
    calibrated_models[model_name] = joblib.load(path)
    meta_features[:, idx] = calibrated_models[model_name].predict_proba(X_cnn)[:, 1]  # Class 1 probabilities

base_feature_names = ["ResNet18", "ConvNeXtTiny", "EfficientNetV2S", "DenseNet121"]
ensemble_feature_names = ["LR", "GB", "RF", "AB", "ET", "LGB"]

# Meta-Model SHAP Analysis
for name, model in calibrated_models.items():
    print(f"[Meta-Model] Computing SHAP values for meta-model {name} using base CNN predictions")
    underlying_model = model.estimator if isinstance(model, CalibratedClassifierCV) else model

    underlying_model.fit(X_cnn, labels)

    # Compute SHAP values using X_cnn as input
    if isinstance(underlying_model, (LogisticRegression, AdaBoostClassifier)):
        explainer = shap.KernelExplainer(underlying_model.predict_proba, shap.sample(X_cnn, 100))
        shap_vals = explainer.shap_values(X_cnn)
        shap_values = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    elif isinstance(underlying_model, (RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier)):
        explainer = shap.TreeExplainer(underlying_model)
        shap_vals = explainer.shap_values(X_cnn)
        shap_values = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    elif isinstance(underlying_model, LGBMClassifier):
        contrib = underlying_model.predict(X_cnn, pred_contrib=True)
        shap_values = contrib[:, :-1] 
    else:
        explainer = shap.Explainer(underlying_model, X_cnn)
        shap_values = explainer(X_cnn).values

    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    print(f"[Meta-Model] Adjusted SHAP values shape for {name}: {shap_values.shape}")

    # Generate and save the violin-based SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_cnn, feature_names=base_feature_names, show=False, plot_type='violin')
    plt.title(f"How Different CNN Models Contribute to {name} Diagnosis\nImportance of Each X-Ray Analysis Model", 
              fontsize=16, fontweight='bold')
    plt.xlabel("Model's Influence on Diagnosis", fontsize=12)
    plt.ylabel("X-Ray Analysis Models", fontsize=12)
    save_path = os.path.join(shap_plot_dir_meta_model, f'SHAP_Meta-Model_{name}.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Meta-Model Partial Dependence Plots
print("Generating Meta-Model Partial Dependence Plots")
for name, model in calibrated_models.items():
    print(f"[Meta-Model] Generating PDP plots for meta-model {name} using base CNN predictions")
    underlying_model = model.estimator if isinstance(model, CalibratedClassifierCV) else model
    underlying_model.fit(X_cnn, labels)  # Re-fit!
    
    fig, axes = plt.subplots(1, len(base_feature_names), figsize=(20, 6))
    PartialDependenceDisplay.from_estimator(
        underlying_model,
        X_cnn,
        features=[0, 1, 2, 3],
        feature_names=base_feature_names,
        ax=axes
    )
    plt.suptitle(f"How {name} Model Decides: Impact of Individual X-Ray Models\non Diagnosis", 
                 fontsize=16, fontweight='bold')
    
    for ax, feature_name in zip(axes, base_feature_names):
        ax.set_title(f"Role of {feature_name} in Diagnosis")
        ax.set_xlabel("Model's Prediction Confidence")
        ax.set_ylabel("Chance of Diagnosis")
    
    save_path = os.path.join(pdp_plot_dir_meta_model, f'PDP_Meta-Model_{name}.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Final Ensemble Analysis
print("Starting Final Ensemble Analysis")
final_ensemble = joblib.load(meta_ensemble_model_path)

# Compute SHAP values for the final ensemble using meta_features as input
explainer_final = shap.KernelExplainer(final_ensemble.predict_proba, shap.sample(meta_features, 100))
shap_vals_final = explainer_final.shap_values(meta_features)
shap_values_final = shap_vals_final[1] if isinstance(shap_vals_final, list) else shap_vals_final
shap_values_final = np.array(shap_values_final)
if shap_values_final.ndim == 3:
    shap_values_final = shap_values_final[:, :, 1]
print(f"[Meta-Ensemble] Meta-Ensemble SHAP values shape: {shap_values_final.shape}")

# Final Ensemble SHAP Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_final, meta_features, feature_names=ensemble_feature_names, show=False, plot_type='violin')
plt.title("Meta-Ensemble: Which Meta-Models Matter Most\nin X-Ray Diagnosis", 
          fontsize=16, fontweight='bold')
plt.xlabel("Model's Impact on Final Diagnosis", fontsize=12)
plt.ylabel("Different Meta-Models", fontsize=12)
save_path = os.path.join(shap_plot_dir_meta_ensemble, 'SHAP_Meta-Ensemble.png')
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')
plt.close()

# Select top 3 features based on the mean absolute SHAP values.
feature_importance = np.abs(shap_values_final).mean(0)
top_feature_indices = np.argsort(feature_importance)[-3:]
top_feature_names = [ensemble_feature_names[i] for i in top_feature_indices]

# Final Ensemble PDP Plot
fig, axes = plt.subplots(1, len(top_feature_indices), figsize=(15, 6))
PartialDependenceDisplay.from_estimator(
    final_ensemble,
    meta_features,
    features=top_feature_indices,
    feature_names=ensemble_feature_names,
    ax=axes
)

plt.suptitle(f"Top 3 Most Important Meta-Models in X-Ray Diagnosis:\n{', '.join(top_feature_names)}", 
             fontsize=16, fontweight='bold')

for ax, feature_name in zip(axes, [top_feature_names[i] for i in range(len(top_feature_indices))]):
    ax.set_title(f"How {feature_name} Influences Diagnosis")
    ax.set_xlabel("Model's Prediction Confidence")
    ax.set_ylabel("Chance of Diagnosis")

save_path = os.path.join(pdp_plot_dir_meta_ensemble, 'PDP_Meta-Ensemble.png')
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print("XAI FINISHED")
