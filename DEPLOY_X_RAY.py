# Import required libraries
import json
import os
import torch
import numpy as np
import joblib
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from torchvision import transforms, models
from lime.wrappers.scikit_image import SegmentationAlgorithm
import random
import cv2

# Set random seed for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the best saved epoch states (hand picked based on validation AUC and Loss)
model_paths = {
    "ResNet18": "./X-Ray_Modality/ResNet18_CXR14/epoch_21_model.pth",
    "ConvNeXtTiny": "./X-Ray_Modality/ConvNeXtTiny_CXR14/epoch_9_model.pth",
    "EfficientNetV2S": "./X-Ray_Modality/EfficientNetV2S_CXR14/epoch_8_model.pth", 
    "DenseNet121": "./X-Ray_Modality/DenseNet121_CXR14/epoch_10_model.pth" 
}

# Meta-model paths
meta_model_paths = {
    'LR': "./X-Ray_Modality/Hyperparameters_CXR14/LR_Model.joblib",
    'GB': "./X-Ray_Modality/Hyperparameters_CXR14/GB_Model.joblib",
    'RF': "./X-Ray_Modality/Hyperparameters_CXR14/RF_Model.joblib",
    'AB': "./X-Ray_Modality/Hyperparameters_CXR14/AB_Model.joblib",
    'ET': "./X-Ray_Modality/Hyperparameters_CXR14/ET_Model.joblib",
    'LGB': "./X-Ray_Modality/Hyperparameters_CXR14/LGB_Model.joblib",
}


# Path for the final ensemble (Meta-Ensemble) model
meta_ensemble_model_path = "./X-Ray_Modality/Meta-Ensemble_CXR14/Stacked_Model_GB.joblib"
F1_threshold_path = "./X-Ray_Modality/Meta-Ensemble_CXR14/F1_Threshold.json"

# Load the best threshold
with open(F1_threshold_path, 'r') as f:
    F1_threshold = json.load(f)['best_threshold']

# Transformations for the input image
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# Function to load model based on name
def load_model(model_name):
    
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
    
    model.load_state_dict(torch.load(model_paths[model_name],  map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def predict_cnn_features(image_tensor):
    models_list = [load_model(name) for name in model_paths.keys()]
    cnn_features = []
    for mdl in models_list:
        with torch.no_grad():
            output = mdl(image_tensor.unsqueeze(0).to(device))
            # Get probability for class 1
            probability = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            cnn_features.append(probability[0])  
    return np.array(cnn_features).reshape(1, -1)

def base_learner_predictions(cnn_features):
    # Initialize meta feature vector with length equal to number of base learners which is 6
    meta_features = np.zeros((1, len(meta_model_paths)))
    
    # Populate meta_features with predictions from each base learner
    for idx, (model_key, path) in enumerate(meta_model_paths.items()):
        model = joblib.load(path)
        meta_features[:, idx] = model.predict_proba(cnn_features)[:, 1]  # Probability for Class 1
    return meta_features

def plot_meta_learner_influence(meta_features, prediction_label, output_dir):
    learner_names = list(meta_model_paths.keys())
    learner_influences = {
        learner: meta_features[0, idx] 
        for idx, learner in enumerate(learner_names)
    }

    plt.figure(figsize=(10, 6))
    plt.bar(learner_influences.keys(), learner_influences.values(), color='#4A90E2')
    plt.xlabel('Meta Learners')
    plt.ylabel(f'Probability for Class "{prediction_label}"')
    plt.title(f'Meta Learner Influence for Class "{prediction_label}"')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'X-Ray_Meta_Learner_Influence.png')
    plt.savefig(save_path)
    plt.close()

prediction_results = {}

def predict():
    global prediction_results
    # Load and preprocess image
    input_dir = "./Input_X-ray" 
    os.makedirs(input_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if len(image_files) != 1:
        raise ValueError(f"Expected exactly one image in '{input_dir}', but found {len(image_files)}.")
    
    image_path = os.path.join(input_dir, image_files[0])
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transforms(image)
    
    # Get CNN feature predictions!
    cnn_features = predict_cnn_features(image_tensor)

    meta_features = base_learner_predictions(cnn_features)

    # Load the meta-ensemble model!
    meta_model = joblib.load(meta_ensemble_model_path)

    # Final ensemble prediction
    final_pred_prob = meta_model.predict_proba(meta_features)[:, 1]
    final_pred = (final_pred_prob >= F1_threshold).astype(int)
    prediction_label = 'Normal' if final_pred == 1 else 'Nodule'
    target_label = 1 if final_pred == 1 else 0

    # Nodule is class 0
    # Normal is class 1

    print(f"[DEBUG] PREDICTION: {prediction_label}") # Prints out predicted class
    probabilities = meta_model.predict_proba(meta_features)[0]
    print(f"[DEBUG] CONFIDENCE: {probabilities[target_label]*100:.2f}%") # Prints out confidence score

    prediction_results = {
        'xray_prediction': prediction_label,
        'xray_confidence': probabilities[target_label]*100
    }

    # Plot and save meta-learner influence chart
    output_dir = "./XAI_Output_1"
    os.makedirs(output_dir, exist_ok=True)
    plot_meta_learner_influence(meta_features, prediction_label, output_dir)

    # Define LIME explainer for the meta model
    explainer = lime_image.LimeImageExplainer(random_state=1)

    # LIME prediction function for the meta ensemble
    def meta_predict(images):
        preds = []
        for img in images:
            img_tensor = val_transforms(Image.fromarray(img)).to(device)
            cnn_feats = predict_cnn_features(img_tensor)
            meta_feats = base_learner_predictions(cnn_feats)
            probas = meta_model.predict_proba(meta_feats)  # Probabilities for both classes
            preds.append(probas[0])
        return np.array(preds)

    segmentation_fn = SegmentationAlgorithm(
        'slic',
        n_segments=50,
        compactness=20,
        sigma=1,
        random_seed=1
    )

    explanation = explainer.explain_instance(
        np.array(image),
        meta_predict,
        labels=[0, 1],
        hide_color=0,
        num_samples=100,
        segmentation_fn=segmentation_fn
    )

    if target_label in explanation.top_labels:
        class_name = "Nodule" if target_label == 0 else "Normal"
        temp, mask = explanation.get_image_and_mask(
            label=target_label,
            positive_only=False,
            num_features=5,
            hide_rest=False
        )

        exp_list = explanation.local_exp[target_label]
        exp_list = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)

        seg_ids = [str(x[0]) for x in exp_list]
        seg_weights = [x[1] for x in exp_list]
        bar_colors = ["green" if w > 0 else "red" for w in seg_weights]

        plt.figure(figsize=(8, 6))
        plt.bar(seg_ids, seg_weights, color=bar_colors)
        plt.title(f"LIME Local Feature Importances for '{class_name}'")
        plt.xlabel("Super-Pixel ID")
        plt.ylabel("Importance Weight")
        plt.xticks(rotation=45, fontsize=8)
        lime_bar_path = os.path.join(output_dir, "X-Ray_LIME_Bar.png")
        plt.savefig(lime_bar_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Only plots the top 5 most important features
        k = 5
        top_segments = exp_list[:k]
        segments = explanation.segments

        lime_overlay = temp.astype(np.float32)
        alpha = 0.4
        for (seg_id, weight) in top_segments:
            mask_area = (segments == seg_id)
            color = (0, 255, 0) if weight > 0 else (255, 0, 0)
            lime_overlay[mask_area] = alpha * lime_overlay[mask_area] + (1 - alpha) * np.array(color, dtype=np.float32)
            y_indices, x_indices = np.where(mask_area)
            if len(y_indices) > 0:
                y_mean = int(np.mean(y_indices))
                x_mean = int(np.mean(x_indices))
                label_text = f"ID:{seg_id}"
                cv2.putText(lime_overlay, label_text, (x_mean, y_mean), 
                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.3, 
                            color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        lime_overlay = np.clip(lime_overlay, 0, 255).astype(np.uint8)
        overlay_path = os.path.join(output_dir, "X-Ray_LIME_Overlay.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(lime_overlay)
        plt.axis('off')
        plt.title(f"LIME Overlay (Top {k} Super-Pixels) for '{class_name}'")
        plt.savefig(overlay_path, bbox_inches='tight', dpi=100)
        plt.close()
    else:
        print(f"LIME did not generate an explanation for the predicted class '{prediction_label}'.")

if __name__ == '__main__':
    predict()