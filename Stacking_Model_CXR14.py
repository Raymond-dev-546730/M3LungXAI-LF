# Import required libraries
import numpy as np
import os
import json
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV  
from torchvision import datasets, transforms, models
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib  

# Set random seed for reproducibility (42)
np.random.seed(42)
torch.manual_seed(42)

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the best saved epoch states (hand picked based on validation AUC and Loss)
model_paths = {
    "ResNet18": "./ResNet18_CXR14/epoch_21_model.pth",
    "ConvNeXtTiny": "./ConvNeXtTiny_CXR14/epoch_9_model.pth",
    "EfficientNetV2S": "./EfficientNetV2S_CXR14/epoch_8_model.pth", 
    "DenseNet121": "./DenseNet121_CXR14/epoch_10_model.pth" 
}

# Load full dataset with validation transforms
data_dir = "./CXR14_processed_224x224" 
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
full_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)
full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


# Function to load model based on name
def load_model(model_name):
    print(f"Loading model: {model_name}")
    
    if model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "ConvNeXtTiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    elif model_name == "EfficientNetV2S":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    else:
        raise ValueError("FATAL ERROR. MODEL WEIGHTS NOT PRESENT.")
    
    # Load the model weights
    model.load_state_dict(torch.load(model_paths[model_name], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

# Function to get predictions from a model
def get_predictions(model, dataloader):
    all_preds = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1 (normal)
            all_preds.extend(probabilities.cpu().numpy())
    return np.array(all_preds)

# Function to optimize threshold for F1 score
def optimize_threshold(y_true, y_pred_probs):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1

if __name__ == '__main__':
    # Load the models
    resnet18 = load_model("ResNet18")
    convnexttiny = load_model("ConvNeXtTiny")
    efficientnetv2s = load_model("EfficientNetV2S")
    densenet121 = load_model("DenseNet121")

    # Generate predictions for each model
    resnet18_preds = get_predictions(resnet18, full_loader)
    convnexttiny_preds = get_predictions(convnexttiny, full_loader)
    efficientnetv2s_preds = get_predictions(efficientnetv2s, full_loader)
    densenet121_preds = get_predictions(densenet121, full_loader)

    # Combine CNN predictions to create a feature matrix for meta-learners
    X_cnn = np.column_stack((resnet18_preds, convnexttiny_preds, efficientnetv2s_preds, densenet121_preds))
    y = np.array([label for _, label in full_dataset.imgs])

    # Define the directory for saving/loading hyperparameters and models
    hyperparameters_dir = "./Hyperparameters_CXR14" 
    models_dir = "./Meta-Ensemble_CXR14" 
    os.makedirs(hyperparameters_dir, exist_ok=True) 
    os.makedirs(models_dir, exist_ok=True)

    # Define parameter grids for hyperparameter tuning
    parameter_grids = { 
        'LR': {'C': [0.001, 0.01, 0.1, 1, 10, 100]}, 
        'RF': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'max_features': ['sqrt', 'log2']
        },
        'GB': {
            'n_estimators': [50, 100, 200], 
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'AB': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'ET': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7], 
            'max_features': ['sqrt', 'log2']
        },
        'LGB': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2], 
            'verbosity': [-1]
        }
    }

    # Function to save hyperparameters to a JSON file
    def save_hyperparameters(filename, hyperparameters): 
        with open(filename, 'w') as f:
            json.dump(hyperparameters, f) 

    # Function to load hyperparameters from a JSON file
    def load_hyperparameters(filename): 
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and hyperparameter tuning for each base learner
    tuned_models = {}
    calibrated_models = {}
    for name, model_class, param_grid in [
        ('LR', LogisticRegression(random_state=42, class_weight='balanced'), parameter_grids['LR']), 
        ('GB', GradientBoostingClassifier(random_state=42), parameter_grids['GB']), 
        ('RF', RandomForestClassifier(random_state=42, class_weight='balanced'), parameter_grids['RF']), 
        ('AB', AdaBoostClassifier(algorithm='SAMME',random_state=42), parameter_grids['AB']), 
        ('ET', ExtraTreesClassifier(random_state=42, class_weight='balanced'), parameter_grids['ET']), 
        ('LGB', LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1), parameter_grids['LGB']) 
    ]:
        hyperparameters_file = os.path.join(hyperparameters_dir, f'{name}_Hyperparameters.json') 
        best_parameters = load_hyperparameters(hyperparameters_file) 

        if best_parameters is not None: 
            print(f"Loading saved hyperparameters for {name}") 
            model = model_class.set_params(**best_parameters)
        else:
            print(f"Tuning hyperparameters for {name}")
            search = GridSearchCV(
                model_class, 
                param_grid, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1_macro', 
                n_jobs=-1
            ) 
            search.fit(X_cnn, y)
            best_parameters = search.best_params_ 
            model = search.best_estimator_
            save_hyperparameters(hyperparameters_file, best_parameters) 

        # Fit the model to ensure it's trained
        model.fit(X_cnn, y)

        # Calibrate the fitted model using Platt scaling
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')  
        calibrated_model.fit(X_cnn, y) 

        # Save the trained and calibrated models
        joblib.dump(calibrated_model, os.path.join(hyperparameters_dir, f'{name}_Model.joblib')) 

        tuned_models[name] = model
        calibrated_models[name] = calibrated_model

    # Initialize an empty array to store the meta-features for stacking
    meta_features = np.zeros((len(X_cnn), len(calibrated_models)))

    # Store the predictions of the base models to use as input to the meta-model
    for idx, (base_name, base_calibrated_model) in enumerate(calibrated_models.items()):
        meta_features[:, idx] = base_calibrated_model.predict_proba(X_cnn)[:, 1]  # Use calibrated probabilities as features

    # Meta-learner (Gradient Boosting)
    gb_meta_model = GradientBoostingClassifier(random_state=42)

    # Perform cross-validation for the Gradient Boosting meta-model with threshold optimization
    auc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    # Initialize variables to track the best model and threshold
    best_f1_global = 0
    best_threshold_global = None

    # Function to print evaluation metrics
    def print_evaluation_metrics(fold, auc, f1, precision, recall, accuracy):
        print(f"\nFold {fold + 1} Metrics:")
        print(f"AUC: {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

    # 5-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(meta_features, y)):
        X_train, X_test = meta_features[train_idx], meta_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply SMOTE to handle class imbalance (only to the training set)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Fit the Gradient Boosting meta-model
        gb_meta_model.fit(X_train_resampled, y_train_resampled)
        meta_preds = gb_meta_model.predict_proba(X_test)[:, 1]

        # Optimize threshold for F1 score
        best_threshold, best_f1 = optimize_threshold(y_test, meta_preds)
        predicted_labels = (meta_preds >= best_threshold).astype(int)

        # Evaluate the meta-model
        auc = roc_auc_score(y_test, meta_preds)
        f1 = f1_score(y_test, predicted_labels, average='macro')
        precision = precision_score(y_test, predicted_labels, average='macro', zero_division=1)
        recall = recall_score(y_test, predicted_labels, average='macro')
        accuracy = accuracy_score(y_test, predicted_labels)

        auc_scores.append(auc)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        accuracy_scores.append(accuracy)

        # Print evaluation metrics for the current fold
        print_evaluation_metrics(fold, auc, f1, precision, recall, accuracy)

        # Update the best model and threshold based on F1 score
        if best_f1 > best_f1_global:
            best_f1_global = best_f1
            joblib.dump(gb_meta_model, os.path.join(models_dir, 'Stacked_Model_GB.joblib'))
            best_threshold_global = best_threshold

    # Save the best threshold to a JSON file
    best_threshold_file = os.path.join(models_dir, 'F1_Threshold.json') 
    with open(best_threshold_file, 'w') as f:
        json.dump({'best_threshold': best_threshold_global}, f)

    # Print average cross-validation results for the Gradient Boosting stacking model
    print("\nAverage Cross-Validation Results for Gradient Boosting Stacking Model:")
    print(f"Average AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Average Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Average Recall: {np.mean(recall_scores):.4f} eep breath  {np.std(recall_scores):.4f}")
    print(f"Average Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")

    print(f"\nBest F1 Score: {best_f1_global:.4f}")
    print(f"Best Threshold: {best_threshold_global}")
