# Import required libraries
import torch
from torchvision import models, transforms, datasets
from PIL import Image
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np  

# Define the CT MobileNetV2 class
class CT_MobileNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super(CT_MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# Define data directories
data_dir = './CT_scan_processed_128x128'

# Define the transform to preprocess the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Create dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CT_MobileNetV2()
model.to(device)

weights_path = './MobileNetV2_CT_Scan/MobileNetV2_fold_1_best.pth'
state_dict = torch.load(weights_path, map_location=device, weights_only=True)

# Adjust state_dict keys to prevent model loading failure
adjusted_state_dict = {}
for key, value in state_dict.items():
    new_key = key
    if key.startswith('classifier.1.1'):
        new_key = key.replace('classifier.1.1', 'model.classifier.1')
    else:
        new_key = f"model.{key}"
    adjusted_state_dict[new_key] = value

# Load the adjusted state_dict into the model
model.load_state_dict(adjusted_state_dict)
model.eval()

class_names = dataset.classes

# Initialize lists to store true labels and predicted scores
true_labels = []
predicted_scores = []

# Iterate through the dataset and make predictions
for i, (inputs, labels) in enumerate(dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
        true_class = labels.item()

        # Store true labels and probabilities
        true_labels.append(true_class)
        predicted_scores.append(probabilities)  # Store all class probabilities

true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(class_names)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], np.array(predicted_scores)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_bin.ravel(), np.array(predicted_scores).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Average & compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Set the style for all plots
plt.style.use('bmh')

# For the Micro-Average plot
plt.figure(figsize=(10, 8))  
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-Average ROC curve'.format(roc_auc["micro"]),
         color='#4A90E2', linestyle='-', linewidth=2)

# Create random classifier line for 4 class classification
x = np.linspace(0, 1, 100)
y = x/4  # Equation for random classifer. AUC should be 0.25
plt.plot(x, y, 'k--', lw=2, label='Random Classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Micro-Averaged ROC Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)
plt.tight_layout() 

# Save the Micro-Average plot as PNG
micro_roc_png_path = './MobileNetV2_CT_Scan/Micro-Average_ROC_Curve_CT_Scan.png'
plt.savefig(micro_roc_png_path, format='png', dpi=300, bbox_inches='tight')


# For the Macro-Average plot
plt.figure(figsize=(10, 8)) 
plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-Average ROC curve'.format(roc_auc["macro"]),
         color='#E14758', linestyle='-', linewidth=2)

# Add random classifier line
plt.plot(x, y, 'k--', lw=2, label='Random Classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Macro-Averaged ROC Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save the Macro-Average plot as PNG
macro_roc_png_path = './MobileNetV2_CT_Scan/Macro-Average_ROC_Curve_CT_Scan.png'
plt.savefig(macro_roc_png_path, format='png', dpi=300, bbox_inches='tight')


print(f"Micro-Averaged ROC curve saved to {micro_roc_png_path}")
print(f"Macro-Averaged ROC curve saved to {macro_roc_png_path}") # Macro-Averaged ROC curve is less biased to class imbalance
