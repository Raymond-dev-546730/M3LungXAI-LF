# 🫁 Lung Cancer Diagnosis via Multimodal AI

This project presents a multimodal deep learning framework for **transparent and accurate lung cancer diagnosis** using a fusion of **X-rays, CT scans, and clinical symptom data**. Our system leverages state-of-the-art computer vision and NLP techniques to provide interpretable, evidence-based predictions with heatmaps, annotated regions, and auto-generated medical summaries.

## 🚀 Key Features

- 🔍 **Multi-Modal Architecture**  
  Combines imaging data (X-ray, CT) and clinical records (text-based) using deep learning and late fusion strategies.

- 🧠 **Custom Deep Learning Models**  
  - CNN-based vision backbones (e.g., ConvNeXt, ResNet, EfficientNet)  
  - Transformer-based fusion and text encoders (e.g., BERT)  
  - Joint representation space for final diagnosis

- 🌐 **Explainable AI (XAI)**  
  - Grad-CAM and LIME visualizations over medical images  
  - Symptom-based attention heatmaps  
  - Automated medical report generation

- 📊 **Evaluation Metrics**  
  - Accuracy, Precision, Recall, F1-score, AUC per modality   
  - Per-sample explainability and overlay visualizations

## 🧬 Dataset

We trained and validated on a combination of publicly available datasets, including:

- **NIH ChestX-ray14**  
- **LIDC-IDRI CT scans**  
- **Self developed clinical text samples** using locally run LLM with symptoms, history, and risk factors

> ⚠️ *Due to licensing restrictions, dataset links are not included.

## 💡 Future Work

- Expand to PET scans and genomic data  
- Integrate with hospital EMRs  
- Active learning for real-world deployment  
- Human-in-the-loop feedback system

## 👨‍💻 Authors

- **Rehaan Kadhar** – Aerospace Engineering @ UC Berkeley  
- **Raymond Lee** – Senior @ Westview High School 

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.
