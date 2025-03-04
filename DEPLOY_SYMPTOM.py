# Import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from fuzzywuzzy import fuzz
from typing import Dict, List, Optional, Tuple, Set
import torch.nn.functional as F

# Clinical note input
Clinical_Note = """

"""

# Symptom weights mapping
SYMPTOM_WEIGHTS = {
    "Recurring lung infections": 5.0,
    "Unexpected wheezing onset": 5.0,
    "Hemoptysis": 5.0,
    "Pleuritic chest pain": 5.0,
    "Persistent worsening cough": 5.0,
    "Hoarseness": 5.0,
    "Dyspnea": 5.0,
    "Extreme fatigue": 5.0,
    "Cervical/Axillary lymphadenopathy": 5.0,
    "Swollen veins in the Neck & Chest": 2.5,
    "Headache": 5.0,
    "Facial and cervical edema": 5.0,
    "Loss of appetite": 5.0,
    "Unexplained weight loss": 5.0,
    "Bone pain": 5.0,
    "Hippocratic fingers": 5.0,
    "Jaundice": 5.0,
    "Dysphagia": 2.5,
    "Ptosis": 2.5,
    "Ipsilateral Anhidrosis": 2.5,
    "New-onset seizures": 5.0,
    "Ipsilateral Miosis": 2.5
}

# Symptom Clusters 
SYMPTOM_CLUSTERS = {
    "Respiratory Distress Cluster": {
        "symptoms": {"Persistent worsening cough", "Unexpected wheezing onset", "Dyspnea"},
        "required_count": 3,
        "bonus_weight": 10.0,
        "reasoning": "This combination suggests significant respiratory distress and warrants urgent evaluation."
    },
    "Hemoptysis Cluster": {
        "symptoms": {"Hemoptysis", "Persistent worsening cough"},
        "required_count": 2,
        "bonus_weight": 15.0,
        "reasoning": "Coughing up blood combined with a persistent cough raises serious concerns for lung cancer or other severe conditions."
    },
    "Systemic Illness Cluster": {
        "symptoms": {"Extreme fatigue", "Unexplained weight loss", "Loss of appetite"},
        "required_count": 3,
        "bonus_weight": 10.0,
        "reasoning": "This combination suggests a systemic illness, which could be due to cancer."
    },
    "Pain Cluster": {
        "symptoms": {"Pleuritic chest pain", "Bone pain"},
        "required_count": 2,
        "bonus_weight": 20.0,
        "reasoning": "Chest pain combined with referred pain to the bones could indicate tumor invasion or other serious lung issues."
    },
    "Voice & Swallowing Changes Cluster": {
        "symptoms": {"Hoarseness", "Dysphagia"},
        "required_count": 2,
        "bonus_weight": 20.0,
        "reasoning": "These symptoms can indicate a tumor pressing on the laryngeal nerve (hoarseness) or the esophagus (difficulty swallowing)."
    },
    "Advanced Disease Red Flags Cluster": {
        "symptoms": {"Cervical/Axillary lymphadenopathy", "Facial and cervical edema", "Hippocratic fingers", "Jaundice"},
        "required_count": 2,
        "bonus_weight": 20.0,
        "reasoning": "The presence of two or more of these signs suggests a more advanced stage of disease, possibly with spread beyond the lungs."
    },
    "Superior Vena Cava Syndrome Cluster": {
        "symptoms": {"Facial and cervical edema", "Swollen veins in the Neck & Chest", "Headache"},
        "required_count": 3,
        "bonus_weight": 25.0,
        "reasoning": "This combination suggests Superior Vena Cava Syndrome (SVCS), often due to a lung tumor compressing the superior vena cava."
    },
    "Horner Syndrome Cluster": {
        "symptoms": {"Ptosis", "Ipsilateral Miosis", "Ipsilateral Anhidrosis"},
        "required_count": 3,
        "bonus_weight": 25.0,
        "reasoning": "This combination indicates Horner syndrome, which can result from damage to the sympathetic nerve pathway due to a tumor."
    }
}

# Load the trained model and tokenizer
model_path = "./Symptom_Modality/SpanBERT-SCM-Large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def load_symptom_dataset(file_path: str) -> Dict[str, List[str]]:
    with open(file_path, 'r') as f:
        return json.load(f)

def find_matching_head_symptom(
    symptom_text: str,
    symptom_dataset: Dict[str, List[str]],
    threshold: int = 80 # 80%
) -> Optional[str]:
    best_match_score = 0
    best_match_head = None
    
    for head_symptom, synonyms in symptom_dataset.items():
        # Check match with head symptom
        head_score = fuzz.token_sort_ratio(symptom_text.lower(), head_symptom.lower())
        if head_score > best_match_score and head_score >= threshold:
            best_match_score = head_score
            best_match_head = head_symptom
            
        # Check match with synonyms
        for synonym in synonyms:
            synonym_score = fuzz.token_sort_ratio(symptom_text.lower(), synonym.lower())
            if synonym_score > best_match_score and synonym_score >= threshold:
                best_match_score = synonym_score
                best_match_head = head_symptom
    
    return best_match_head

def analyze_symptom_clusters(identified_symptoms: Set[str]) -> List[Dict]:
    activated_clusters = []
    
    for cluster_name, cluster_info in SYMPTOM_CLUSTERS.items():
        matching_symptoms = cluster_info["symptoms"].intersection(identified_symptoms)
        
        if len(matching_symptoms) >= cluster_info["required_count"]:
            activated_clusters.append({
                "name": cluster_name,
                "matching_symptoms": list(matching_symptoms),
                "bonus_weight": cluster_info["bonus_weight"],
                "reasoning": cluster_info["reasoning"]
            })
    
    return activated_clusters

def calculate_confidence_score(logits: torch.Tensor, pred_idx: int) -> float:
    probabilities = F.softmax(logits, dim=-1)
    confidence = probabilities[pred_idx].item()
    return round(confidence * 100, 2)

def extract_spans_and_calculate_weights(
    text: str,
    tokenizer,
    model,
    device,
    symptom_dataset: Dict[str, List[str]]
) -> Tuple[List[Dict], Dict[str, float], List[Dict], float, float]:
    # offset_mapping returns character indices for each token
    # These indices represent the character positions in the original text where each token begins and ends
    # So a example would be: "hello" --> [(0,1), (1,2), (2,3), (3,4), (4,5)] for tokenization
    tokenized = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    offsets = tokenized["offset_mapping"].squeeze()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

    verified_spans = []
    identified_weights = {symptom: 0.0 for symptom in SYMPTOM_WEIGHTS.keys()}
    used_head_symptoms = set()
    identified_symptoms = set()
    confidence_scores = []
    in_span = False
    span_start = None

    for idx, (pred, (start, end)) in enumerate(zip(predictions, offsets)):
        if pred == 1 and not in_span:  # Start of a new span (1 for SYMPTOM)
            in_span = True
            span_start = int(start)
        elif (pred == 0 or idx == len(predictions)-1) and in_span:  # End of current span
            span_text = text[span_start:int(end)]
            if span_text.strip():
                head_symptom = find_matching_head_symptom(span_text, symptom_dataset)
                confidence_score = calculate_confidence_score(logits[0, idx], pred)
                confidence_scores.append(confidence_score)
                
                if head_symptom and head_symptom not in used_head_symptoms:
                    weight = SYMPTOM_WEIGHTS.get(head_symptom, 0.0)
                    verified_spans.append({
                        "text": span_text,
                        "start": span_start,
                        "end": int(end),
                        "label": "SYMPTOM",
                        "head_symptom": head_symptom,
                        "weight": weight,
                        "confidence_score": confidence_score
                    })
                    identified_weights[head_symptom] = weight
                    identified_symptoms.add(head_symptom)
                    used_head_symptoms.add(head_symptom)
            in_span = False

    # Analyze clusters and calculate total weight
    activated_clusters = analyze_symptom_clusters(identified_symptoms)
    base_weight = sum(identified_weights.values())
    cluster_bonus = sum(cluster["bonus_weight"] for cluster in activated_clusters)
    total_weight = base_weight + cluster_bonus
    
    # Calculate overall confidence
    overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    return verified_spans, identified_weights, activated_clusters, total_weight, overall_confidence

def assess_risk_level(total_weight: float) -> Tuple[str, str]:
    if total_weight <= 50:
        return "Low Risk" # Total weight ≤ 50%
    elif 51 <= total_weight <= 100:
        return "Moderate Risk" # 51% ≤ Total weight ≤ 100%
    else:
        return "High Risk" # Total weight > 100%

def set_clinical_note(text: str):
    global Clinical_Note
    Clinical_Note = text


prediction_results = {}


def predict():
    global Clinical_Note
    global prediction_results
    prediction_results = {}
    # Load symptom dataset
    symptom_dataset = load_symptom_dataset("SYMPTOM_DATASET.json")

    # Get predictions
    verified_predictions, weights, activated_clusters, total_weight, overall_confidence = extract_spans_and_calculate_weights(
        Clinical_Note,
        tokenizer,
        model,
        device,
        symptom_dataset
    )

    # Get risk level
    risk_level = assess_risk_level(total_weight)

    # Print results
    print(f"[DEBUG] CONFIDENCE: {overall_confidence}%")
    print("\n === Risk Assessment === ")
    print(f"Risk Level: {risk_level}")
    print(f"Total Weight: {total_weight}%")
    print("\n === Detected Symptoms === ")
    for symptom in verified_predictions:
        print(f"\nDetected Text: \"{symptom['text']}\"")
        print(f"Location: Characters {symptom['start']} to {symptom['end']}")
        print(f"Matched Head Symptom: {symptom['head_symptom']}")
    
    print("\n === Activated Symptom Clusters === ")
    if activated_clusters:
        for cluster in activated_clusters:
            print(f"\nCluster: {cluster['name']}")
            print(f"Matching Symptoms: {', '.join(cluster['matching_symptoms'])}")
            print(f"Clinical Significance: {cluster['reasoning']}")
    else:
        print("No symptom clusters rules activated")
    
    # Clear note after prediction
    Clinical_Note = """"""
    
    prediction_results = {
        'clinical_results': {
            'symptoms': verified_predictions,
            'clusters': activated_clusters,
            'risk_level': risk_level,
            'risk_weight': total_weight,
            's_confidence': overall_confidence 
        }
    }

if __name__ == "__main__":
    predict()