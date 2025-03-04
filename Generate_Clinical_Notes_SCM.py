# Import required libraries
import random
import os
from llama_cpp import Llama
from transformers import AutoTokenizer
import logging
import json 
import re
from fuzzywuzzy import fuzz

# Using an Mistral 7B Instruct v0.3 6-bit quantized model

# Model path and settings
mistral_model_path = "./Mistral_7B/Mistral-7B-Instruct-v0.3-Q6_K.gguf"
mistral_context_limit = 10000 # Limited context window

# Logging setup
logging.basicConfig(
    filename="clinical_note_labeling.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Query model with specified parameters: temperature controls randomness, top_p controls sampling diversity
def query_model(model, prompt, temperature=0.5, top_p=0.7):
    max_tokens = mistral_context_limit - len(tokenizer.tokenize(prompt))
    max_tokens = max(max_tokens, 1) 

    response = model(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    return response["choices"][0]["text"].strip()

# Initialize the tokenizer (BIO BERT)
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


# Generate random patient names
def generate_random_name():
    first_names = [
        "John", "Jane", "Michael", "Sarah", "Emily", "Robert", "Linda", "David", "Laura",
        "William", "Olivia", "James", "Emma", "Benjamin", "Sophia", "Henry", "Isabella",
        "Alexander", "Charlotte", "Ethan", "Amelia", "Mason", "Mia", "Logan", "Harper",
        "Lucas", "Evelyn", "Jackson", "Abigail", "Levi", "Ella", "Sebastian", "Avery",
        "Jack", "Scarlett", "Owen", "Grace", "Elijah", "Zoe", "Noah", "Lily", "Liam",
        "Hannah", "Aiden", "Chloe", "Caleb", "Victoria", "Matthew", "Ellie", "Nathan",
        "Nora", "Samuel", "Addison", "Andrew", "Natalie", "Joseph", "Aria", "Joshua", "Lucy",
        "Thomas", "Ruby", "Gabriel", "Layla", "Ryan", "Alice", "Christopher", "Eva", 
        "Julian", "Luna", "Daniel", "Stella", "Isaac", "Maya", "Adam", "Hazel", "Jonathan", 
        "Penelope", "Connor", "Savannah", "Hunter", "Aurora", "Dylan", "Madison", 
        "Christian", "Elliana", "Carter", "Peyton", "Anthony", "Violet", "Nathaniel", 
        "Rose", "Oliver", "Claire", "Brayden", "Faith", "Jaxon", "Elena", "Grayson", 
        "Aubrey", "Eli", "Willow", "Aaron", "Paisley", "Landon", "Samantha", "Miles", 
        "Jasmine", "Isaiah", "Sophia", "Evan", "Skylar"
    ]

    last_names = [
        "Smith", "Johnson", "Brown", "Taylor", "Anderson", "Lee", "Martin", "Garcia",
        "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Moore",
        "Thomas", "Jackson", "White", "Harris", "Thompson", "Young", "Scott", "Green",
        "Walker", "Hall", "Allen", "King", "Wright", "Hill", "Torres", "Campbell", "Mitchell",
        "Perez", "Roberts", "Turner", "Phillips", "Parker", "Evans", "Edwards", "Collins",
        "Stewart", "Sanchez", "Morris", "Rogers", "Reed", "Cook", "Morgan", "Bell", "Murphy",
        "Bailey", "Rivera", "Cooper", "Richardson", "Cox", "Howard", "Ward", "Torres", "Peterson",
        "Barnes", "Foster", "Powell", "Henderson", "Ross", "Coleman", "Jenkins", "Perry",
        "Reynolds", "Griffin", "Russell", "Wood", "Watson", "Brooks", "Kelly", "Sanders",
        "Price", "Bennett", "Butler", "Fisher", "Hendrix", "Fleming", "Hoffman", "McCarthy",
        "Stone", "Adams", "Gonzalez", "Ortiz", "Ramirez", "Castro", "Morales", "Sullivan",
        "Murillo", "Ramos", "Gomez", "Alvarez", "Vargas", "Cruz", "Flores", "Guzman"
    ]

    return f"{random.choice(first_names)} {random.choice(last_names)}"

# Define symptom synonyms (prevent harmful hallucinations)
symptom_synonyms = {
    "Recurring lung infections": [
        "frequent respiratory infections",
        "chronic lung issues",
        "recurrent respiratory infections",
        "repeated chest infections",
        "persistent bronchial infections",
        "lung infections that come back",
        "recurrent lower respiratory tract infections",
        "recurrent upper respiratory tract infections",
        "frequent episodes of pneumonia",
        "bronchitis exacerbations",
        "recurrent pulmonary infections",
        "history of multiple lung infections",
        "prone to chest infections",
        "repetitive respiratory illness",
        "bronchial infection"
    ],
    "Unexpected wheezing onset": [
        "new onset wheezing",
        "sudden wheezing episodes",
        "unexpected breathing whistling",
        "abrupt wheezing",
        "newly developed wheezing",
        "random wheezing onset",
        "acute wheezing development",
        "wheezing without prior history",
        "spontaneous wheezing",
        "unanticipated onset of wheeze",
        "wheezing of recent onset",
        "wheezing present",
        "wheezing",
        "acute wheezing"
    ],
    "Hemoptysis": [
        "coughing up blood",
        "bloody sputum",
        "blood in phlegm",
        "red-streaked mucus",
        "spitting up blood",
        "blood found in phlegum",
        "bloody phlegum",
        "hemoptysis noted",
        "expectoration with blood",
        "blood-tinged expectorate",
        "frank hemoptysis",
        "minor hemoptysis",
        "cough with bloody discharge",
        "cough with blood"
    ],
    "Pleuritic chest pain": [
        "sharp chest pain",
        "pain with deep breaths",
        "chest pain when breathing",
        "stabbing pain in the chest",
        "chest pain aggravated by breathing or coughing",
        "worsening left-sided chest pain",
        "worsening right-sided chest pain",
        "worsening one-sided chest pain",
        "pleuritic chest discomfort",
        "inspiratory chest pain",
        "respiratory-related chest pain",
        "sharp, localized chest pain",
        "chest pain on inspiration",
        "pleuritic pain",
        "pleurisy"
    ],
    "Persistent worsening cough": [
        "chronic cough",
        "unrelenting cough",
        "recurring cough",
        "constant cough",
        "cough that won't go away",
        "progressively worsening cough",
        "persistent productive cough",
        "persistent cough",
        "cough, non-resolving",
        "refractory cough",
        "cough, persistent and worsening",
        "ongoing cough",
        "incessant cough",
        "cough worsening",
        "chronic dry cough"
    ],
    "Hoarseness": [
        "raspy voice",
        "strained voice",
        "scratchy throat",
        "gravelly voice",
        "loss of normal voice tone",
        "rough or husky voice",
        "hoarse voice",
        "husky voice",
        "rough voice",
        "husky or rough voice",
        "dysphonia",
        "voice changes",
        "altered vocal quality",
        "hoarseness present",
        "vocal hoarseness",
        "hoarse sounding voice"
    ],
    "Dyspnea": [
        "shortness of breath",
        "difficulty breathing",
        "breathlessness",
        "trouble catching breath",
        "air hunger",
        "labored breathing",
        "dyspnea on exertion",
        "resting dyspnea",
        "SOB (shortness of breath)",
        "increased work of breathing",
        "respiratory distress",
        "short of breath",
        "increased effort breathing"
    ],
    "Extreme fatigue": [
        "severe tiredness",
        "profound exhaustion",
        "overwhelming fatigue",
        "chronic fatigue",
        "feeling drained",
        "unusual tiredness",
        "chronic exhaustion",
        "generalized weakness",
        "debilitating fatigue",
        "malaise",
        "lethargy",
        "asthenia",
        "easy fatigability",
        "feeling exhausted",
        "feelings of exhaustion"
    ],
    "Cervical/Axillary lymphadenopathy": [
        "swollen lymph nodes in neck",
        "enlarged lymph nodes in the armpit",
        "lumps in neck or armpit",
        "tender lymph nodes",
        "swelling in lymph node regions",
        "palpable lymph nodes",
        "large lymph nodes",
        "persistent lumps",
        "lumps in the neck",
        "lumps in the armpit",
        "cervical lymphadenitis",
        "axillary lymph node enlargement",
        "palpable cervical/axillary nodes",
        "lymphadenopathy, cervical and axillary",
        "lymph node swelling",
        "lymphadenopathy",
        "lymphadenitis"
    ],
    "Swollen veins in the Neck & Chest": [
        "visible neck veins",
        "bulging veins in the chest",
        "engorged chest veins",
        "distended neck veins",
        "prominent veins on chest and neck",
        "jugular venous distension",
        "chest wall vein prominence",
        "superficial vein engorgement",
        "neck and chest vein distention",
        "vascular engorgement in neck and chest",
        "visible veins",
        "bulging veins",
        "JVD (Jugular Venous Distension)",
        "JVP (Jugular Venous Pressure)",
        "Jugular Venous Distension",
        "Jugular Venous Pressure",
        "JVD",
        "JVP"
    ],
    "Headache": [
        "severe headache",
        "persistent head pain",
        "throbbing headache",
        "continuous headache",
        "pressure in the head",
        "splitting headache",
        "migraine",
        "intense headache",
        "agonizing headache",
        "cephalgia",
        "headache, severe in nature",
        "unremitting headache",
        "chronic daily headache",
        "head pain",
        "debilitating headache",
        "headaches"
    ],
    "Facial and cervical edema": [
        "facial swelling",
        "neck swelling",
        "puffy face",
        "swollen neck",
        "swelling in face and neck",
        "enlarged facial tissues",
        "lymphatic obstruction in the facial region",
        "facial puffiness",
        "facial and neck edema",
        "facial edema",
        "neck edema",
        "periorbital edema",
        "generalized facial swelling",
        "non-pitting edema, face and neck",
        "facial and neck fullness",
        "swelling of the face",
        "swelling of the neck",
        "edema of the face and neck"
    ],
    "Loss of appetite": [
        "reduced appetite",
        "no interest in eating",
        "decreased hunger",
        "lack of desire for food",
        "eating less than usual",
        "avoiding meals",
        "decreased appetite",
        "less appetite",
        "decline in appetite",
        "appetite decline",
        "decreased appetite",
        "aversion to food",
        "diminished appetite",
        "anorexia",
        "poor oral intake",
        "inappetence",
        "hyporexia",
        "loss of desire to eat",
        "poor intake of food",
        "poor appetite"
    ],
    "Unexplained weight loss": [
        "unintentional weight loss",
        "sudden weight drop",
        "losing weight without trying",
        "unexpected slimming down",
        "rapid weight reduction",
        "unexplained weight drop",
        "sudden weight drop",
        "unexpected weight loss",
        "unexpected weight drop",
        "extreme weight loss",
        "extreme weight drop",
        "weight loss, unintentional",
        "significant weight loss",
        "cachexia",
        "unexplained decrease in weight",
        "weight reduction without diet or exercise",
        "decreased body mass",
        "weight loss present"
    ],
    "Bone pain": [
        "aching bones",
        "deep bone pain",
        "persistent bone discomfort",
        "sore bones",
        "pain within the bones",
        "bone tenderness",
        "osseous pain",
        "bone pain, localized",
        "generalized bone pain",
        "skeletal pain",
        "deep aching pain",
        "pain in the bones",
        "bone pain present",
        "widespread bone pain"
    ],
    "Hippocratic fingers": [
        "clubbing of fingers",
        "rounded fingertips",
        "bulbous finger tips",
        "enlarged finger ends",
        "curved nails with widened fingertips",
        "digital clubbing",
        "abnormal enlargement of the fingertips",
        "finger clubbing",
        "clubbing",
        "clubbing noted",
        "drumstick fingers",
        "watch-glass nails",
        "Hippocratic nails",
        "digital widening",
        "clubbed digits",
        "clubbing of the digits"
    ],
    "Jaundice": [
        "yellowing of the skin",
        "yellow eyes",
        "skin discoloration",
        "yellow-tinted skin",
        "icteric appearance",
        "yellow pigmentation",
        "yellowing of the eyes",
        "scleral icterus",
        "jaundice present",
        "hyperbilirubinemia",
        "yellowish discoloration of skin and sclera",
        "skin with yellow hue",
        "jaundice noted",
        "icterus present",
        "icterus"
    ],
    "Dysphagia": [
        "difficulty swallowing",
        "trouble eating",
        "pain when swallowing",
        "choking while eating",
        "hard to swallow",
        "difficulty passing food down throat",
        "odynophagia",
        "transfer dysphagia",
        "esophageal dysphagia",
        "swallowing impairment",
        "difficulty with solids or liquids",
        "painful swallowing",
        "difficulty swallowing solids",
        "difficulty swallowing liquids",
        "choking on food"
    ],
    "Ptosis": [
        "drooping eyelid",
        "sagging eyelid",
        "lowered eyelid",
        "eyelid hanging down",
        "partial eyelid closure",
        "falling upper eyelid",
        "falling right upper eyelid",
        "falling left upper eyelid",
        "blepharoptosis",
        "ptosis, right eye",
        "ptosis, left eye",
        "upper eyelid ptosis",
        "drooping of upper eyelid",
        "ptosis present",
        "droopy eyelid"
    ],
    "Ipsilateral Anhidrosis": [
        "lack of sweating on one side",
        "unilateral absence of sweating",
        "dry skin on one side",
        "one-sided sweat reduction",
        "non-sweating on one side of body",
        "anhidrosis on one side",
        "anhidrosis present on one side",
        "one-sided anhidrosis",
        "localized anhidrosis",
        "segmental anhidrosis",
        "reduced sweating on affected side",
        "absence of sweating, unilateral",
        "asymmetric sweating",
        "facial anhidrosis",
        "unilateral reduced sweating"
    ],
    "New-onset seizures": [
        "first-time seizures",
        "recent seizure activity",
        "new seizure episodes",
        "seizures starting recently",
        "initial seizures",
        "recently developed seizures",
        "recurrent seizures",
        "new seizures",
        "unprovoked seizure",
        "seizure, first episode",
        "de novo seizures",
        "recent onset of seizure disorder",
        "newly diagnosed seizure",
        "acute symptomatic seizure",
        "seizures",
        "seizure"
    ],
    "Ipsilateral Miosis": [
        "one-sided pupil constriction",
        "small pupil on one side",
        "narrow pupil on one side",
        "shrunken pupil on one side",
        "unilateral pupil constriction",
        "left-sided miosis",
        "right-sided miosis",
        "one pupil smaller than the other",
        "pupil smaller",
        "smaller pupil",
        "differing sizes of pupils",
        "miotic pupil, unilaterally",
        "constricted pupil on affected side",
        "asymmetric pupils",
        "unequal pupil size",
        "anisocoria",
        "miosis present"
    ]
}

# Constraints for symptoms (prevent harmful hallucinations)
symptom_constraints = {
    "Recurring lung infections": {
        "recommendations": [
            "Recommend a chest X-ray to check for signs of chronic infections or lung damage.",
            "Order sputum culture to identify the causative organism.",
            "Refer to a pulmonologist for further evaluation if recurrent infections persist."
        ],
        "avoid": ["generic antibiotics without identified cause"]
    },
    "Unexpected wheezing onset": {
        "recommendations": [
            "Perform spirometry or peak flow measurement to assess airflow obstruction.",
            "Consider a trial of bronchodilators (e.g., albuterol).",
            "Order a chest X-ray to rule out structural abnormalities."
        ],
        "avoid": ["steroids without diagnosing asthma or inflammation"]
    },
    "Hemoptysis": {
        "recommendations": [
            "Order a chest X-ray or CT scan to investigate potential causes (e.g., infection, malignancy).",
            "Refer to a pulmonologist for further evaluation.",
            "Request complete blood count and coagulation profile to rule out bleeding disorders."
        ],
        "avoid": ["antibiotics unless infection is confirmed"]
    },
    "Pleuritic chest pain": {
        "recommendations": [
            "Order a chest X-ray to rule out pleural effusion or pneumothorax.",
            "Perform an ECG to exclude cardiac causes.",
            "Prescribe NSAIDs for symptomatic relief if inflammation is confirmed."
        ],
        "avoid": ["opioids as first-line pain management"]
    },
    "Persistent worsening cough": {
        "recommendations": [
            "Request a chest X-ray to rule out infections or malignancy.",
            "Consider testing for tuberculosis in high-risk populations.",
            "Evaluate for asthma, GERD, or postnasal drip as potential causes."
        ],
        "avoid": ["cough suppressants without identifying underlying cause"]
    },
    "Hoarseness": {
        "recommendations": [
            "Recommend a laryngoscopy to examine the vocal cords.",
            "Advise voice rest and hydration for symptomatic relief.",
            "Refer to an ENT specialist if symptoms persist for more than two weeks."
        ],
        "avoid": ["antibiotics unless laryngitis is bacterial"]
    },
    "Dyspnea": {
        "recommendations": [
            "Perform spirometry to assess for obstructive or restrictive lung diseases.",
            "Order a chest X-ray or CT scan to evaluate for pulmonary or cardiac causes.",
            "Monitor oxygen saturation and provide supplemental oxygen if hypoxia is detected."
        ],
        "avoid": ["excessive physical exertion during episodes"]
    },
    "Extreme fatigue": {
        "recommendations": [
            "Order a complete blood count to check for anemia or infection.",
            "Screen for hypothyroidism using TSH levels.",
            "Evaluate for chronic fatigue syndrome if no other causes are identified."
        ],
        "avoid": ["stimulants without identifying the underlying cause"]
    },
    "Cervical/Axillary lymphadenopathy": {
        "recommendations": [
            "Perform a fine needle aspiration or biopsy for lymph nodes persisting over 4 weeks.",
            "Order a complete blood count and peripheral smear to rule out hematological malignancies.",
            "Consider imaging (e.g., ultrasound, CT) to evaluate the lymph node characteristics."
        ],
        "avoid": ["antibiotics unless infectious lymphadenopathy is suspected"]
    },
    "Swollen veins in the Neck & Chest": {
        "recommendations": [
            "Perform a CT angiography to evaluate for superior vena cava syndrome.",
            "Assess for thoracic malignancies or large mediastinal masses.",
            "Consider an echocardiogram to rule out cardiac causes."
        ],
        "avoid": ["diuretics without confirming fluid overload"]
    },
    "Headache": {
        "recommendations": [
            "Assess for red flags such as sudden onset, focal neurological signs, or worsening with Valsalva.",
            "Suggest an MRI or CT scan if concerning features are present.",
            "Treat migraines with triptans if diagnostic criteria are met."
        ],
        "avoid": ["routine opioids for headache management"]
    },
    "Facial and cervical edema": {
        "recommendations": [
            "Order a CT scan of the neck to evaluate for masses or lymphatic obstruction.",
            "Consider testing for thyroid dysfunction (TSH, T4).",
            "Assess for superior vena cava syndrome or venous obstruction."
        ],
        "avoid": ["empirical diuretics without a clear cause"]
    },
    "Loss of appetite": {
        "recommendations": [
            "Evaluate for gastrointestinal causes such as GERD or ulcers.",
            "Screen for depression or anxiety contributing to appetite loss.",
            "Consider a trial of appetite stimulants if no reversible cause is found."
        ],
        "avoid": ["force-feeding without addressing underlying issues"]
    },
    "Unexplained weight loss": {
        "recommendations": [
            "Order a comprehensive metabolic panel and thyroid function tests.",
            "Evaluate for malignancies with imaging (e.g., CT or PET scan).",
            "Screen for chronic infections such as tuberculosis or HIV."
        ],
        "avoid": ["nutritional supplements without identifying the underlying cause"]
    },
    "Bone pain": {
        "recommendations": [
            "Order imaging (e.g., X-ray, MRI) to assess for fractures, malignancies, or other abnormalities.",
            "Evaluate for osteoporosis in at-risk populations.",
            "Check calcium, phosphate, and vitamin D levels to rule out metabolic bone disease."
        ],
        "avoid": ["steroids unless inflammation or autoimmune causes are identified"]
    },
    "Hippocratic fingers": {
        "recommendations": [
            "Investigate for chronic hypoxia with pulse oximetry and arterial blood gas analysis.",
            "Order a chest X-ray or CT scan to assess for interstitial lung disease or malignancies.",
            "Refer to a pulmonologist or cardiologist based on findings."
        ],
        "avoid": ["empirical treatment without identifying cause"]
    },
    "Jaundice": {
        "recommendations": [
            "Order a liver function panel and ultrasound to evaluate for hepatic or biliary causes.",
            "Screen for hemolytic anemia with a complete blood count and reticulocyte count.",
            "Refer to a gastroenterologist for persistent or worsening jaundice."
        ],
        "avoid": ["empirical antibiotics unless infection is suspected"]
    },
    "Dysphagia": {
        "recommendations": [
            "Request a barium swallow or upper endoscopy to evaluate structural abnormalities.",
            "Order a modified barium swallow if neurological causes are suspected.",
            "Refer to a speech therapist for swallowing rehabilitation if indicated."
        ],
        "avoid": ["antibiotics unless infection is suspected"]
    },
    "Ptosis": {
        "recommendations": [
            "Perform a neurological exam to rule out Horner's syndrome or myasthenia gravis.",
            "Order imaging (e.g., MRI or CT) to assess for cranial nerve abnormalities.",
            "Refer to a neurologist for persistent or worsening ptosis."
        ],
        "avoid": ["empirical steroids without diagnosis"]
    },
    "Ipsilateral Anhidrosis": {
        "recommendations": [
            "Evaluate for Horner's syndrome with a thorough neurological and ophthalmological exam.",
            "Order imaging of the neck and thorax to assess for nerve compression or damage.",
            "Refer to a neurologist for further evaluation."
        ],
        "avoid": ["empirical antiperspirants without cause identification"]
    },
    "New-onset seizures": {
        "recommendations": [
            "Order an MRI of the brain and EEG to evaluate for structural or electrical abnormalities.",
            "Screen for metabolic causes (e.g., hypoglycemia, electrolyte imbalance).",
            "Refer to a neurologist for long-term management."
        ],
        "avoid": ["empirical anticonvulsants without diagnostic workup"]
    },
    "Ipsilateral Miosis": {
        "recommendations": [
            "Evaluate for Horner's syndrome with imaging of the neck and thorax.",
            "Perform a neurological exam to assess for additional cranial nerve deficits.",
            "Refer to a neurologist for further assessment."
        ],
        "avoid": ["empirical dilation drops without diagnosis"]
    },
}


# Select random symptoms for the clinical note
def select_random_symptoms(symptom_dict, num_symptoms):
    selected_symptoms = random.sample(list(symptom_dict.keys()), num_symptoms)
    symptoms_text = ", ".join([random.choice(symptom_dict[symptom]) for symptom in selected_symptoms])
    return selected_symptoms, symptoms_text


# Get constraints based on selected symptoms
def get_constraints_for_prompt(selected_symptoms):
    constraints_text = []
    for symptom in selected_symptoms:
        if symptom in symptom_constraints:
            constraint = symptom_constraints[symptom]
            recommendations = " ".join(constraint.get("recommendations", []))
            avoidances = " ".join(constraint.get("avoid", []))
            constraints_text.append(f"- {symptom}:\n  Recommendations: {recommendations}\n  Avoid: {avoidances}")
    return "\n".join(constraints_text)


def extract_and_validate_symptoms(note, selected_symptoms, symptom_synonyms):
    # Extract symptoms wrapped in <<...>> and validate against selected symptoms using fuzzywuzzy matching.
    pattern = r'<<(.*?)>>'
    matches = re.finditer(pattern, note)
    
    found_spans = []
    valid_symptoms = set()
    found_symptoms = []
    FUZZY_THRESHOLD = 80  # 80% 
    
    # Extract spans and validate symptoms
    for match in matches:
        span_text = match.group(1)
        start_idx = match.start(1)
        end_idx = match.end(1)
        found_symptoms.append(span_text)
        
        # Check against each symptom and its synonyms using fuzzywuzzy matching
        matched_symptom = None
        highest_ratio = 0
        for symptom in selected_symptoms:
            variants = [symptom] + symptom_synonyms.get(symptom, [])
            
            for variant in variants:
                ratio = fuzz.token_set_ratio(span_text.lower(), variant.lower())
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    if ratio >= FUZZY_THRESHOLD:
                        matched_symptom = symptom
        
        if matched_symptom:
            found_spans.append({
                "span": span_text,
                "start": start_idx - 2,
                "end": end_idx + 2,
                "label": "SYMPTOM",
                "matched_original": matched_symptom,
                "match_confidence": highest_ratio
            })
            valid_symptoms.add(matched_symptom)
            # Log successful symptom match with confidence score for validation tracking
            logging.info(f"Matched symptom '{span_text}' to '{matched_symptom}' with {highest_ratio}% confidence")
        else:
            # Log unmatched symptoms to identify potential fuzzywuzzy matching issues
            logging.warning(f"Found unmatched symptom marker: '{span_text}' (highest similarity: {highest_ratio}%)")
    
    # Check for missing symptoms
    missing_symptoms = set(selected_symptoms) - valid_symptoms
    if missing_symptoms:
        logging.warning(f"Missing symptoms in note: {', '.join(missing_symptoms)}") # Showcase missing symptoms (if any)
        return None
    
    # Process valid note
    clean_note = re.sub(r'<<|>>', '', note)
    
    # Adjust spans for removed markers
    adjusted_spans = []
    marker_count = 0
    for span in found_spans:
        adjusted_start = span["start"] - (marker_count * 4)
        adjusted_end = span["end"] - (marker_count * 4) - 4
        
        adjusted_spans.append({
            "span": span["span"],
            "start": adjusted_start,
            "end": adjusted_end,
            "label": "SYMPTOM",
            "matched_original": span["matched_original"],
            "match_confidence": span["match_confidence"]
        })
        marker_count += 1
    
    return {
        "text": clean_note,
        "spans": adjusted_spans
    }


def save_human_readable_clinical_note(note, note_id, metadata_path="clinical_notes/metadata.jsonl"):
    # Saves clean version without markup
    clean_note = re.sub(r'<<|>>', '', note)
    
    raw_note_path = f"clinical_notes/raw/note_{note_id}.txt"
    os.makedirs("clinical_notes/raw", exist_ok=True)
    
    with open(raw_note_path, "w") as note_file:
        note_file.write(clean_note)
    
    metadata = {
        "note_id": note_id,
        "raw_note_path": raw_note_path,
        "labeled_path": f"clinical_notes/labeled/note_{note_id}.json"
    }
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "a") as meta_file:
        meta_file.write(json.dumps(metadata) + "\n")


def save_labeled_clinical_note(labeled_data, note_id, metadata_path="clinical_notes/metadata.jsonl"):
    # Saves version with symptom annotations and metadata
    labeled_path = f"clinical_notes/labeled/note_{note_id}.json"
    os.makedirs("clinical_notes/labeled", exist_ok=True)
    
    with open(labeled_path, "w") as f:
        json.dump(labeled_data, f, indent=2)
    
    if not os.path.exists(metadata_path):
        metadata = {
            "note_id": note_id,
            "raw_note_path": f"clinical_notes/raw/note_{note_id}.txt",
            "labeled_path": labeled_path
        }
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "a") as meta_file:
            meta_file.write(json.dumps(metadata) + "\n")


def generate_clinical_note(model, note_id):
    while True:
        num_symptoms = random.randint(2, 5)
        selected_symptoms, symptoms_text = select_random_symptoms(symptom_synonyms, num_symptoms)
        patient_name = generate_random_name()
        constraints_text = get_constraints_for_prompt(selected_symptoms)

        logging.info(f"Generating note {note_id} - Symptoms ({num_symptoms}): {', '.join(selected_symptoms)}") # Log symptoms before prompt generation
        
        # Example clinical note for better model guidance
        example_note = """
*Clinical Note*

Patient Name: Sarah Thompson  
Date: 2025-01-01  

Subjective:  
- Chief Complaint: <<Hippocratic fingers>> and <<jaundice>>.  
- History of Present Illness:  
  Sarah Thompson, a 42-year-old female, presents with the above symptoms. She describes a gradual onset of changes over the past few weeks. She denies any significant recent events or exposures. Family history is notable for relevant conditions.

Objective:  
- Vital Signs: BP: 128/76 mmHg, HR: 78 bpm, RR: 16 breaths/min, Temp: 98.7°F.  
- Physical Examination Findings: Consistent with the symptoms described.  

Assessment:  
1. Chronic condition contributing to the reported symptoms.  
2. Potential systemic causes requiring further investigation.  
3. Rule out underlying conditions, including malignancy or other organ system involvement.  

Plan:  
1. Diagnostics:  
   - Order relevant laboratory tests to evaluate organ function and systemic markers.  
   - Conduct imaging to assess potential underlying causes.  

2. Referrals:  
   - Refer to specialists as appropriate for further workup.  

3. Treatment:  
   - Initiate general supportive care measures.  
   - Provide guidance on lifestyle adjustments and symptom monitoring.  

4. Follow-Up:  
   - Schedule follow-up in one week to review results and reassess.  
   - Educate the patient on warning signs requiring immediate medical attention.  
   - Maintain open communication for any new or worsening concerns.  
"""
        
        # Model prompt that instructs it to use <<...>> markers around symptoms
        prompt = f"""
You are a trusted medical assistant tasked with creating realistic and concise clinical notes for patients. Below is an example clinical note to guide your structure and format:

{example_note}

Now, generate a clinical note based on the following information:

Patient Name: {patient_name}
Symptoms: {symptoms_text}

CRITICAL INSTRUCTION: You MUST wrap EVERY SINGLE symptom mentioned in the symptoms above with << >> markers.
Example of correct symptom formatting: Patient presents with <<frequent respiratory infections>> and <<chronic cough>>.

Guidelines for the note:
- Subjective: State the chief complaint and provide a concise history of present illness, including symptom timeline, associated factors, and relevant context.
- Objective: Document key findings from the physical exam, including vital signs and observations (e.g., "BP: 120/80 mmHg, bilateral wheezing").
- Assessment: List likely diagnoses or differential diagnoses with brief reasoning for each.
- Plan: Outline actionable steps, including diagnostics, treatments, referrals, and follow-up plans.

Additional constraints for the symptoms:
{constraints_text}

Requirements for the note:
- Please ensure your finished clinical note is in the same structure as the provided example clinical note.
- EACH symptom MUST be wrapped in << >> markers. FAILURE TO DO THIS WILL RESULT IN UNSATISFACTORY OUTCOMES.
- All symptoms must be included in the final clinical note, and their phrasing should remain consistent with or closely resemble the original format. 
- UNDER NO CIRCUMSTANCES should you include a disclaimer of ANY KIND in the finished clinical note.
- Respond only with your finished clinical note.
"""
        clinical_note = query_model(model, prompt)
        labeled_data = extract_and_validate_symptoms(clinical_note, selected_symptoms, symptom_synonyms)
        
        if labeled_data:
            save_human_readable_clinical_note(clinical_note, note_id)
            save_labeled_clinical_note(labeled_data, note_id)
            logging.info(f"SUCCESS: Generated and validated note {note_id}")
            return clinical_note
        else:
            logging.warning(f"FAILURE: Regenerating note {note_id} due to validation failure")
            # Continues generating until a valid note is produced


# Generate the specified number of clinical notes in num_notes (10k)
def generate_multiple_notes(num_notes):
    model = Llama(
        model_path=mistral_model_path,
        n_gpu_layers=33, # Layers needed to load the entire LLM onto the GPU
        n_ctx=mistral_context_limit
    )
    for i in range(1, num_notes + 1):
        generate_clinical_note(model, note_id=i)
    del model

if __name__ == "__main__":
    num_notes = 10000 # 10k clinical notes
    generate_multiple_notes(num_notes)
