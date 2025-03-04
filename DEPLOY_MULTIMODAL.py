# Import required libraries
import streamlit as st
from streamlit_option_menu import option_menu
import os
import glob
import shutil
import pandas as pd
import time
from datetime import datetime
from PIL import Image
import PROCESS_X_RAY 
import PROCESS_CT_SCAN 
import DEPLOY_X_RAY 
import DEPLOY_CT_SCAN 
import DEPLOY_SYMPTOM 
from streamlit.components.v1 import html

# Cleans up directories for re-runs
def cleanup_directories():
    directories = [
        './Pre_Input_X-ray/*', './Input_X-ray/*', 
        './Pre_Input_CT-Scan/*', './Input_CT-Scan/*',
        './XAI_Output_1/*', './XAI_Output_2/*'
    ]
    for pattern in directories:
        for f in glob.glob(pattern):
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)

# Ensures needed folders are created
def ensure_directories():
    for dir_ in ['./Pre_Input_X-ray', './Input_X-ray', 
                 './Pre_Input_CT-Scan', './Input_CT-Scan',
                 './XAI_Output_1', './XAI_Output_2']:
        os.makedirs(dir_, exist_ok=True)

# Processes and predicts using all 3 modalities
def process_and_predict():
    progress = st.progress(0)
    status = st.empty()
    results = {}
    time.sleep(0.3)

    # Process and predict for each modality
    progress.progress(10)
    if os.path.exists('./Pre_Input_X-ray') and os.listdir('./Pre_Input_X-ray'):
        status.text("Processing X-Ray...")
        PROCESS_X_RAY.process()
        status.text("Running X-Ray analysis...")
        DEPLOY_X_RAY.predict()
        results['xray_results'] = DEPLOY_X_RAY.prediction_results
        progress.progress(21)

    if os.path.exists('./Pre_Input_CT-Scan') and os.listdir('./Pre_Input_CT-Scan'):
        status.text("Processing CT Scan...")
        PROCESS_CT_SCAN.process()
        status.text("Running CT Scan analysis...")
        DEPLOY_CT_SCAN.predict()
        results['ct_results'] = DEPLOY_CT_SCAN.prediction_results
        progress.progress(35)
        time.sleep(0.2)
        progress.progress(66)

    if st.session_state.get('clinical_text'):
        status.text("Analyzing clinical notes...")
        DEPLOY_SYMPTOM.Clinical_Note = st.session_state.clinical_text
        DEPLOY_SYMPTOM.predict()
        results['symptom_results'] = DEPLOY_SYMPTOM.prediction_results
        progress.progress(80)
        time.sleep(0.1)
        progress.progress(100)

    # Integrate results only after all modalities are processed
    integrated_results = integrate_prediction_results(
        results.get('ct_results', {}),
        results.get('xray_results', {}),
        results.get('symptom_results', {})
    )

    status.text("Analysis complete!")
    st.session_state.button_clicked = False
    block=True
    time.sleep(1)
    st.session_state.results = integrated_results
    st.rerun()
    return integrated_results

def calculate_base_weights(
    ct_class: str, 
    x_class: str, 
    risk_level: str
) -> dict:
    # CT modality weight calculation (50%)
    ct_cancer_types = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma'] 
    ct_weight = 50 if ct_class in ct_cancer_types else 0 
    
    # X-ray modality weight calculation (40%)
    x_weight = 40 if x_class == 'Nodule' else 0 
     
    # Risk level weight mapping for symptoms
    risk_weight_map = {
        'High Risk': 10.0,
        'Moderate Risk': 7.5,
        'Low Risk': 5.0
    }
    risk_weight = risk_weight_map.get(risk_level, 5.0)  # Default to Low Risk if unknown (JUST IN CASE)
    
    weights = {
        'ct_weight': ct_weight,
        'xray_weight': x_weight,
        'symptom_weight': risk_weight,
        'total': ct_weight + x_weight + risk_weight
    }
    
    # Debug output for base weights (shoul be somewhat consistent across differnet runs as long as predictions remain the same)
    print("-"*25)
    print(f"[DEBUG] CT Weight: {ct_weight}%")
    print(f"[DEBUG] X-Ray Weight: {x_weight}%")
    print(f"[DEBUG] Symptom Weight: {risk_weight}%")
    print(f"[DEBUG] Total Base Weight: {weights['total']}%")

    return weights

def calculate_confidence_adjustments(
    ct_confidence: float, 
    x_confidence: float, 
    s_confidence: float,
    risk_level: str  
) -> dict:
    # Define baseline weights with dynamic symptom weight stuff
    risk_weight_map = {
        'High Risk': 10.0,
        'Moderate Risk': 7.5,
        'Low Risk': 5.0
    }
    symptom_weight = risk_weight_map.get(risk_level, 5.0)
    
    baseline_weights = {
        'CT': 50,
        'X-ray': 40,
        'Symptoms': symptom_weight 
    }
    
    confidence_list = [
        {'modality': 'CT', 'confidence': ct_confidence},
        {'modality': 'X-ray', 'confidence': x_confidence}, 
        {'modality': 'Symptoms', 'confidence': s_confidence} 
    ] 
    
    # Sort confidences from highest to lowest
    sorted_confidences = sorted(confidence_list, key=lambda x: x['confidence'], reverse=True)
    
    # Calculate proportional adjustments
    adjustments = {
        'highest': {
            'modality': sorted_confidences[0]['modality'],
            'adjustment': baseline_weights[sorted_confidences[0]['modality']] * 0.2
        },
        'middle': {
            'modality': sorted_confidences[1]['modality'],
            'adjustment': baseline_weights[sorted_confidences[1]['modality']] * 0.1
        },
        'lowest': {
            'modality': sorted_confidences[2]['modality'],
            'adjustment': baseline_weights[sorted_confidences[2]['modality']] * 0.05
        }
    }
    
    # Calculate total adjustment
    total_adjustment = (adjustments['highest']['adjustment'] + 
                       adjustments['middle']['adjustment'] + 
                       adjustments['lowest']['adjustment'])
    
    adjustments['total'] = total_adjustment
    
    # Debug output for confidence adjustments
    print("-"*25)
    print(f"[DEBUG] CT Confidence: {ct_confidence}%")
    print(f"[DEBUG] X-Ray Confidence: {x_confidence}%")
    print(f"[DEBUG] Symptom Confidence: {s_confidence}%")
    
    print("-"*25)
    print(f"[DEBUG] Highest Confidence ({adjustments['highest']['modality']}): +{adjustments['highest']['adjustment']:.2f}%")
    print(f"[DEBUG] Middle Confidence ({adjustments['middle']['modality']}): +{adjustments['middle']['adjustment']:.2f}%")
    print(f"[DEBUG] Lowest Confidence ({adjustments['lowest']['modality']}): +{adjustments['lowest']['adjustment']:.2f}%")
    print(f"[DEBUG] Total Adjustment: +{adjustments['total']:.2f}%")
    
    return adjustments

def calculate_risk_weight_adjustment(
    ct_class: str, 
    x_class: str, 
    risk_level: str
) -> float:
    weights = calculate_base_weights(ct_class, x_class, risk_level)
    return weights['total']

def calculate_modality_confidence_adjustment(
    ct_confidence: float, 
    x_confidence: float, 
    s_confidence: float,
    risk_level: str
) -> float:
    adjustments = calculate_confidence_adjustments(ct_confidence, x_confidence, s_confidence, risk_level)
    return adjustments['total']

def calculate_overall_cancer_suspicion(results):
    
    # Get predictions
    ct_pred = results.get('ct_prediction', '').strip()
    x_pred = results.get('xray_prediction', '').strip()
    
    # Get clinical results
    clinical_results = results.get('clinical_results', {})
    risk_level = clinical_results.get('risk_level', 'Low Risk')
    
    # Get confidence scores
    ct_confidence = results.get('ct_confidence', 0)
    x_confidence = results.get('xray_confidence', 0)
    s_confidence = clinical_results.get('s_confidence', 0) 
    
    print("-"*25)
    print(f"[DEBUG] CT Prediction: {ct_pred}")
    print(f"[DEBUG] X-Ray Prediction: {x_pred}")
    print(f"[DEBUG] Symptom Risk Level: {risk_level}")

    
    # Risk weight adjustment shit
    risk_weight = calculate_risk_weight_adjustment(ct_pred, x_pred, risk_level)
    
    # Calculate confidence adjustment with risk level 
    confidence_adjustment = calculate_modality_confidence_adjustment(
        ct_confidence, x_confidence, s_confidence, risk_level
    )
    
    # Combine confidence adjustment and weight
    total_score = risk_weight + confidence_adjustment
    
    print("-"*25)
    print(f"[DEBUG] Base Risk Weight: {risk_weight}%")
    print(f"[DEBUG] Confidence Adjustment: {confidence_adjustment}%")
    print(f"[DEBUG] Final Total Score: {total_score}%")
    
    # Diagnosis logic
    is_suspicious = total_score > 100
    
    # Suspicion descriptions
    suspicion_description = (
        "Lung Cancer is Suspected" if is_suspicious else 
        "Results are Inconclusive"
    )

    print("-"*25)
    print(f"\n[DEBUG] Final Assessment: {suspicion_description}")
    
    return is_suspicious, suspicion_description, total_score

def integrate_prediction_results(
    ct_results: dict, 
    x_results: dict, 
    s_results: dict
) -> dict:
    
    # Combine results
    integrated_results = {}
    
    # Add CT results
    if ct_results:
        print("\nAdding CT Results...")
        integrated_results.update({
            'ct_prediction': ct_results.get('ct_prediction', ''),
            'ct_confidence': ct_results.get('ct_confidence', 0)
        })
    
    # Add X-ray results
    if x_results:
        print("\nAdding X-Ray Results...")
        integrated_results.update({
            'xray_prediction': x_results.get('xray_prediction', ''),
            'xray_confidence': x_results.get('xray_confidence', 0)
        })
    
    # Add Symptom results
    if s_results and 'clinical_results' in s_results:
        print("\nAdding Clinical Results...")
        clinical_results = s_results['clinical_results']
        integrated_results['clinical_results'] = clinical_results
    
    # Calculate cancer suspicion level
    print("\nCalculating Overall Cancer Suspicion Level...")
    is_suspicious, suspicion_description, adjusted_confidence = calculate_overall_cancer_suspicion(integrated_results)
    
    # Add suspicion details
    integrated_results.update({
        'is_suspicious': is_suspicious,
        'suspicion_description': suspicion_description,
        'adjusted_confidence': adjusted_confidence
    })

    return integrated_results

# Generative Medical Report
def display_medical_report(results, patient_name):

    st.title("Medical Analysis Report")
    display_name = patient_name if patient_name else "Anonymous"

    def stream_data01():
     
     name= f"**Patient Name:** {display_name}"
     for word in name.split(" "):
            yield word + " "
            time.sleep(0.1)
    
    def stream_data02():
     date=f"**Date of Test:** {datetime.now().strftime('%B %d, %Y')}"
     for word in date.split(" "):
            yield word + " "
            time.sleep(0.1)

    
    # Patient and Report Header
    col1, col2 = st.columns([3, 1])
    with col1:
        
        st.write_stream(stream_data01)
        st.write_stream(stream_data02)
    with col2:
        # Download report as txt file
        report_text = generate_detailed_report_text(results, display_name)
        st.download_button(
            "Download Report",
            report_text,
            f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )

    # Add cancer suspicion section 
    is_suspicious = results.get('is_suspicious', False)
    suspicion_description = results.get('suspicion_description', 'Results Pending')
    adjusted_confidence = results.get('adjusted_confidence', 0)
    
    if is_suspicious:
        st.error(f" **{suspicion_description}** (Confidence: {adjusted_confidence:.2f}%) ")
    else:
        st.warning(f" **{suspicion_description}** (Confidence: {adjusted_confidence:.2f}%)")

    st.divider()

    # 1. X-Ray Analysis Section
    if 'xray_prediction' in results:
    
        def stream_data1():
          st.header("1. X-Ray Analysis")
          x_ray=(f"""
            
           The AI model predicted the presence of **{results['xray_prediction']}** based on the X-ray scan provided with {results['xray_confidence']:.2f}% confidence. 
           To help better understand this prediction, the image has been overlaid with color-coded highlights. 
           Green areas indicate regions that increased the model’s confidence in the diagnosis, while red areas highlight regions that reduced the confidence level. The top 5 super-pixels have been overlaid on the X-ray image provided. 
            
            """)

          for word in x_ray.split(" "):
            yield word + " "
            time.sleep(0.02)

        def stream_data101():
          x_ray2=(f"""The green regions represent features commonly associated with **{results['xray_prediction']}**, such as specific patterns, densities, or structural changes that align with the diagnosis. 
          These areas of the image show where the model found characteristics that strongly support the presence of **{results['xray_prediction']}**. 
          The more green areas that appear, the higher the confidence the model has in this result. On the other hand, the red areas represent regions that made the model less confident in the prediction. 
          These could be areas of the image where the features do not match the typical patterns seen in the predicted result, or where the model found inconsistencies in the data that lowered the overall confidence level. 
          
          """)

          for word in x_ray2.split(" "):
            yield word + " "
            time.sleep(0.02)

        def stream_data103():
          x_ray4=(f"""The combination of green and red areas offers a visual understanding of how the model arrived at its conclusion. 
          The green areas suggest a strong indication of the predicted outcome, while the red areas highlight areas that might have contributed to uncertainty. 
          This visual guide helps you better understand the specific parts of the X-ray that influenced the model’s prediction, providing a clearer picture of the result.
          """)

          for word in x_ray4.split(" "):
            yield word + " "
            time.sleep(0.02)


        col1, col2 = st.columns([2, 1])
        with col1:
           st.write_stream(stream_data1)
        
        
        with col2:
            # Small version of LIME Overlay with expandable original
            if os.path.exists('./XAI_Output_1/X-Ray_LIME_Overlay.png'):
                image = Image.open('./XAI_Output_1/X-Ray_LIME_Overlay.png')
                st.image(image, caption="X-Ray Region Overlay", use_container_width=True)

        st.write_stream(stream_data101)

        st.write_stream(stream_data103)

        def stream_data2():
         lime= """
                The bar chart provides an overview of how different regions contribute to the overall prediction. It summarises the region's contributions highlighted in the overlay chart. 
                Each bar represents a specific region, and the height of the bar indicates the strength of that region’s contribution. 
                Green corresponds to regions that have a reinforcing effect on the model’s prediction, effectively supporting the decision. 
                Red represents regions that work against the prediction, indicating areas that challenge or contradict the outcome. 
                """
         for word in lime.split(" "):
            yield word + " "
            time.sleep(0.02)

        def stream_data3():
          meta= """
                The meta-learner influence graph provides a clear visualization of how the base models contribute to the final prediction. 
                It displays the relative impact and effect of each individual model, showing the extent to which each one influences the final model and overall decision. 
                By analyzing this graph, insights are gained into the strengths and weaknesses of the models involved, as well as understanding how their combined contributions shape the final prediction outcome.""" 

          for word in meta.split(" "):
            yield word + " "
            time.sleep(0.02)
        
        
        # Bar chart and meta learner influence
        col1, col2 = st.columns(2)
        with col1:
            # LIME Bar Chart
            if os.path.exists('./XAI_Output_1/X-Ray_LIME_Bar.png'):
                st.image('./XAI_Output_1/X-Ray_LIME_Bar.png', 
                         caption="Feature Importance Analysis", 
                         use_container_width=True)
                
        
        with col2:
            # Meta Learner Influence
            st.write_stream(stream_data2)
            if os.path.exists('./XAI_Output_1/X-Ray_Meta_Learner_Influence.png'):
                st.image('./XAI_Output_1/X-Ray_Meta_Learner_Influence.png', 
                         caption="Model Contribution Analysis", 
                         use_container_width=True)
        with col1:      
                st.write_stream(stream_data3)
        
        st.divider()

    # 2. CT Scan Analysis Section
    if 'ct_prediction' in results:

        def stream_data4():
          st.header("2. CT Scan Analysis")

          ct_scan =(f"""The AI model analyzed the CT scan provided and predicted a **{results['ct_prediction']}** diagnosis with {results['ct_confidence']:.2f}% confidence. 
          This level of confidence indicates the model's certainty, with higher confidence suggesting stronger evidence supporting the prediction. 
          To help better understand this result, a heatmap is developed to highlight the areas that most influenced the prediction. It uses a color-gradient to represent the relevance of different regions within the scan. """)
        
          for word in ct_scan.split(" "):
            yield word + " "
            time.sleep(0.02)

        def stream_data401():
          ct_1=(f""" Warmer colors, such as red and yellow, indicate the regions most strongly associated with the predicted diagnosis, showing where the model detected features that align with a **{results['ct_prediction']}** outcome. 
          In particular, the regions highlighted in warmer colors reflect tissue densities and structural patterns that suggest no signs of abnormalities or support for the predicted condition, depending on the diagnosis. 
          These areas represent healthy tissue or other patterns that reinforce the prediction. 
          The heatmap serves as a visual guide to show which parts of the scan contributed most to the prediction, helping analyse how the model arrived at its conclusion. 
          This visualization of the key factors that influenced the prediction offers greater clarity in interpreting the results.""")

          for word in ct_1.split(" "):
            yield word + " "
            time.sleep(0.02)

        
        col1, col2 = st.columns([2, 1])
        with col1:
           st.write_stream(stream_data4)

        with col2:
            # Small version of GradCAM++
            if os.path.exists('./XAI_Output_2/CT_GradCAM++.png'):
                image = Image.open('./XAI_Output_2/CT_GradCAM++.png')
                st.image(image, caption="CT Scan Heatmap", use_container_width=True)

        st.write_stream(stream_data401)

        
        # CT Overlay and Bar Chart
        col1, col2 = st.columns(2)

        def stream_data5():
         ct_overlay= """
                The overlay on the CT scan pinpoints specific regions that increase or decrease the likelihood of the prediction. The combination of green and red areas offers a visual understanding of how the model arrived at its conclusion. 
                The overlay acts as a visual tool to indicate how different areas of the scan have contributed to the model’s decision. 
                It highlights regions that either support or challenge the prediction, offering a more transparent view of the underlying factors that influenced the diagnosis. 
                This helps in understanding which structures or abnormalities played a significant role in shaping the model’s conclusion. 
                Whether it’s an indication of normalcy or the presence of a condition, the overlay allows for easy identification of critical areas in the CT scan that contributed to the final prediction. 
                 """
         for word in ct_overlay.split(" "):
            yield word + " "
            time.sleep(0.02)

        def stream_data6():
          ct_bar= """
                The bar chart provides a detailed summary of the regional contributions to the CT scan prediction. It visually represents the impact of different regions within the scan, showing how each area either supports or challenges the final prediction. 
                This graphical representation works in conjunction with the overlay image, offering a clearer understanding of the regions that influenced the AI model’s decision-making process. 
                Each bar in the chart corresponds to a specific region of the scan and indicates the degree to which that region contributed to the prediction. 
                It makes it easier to compare the contributions of different regions, allowing for a more precise understanding of which areas played a more significant role in supporting or opposing the predicted diagnosisPositive values suggest areas that reinforce the predicted diagnosis, while negative values represent regions that reduce the model’s confidence in the prediction.

                """
          for word in ct_bar.split(" "):
            yield word + " "
            time.sleep(0.02)


        with col1:
            if os.path.exists('./XAI_Output_2/CT_LIME_Overlay.png'):
                st.image('./XAI_Output_2/CT_LIME_Overlay.png', 
                         caption="CT Scan Region Overlay", 
                         use_container_width=True)
                st.write_stream(stream_data5)
        
        with col2:
            if os.path.exists('./XAI_Output_2/CT_LIME_Bar.png'):
                st.image('./XAI_Output_2/CT_LIME_Bar.png', 
                         caption="Region Contribution Analysis", 
                         use_container_width=True)
                st.write_stream(stream_data6)
        
        st.divider()

    # 3. Clinical Symptom Analysis 
    if 'clinical_results' in results:
        clinical_data = results['clinical_results']
        st.header("3. Clinical Symptom Analysis")

        def stream_data7():
         symptom_analysis= f"""
        The patient's clinical notes were analyzed to identify relevant symptoms and assess risk. 
        This thorough analysis involved extracting key symptoms that could provide important context for the diagnosis. 
        Several key symptoms were detected, including {', '.join(symptoms_list)}, which were then grouped into clusters based on their relationship to each other and their potential connection to specific conditions. 
        These clusters of symptoms help to create a more comprehensive diagnostic context, enabling the model to evaluate the patient's condition more effectively.
        
        **Risk Assessment**
        - Risk Level: **{clinical_data['risk_level']}**
        - Risk Weight: **{clinical_data['risk_weight']}%**
        """

         for word in symptom_analysis.split(" "):
            yield word + " "
            time.sleep(0.07)

        def stream_data8():
          symptom_cluster= f"""
                **{cluster['name']}**
                - Symptoms: {', '.join(cluster['matching_symptoms'])}
                - Clinical Significance: {cluster['reasoning']}
                """

          for word in symptom_cluster.split(" "):
            yield word + " "
            time.sleep(0.07)

        
        # Clinical Analysis Text Report
        symptoms_list = [f"**{s['head_symptom']}**" for s in clinical_data['symptoms']]

        st.write_stream(stream_data7)
        
        # Symptom Clusters
        if clinical_data['clusters']:
            for cluster in clinical_data['clusters']:

                st.write_stream(stream_data8)
        
        # Expandable Clinical Note Annotation 
        with st.expander("View Annotated Clinical Note"):
            clinical_text = st.session_state.clinical_text
            final_text = clinical_text
            
            # Process all symptoms and sort by start index in reverse
            highlights = []
            for symptom in clinical_data['symptoms']:
                highlights.append({
                    'start': symptom['start'],
                    'end': symptom['end'],
                    'text': symptom['text'],
                    'head_symptom': symptom['head_symptom']
                })
            
            # Sort in reverse order to mark from end to start
            highlights.sort(key=lambda x: x['start'], reverse=True)
            
            # Apply highlighting 
            for highlight in highlights:
                before = final_text[:highlight['start']]
                after = final_text[highlight['end']:]
                highlighted = final_text[highlight['start']:highlight['end']]
                final_text = before + f"<mark title='{highlight['head_symptom']}'>{highlighted}</mark>" + after
            
            st.markdown(final_text, unsafe_allow_html=True)

# Saved Generative Medical Report
def display_medical_report2(results, patient_name):

    st.title("Medical Analysis Report")
    display_name = patient_name if patient_name else "Anonymous"

    def stream_data01():
     
     name= f"**Patient Name:** {display_name}"
     for word in name.split(" "):
            yield word + " "
            time.sleep(0)
    
    def stream_data02():
     date=f"**Date of Test:** {datetime.now().strftime('%B %d, %Y')}"
     for word in date.split(" "):
            yield word + " "
            time.sleep(0)

    
    # Patient and Report Header
    col1, col2 = st.columns([3, 1])
    with col1:
        
        st.write_stream(stream_data01)
        st.write_stream(stream_data02)
    with col2:
        # Download report as txt file
        report_text = generate_detailed_report_text(results, display_name)
        st.download_button(
            "Download Report",
            report_text,
            f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )

    # Add cancer suspicion section 
    is_suspicious = results.get('is_suspicious', False)
    suspicion_description = results.get('suspicion_description', 'Results Pending')
    adjusted_confidence = results.get('adjusted_confidence', 0)
    
    if is_suspicious:
        st.error(f" **{suspicion_description}** (Confidence: {adjusted_confidence:.2f}%) ")
    else:
        st.warning(f" **{suspicion_description}** (Confidence: {adjusted_confidence:.2f}%)")

    st.divider()

    # 1. X-Ray Analysis Section 
    if 'xray_prediction' in results:
    
        def stream_data1():
          st.header("1. X-Ray Analysis")
          x_ray=(f"""
            
           The AI model predicted the presence of **{results['xray_prediction']}** based on the X-ray scan provided with {results['xray_confidence']:.2f}% confidence. 
           To help better understand this prediction, the image has been overlaid with color-coded highlights. 
           Green areas indicate regions that increased the model’s confidence in the diagnosis, while red areas highlight regions that reduced the confidence level. The top 5 super-pixels have been overlaid on the X-ray image provided. 
            
            """)

          for word in x_ray.split(" "):
            yield word + " "
            time.sleep(0.0)

        def stream_data101():
          x_ray2=(f"""The green regions represent features commonly associated with **{results['xray_prediction']}**, such as specific patterns, densities, or structural changes that align with the diagnosis. 
          These areas of the image show where the model found characteristics that strongly support the presence of **{results['xray_prediction']}**. 
          The more green areas that appear, the higher the confidence the model has in this result. On the other hand, the red areas represent regions that made the model less confident in the prediction. 
          These could be areas of the image where the features do not match the typical patterns seen in the predicted result, or where the model found inconsistencies in the data that lowered the overall confidence level. 
          
          """)

          for word in x_ray2.split(" "):
            yield word + " "
            time.sleep(0.0)

        

        def stream_data103():
          x_ray4=(f"""The combination of green and red areas offers a visual understanding of how the model arrived at its conclusion. 
          The green areas suggest a strong indication of **{results['xray_prediction']}**, while the red areas highlight areas that might have contributed to uncertainty. 
          This visual guide helps you better understand the specific parts of the X-ray that influenced the model’s prediction, providing a clearer picture of the result.
          """)

          for word in x_ray4.split(" "):
            yield word + " "
            time.sleep(0.0)


        col1, col2 = st.columns([2, 1])
        with col1:
           st.write_stream(stream_data1)
        
        
        with col2:
            # Small version of LIME Overlay with expandable original
            if os.path.exists('./XAI_Output_1/X-Ray_LIME_Overlay.png'):
                image = Image.open('./XAI_Output_1/X-Ray_LIME_Overlay.png')
                st.image(image, caption="X-Ray Region Overlay", use_container_width=True)

        st.write_stream(stream_data101)

        st.write_stream(stream_data103)

        def stream_data2():
         lime= """
                The bar chart provides an overview of how different regions contribute to the overall prediction. It summarises the region's contributions highlighted in the overlay chart. 
                Each bar represents a specific region, and the height of the bar indicates the strength of that region’s contribution. 
                Green corresponds to regions that have a reinforcing effect on the model’s prediction, effectively supporting the decision. 
                Red represents regions that work against the prediction, indicating areas that challenge or contradict the outcome. 
                """
         for word in lime.split(" "):
            yield word + " "
            time.sleep(0.0)

        def stream_data3():
          meta= """
                The meta-learner influence graph provides a clear visualization of how the base models contribute to the final prediction. 
                It displays the relative impact and effect of each individual model, showing the extent to which each one influences the final model and overall decision. 
                By analyzing this graph, insights are gained into the strengths and weaknesses of the models involved, as well as understanding how their combined contributions shape the final prediction outcome. 
                """
          for word in meta.split(" "):
            yield word + " "
            time.sleep(0.0)
        
        
        # Bar chart and meta learner influence
        col1, col2 = st.columns(2)
        with col1:
            # LIME Bar Chart
            if os.path.exists('./XAI_Output_1/X-Ray_LIME_Bar.png'):
                st.image('./XAI_Output_1/X-Ray_LIME_Bar.png', 
                         caption="Feature Importance Analysis", 
                         use_container_width=True)
                
        
        with col2:
            # Meta Learner Influence
            st.write_stream(stream_data2)
            if os.path.exists('./XAI_Output_1/X-Ray_Meta_Learner_Influence.png'):
                st.image('./XAI_Output_1/X-Ray_Meta_Learner_Influence.png', 
                         caption="Model Contribution Analysis", 
                         use_container_width=True)
        with col1:      
                st.write_stream(stream_data3)
        
        st.divider()

    # 2. CT Scan Analysis Section 
    if 'ct_prediction' in results:

        def stream_data4():
          st.header("2. CT Scan Analysis")

          ct_scan =(f"""The AI model analyzed the CT scan provided and predicted a **{results['ct_prediction']}** diagnosis with {results['ct_confidence']:.2f}% confidence. 
          This level of confidence indicates the model's certainty, with higher confidence suggesting stronger evidence supporting the prediction. 
          To help better understand this result, a heatmap is developed to highlight the areas that most influenced the prediction. It uses a color-gradient to represent the relevance of different regions within the scan. """)
        

          for word in ct_scan.split(" "):
            yield word + " "
            time.sleep(0.0)

        def stream_data401():
          ct_1=(f"""Warmer colors, such as red and yellow, indicate the regions most strongly associated with the predicted diagnosis, showing where the model detected features that align with a **{results['ct_prediction']}** outcome. 
          In particular, the regions highlighted in warmer colors reflect tissue densities and structural patterns that suggest no signs of abnormalities or support for the predicted condition, depending on the diagnosis. 
          These areas represent healthy tissue or other patterns that reinforce the prediction. 
          The heatmap serves as a visual guide to show which parts of the scan contributed most to the prediction, helping analyse how the model arrived at its conclusion. 
          This visualization of the key factors that influenced the prediction offers greater clarity in interpreting the results.""")

          for word in ct_1.split(" "):
            yield word + " "
            time.sleep(0.0)

        
        col1, col2 = st.columns([2, 1])
        with col1:
           st.write_stream(stream_data4)

        with col2:
            # Small version of GradCAM++
            if os.path.exists('./XAI_Output_2/CT_GradCAM++.png'):
                image = Image.open('./XAI_Output_2/CT_GradCAM++.png')
                st.image(image, caption="CT Scan Heatmap", use_container_width=True)

        st.write_stream(stream_data401)

        
        # CT Overlay and Bar Chart
        col1, col2 = st.columns(2)

        def stream_data5():
         ct_overlay= """
                The overlay on the CT scan pinpoints specific regions that increase or decrease the likelihood of the prediction. The combination of green and red areas offers a visual understanding of how the model arrived at its conclusion. 
                The overlay acts as a visual tool to indicate how different areas of the scan have contributed to the model’s decision. 
                It highlights regions that either support or challenge the prediction, offering a more transparent view of the underlying factors that influenced the diagnosis. 
                This helps in understanding which structures or abnormalities played a significant role in shaping the model’s conclusion. 
                Whether it’s an indication of normalcy or the presence of a condition, the overlay allows for easy identification of critical areas in the CT scan that contributed to the final prediction. 
                """
         for word in ct_overlay.split(" "):
            yield word + " "
            time.sleep(0.0)

        def stream_data6():
          ct_bar= """
                The bar chart provides a detailed summary of the regional contributions to the CT scan prediction. It visually represents the impact of different regions within the scan, showing how each area either supports or challenges the final prediction. 
                This graphical representation works in conjunction with the overlay image, offering a clearer understanding of the regions that influenced the AI model’s decision-making process. 
                Each bar in the chart corresponds to a specific region of the scan and indicates the degree to which that region contributed to the prediction. 
                It makes it easier to compare the contributions of different regions, allowing for a more precise understanding of which areas played a more significant role in supporting or opposing the predicted diagnosisPositive values suggest areas that reinforce the predicted diagnosis, while negative values represent regions that reduce the model’s confidence in the prediction.

                """
          for word in ct_bar.split(" "):
            yield word + " "
            time.sleep(0.0)

        with col1:
            if os.path.exists('./XAI_Output_2/CT_LIME_Overlay.png'):
                st.image('./XAI_Output_2/CT_LIME_Overlay.png', 
                         caption="CT Scan Region Overlay", 
                         use_container_width=True)
                st.write_stream(stream_data5)
        
        with col2:
            if os.path.exists('./XAI_Output_2/CT_LIME_Bar.png'):
                st.image('./XAI_Output_2/CT_LIME_Bar.png', 
                         caption="Region Contribution Analysis", 
                         use_container_width=True)
                st.write_stream(stream_data6)
        
        st.divider()

    # 3. Clinical Symptom Analysis 
    if 'clinical_results' in results:
        clinical_data = results['clinical_results']
        st.header("3. Clinical Symptom Analysis")

        def stream_data7():
         symptom_analysis= f"""
        The patient's clinical notes were analyzed to identify relevant symptoms and assess risk. 
        This thorough analysis involved extracting key symptoms that could provide important context for the diagnosis. 
        Several key symptoms were detected, including {', '.join(symptoms_list)}, which were then grouped into clusters based on their relationship to each other and their potential connection to specific conditions. 
        These clusters of symptoms help to create a more comprehensive diagnostic context, enabling the model to evaluate the patient's condition more effectively.
        
        **Risk Assessment**
        - Risk Level: **{clinical_data['risk_level']}**
        - Risk Weight: **{clinical_data['risk_weight']}%**
        """

         for word in symptom_analysis.split(" "):
            yield word + " "
            time.sleep(0.0)

        def stream_data8():
          symptom_cluster= f"""
                **{cluster['name']}**
                - Symptoms: {', '.join(cluster['matching_symptoms'])}
                - Clinical Significance: {cluster['reasoning']}
                """

          for word in symptom_cluster.split(" "):
            yield word + " "
            time.sleep(0.0)

        
        # Clinical Analysis Text Report+
        symptoms_list = [f"**{s['head_symptom']}**" for s in clinical_data['symptoms']]

        st.write_stream(stream_data7)
        
        # Symptom Clusters
        if clinical_data['clusters']:
            for cluster in clinical_data['clusters']:

                st.write_stream(stream_data8)
        
        # Expandable Clinical Note Annotation 
        with st.expander("View Annotated Clinical Note"):
            clinical_text = st.session_state.clinical_text
            final_text = clinical_text
            
            # Process all symptoms and sort by start index in reverse
            highlights = []
            for symptom in clinical_data['symptoms']:
                highlights.append({
                    'start': symptom['start'],
                    'end': symptom['end'],
                    'text': symptom['text'],
                    'head_symptom': symptom['head_symptom']
                })
            
            # Sort in reverse order to mark from end to start
            highlights.sort(key=lambda x: x['start'], reverse=True)
            
            # Apply highlighting (Symptom XAI)
            for highlight in highlights:
                before = final_text[:highlight['start']]
                after = final_text[highlight['end']:]
                highlighted = final_text[highlight['start']:highlight['end']]
                final_text = before + f"<mark title='{highlight['head_symptom']}'>{highlighted}</mark>" + after
            
            st.markdown(final_text, unsafe_allow_html=True)


# Generation function  for downloadable medical report
def generate_detailed_report_text(results, patient_name):
    is_suspicious = results.get('is_suspicious', False)
    suspicion_description = results.get('suspicion_description', 'Results Pending')
    adjusted_confidence = results.get('adjusted_confidence', 0)
    
    report_text = f"""Medical Analysis Report
Patient Name: {patient_name}
Date: {datetime.now().strftime('%B %d, %Y')}
Overall Assessment: {suspicion_description}
Confidence: {adjusted_confidence:.2f}%

Analysis Results:
"""
    if 'xray_prediction' in results:
        report_text += f"\n1. X-Ray Analysis:\n"
        report_text += f"- Prediction: {results['xray_prediction']}\n"
        report_text += f"- Confidence: {results['xray_confidence']:.2f}%\n"
        report_text += "- Detailed findings available in the detailed report.\n"

    if 'ct_prediction' in results:
        report_text += f"\n2. CT Scan Analysis:\n"
        report_text += f"- Prediction: {results['ct_prediction']}\n"
        report_text += f"- Confidence: {results['ct_confidence']:.2f}%\n"
        report_text += "- Detailed findings available in the detailed report.\n"

    if 'clinical_results' in results:
        clinical = results['clinical_results']
        report_text += f"\n3. Clinical Analysis:\n"
        report_text += f"- Risk Level: {clinical['risk_level']}\n"
        report_text += f"- Risk Weight: {clinical['risk_weight']}%\n"
        
        # Add the symptoms
        if 'symptoms' in clinical:
            report_text += "- Key Symptoms:\n"
            for symptom in clinical['symptoms']:
                report_text += f"  * {symptom['head_symptom']}\n"

    return report_text

# XAI Generative Report
def display_global_xai():

    # CT Scan Section
    st.title("CT Scan G-XAI Analysis")

    def stream_data15():
          gxai_ct=(f"""For CT scan analysis, SHAP was employed at the feature level to identify critical regions impacting lung cancer subtype classification. Using a GradientExplainer, feature attributions were approximated by analyzing the impact of regional perturbations on model predictions. The SHAP Summary Violin Plot highlighted the contribution of specific anatomical features (CT Scan Feature A, B, C) to diagnosis, offering guidance in scan interpretation. The analysis was enhanced with PDP through Captum's Feature Ablation, which employed Gaussian noise perturbation to assess classification confidence changes when specific image regions were removed. A Feature Importance Bar Chart provided a global perspective on regional influence, using a color-coded system—green for high importance, yellow for moderate, and red for minimal contribution—to categorize the impact of different CT features on diagnosis and improve the readability of the PDP plots. """)

          for word in gxai_ct.split(" "):
            yield word + " "
            time.sleep(0.05)

    st.write_stream(stream_data15)

    ct_scan_dir = './G-XAI/SHAP_Plots_CT'
    pdp_ct_scan_dir = './G-XAI/PDP_Plots_CT'
    pdp_ct_scan_dir2 = './G-XAI/PDP_Plots_CT2'

    if os.path.exists(ct_scan_dir):
        shap_ct_scan_path = os.path.join(ct_scan_dir, 'SHAP_CT_Scan.png')
        if os.path.exists(shap_ct_scan_path):
            st.image(shap_ct_scan_path, caption="SHAP Summary Plot for CT Scan", use_container_width=True)


    col1, col2 = st.columns([2, 2.025])
    with col1:
     if os.path.exists(pdp_ct_scan_dir):
        pdp_ct_files = [
            os.path.join(pdp_ct_scan_dir, f) for f in os.listdir(pdp_ct_scan_dir) if f.endswith('.png')
        ]
        for pdp_file in pdp_ct_files:
            st.image(pdp_file, caption=f"{os.path.basename(pdp_file)}", use_container_width=True)


    with col2:
     if os.path.exists(pdp_ct_scan_dir2):
        pdp_ct_files = [
            os.path.join(pdp_ct_scan_dir2, f) for f in os.listdir(pdp_ct_scan_dir2) if f.endswith('.png')
        ]
        for pdp_file in pdp_ct_files:
            st.image(pdp_file, caption=f"{os.path.basename(pdp_file)}", use_container_width=True)

    st.divider()

    # X-Ray Meta-Ensemble Section
    st.title("X-Ray G-XAI Analysis")

    def stream_data16():
          gxai_xray=(f"""In the X-ray modality analysis, SHAP was implemented at two distinct levels: meta-model feature attribution and meta-ensemble decision analysis. The approach quantified contributions from individual base CNN models (ResNet18, ConvNeXtTiny, EfficientNetV2S, DenseNet121) and meta-learners (Logistic Regression, Random Forest, Gradient Boosting, among others) to the final prediction. Violin SHAP Summary Plots revealed the relative influence of CNNs and meta-learners on classification outcomes. PDP analysis complemented these findings by visualizing the relationship between CNN outputs and meta-model decisions, as well as demonstrating how meta-learners shaped the meta-ensemble's probability scores. Together, these techniques provided clear insights into model influence patterns and their collaborative impact on classification accuracy.""")

          for word in gxai_xray.split(" "):
            yield word + " "
            time.sleep(0.05)

    st.write_stream(stream_data16)

    st.header("1. X-Ray Stacked-Ensemble G-XAI Analysis")

    xray_ensemble_shap_dir = './G-XAI/SHAP_Plots_X/Meta-Ensemble'
    xray_ensemble_pdp_dir = './G-XAI/PDP_Plots_X/Meta-Ensemble'

    

    if os.path.exists(xray_ensemble_shap_dir):
        shap_xray_ensemble_files = [
            os.path.join(xray_ensemble_shap_dir, f) for f in os.listdir(xray_ensemble_shap_dir) if f.endswith('.png')
        ]
        for shap_file in shap_xray_ensemble_files:
            st.image(shap_file, caption=f"{os.path.basename(shap_file)}", use_container_width=True)
    
    if os.path.exists(xray_ensemble_pdp_dir):
        pdp_xray_ensemble_files = [
            os.path.join(xray_ensemble_pdp_dir, f) for f in os.listdir(xray_ensemble_pdp_dir) if f.endswith('.png')
        ]
        for pdp_file in pdp_xray_ensemble_files:
            st.image(pdp_file, caption=f"{os.path.basename(pdp_file)}", use_container_width=True)

    # X-Ray Meta-Model Section 
    st.header("2. X-Ray Meta-Model G-XAI Analysis")
    
    xray_shap_dir = './G-XAI/SHAP_Plots_X/Meta-Model'
    xray_pdp_dir = './G-XAI/PDP_Plots_X/Meta-Model'

    col1, col2 = st.columns([2, 2.525])
    with col1:

     if os.path.exists(xray_shap_dir):
        shap_xray_files = [
            os.path.join(xray_shap_dir, f) for f in os.listdir(xray_shap_dir) if f.endswith('.png')
        ]
        for shap_file in shap_xray_files:
            st.image(shap_file, caption=f" {os.path.basename(shap_file)}", use_container_width=True)
    with col2:
         
     if os.path.exists(xray_pdp_dir):
        pdp_xray_files = [
            os.path.join(xray_pdp_dir, f) for f in os.listdir(xray_pdp_dir) if f.endswith('.png')
        ]
        for pdp_file in pdp_xray_files:
            st.image(pdp_file, caption=f"{os.path.basename(pdp_file)}", use_container_width=True)

# XAI saved generative report
def display_global_xai2():
 # CT Scan Section 
    st.title("CT Scan G-XAI Analysis")

    def stream_data15():
          gxai_ct=(f"""For CT scan analysis, SHAP was employed at the feature level to identify critical regions impacting lung cancer subtype classification. Using a GradientExplainer, feature attributions were approximated by analyzing the impact of regional perturbations on model predictions. The SHAP Summary Violin Plot highlighted the contribution of specific anatomical features (CT Scan Feature A, B, C) to diagnosis, offering guidance in scan interpretation. The analysis was enhanced with PDP through Captum's Feature Ablation, which employed Gaussian noise perturbation to assess classification confidence changes when specific image regions were removed. A Feature Importance Bar Chart provided a global perspective on regional influence, using a color-coded system—green for high importance, yellow for moderate, and red for minimal contribution—to categorize the impact of different CT features on diagnosis and improve the readability of the PDP plots. """)

          for word in gxai_ct.split(" "):
            yield word + " "
            time.sleep(0.0)

    st.write_stream(stream_data15)

    ct_scan_dir = './G-XAI/SHAP_Plots_CT'
    pdp_ct_scan_dir = './G-XAI/PDP_Plots_CT'
    pdp_ct_scan_dir2 = './G-XAI/PDP_Plots_CT2'

    if os.path.exists(ct_scan_dir):
        shap_ct_scan_path = os.path.join(ct_scan_dir, 'SHAP_CT_Scan.png')
        if os.path.exists(shap_ct_scan_path):
            st.image(shap_ct_scan_path, caption="SHAP Summary Plot for CT Scan", use_container_width=True)


    col1, col2 = st.columns([2, 2.025])
    with col1:
     if os.path.exists(pdp_ct_scan_dir):
        pdp_ct_files = [
            os.path.join(pdp_ct_scan_dir, f) for f in os.listdir(pdp_ct_scan_dir) if f.endswith('.png')
        ]
        for pdp_file in pdp_ct_files:
            st.image(pdp_file, caption=f"{os.path.basename(pdp_file)}", use_container_width=True)


    with col2:
     if os.path.exists(pdp_ct_scan_dir2):
        pdp_ct_files = [
            os.path.join(pdp_ct_scan_dir2, f) for f in os.listdir(pdp_ct_scan_dir2) if f.endswith('.png')
        ]
        for pdp_file in pdp_ct_files:
            st.image(pdp_file, caption=f"{os.path.basename(pdp_file)}", use_container_width=True)

    st.divider()

    # X-Ray Meta-Ensemble Section
    st.title("X-Ray G-XAI Analysis")

    def stream_data16():
          gxai_xray=(f"""In the X-ray modality analysis, SHAP was implemented at two distinct levels: meta-model feature attribution and meta-ensemble decision analysis. The approach quantified contributions from individual base CNN models (ResNet18, ConvNeXtTiny, EfficientNetV2S, DenseNet121) and meta-learners (Logistic Regression, Random Forest, Gradient Boosting, among others) to the final prediction. Violin SHAP Summary Plots revealed the relative influence of CNNs and meta-learners on classification outcomes. PDP analysis complemented these findings by visualizing the relationship between CNN outputs and meta-model decisions, as well as demonstrating how meta-learners shaped the meta-ensemble's probability scores. Together, these techniques provided clear insights into model influence patterns and their collaborative impact on classification accuracy.""")

          for word in gxai_xray.split(" "):
            yield word + " "
            time.sleep(0.0)

    st.write_stream(stream_data16)

    st.header("1. X-Ray Stacked-Ensemble G-XAI Analysis")

    xray_ensemble_shap_dir = './G-XAI/SHAP_Plots_X/Meta-Ensemble'
    xray_ensemble_pdp_dir = './G-XAI/PDP_Plots_X/Meta-Ensemble'

    

    if os.path.exists(xray_ensemble_shap_dir):
        shap_xray_ensemble_files = [
            os.path.join(xray_ensemble_shap_dir, f) for f in os.listdir(xray_ensemble_shap_dir) if f.endswith('.png')
        ]
        for shap_file in shap_xray_ensemble_files:
            st.image(shap_file, caption=f"{os.path.basename(shap_file)}", use_container_width=True)
    
    if os.path.exists(xray_ensemble_pdp_dir):
        pdp_xray_ensemble_files = [
            os.path.join(xray_ensemble_pdp_dir, f) for f in os.listdir(xray_ensemble_pdp_dir) if f.endswith('.png')
        ]
        for pdp_file in pdp_xray_ensemble_files:
            st.image(pdp_file, caption=f" {os.path.basename(pdp_file)}", use_container_width=True)

    # X-Ray Meta-Model Section
    st.header("2. X-Ray Meta-Model G-XAI Analysis")

    xray_shap_dir = './G-XAI/SHAP_Plots_X/Meta-Model'
    xray_pdp_dir = './G-XAI/PDP_Plots_X/Meta-Model'

    col1, col2 = st.columns([2, 2.525])
    with col1:

     if os.path.exists(xray_shap_dir):
        shap_xray_files = [
            os.path.join(xray_shap_dir, f) for f in os.listdir(xray_shap_dir) if f.endswith('.png')
        ]
        for shap_file in shap_xray_files:
            st.image(shap_file, caption=f" {os.path.basename(shap_file)}", use_container_width=True)
    with col2:
         
     if os.path.exists(xray_pdp_dir):
        pdp_xray_files = [
            os.path.join(xray_pdp_dir, f) for f in os.listdir(xray_pdp_dir) if f.endswith('.png')
        ]
        for pdp_file in pdp_xray_files:
            st.image(pdp_file, caption=f"{os.path.basename(pdp_file)}", use_container_width=True)



# MAIN FUNCTION
def run():
   # Initialize session state on first run
   if 'initialized' not in st.session_state:
       # Clean up any previous analysis directories
       cleanup_directories()
       # Ensure necessary directories exist
       ensure_directories()
       
       # Set initial session state variables
       st.session_state.step = 'upload'
       st.session_state.step1 = 'No'
       st.session_state.final2 = False
       st.session_state.final3 = False
       st.session_state.final4 = False
       st.session_state.initialized = True
       st.session_state.run_analysis_clicked = False
       st.session_state.final= False
       st.session_state.results = None
       st.session_state.patient_name = None
       st.session_state.uploaded_xray = None
       st.session_state.uploaded_ct = None

def upload():
   # Upload page logic
       
       # Page title and description

       st.title("M3 LungXAI-LF-v3")
       st.markdown("Welcome to the **M3 LungXAI-LF-v3**. Please upload your data below to begin.")

       # Patient name input
       patient_name = st.text_input(
           "Patient Name",
           value=st.session_state.get('patient_name', ''),
           placeholder="Enter the patient's name",
           key="patient_name_input",
           disabled=st.session_state.get('run_analysis_clicked', False)
       )
       st.session_state.patient_name = patient_name

       
       st.write("\n\n")
       
       # Create columns for file uploaders
       col1, spacer1, col2, spacer2, col3 = st.columns([3, 0.5, 3, 0.5, 3])

       # X-Ray upload section
       with col1:
           st.subheader("X-Ray Upload")
           # Allow upload only if analysis hasn't started
           if not st.session_state.get('run_analysis_clicked', False):
               xray_file = st.file_uploader("Upload X-Ray Image", 
                                            type=['png', 'jpg', 'jpeg'], 
                                            key="xray")
               # Display uploaded X-Ray image
               if xray_file:
                   st.session_state.uploaded_xray = xray_file
                   image = Image.open(xray_file)
                   st.image(image.resize((400, 400)), caption="Uploaded X-Ray Image", use_container_width=True)
           else:
               # Show locked image during analysis
               st.warning("X-Ray image is locked for analysis")
               if st.session_state.uploaded_xray:
                   image = Image.open(st.session_state.uploaded_xray)
                   st.image(image.resize((400, 400)), caption="Uploaded X-Ray Image", use_container_width=True)

       # CT Scan upload section (similar to X-Ray)
       with col2:
           st.subheader("CT Upload")
           if not st.session_state.get('run_analysis_clicked', False):
               ct_file = st.file_uploader("Upload CT Scan Image", 
                                          type=['png', 'jpg', 'jpeg'], 
                                          key="ct")
               if ct_file:
                   st.session_state.uploaded_ct = ct_file
                   image = Image.open(ct_file)
                   st.image(image.resize((400, 400)), caption="Uploaded CT Scan Image", use_container_width=True)
           else:
               st.warning("CT Scan image is locked for analysis")
               if st.session_state.uploaded_ct:
                   image = Image.open(st.session_state.uploaded_ct)
                   st.image(image.resize((400, 400)), caption="Uploaded CT Scan Image", use_container_width=True)

       # Clinical notes section
       with col3:
           st.subheader("Clinical Notes")
           clinical_text = st.text_area("Enter Clinical Notes", 
                                      placeholder="Type clinical notes here", 
                                      key="clinical_notes",
                                      height=400,
                                      disabled=st.session_state.get('run_analysis_clicked', False))  

       st.divider()
     
       # Prevent multiple button clicks
       if 'button_clicked' not in st.session_state:
           st.session_state.button_clicked = False
        

       col1, col2 = st.columns([2,9])
       with col2:
           if st.button("Start New Analysis",disabled=st.session_state.button_clicked):
               st.session_state.step = 'upload'
               st.session_state.button_clicked = False
               st.session_state.run_analysis_clicked = False
               
               st.session_state.results = None
               st.session_state.uploaded_xray = None
               st.session_state.uploaded_ct = None
               st.session_state.clinical_text = None
               st.session_state.patient_name = None
               cleanup_directories()
               ensure_directories()
               st.rerun()

       # Run Analysis button
       with col1:
        run_button = st.button("Run Analysis", disabled=st.session_state.get('run_analysis_clicked', False))  

       if run_button and not st.session_state.button_clicked:
           # Lock UI and prepare for analysis
           st.session_state.button_clicked = True
           st.session_state.run_analysis_clicked = True
           st.rerun()


        
       # Process files when analysis is triggered
       if st.session_state.button_clicked:
           # Clear previous analysis directories
           cleanup_directories()

           # Save uploaded files
           if st.session_state.uploaded_xray:
               with open(os.path.join('./Pre_Input_X-ray', st.session_state.uploaded_xray.name), 'wb') as f:
                   f.write(st.session_state.uploaded_xray.getbuffer())
           if st.session_state.uploaded_ct:
               with open(os.path.join('./Pre_Input_CT-Scan', st.session_state.uploaded_ct.name), 'wb') as f:
                   f.write(st.session_state.uploaded_ct.getbuffer())
           if clinical_text:
               st.session_state.clinical_text = clinical_text

           # Run prediction process
           results = process_and_predict()
           st.session_state.results = results
           st.session_state.step = 'upload'
           st.session_state.button_clicked = False
           st.session_state.run_analysis_clicked = False
           st.rerun()

       if not st.session_state.get('run_analysis_clicked', False):
          filler=1
          
       else:
          st.warning("Analysis Complete; Go to Explore Page")
          st.session_state.final2 = False
          st.session_state.final3 = False
          st.session_state.final4 = False
          st.session_state.final = False

          
# Explore page logic: Medical Report
def results():

       col1, col2 = st.columns([2,0.45])

       with col1:
           if not st.session_state.get("final",False):
              
            if st.button("Generate Medical Report"):
               st.session_state.step1 = 'results'
               st.session_state.final = True
           
               st.rerun()
           elif st.session_state.get("final2",False):  
              st.button("Generate Medical Report")   
              st.session_state.step1 = 'results_stay'
           else:
              st.button("Generate Medical Report") 
              st.write("")
       with col2:   
           if st.button("Close Report"):
               st.session_state.step1 = 'No'
               st.session_state.final = False
               st.session_state.final2 = True
               st.rerun() 
    
       if st.session_state.step1 == 'results':
        if st.session_state.results is not None:
            display_medical_report(st.session_state.results, st.session_state.patient_name)
            st.session_state.final2 = True
        else:
           st.error("No analysis results available")
       if st.session_state.step1 == 'results_stay':
          if st.session_state.results is not None:
            display_medical_report2(st.session_state.results, st.session_state.patient_name)
            st.session_state.final2 = True
          else:
           st.error("No analysis results available")
          
       if st.session_state.step1 == 'No':
          st.write("")

# Upload page logic: Model Analysis      
def gxai():   
    col1, col2 = st.columns([2,0.45])
    with col1:
           if not st.session_state.get("final3",False):
              
            if st.button("Generate Model Analysis"):
               st.session_state.step = 'results'
               st.session_state.final3 = True
           
               st.rerun()
           elif st.session_state.get("final4",False):  
              st.button("Generate Medical Report")   
              st.session_state.step = 'results_stay'
           else:
              st.button("Generate Medical Report") 
              st.write("")
    with col2:   
           if st.button("Close Report"):
               st.session_state.step = 'No'
               st.session_state.final3 = False
               st.session_state.final4 = True
               st.rerun() 
  
    if st.session_state.step == 'results':
        if st.session_state.results is not None:
            display_global_xai()
            st.session_state.final4 = True
        else:
           st.error("No analysis results available")
    if st.session_state.step == 'results_stay':
          if st.session_state.results is not None:
            display_global_xai2()
            st.session_state.final4 = True
          else:
           st.error("No analysis results available")
         
    if st.session_state.step == 'No':
          st.write("")

def home():
   # Home page logic
       
       def stream_home1():
          home1=(f"""Lung cancer was almost nonexistent in medical literature before the 19th century. Early cases were reported sporadically, with physicians regarding them as rare anomalies. However, the Industrial Revolution introduced widespread air pollution, and by the late 1800s, lung diseases became more prevalent. Despite this, lung cancer remained largely unrecognized as a distinct condition. It wasn't until the early 20th century that lung cancer diagnoses began to rise significantly, coinciding with the mass production and consumption of cigarettes.""")
          for word in home1.split(" "):
            yield word + " "
            time.sleep(0.0)


       def stream_home2():
          home2=(f"""The early 20th century saw a dramatic increase in lung cancer cases, correlating with the rise of cigarette smoking. In the 1920s and 1930s, German researchers established a statistical link between smoking and lung cancer, but tobacco companies aggressively denied these findings. It wasn’t until large epidemiological studies in the 1950s that the connection was widely accepted.
          By the 1950s and 1960s, studies like the British Doctors’ Study and the U.S. Surgeon General’s 1964 report confirmed smoking as the primary cause of lung cancer. This led to public health initiatives, warning labels on cigarette packs, and restrictions on tobacco advertising. However, lung cancer rates continued to rise due to widespread smoking habits.
          """)
          for word in home2.split(" "):
            yield word + " "
            time.sleep(0.0)

       def stream_home3():
          home3=(f""" """)
          for word in home3.split(" "):
            yield word + " "
            time.sleep(0.0)

       def stream_home4():
          home4=(f"""Medical advancements in the 1970s and 1980s, such as chest X-rays and CT scans, improved early detection. The classification of lung cancer into two main types—small cell lung cancer (SCLC) and non-small cell lung cancer (NSCLC)—allowed for more targeted treatment strategies, including surgery, chemotherapy, and radiation therapy.The late 20th and early 21st centuries saw significant breakthroughs in molecular oncology. Targeted therapies, such as EGFR inhibitors and ALK inhibitors, offered personalized treatment based on genetic mutations. Immunotherapy, particularly checkpoint inhibitors like PD-1/PD-L1 blockers, revolutionized treatment by harnessing the immune system to attack cancer cells.""")
          for word in home4.split(" "):
            yield word + " "
            time.sleep(0.0)
        
       def stream_home5():
          home5=(f""" """)
          for word in home5.split(" "):
            yield word + " "
            time.sleep(0.0)

       def stream_home6():
          home6=(f"""Today, advancements in artificial intelligence (AI) and precision medicine are transforming lung cancer diagnosis and treatment. AI-powered imaging models assist radiologists in detecting tumors at an early stage, while genomic profiling enables highly personalized therapies. Low-dose CT scans for high-risk individuals are improving early detection, potentially reducing lung cancer mortality rates. These technological advancements are enhancing diagnostic accuracy, optimizing treatment plans, and ultimately improving patient outcomes.""")
          for word in home6.split(" "):
            yield word + " "
            time.sleep(0.0)

       def stream_home7():
          home7=(f"""Adenocarcinoma is the most common form of lung cancer, particularly among non-smokers. It originates in the mucus-producing cells of the lung and tends to grow in the outer regions. This subtype is often detected at an early stage due to slow growth, making it more amenable to targeted therapies. Advances in molecular testing have led to the development of targeted drugs that specifically address genetic mutations associated with adenocarcinoma, improving treatment effectiveness. It is also the most frequently diagnosed subtype in women.""")
          for word in home7.split(" "):
            yield word + " "
            time.sleep(0.0)

       def stream_home8():
          home8=(f"""Squamous cell carcinoma tends to grow more aggressively than adenocarcinoma and may cause symptoms such as coughing and airway obstruction early on. This type of lung cancer typically develops in the central airways and is strongly linked to smoking. It is often treated with chemotherapy, radiation, or immunotherapy. Recent advancements in immunotherapy have shown promising results in treating squamous cell carcinoma, particularly for patients with high PD-L1 expression. It is more common in men than in women.""")
          for word in home8.split(" "):
            yield word + " "
            time.sleep(0.0)

       def stream_home9():
          home9=(f"""Large cell carcinoma is an aggressive and less common form of NSCLC that can appear in any part of the lung. It grows and spreads quickly, making early detection crucial. Due to its fast progression, treatment typically involves a combination of surgery, chemotherapy, and immunotherapy. This subtype is known for its tendency to resist standard treatments, often requiring a multimodal approach for better outcomes. Researchers are exploring targeted therapies and novel drug combinations to improve survival rates for patients with large cell carcinoma.""")
          for word in home9.split(" "):
            yield word + " "
            time.sleep(0.0)

       def stream_home10():
          home10=(f"""Artificial intelligence is playing an increasingly vital role in lung cancer detection and treatment planning. AI algorithms can analyze CT scans and X-rays with high accuracy, helping radiologists identify potential malignancies at an early stage. Machine learning models assist in risk assessment by analyzing a patient’s medical history, genetic markers, and lifestyle factors. AI also aids in treatment selection by predicting responses to specific therapies, leading to more personalized and effective care. """)
          for word in home10.split(" "):
            yield word + " "
            time.sleep(0.0)


       st.title("History of Lung Cancer")

       col1, col2 = st.columns([2, 2.025])
       with col1:
        st.write_stream(stream_home1)
       
       with col2:
        if os.path.exists('./Home/Cancer_Death.jpeg'):
                st.image('./Home/Cancer_Death.jpeg', 
                         caption="Cancer Diagnosis and Death Comparision", 
                         use_container_width=True)


       st.write_stream(stream_home2)



       if os.path.exists('./Home/Lung_type.png'):
                st.image('./Home/Lung_type.png', 
                         caption="Types of Lung Cancer", 
                         use_container_width=True)
     
       st.write_stream(stream_home4)

       col1, col2 = st.columns([2, 2.025])
       with col1:
        st.write_stream(stream_home6)

       with col2:
        if os.path.exists('./Home/lung_pi.png'):
                st.image('./Home/lung_pi.png', 
                         caption="Smoker vs Non-smoker effects on Lung Cancer Subtypes", 
                         use_container_width=True)

       st.write_stream(stream_home10)


       st.divider()


       st.title("Non-Small Cell Lung Cancer (NSCLC)")

       col1, col2 = st.columns([2, 2.025])
       with col1:
        st.write_stream(stream_home7)
       
       with col2:
         if os.path.exists('./Home/adeno.jpeg'):
                st.image('./Home/adeno.jpeg', 
                         caption="Adenocarcinoma", 
                         use_container_width=True)

       col1, col2 = st.columns([2, 2.025])
       with col1:
        st.write_stream(stream_home8)

       with col2:
        if os.path.exists('./Home/s_cell.jpeg'):
                st.image('./Home/s_cell.jpeg', 
                         caption="Squamous Cell Carcinoma", 
                         use_container_width=True)
    
       col1, col2 = st.columns([2, 2.025])
       with col1:
        st.write_stream(stream_home9)

       with col2:
        if os.path.exists('./Home/large.jpeg'):
                st.image('./Home/large.jpeg', 
                         caption="Large Cell Carcinoma", 
                         use_container_width=True)

        

# Sidebar logic
with st.sidebar:
    st.title("💬 M3 LungXAI-LF-v3 Lung Cancer Diagnosis Tool")
    st.caption("AI-powered precision for early and accurate lung cancer diagnosis")


    
with st.sidebar:      
 selected0 = option_menu(
 menu_title = None,
 options=["Model"],
 icons=["lungs-fill","folder-plus"],
 menu_icon="cast",
 default_index=0,    
 styles ={
    "container":{"padding": "01important", "background-color": "transparent"},
    "icon":{"color":"white", "Font-size": "22px"},
    "nav-link":{
        "font-size":"22px",
        "text-align":"left",
        "margin":"0px",
        "--hover-color":"#eee",
        },
        "nav-link-selected": {"background-color": "darkblue"},
        },
   )
       
# Connection between pages
if selected0 == "Model":
  
 # Horizontal menu tab
 selected1 = option_menu(
 menu_title = None,
 options=["Home", "Predict", "Explore"],
 icons=["house","heart-pulse-fill","bar-chart-steps"],
 menu_icon="cast",
 default_index=0,
 orientation="horizontal",    
 styles ={
    "container":{"padding": "01important", "background-color": "black"},
    "icon":{"color":"white", "Font-size": "25px"},
    "nav-link":{
        "font-size":"25px",
        "text-align":"left",
        "margin":"0px",
        "--hover-color":"#eee",
        },
        "nav-link-selected": {"background-color": "blue"},
        },
   )

 if selected1 == "Explore":

  selected2 = option_menu(
  menu_title = None,
  options=["Medical Reporting", "Model Analysis"],
  icons=["layout-text-window-reverse","database-fill-up"],
  menu_icon="cast",
  default_index=0,
  orientation="horizontal",    
  styles ={
      "container":{"padding": "01important", "background-color": "black"},
      "icon":{"color":"white", "Font-size": "25px"},
      "nav-link":{
          "font-size":"25px",
          "text-align":"left",
          "margin":"0px",
          "--hover-color":"#eee",
          },
          "nav-link-selected": {"background-color": "darkblue"},
        },
  )

  if selected2 == "Medical Reporting":
       run()
       results()
       
  if selected2 == "Model Analysis":
       run()
       gxai()

 if selected1 == "Home": 
    home()


 if selected1 == "Predict":
    if __name__ == "__main__":
      run()
      upload()
