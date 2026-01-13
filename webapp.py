import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import tempfile
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import json
import re
import yaml

st.set_page_config(layout="wide", page_title="Radiomics Model for The Prediction of Hematoma Expansion & SHAP Analysis")

SERIALIZED_MODELS_DIR = "serialized_models"
STANDARDIZATION_PARAMS_PATH = "standardization_parameters.csv"
SHAP_BACKGROUND_DATA_PATH = "shap_background_data.csv"

if 'radiomic_features' not in st.session_state:
    st.session_state.radiomic_features = {}
if 'clinical_features' not in st.session_state:
    st.session_state.clinical_features = {}
if 'all_extracted_features' not in st.session_state:
    st.session_state.all_extracted_features = {}
if 'show_standardization_details' not in st.session_state:
    st.session_state.show_standardization_details = False
if 'clear_clinical_counter' not in st.session_state:
    st.session_state.clear_clinical_counter = 0

def create_optimized_waterfall_plot(shap_values, feature_names=None, max_display=15):
    if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 1:
        if shap_values.values.shape[-1] == 2:
            if hasattr(shap_values.base_values, '__len__') and len(shap_values.base_values) == 2:
                base_value = shap_values.base_values[1]
            else:
                base_value = shap_values.base_values
            
            shap_values = shap.Explanation(
                values=shap_values.values[..., 1],
                base_values=base_value,
                data=shap_values.data if hasattr(shap_values, 'data') else None,
                feature_names=shap_values.feature_names if hasattr(shap_values, 'feature_names') else feature_names
            )
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(10, min(max_display * 0.5 + 2, 8)))
    ax = plt.gca()
    
    shap.waterfall_plot(
        shap_values, 
        show=False, 
        max_display=max_display
    )
    
    plt.subplots_adjust(left=0.4, right=0.95, top=0.92, bottom=0.08)
    
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=11)
    
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    ax.set_title('SHAP Feature Contribution Analysis', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
    
    return fig

@st.cache_data
def load_csv_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"CSV file '{path}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading CSV file '{path}': {e}")
        return None

@st.cache_data
def load_standardization_parameters(path):
    try:
        params_df = pd.read_csv(path)
        
        standardization_info = {
            'params_df': params_df,
            'continuous_features': params_df[params_df['feature_type'] == 'continuous']['feature_name'].tolist(),
            'binary_features': params_df[params_df['feature_type'] == 'binary']['feature_name'].tolist(),
            'mean_dict': dict(zip(
                params_df[params_df['feature_type'] == 'continuous']['feature_name'],
                params_df[params_df['feature_type'] == 'continuous']['mean']
            )),
            'std_dict': dict(zip(
                params_df[params_df['feature_type'] == 'continuous']['feature_name'],
                params_df[params_df['feature_type'] == 'continuous']['std']
            ))
        }
        
        return standardization_info
    except FileNotFoundError:
        st.error(f"Standardization parameters file '{path}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading standardization parameters file: {e}")
        return None

def _get_extraction_config():
    _base_settings = {
        'binWidth': 25,
        'resampledPixelSpacing': [1, 1, 1],
        'interpolator': 'sitkBSpline',
        'label': 1,
        'normalize': True
    }
    
    _image_types = {
        'Original': {},
        'Wavelet': {},
        'LoG': {
            'sigma': [1.0, 2.0, 3.0]
        },
        'SquareRoot': {}
    }
    
    return _base_settings, _image_types

@st.cache_resource
def load_model_file(path):
    try:
        import joblib
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Model file '{path}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model file '{path}': {e}")
        return None

standardization_info = load_standardization_parameters(STANDARDIZATION_PARAMS_PATH)
shap_background_df = load_csv_data(SHAP_BACKGROUND_DATA_PATH)

if standardization_info is not None:
    st.sidebar.success(f"✔ Successfully loaded standardization parameters: {len(standardization_info['continuous_features'])} continuous features, {len(standardization_info['binary_features'])} binary features")
else:
    st.sidebar.error("Unable to load standardization parameters file")

def classify_features(feature_names):
    if standardization_info is None:
        radiomic_keywords = [
            'original_', 'wavelet_', 'wavelet-', 'log_', 'log-sigma-', 
            'logarithm_', 'gradient_', 'gradient-', 'exponential_', 
            'square_', 'squareroot_', 'shape_', 'shape2D_',
            'firstorder_', 'glcm_', 'glrlm_', 'glszm_', 
            'ngtdm_', 'gldm_', 'glrm_'
        ]
        
        radiomic_features = []
        clinical_features = []
        
        for feature in feature_names:
            if feature.upper() == 'STATUS':
                continue
            
            is_radiomic = any(keyword in feature.lower() for keyword in radiomic_keywords)
            
            if not is_radiomic:
                pyradiomics_patterns = [
                    r'^(log|wavelet|gradient|exponential|square|logarithm|original)-.*_(firstorder|glcm|glrlm|glszm|ngtdm|gldm|shape)',
                    r'^(log-sigma-[\d-]+mm-\dD)_',
                    r'^wavelet-[LH]+_'
                ]
                for pattern in pyradiomics_patterns:
                    if re.match(pattern, feature, re.IGNORECASE):
                        is_radiomic = True
                        break
            
            if is_radiomic:
                radiomic_features.append(feature)
            else:
                clinical_features.append(feature)
    else:
        all_features_in_params = standardization_info['continuous_features'] + standardization_info['binary_features']
        radiomic_features = []
        clinical_features = []
        
        for feature in feature_names:
            if feature.upper() == 'STATUS':
                continue
            
            if feature in all_features_in_params:
                radiomic_keywords = ['original_', 'wavelet', 'log-sigma']
                if any(keyword in feature.lower() for keyword in radiomic_keywords):
                    radiomic_features.append(feature)
                else:
                    clinical_features.append(feature)
            else:
                clinical_features.append(feature)
    
    return radiomic_features, clinical_features

def identify_variable_type(series):
    non_null_values = series.dropna()
    
    if len(non_null_values) == 0:
        return 'continuous', None
    
    unique_values = non_null_values.unique()
    
    if len(unique_values) == 2:
        return 'binary', sorted(unique_values.tolist())
    
    if len(unique_values) <= 5 and all(isinstance(x, (int, np.integer)) for x in unique_values):
        if set(unique_values).issubset({0, 1}):
            return 'binary', sorted(unique_values.tolist())
    
    return 'continuous', None

if shap_background_df is not None:
    FEATURE_ORDER = list(shap_background_df.columns)
    FEATURE_ORDER = [f for f in FEATURE_ORDER if f.upper() != 'STATUS']
    RADIOMIC_FEATURES, CLINICAL_FEATURES = classify_features(FEATURE_ORDER)
    
    with st.sidebar:
        st.info(f"Detected feature statistics:\n- Radiomic features: {len(RADIOMIC_FEATURES)}\n- Clinical features: {len(CLINICAL_FEATURES)}")
else:
    FEATURE_ORDER = []
    RADIOMIC_FEATURES = []
    CLINICAL_FEATURES = []

def extract_radiomics_features(image_path, mask_path, params=None):
    try:
        _settings, _image_configs = _get_extraction_config()
        
        _config_dict = {
            'imageType': _image_configs,
            'setting': _settings
        }
        
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_yaml:
            yaml.dump(_config_dict, temp_yaml)
            temp_yaml_path = temp_yaml.name
        
        extractor = featureextractor.RadiomicsFeatureExtractor(temp_yaml_path)
        
        os.unlink(temp_yaml_path)
        
        extractor.enableAllFeatures()
        st.info("All feature classes enabled")
        
        enabled_image_types = extractor.enabledImagetypes
        if enabled_image_types:
            st.info(f"Enabled image types: {list(enabled_image_types.keys())}")
        
        st.info("Extracting radiomics features, please wait...")
        feature_vector = extractor.execute(image_path, mask_path)
        
        feature_dict = {}
        for key, value in feature_vector.items():
            if not key.startswith('diagnostics_'):
                try:
                    feature_dict[key] = float(value)
                except Exception:
                    feature_dict[key] = value
        
        st.success(f"Successfully extracted {len(feature_dict)} features")
        return feature_dict
        
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        import traceback
        st.error(f"Detailed error information: {traceback.format_exc()}")
        return None

def filter_and_standardize_features(extracted_features, background_features, standardization_params):
    st.session_state.all_extracted_features = extracted_features
    
    if standardization_params is not None:
        expected_features = standardization_params['continuous_features'] + standardization_params['binary_features']
        st.info(f"Standardization parameters expect {len(expected_features)} features")
        
        filtered_features = {}
        if background_features is not None:
            background_columns = list(background_features.columns)
            background_radiomic_features = []
            for col in background_columns:
                if any(keyword in col.lower() for keyword in ['original_', 'wavelet', 'log-sigma', 'squareroot_']):
                    background_radiomic_features.append(col)
            
            st.info(f"Background dataset contains {len(background_radiomic_features)} radiomic features")
            
            for feature in background_radiomic_features:
                if feature in extracted_features:
                    filtered_features[feature] = extracted_features[feature]
                else:
                    filtered_features[feature] = background_features[feature].median()
        
        st.success(f"Filtered {len(filtered_features)} radiomic features for display from extracted features")
    else:
        filtered_features = {}
        for feature in RADIOMIC_FEATURES:
            if feature in extracted_features:
                filtered_features[feature] = extracted_features[feature]
            else:
                if background_features is not None and feature in background_features.columns:
                    filtered_features[feature] = background_features[feature].median()
                else:
                    filtered_features[feature] = 0.0
    
    return filtered_features

def preprocess_input(input_data_dict, feature_order_list, standardization_params=None):
    try:
        if standardization_params is None:
            st.error("Standardization parameters not loaded, unable to standardize features!")
            return None
            
        continuous_features = standardization_params['continuous_features']
        mean_dict = standardization_params['mean_dict']
        std_dict = standardization_params['std_dict']
        
        all_raw_features = {}
        all_extracted = getattr(st.session_state, 'all_extracted_features', {})
        
        for feature, value in input_data_dict.items():
            all_raw_features[feature] = value
        
        for feature in st.session_state.radiomic_features:
            if feature not in all_raw_features:
                all_raw_features[feature] = st.session_state.radiomic_features[feature]
        
        for feature in all_extracted:
            if feature not in all_raw_features:
                all_raw_features[feature] = all_extracted[feature]
        
        final_features = {}
        standardization_log = []
        
        for feature in feature_order_list:
            if feature in mean_dict and feature in std_dict:
                if feature in all_raw_features:
                    raw_value = all_raw_features[feature]
                    standardized_value = (raw_value - mean_dict[feature]) / std_dict[feature]
                    final_features[feature] = standardized_value
                    standardization_log.append({
                        'feature': feature,
                        'raw_value': raw_value,
                        'mean': mean_dict[feature],
                        'std': std_dict[feature],
                        'standardized_value': standardized_value,
                        'status': 'Standardized'
                    })
                else:
                    if shap_background_df is not None and feature in shap_background_df.columns:
                        raw_value = shap_background_df[feature].median()
                        standardized_value = (raw_value - mean_dict[feature]) / std_dict[feature]
                        final_features[feature] = standardized_value
                    else:
                        final_features[feature] = 0.0
            elif feature in all_raw_features:
                final_features[feature] = all_raw_features[feature]
                standardization_log.append({
                    'feature': feature,
                    'raw_value': all_raw_features[feature],
                    'mean': 'N/A',
                    'std': 'N/A',
                    'standardized_value': all_raw_features[feature],
                    'status': 'Binary Feature'
                })
            else:
                if shap_background_df is not None and feature in shap_background_df.columns:
                    final_features[feature] = shap_background_df[feature].median()
                else:
                    final_features[feature] = 0.0
        
        st.session_state.standardization_log = standardization_log
        
        final_df = pd.DataFrame([final_features])
        final_df = final_df[feature_order_list]
        
        if final_df.isnull().values.any():
            nan_columns = final_df.columns[final_df.isnull().any()].tolist()
            st.error(f"The following features contain NaN values: {nan_columns}")
            final_df.fillna(0, inplace=True)
        
        return final_df
            
    except Exception as e:
        st.error(f"Preprocessing failed: {str(e)}")
        import traceback
        with st.expander("View detailed error information"):
            st.code(traceback.format_exc())
        return None

st.title("Radiomics Model for The Prediction of Hematoma Expansion & SHAP Analysis")

st.sidebar.header("Model Selection")
model_files = [f for f in os.listdir(SERIALIZED_MODELS_DIR) if f.endswith(".joblib")]
if not model_files:
    st.error(f"No model files found in '{SERIALIZED_MODELS_DIR}' directory.")
    st.stop()

model_display_names = [os.path.splitext(f)[0].replace("_", " ") for f in model_files]
model_name_to_file_map = dict(zip(model_display_names, model_files))

selected_model_display_name = st.sidebar.selectbox(
    "Select prediction model:",
    options=model_display_names,
    index=model_display_names.index("Logistic Regression") if "Logistic Regression" in model_display_names else 0
)

selected_model_filename = model_name_to_file_map[selected_model_display_name]
model_path = os.path.join(SERIALIZED_MODELS_DIR, selected_model_filename)
model = load_model_file(model_path)

if model is None:
    st.error("Model loading failed.")
    st.stop()

st.sidebar.success(f"✔ Model loaded: {selected_model_display_name}")

st.sidebar.header("Image File Upload")
st.sidebar.info("Please upload NIfTI format image and mask files (.nii or .nii.gz)")

image_file = st.sidebar.file_uploader(
    "Select original image file:",
    type=['nii', 'gz'],
    key="image_uploader"
)

mask_file = st.sidebar.file_uploader(
    "Select mask file:",
    type=['nii', 'gz'],
    key="mask_uploader"
)

if image_file and mask_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, image_file.name)
        mask_path = os.path.join(temp_dir, mask_file.name)
        
        with open(image_path, 'wb') as f:
            f.write(image_file.getbuffer())
        with open(mask_path, 'wb') as f:
            f.write(mask_file.getbuffer())
        
        if st.sidebar.button("Extract Radiomics Features", type="secondary"):
            with st.spinner("Extracting features..."):
                extracted_features = extract_radiomics_features(
                    image_path, mask_path, None
                )
                
                if extracted_features:
                    filtered_features = filter_and_standardize_features(
                        extracted_features, shap_background_df, standardization_info
                    )
                    
                    st.session_state.radiomic_features = filtered_features
                    st.sidebar.success(f"Successfully extracted {len(filtered_features)} radiomic features (raw values)")
                else:
                    st.sidebar.error("Feature extraction failed")

st.sidebar.header("Feature Value Input")

input_values = {}

default_input_values = {}

if RADIOMIC_FEATURES:
    st.sidebar.subheader("Radiomic Features")
    radiomic_display = st.sidebar.checkbox("Show radiomic feature input fields", value=False)
    
    if radiomic_display:
        for feature in RADIOMIC_FEATURES:
            if feature in st.session_state.radiomic_features:
                default_val = st.session_state.radiomic_features[feature]
            else:
                default_val = 0.0
            
            input_values[feature] = st.sidebar.number_input(
                f"{feature}",
                value=float(default_val),
                format="%.6f",
                key=f"radiomic_{feature}_{st.session_state.clear_clinical_counter}"
            )
    else:
        for feature in RADIOMIC_FEATURES:
            if feature in st.session_state.radiomic_features:
                input_values[feature] = st.session_state.radiomic_features[feature]
            else:
                input_values[feature] = 0.0

if CLINICAL_FEATURES:
    st.sidebar.subheader("Clinical Features (Please enter raw values manually)")
    
    if st.sidebar.button("Clear All Clinical Values", type="secondary", use_container_width=True):
        st.session_state.clear_clinical_counter += 1
        st.rerun()
    
    clinical_feature_types = {}
    if shap_background_df is not None:
        for feature in CLINICAL_FEATURES:
            if feature in shap_background_df.columns:
                var_type, unique_vals = identify_variable_type(shap_background_df[feature])
                clinical_feature_types[feature] = (var_type, unique_vals)
            else:
                clinical_feature_types[feature] = ('continuous', None)
    
    for feature in CLINICAL_FEATURES:
        default_val = ""
        
        var_type, unique_vals = clinical_feature_types.get(feature, ('continuous', None))
        
        if var_type == 'binary' and unique_vals is not None:
            input_values[feature] = st.sidebar.selectbox(
                f"{feature}",
                options=unique_vals,
                index=0,
                key=f"clinical_{feature}_{st.session_state.clear_clinical_counter}"
            )
        else:
            text_value = st.sidebar.text_input(
                f"{feature}",
                value=default_val,
                key=f"clinical_{feature}_{st.session_state.clear_clinical_counter}",
                placeholder="Enter value",
                help="Enter any numerical value (e.g., 123.456)"
            )
            
            try:
                if text_value.strip():
                    input_values[feature] = float(text_value)
                else:
                    input_values[feature] = 0.0
            except ValueError:
                st.sidebar.error(f"Invalid input for {feature}. Using 0.0")
                input_values[feature] = 0.0

if st.sidebar.button("Run Prediction & Analysis", type="primary", use_container_width=True):
    tab1, tab2 = st.tabs(["Feature Preprocessing", "Prediction Results & SHAP Analysis"])
    
    processed_input_df = preprocess_input(input_values, FEATURE_ORDER, standardization_info)
    
    with tab1:
        st.subheader("Feature Preprocessing Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Features", len(FEATURE_ORDER))
        with col2:
            st.metric("Continuous Features", len(standardization_info['continuous_features']) if standardization_info else 0)
        with col3:
            st.metric("Binary Features", len(standardization_info['binary_features']) if standardization_info else 0)
        
        with st.expander("View standardized radiomic features", expanded=False):
            if RADIOMIC_FEATURES and hasattr(st.session_state, 'standardization_log'):
                radiomic_log = [log for log in st.session_state.standardization_log if log['feature'] in RADIOMIC_FEATURES and log['status'] == 'Standardized']
                if radiomic_log:
                    for i, log in enumerate(radiomic_log[:5]):
                        st.success(f"✔ Feature '{log['feature']}' standardized: {log['raw_value']:.4f} → {log['standardized_value']:.4f}")
                    if len(radiomic_log) > 5:
                        st.info(f"... and {len(radiomic_log)-5} more features standardized")
        
        with st.expander("View binary features", expanded=False):
            binary_features_in_input = [f for f in input_values.keys() if f in standardization_info['binary_features']]
            if binary_features_in_input:
                for feature in binary_features_in_input:
                    st.info(f"✔ Feature '{feature}' using raw value (binary feature): {input_values[feature]}")
    
    with tab2:
        if processed_input_df is not None:
            col_result, col_shap = st.columns([1, 2])
            
            with col_result:
                st.subheader("Model Prediction Results")
                try:
                    model_type = str(type(model))
                    
                    if "XGBoost" in model_type or "xgboost" in model_type.lower():
                        try:
                            import xgboost as xgb
                            prediction_proba = model.predict_proba(processed_input_df)
                            prediction = model.predict(processed_input_df)
                        except Exception as xgb_error:
                            st.warning("Attempting to use XGBoost DMatrix format...")
                            dmatrix = xgb.DMatrix(processed_input_df)
                            prediction = model.predict(dmatrix)
                            prediction_proba = np.array([[1-prediction[0], prediction[0]]])
                            prediction = np.array([int(prediction[0] > 0.5)])
                    else:
                        prediction_proba = model.predict_proba(processed_input_df)
                        prediction = model.predict(processed_input_df)
                    
                    st.metric(label="Predicted Class", value=str(prediction[0]))
                    
                    if len(prediction_proba[0]) > 1:
                        prob_positive_class = prediction_proba[0][1]
                    else:
                        prob_positive_class = prediction_proba[0][0]
                    
                    prob_positive_class = float(prob_positive_class)
                    
                    st.metric(label="Positive Class Probability", value=f"{prob_positive_class:.4f}")
                    st.progress(prob_positive_class)
                    
                except Exception as e:
                    st.error(f"Error occurred during prediction: {e}")
                    prediction = None
            
            with col_shap:
                if prediction is not None:
                    st.subheader("SHAP Interpretability Analysis")
                    try:
                        explainer = None
                        model_type_str = str(type(model)).lower()
                        
                        shap_background_processed = shap_background_df[FEATURE_ORDER].copy()
                        
                        if "forest" in model_type_str or "tree" in model_type_str:
                            explainer = shap.TreeExplainer(model)
                        elif "logistic" in model_type_str or "linear" in model_type_str:
                            explainer = shap.LinearExplainer(model, shap_background_processed)
                        elif "xgboost" in model_type_str:
                            try:
                                explainer = shap.TreeExplainer(model)
                            except:
                                def xgb_predict(data):
                                    return model.predict_proba(data)[:, 1]
                                background_sample = shap_background_processed.sample(min(100, len(shap_background_processed)))
                                explainer = shap.KernelExplainer(xgb_predict, background_sample)
                        else:
                            def model_predict_proba_for_shap(data_as_np_array):
                                data_as_df = pd.DataFrame(data_as_np_array, columns=FEATURE_ORDER)
                                if hasattr(model, 'predict_proba'):
                                    return model.predict_proba(data_as_df)[:, 1]
                                else:
                                    return model.predict(data_as_df)
                            
                            background_sample = shap_background_processed.sample(min(100, len(shap_background_processed)))
                            explainer = shap.KernelExplainer(model_predict_proba_for_shap, background_sample)
                        
                        if explainer is not None:
                            shap_values_instance = explainer(processed_input_df)
                            
                            if isinstance(shap_values_instance, shap.Explanation):
                                fig_waterfall = create_optimized_waterfall_plot(
                                    shap_values_instance[0],
                                    feature_names=FEATURE_ORDER,
                                    max_display=15
                                )
                                st.pyplot(fig_waterfall, clear_figure=True)
                                plt.close()
                                
                                with st.expander("View detailed feature importance", expanded=False):
                                    if hasattr(shap_values_instance[0], 'values'):
                                        shap_vals = shap_values_instance[0].values
                                        if len(shap_vals.shape) > 1 and shap_vals.shape[-1] == 2:
                                            shap_vals = shap_vals[:, 1]
                                    else:
                                        shap_vals = shap_values_instance[0]
                                    
                                    if hasattr(shap_vals, 'flatten'):
                                        shap_vals = shap_vals.flatten()
                                    
                                    feature_importance = pd.DataFrame({
                                        'Feature Name': FEATURE_ORDER,
                                        'SHAP Value': shap_vals,
                                        'Feature Value': processed_input_df.iloc[0].values
                                    })
                                    
                                    feature_importance['Absolute SHAP'] = abs(feature_importance['SHAP Value'])
                                    feature_importance = feature_importance.sort_values('Absolute SHAP', ascending=False)
                                    
                                    st.dataframe(
                                        feature_importance.head(15)[['Feature Name', 'SHAP Value', 'Feature Value']],
                                        hide_index=True,
                                        use_container_width=True
                                    )
                            else:
                                st.warning("SHAP value format does not support waterfall plot, trying other visualizations...")
                                fig_summary = plt.figure(figsize=(10, 6))
                                shap.summary_plot(
                                    shap_values_instance, 
                                    processed_input_df, 
                                    feature_names=FEATURE_ORDER, 
                                    show=False,
                                    max_display=15
                                )
                                plt.tight_layout()
                                st.pyplot(fig_summary, clear_figure=True)
                                plt.close()
                        
                    except Exception as e:
                        st.error(f"Error during SHAP analysis: {e}")
                        import traceback
                        with st.expander("View detailed error information"):
                            st.code(traceback.format_exc())

else:
    st.info("Please upload image files or enter feature values, then click 'Run Prediction & Analysis' button.")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Instructions:**\n"
    "1. Select prediction model\n"
    "2. Upload NIfTI format image and mask files\n"
    "3. Click 'Extract Radiomics Features' to auto-fill radiomic features\n"
    "4. Manually enter clinical feature values\n"
    "5. Click 'Run Prediction & Analysis' to view results"
)
