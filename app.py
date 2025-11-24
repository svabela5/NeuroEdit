import streamlit as st
from PIL import Image, ImageEnhance, ImageStat
import numpy as np
import io
import firebase_admin
from firebase_admin import credentials, firestore

# --- Configuration ---
st.set_page_config(
    page_title="NeuroEdit AI",
    page_icon="📸",
    layout="wide"
)

# --- Firebase Setup ---
# We check if the app is already initialized to prevent errors on hot-reloads
if not firebase_admin._apps:
    # Load credentials from Streamlit secrets
    # See README for how to set this up in Streamlit Cloud
    if 'firebase' in st.secrets:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    else:
        st.warning("Firebase secrets not found. App is running in Read-Only Mode or will fail to save.")

# Get Firestore client
try:
    db = firestore.client()
    DB_AVAILABLE = True
except:
    DB_AVAILABLE = False

COLLECTION_NAME = "neuroedit_memory"

# --- Helper Functions ---

def rotate_if_vertical(img):
    width, height = img.size
    if height > width:
        return img.rotate(90, expand=True)
    return img

def get_image_stats(img):
    img_np = np.array(img)
    
    if img.mode == 'RGB':
        luma = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
    else:
        luma = img_np
    mean_brightness = np.mean(luma)
    rms_contrast = np.std(luma)
    
    if img.mode == 'RGB':
        hsv_img = img.convert('HSV')
        hsv_np = np.array(hsv_img)
        mean_saturation = np.mean(hsv_np[:,:,1])
    else:
        mean_saturation = 0 
        
    return mean_brightness, rms_contrast, mean_saturation

def save_training_data(stats, values):
    """
    Saves inputs/labels to Firestore.
    """
    if not DB_AVAILABLE:
        st.error("Database not connected. Cannot save training data.")
        return

    data_point = {
        "inputs": [float(x) for x in stats],
        "labels": [int(x) for x in values],
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    
    # Add to collection
    db.collection(COLLECTION_NAME).add(data_point)
    
    # Clear the cache so next prediction includes this new point
    fetch_memory_data.clear()

@st.cache_data(ttl=600) # Cache data for 10 minutes or until cleared
def fetch_memory_data():
    """
    Fetches all training examples from Firestore.
    Cached to prevent excessive database reads.
    """
    if not DB_AVAILABLE:
        return []
    
    docs = db.collection(COLLECTION_NAME).stream()
    history = []
    for doc in docs:
        history.append(doc.to_dict())
    return history

def predict_from_memory(current_stats):
    """
    Uses K-Nearest Neighbors (KNN) with data from Firestore.
    """
    history = fetch_memory_data()
            
    if len(history) < 3:
        return None
        
    # Extract inputs/labels, handling potential missing keys safely
    try:
        history_inputs = np.array([h['inputs'] for h in history])
        history_labels = np.array([h['labels'] for h in history])
    except KeyError:
        return None

    current_inputs = np.array(current_stats)
    
    # Calculate Euclidean distance
    distances = np.linalg.norm(history_inputs - current_inputs, axis=1)
    
    # Get K nearest neighbors (K=5)
    k = min(5, len(history))
    nearest_indices = distances.argsort()[:k]
    
    nearest_labels = history_labels[nearest_indices]
    prediction = np.mean(nearest_labels, axis=0)
    
    return int(prediction[0]), int(prediction[1]), int(prediction[2])

def calculate_ai_suggestions(mean_b, rms_c, mean_s, use_memory=True):
    if use_memory:
        memory_prediction = predict_from_memory((mean_b, rms_c, mean_s))
        if memory_prediction:
            return memory_prediction[0], memory_prediction[1], memory_prediction[2], "🧠 Cloud Memory"

    # Fallback Heuristics
    target_b = 115.0
    b_factor = target_b / (mean_b if mean_b > 10 else 10)
    b_factor = max(0.5, min(2.0, b_factor))
    suggested_brightness = int(b_factor * 128)
    
    target_c = 60.0
    c_factor = target_c / (rms_c if rms_c > 10 else 10)
    c_factor = max(0.8, min(1.8, c_factor))
    suggested_contrast = int(c_factor * 128)
    
    target_s = 90.0
    s_factor = (target_s / mean_s) if mean_s > 5 else 1.0
    s_factor = max(0.0, min(1.6, s_factor))
    suggested_saturation = int(s_factor * 128)
    
    return suggested_brightness, suggested_contrast, suggested_saturation, "📐 Mathematical Heuristics"

def apply_edits(img, brightness_val, contrast_val, saturation_val):
    b_factor = brightness_val / 128.0
    c_factor = contrast_val / 128.0
    s_factor = saturation_val / 128.0
    
    edited = img.copy()
    edited = ImageEnhance.Color(edited).enhance(s_factor)
    edited = ImageEnhance.Contrast(edited).enhance(c_factor)
    edited = ImageEnhance.Brightness(edited).enhance(b_factor)
    
    return edited

# --- Main UI ---

def main():
    st.title("🧠 NeuroEdit: Cloud AI Photo Enhancer")
    
    # Sidebar Status
    st.sidebar.header("System Status")
    if DB_AVAILABLE:
        st.sidebar.success("Database Connected")
    else:
        st.sidebar.error("Database Disconnected")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        image = rotate_if_vertical(image)
        
        preview_width = 1080
        w_percent = (preview_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        image_preview = image.resize((preview_width, h_size), Image.Resampling.LANCZOS)
        
        # AI Analysis
        st.sidebar.header("AI Configuration")
        mode_selection = st.sidebar.radio("Model Source:", ["Adaptive Memory", "Mathematical Heuristics"])
        use_memory = (mode_selection == "Adaptive Memory")

        file_id = uploaded_file.name + str(uploaded_file.size)
        
        if ('current_file_id' not in st.session_state or 
            st.session_state['current_file_id'] != file_id or
            st.session_state.get('last_mode') != mode_selection):

            with st.spinner(f"Analyzing with {mode_selection}..."):
                stats = get_image_stats(image_preview)
                s_b, s_c, s_s, source = calculate_ai_suggestions(*stats, use_memory=use_memory)
                
                st.session_state['current_file_id'] = file_id
                st.session_state['last_mode'] = mode_selection
                st.session_state['s_brightness'] = s_b
                st.session_state['s_contrast'] = s_c
                st.session_state['s_saturation'] = s_s
                st.session_state['original_stats'] = stats
                st.session_state['source'] = source
                
                st.success(f"Suggestion Source: {source}")

        # Controls
        st.sidebar.header("Adjustment Controls")
        st.sidebar.info(f"Values suggested by: {st.session_state.get('source', 'Unknown')}")
        
        brightness = st.sidebar.slider("Brightness", 0, 255, st.session_state['s_brightness'])
        contrast = st.sidebar.slider("Contrast", 0, 255, st.session_state['s_contrast'])
        saturation = st.sidebar.slider("Saturation", 0, 255, st.session_state['s_saturation'])
        
        # Preview
        edited_preview = apply_edits(image_preview, brightness, contrast, saturation)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_preview, caption="Original", use_container_width=True)
        with col2:
            st.image(edited_preview, caption="Enhanced", use_container_width=True)

        st.markdown("---")
        
        if st.button("Accept, Download & Train"):
            with st.spinner("Saving knowledge to Cloud and processing..."):
                final_image = apply_edits(image, brightness, contrast, saturation)
                
                buf = io.BytesIO()
                fmt = image.format if image.format else 'JPEG'
                final_image.save(buf, format=fmt, quality=95)
                byte_im = buf.getvalue()
                
                # Save to Cloud
                save_training_data(st.session_state['original_stats'], [brightness, contrast, saturation])
                
                st.balloons()
                st.download_button(
                    label="Download Final Image",
                    data=byte_im,
                    file_name=f"enhanced_{uploaded_file.name}",
                    mime=f"image/{fmt.lower()}"
                )
                st.success(f"Training Complete! The Cloud Model has been updated.")

if __name__ == "__main__":
    main()