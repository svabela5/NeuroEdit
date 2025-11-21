import streamlit as st
from PIL import Image, ImageEnhance, ImageStat
import numpy as np
import io
import json
import os

# --- Configuration ---
st.set_page_config(
    page_title="NeuroEdit AI",
    page_icon="📸",
    layout="wide"
)

DATA_FILE = "neuroedit_memory.json"

# --- Helper Functions ---

def rotate_if_vertical(img):
    """
    Ensures the image is landscape (horizontal).
    Rotates 90 degrees counter-clockwise if height > width.
    """
    width, height = img.size
    if height > width:
        return img.rotate(90, expand=True)
    return img

def get_image_stats(img):
    """
    Analyzes the image to find Mean Brightness, RMS Contrast, and Avg Saturation.
    Returns raw metrics.
    """
    # Convert to numpy for faster stats
    img_np = np.array(img)
    
    # 1. Brightness (Luma)
    # Standard Rec 601 luma conversion
    if img.mode == 'RGB':
        luma = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
    else:
        luma = img_np
    mean_brightness = np.mean(luma)
    
    # 2. Contrast (RMS Contrast - Standard Deviation of Luma)
    rms_contrast = np.std(luma)
    
    # 3. Saturation
    if img.mode == 'RGB':
        hsv_img = img.convert('HSV')
        hsv_np = np.array(hsv_img)
        # Channel 1 is Saturation
        mean_saturation = np.mean(hsv_np[:,:,1])
    else:
        mean_saturation = 0 # Grayscale has no saturation
        
    return mean_brightness, rms_contrast, mean_saturation

def save_training_data(stats, values):
    """
    Saves the image stats and user preferred values to a JSON file.
    This persists the 'Memory' of the AI.
    """
    data_point = {
        "inputs": [float(x) for x in stats],
        "labels": [int(x) for x in values]
    }
    
    history = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                history = json.load(f)
        except (json.JSONDecodeError, ValueError):
            history = []
            
    history.append(data_point)
    
    with open(DATA_FILE, 'w') as f:
        json.dump(history, f)

def predict_from_memory(current_stats):
    """
    Uses K-Nearest Neighbors (KNN) to find similar past images
    and suggests values based on what the user chose previously.
    """
    if not os.path.exists(DATA_FILE):
        return None
        
    try:
        with open(DATA_FILE, 'r') as f:
            history = json.load(f)
    except:
        return None
            
    if len(history) < 3: # Need a few points to be reliable
        return None
        
    # Convert history to numpy arrays for vectorized calculation
    history_inputs = np.array([h['inputs'] for h in history])
    history_labels = np.array([h['labels'] for h in history])
    current_inputs = np.array(current_stats)
    
    # Calculate Euclidean distance between current image stats and all past images
    # (We are finding images with similar brightness/contrast/saturation profiles)
    distances = np.linalg.norm(history_inputs - current_inputs, axis=1)
    
    # Get indices of K nearest neighbors (K=5 or less if history is small)
    k = min(5, len(history))
    nearest_indices = distances.argsort()[:k]
    
    # Average the labels (slider values) of the nearest neighbors
    nearest_labels = history_labels[nearest_indices]
    prediction = np.mean(nearest_labels, axis=0)
    
    return int(prediction[0]), int(prediction[1]), int(prediction[2])

def calculate_ai_suggestions(mean_b, rms_c, mean_s):
    """
    Hybrid AI:
    1. Tries to use Memory (KNN) first.
    2. Falls back to Heuristics (Math) if no memory exists.
    """
    
    # Try to learn from history
    memory_prediction = predict_from_memory((mean_b, rms_c, mean_s))
    
    if memory_prediction:
        return memory_prediction[0], memory_prediction[1], memory_prediction[2], "🧠 Learned Memory"

    # --- Fallback: Heuristic Logic ---
    # Target a mean brightness of roughly 100-130 (out of 255)
    target_b = 115.0
    if mean_b < 10: mean_b = 10
    b_factor = target_b / mean_b
    b_factor = max(0.5, min(2.0, b_factor))
    suggested_brightness = int(b_factor * 128)
    
    # Contrast Logic
    target_c = 60.0
    if rms_c < 10: rms_c = 10
    c_factor = target_c / rms_c
    c_factor = max(0.8, min(1.8, c_factor))
    suggested_contrast = int(c_factor * 128)
    
    # Saturation Logic
    target_s = 90.0
    if mean_s < 5: 
        s_factor = 1.0 
    else:
        s_factor = target_s / mean_s
    s_factor = max(0.0, min(1.6, s_factor))
    suggested_saturation = int(s_factor * 128)
    
    return suggested_brightness, suggested_contrast, suggested_saturation, "📐 Mathematical Heuristics"

def apply_edits(img, brightness_val, contrast_val, saturation_val):
    """
    Applies edits based on 0-255 scale.
    """
    # Convert 0-255 scale to 0.0-2.0 multiplier
    b_factor = brightness_val / 128.0
    c_factor = contrast_val / 128.0
    s_factor = saturation_val / 128.0
    
    # Order: Saturation -> Contrast -> Brightness
    edited = img.copy()
    
    enhancer_s = ImageEnhance.Color(edited)
    edited = enhancer_s.enhance(s_factor)
    
    enhancer_c = ImageEnhance.Contrast(edited)
    edited = enhancer_c.enhance(c_factor)
    
    enhancer_b = ImageEnhance.Brightness(edited)
    edited = enhancer_b.enhance(b_factor)
    
    return edited

# --- Main UI ---

def main():
    st.title("🧠 NeuroEdit: Adaptive AI Photo Enhancer")
    st.markdown("""
    Upload an image. The AI will suggest values based on past training data.
    If you adjust the values and accept, the AI will learn from your preferences.
    """)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 1. Load Image
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        # 2. Pre-processing (Scale & Rotate)
        image = rotate_if_vertical(image)
        
        # Create a preview thumbnail (1080p horizontal max)
        preview_width = 1080
        w_percent = (preview_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        image_preview = image.resize((preview_width, h_size), Image.Resampling.LANCZOS)
        
        # 3. AI Analysis
        file_id = uploaded_file.name + str(uploaded_file.size)
        
        if 'current_file_id' not in st.session_state or st.session_state['current_file_id'] != file_id:
            with st.spinner("Consulting neural memory bank..."):
                # Analyze
                stats = get_image_stats(image_preview)
                s_b, s_c, s_s, source = calculate_ai_suggestions(*stats)
                
                # Store in session
                st.session_state['current_file_id'] = file_id
                st.session_state['s_brightness'] = s_b
                st.session_state['s_contrast'] = s_c
                st.session_state['s_saturation'] = s_s
                st.session_state['original_stats'] = stats
                st.session_state['source'] = source
                
                st.success(f"Suggestion Source: {source}")

        # --- Sidebar Controls ---
        st.sidebar.header("Adjustment Controls")
        st.sidebar.info(f"Values suggested by: {st.session_state.get('source', 'Unknown')}")
        
        # Sliders default to the session state AI values
        brightness = st.sidebar.slider("Brightness", 0, 255, st.session_state['s_brightness'])
        contrast = st.sidebar.slider("Contrast", 0, 255, st.session_state['s_contrast'])
        saturation = st.sidebar.slider("Saturation", 0, 255, st.session_state['s_saturation'])
        
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Inputs:** B:{st.session_state['s_brightness']} C:{st.session_state['s_contrast']} S:{st.session_state['s_saturation']}")
        
        # --- Preview Area ---
        edited_preview = apply_edits(image_preview, brightness, contrast, saturation)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            st.image(image_preview, use_container_width=True)
            
        with col2:
            st.subheader("AI Enhanced + User Edits")
            st.image(edited_preview, use_container_width=True)

        # --- Acceptance & Download ---
        st.markdown("---")
        
        if st.button("Accept, Download & Train"):
            with st.spinner("Saving knowledge and processing image..."):
                # 1. Apply to original full size image
                final_image = apply_edits(image, brightness, contrast, saturation)
                
                # 2. Prepare Download
                buf = io.BytesIO()
                fmt = image.format if image.format else 'JPEG'
                final_image.save(buf, format=fmt, quality=95)
                byte_im = buf.getvalue()
                
                # 3. Train (Real persistence)
                # We save the ORIGINAL stats (inputs) and the FINAL user sliders (labels)
                save_training_data(st.session_state['original_stats'], [brightness, contrast, saturation])
                
                # 4. Show Download Button
                st.balloons()
                st.download_button(
                    label="Download Final Image",
                    data=byte_im,
                    file_name=f"enhanced_{uploaded_file.name}",
                    mime=f"image/{fmt.lower()}"
                )
                st.success(f"Training Complete! The AI has learned from your preference for this image type.")

if __name__ == "__main__":
    main()