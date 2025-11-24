import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import requests  # Added for auto-downloading models
from PIL import Image
from streamlit_image_comparison import image_comparison

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Restoration Station",
    page_icon="⚙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    /* Make buttons look professional */
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
    }
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    
    /* Center layout for controls */
    div[data-testid="stVerticalBlock"] > div {
        vertical-align: middle;
    }
    
    /* Remove default top padding to pull header up */
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL CONFIG & AUTO-DOWNLOADER ---
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

PROTOTXT = os.path.join(MODELS_DIR, "colorization_deploy_v2.prototxt")
MODEL_PATH = os.path.join(MODELS_DIR, "colorization_release_v2.caffemodel")
POINTS_PATH = os.path.join(MODELS_DIR, "pts_in_hull.npy")

@st.cache_resource
def download_models():
    """
    Automatically downloads AI models from Intel's safe storage.
    Critical for GitHub/Streamlit Cloud deployment.
    """
    urls = {
        PROTOTXT: "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_deploy_v2.prototxt",
        MODEL_PATH: "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel",
        POINTS_PATH: "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/pts_in_hull.npy"
    }

    for path, url in urls.items():
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)}...")
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    st.error(f"Failed to download {os.path.basename(path)}")
            except Exception as e:
                st.error(f"Error downloading models: {e}")

# Trigger download on app start
download_models()

# --- 4. ENGINE FUNCTIONS ---
def process_image_pipeline(input_image, enable_denoise, enable_colorize, enable_sharpen, color_boost=1.5):
    img = np.array(input_image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if enable_denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    if enable_colorize:
        if os.path.exists(PROTOTXT) and os.path.exists(MODEL_PATH) and os.path.exists(POINTS_PATH):
            try:
                net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_PATH)
                pts = np.load(POINTS_PATH)

                class8 = net.getLayerId("class8_ab")
                conv8 = net.getLayerId("conv8_313_rh")
                pts = pts.transpose().reshape(2, 313, 1, 1)
                net.getLayer(class8).blobs = [pts.astype("float32")]
                net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

                scaled = img.astype("float32") / 255.0
                lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
                resized = cv2.resize(lab, (224, 224))
                L = cv2.split(resized)[0]
                L -= 50

                net.setInput(cv2.dnn.blobFromImage(L))
                ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

                ab = ab * color_boost
                ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
                L_orig = cv2.split(lab)[0]
                colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
                
                colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
                colorized = np.clip(colorized, 0, 1)
                colorized = (255 * colorized).astype("uint8")
                img = colorized
            except Exception as e:
                st.error(f"AI Error: {e}")

    if enable_sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process_video_pipeline(video_file, color_boost):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = "restored_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        processed_pil = process_image_pipeline(pil_frame, False, True, False, color_boost)
        processed_frame = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
        out.write(processed_frame)
        
        frame_idx += 1
        progress_bar.progress(min(frame_idx / total_frames, 1.0))
        status_text.caption(f"Processing Frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    return output_path

# --- 5. SIDEBAR ---
st.sidebar.markdown("### ⚙️ Pipeline Settings")

use_denoise = st.sidebar.checkbox("Step 1: Denoise (Grain Removal)", value=True)
use_colorize = st.sidebar.checkbox("Step 2: Colorize (AI)", value=True)
use_sharpen = st.sidebar.checkbox("Step 3: Sharpen (Edge Enhance)", value=True)

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About the Project"):
    st.markdown("""
    **Restoration Station** uses Deep Learning to hallucinate colors in grayscale media.
    
    - **Model:** Zhang et al. (ECCV 2016)
    - **Color Space:** CIELAB
    - **Training:** 1.3M Images
    """)

# --- 6. MAIN UI ---

# HEADER SECTION (Flexbox for Perfect Alignment)
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <div style="font-size: 4.5rem; margin-right: 20px; line-height: 1;">⚙️</div>
    <div>
        <h1 style="margin: 0; padding: 0; font-size: 3rem;">Restoration Station</h1>
        <h5 style="margin: 0; padding: 0; color: grey; font-weight: normal;">The Intelligent Historical Archive Restoration Engine</h5>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# Upload Section
uploaded_file = st.file_uploader("Upload black & white media", type=["jpg", "png", "jpeg", "mp4"])

# Logic
if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'mp4':
        st.info("🎬 Video Mode Detected")
        
        # Controls for Video
        col_v_btn, col_v_slider = st.columns([2, 1])
        with col_v_slider:
             color_boost = st.slider("Color Confidence", 0.5, 2.5, 1.5, 0.1)
        with col_v_btn:
            start_btn = st.button("Start Processing")
            
        col1, col2 = st.columns(2)
        with col1:
            st.video(uploaded_file)
            st.caption("Original")
        
        if start_btn:
            with st.spinner("Restoring Video..."):
                res_path = process_video_pipeline(uploaded_file, color_boost)
            with col2:
                st.success("Done!")
                with open(res_path, 'rb') as f:
                    st.download_button("⬇️ Download Result", f, "restored.mp4", "video/mp4")
    
    else:
        # Image Mode
        original = Image.open(uploaded_file)
        
        # Controls: Button Left, Slider Right
        st.markdown("##### Restoration Controls")
        col_btn, col_slider = st.columns([2, 1])
        
        with col_slider:
            color_boost = st.slider(
                "Color Confidence", 0.5, 2.5, 1.5, 0.1,
                help="Higher values = More vivid colors"
            )
        
        with col_btn:
            # Adding a bit of spacing to align visually with the slider
            st.write("") 
            run_btn = st.button("✨ RESTORE IMAGE")
        
        if run_btn:
            with st.spinner("Processing..."):
                restored = process_image_pipeline(original, use_denoise, use_colorize, use_sharpen, color_boost)
                
                st.markdown("### Result")
                image_comparison(
                    img1=restored,
                    img2=original,
                    label1="Restored",
                    label2="Original",
                )
                
                # Download
                from io import BytesIO
                buf = BytesIO()
                Image.fromarray(restored).save(buf, format="PNG")
                st.download_button("⬇️ Download High-Res", buf.getvalue(), "restored.png", "image/png")

else:
    # CLEAN Empty State
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <br>
            <h3>👋 Welcome!</h3>
            <p>Drag and drop a black & white photo or video above to begin.</p>
            <p style='font-size: 0.8em;'>Supported Formats: JPG, PNG, MP4</p>
        </div>
        """, 
        unsafe_allow_html=True
    )