import streamlit as st
import os
import sys
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.predict_video import VideoPredictor

# --- Page Config ---
st.set_page_config(
    page_title="Deepfake Defender", 
    layout="wide", 
    page_icon="�️"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #121212;
    }
    
    /* Headers */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #ffffff;
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
        background: -webkit-linear-gradient(45deg, #0d6efd, #0dcaf0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3, h4 {
        color: #e0e0e0 !important;
    }
    
    /* Cards */
    .metric-container {
        background-color: #1e1e1e;
        color: #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        transition: transform 0.2s;
        border: 1px solid #333;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0d6efd, #0d6efd);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 12px rgba(13, 110, 253, 0.4);
    }
    
    /* Upload Area */
    .stFileUploader {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #121212;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for About toggle
if 'show_about' not in st.session_state:
    st.session_state.show_about = False

def toggle_about():
    st.session_state.show_about = not st.session_state.show_about

# Layout: Title (Left) and About Button (Right)
col_header, col_btn = st.columns([6, 1])

with col_header:
    st.markdown("<h1 style='text-align: left; padding: 0;'>🛡️ Deepfake Defender</h1>", unsafe_allow_html=True)

with col_btn:
    st.write("") # Spacer
    if st.button("ℹ️ About", help="Learn about this project"):
        toggle_about()

# --- About "Popup" (Expander style) ---
if st.session_state.show_about:
    with st.container():
        st.info("ℹ️ **About This Project**")
        st.markdown("""
        **Deepfake Defender** uses **EfficientNet-B0** + **Autoencoder** to detect manipulated videos.
        
        **How it works:**
        1.  Extracts faces using MTCNN.
        2.  Attempts to reconstruct them using a model trained ONLY on real faces.
        3.  High reconstruction error = **FAKE** (Anomaly).
        
        *Built with PyTorch & Streamlit.*
        """)
        if st.button("Close"):
            toggle_about()
            st.rerun()
    st.markdown("---")

st.markdown("<h4 style='color: #6c757d;'>Upload a video or image to verify its authenticity</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Drag and drop your file here", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save to temp file (preserving extension)
    ext = os.path.splitext(uploaded_file.name)[1]
    if not ext: ext = ".mp4" # Default fallback
    
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tfile.write(uploaded_file.read())
    file_path = tfile.name
    tfile.close()

    st.markdown("---")
    
    # Layout: Media and Controls
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.subheader("📺 Input Media")
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
             st.image(file_path, use_column_width=True)
        else:
             st.video(file_path)
        
    with c2:
        st.subheader("⚙️ Analysis")
        st.write("Ready to inspect video frames for anomalies.")
        
        # Threshold Slider (Restored for Xception Control)
        # Threshold Slider (Adjusted for MSE)
        threshold = st.slider(
            "Sensitivity Threshold (MSE)", 
            min_value=0.000, 
            max_value=0.150, 
            value=0.065, 
            step=0.001,
            format="%.3f",
            help="Reconstruction Error Cutoff. Lower = Stricter (More Fakes). Higher = Lenient."
        )
        
        analyze_btn = st.button("🚀 Start Deep Scan", use_container_width=True)
    
    if analyze_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load predictor
            status_text.markdown("**Initializing...**")
            predictor = VideoPredictor()
            
            status_text.markdown("**Step 1/3: Extracting Frames & Detecting Faces...**")
            progress_bar.progress(20)
            
            # Predict with user threshold
            results = predictor.predict(file_path, threshold=threshold)
            
            status_text.markdown("**Step 3/3: Finalizing Report...**")
            progress_bar.progress(100)
            time.sleep(0.5)
            status_text.empty()
            
            # --- Results Section ---
            st.markdown("---")
            st.markdown("<h2 style='text-align: center;'>🔍 Analysis Report</h2>", unsafe_allow_html=True)
            
            if results['prediction'] == 'UNKNOWN':
                 st.warning("⚠️ No faces detected in the video. Cannot verify authenticity.")
            elif results['prediction'] == 'ERROR':
                st.error("❌ Model Error: " + "Autoencoder model not found.")
                st.info("Please ensure 'models/trained/autoencoder.pth' exists. You may need to train the model first.")
            else:
                # 1. Final Classification
                pred_color = "#dc3545" if results["prediction"] == "FAKE" else "#198754"
                pred_icon = "🚨" if results["prediction"] == "FAKE" else "✅"
                
                st.markdown(f"""
                <div style="background-color: {pred_color}; padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px;">
                    <h1 style="color: white; margin: 0;">{pred_icon} {results['prediction']}</h1>
                    <p style="font-size: 1.5rem; font-weight: bold; opacity: 1;">Confidence: {results['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

                # 2. Reconstruction Error Metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Mean Reconstruction Error", f"{results['mean_error']:.5f}", help="Lower=Real, Higher=Fake")
                with m2:
                    st.metric("Decision Threshold", f"{results['threshold']:.3f}")
                with m3:
                    decision_text = "Error > Threshold" if results['mean_error'] > results['threshold'] else "Error < Threshold"
                    st.metric("Decision Rule", decision_text)
                
                # Guidance
                if results['prediction'] == "FAKE":
                    st.error(f"Reason: Since Mean Error ({results['mean_error']:.5f}) > Threshold ({results['threshold']:.3f}), this video is flagged as FAKE.")
                else:
                    st.success(f"Reason: Since Mean Error ({results['mean_error']:.5f}) < Threshold ({results['threshold']:.5f}), this video is classified as REAL.")
                
                st.markdown("---")
                
                # 3. Frame-wise Error Graph
                st.subheader("📈 Frame-wise Reconstruction Error")
                frame_errors = results.get("frame_errors", [])
                if frame_errors:
                    fig, ax = plt.subplots(figsize=(10, 3.5))
                    # Plot Errors
                    ax.plot(frame_errors, color='#0d6efd', linewidth=2, label='Reconstruction Error')
                    # Plot Threshold
                    ax.axhline(y=results['threshold'], color='#dc3545', linestyle='--', linewidth=2, label='Threshold')
                    
                    # Highlight Danger Zone
                    if results['mean_error'] > results['threshold']:
                         ax.fill_between(range(len(frame_errors)), results['threshold'], max(frame_errors + [results['threshold'] * 1.5]), color='#dc3545', alpha=0.1)

                    ax.set_ylabel("Error (MSE)")
                    ax.set_xlabel("Video Frame Index")
                    ax.legend(loc='upper right')
                    
                    ax.set_ylim(0, max(max(frame_errors), results['threshold']) * 1.2)
                    ax.grid(True, linestyle=':', alpha=0.6)
                    
                    # Style
                    ax.set_facecolor('#f8f9fa')
                    fig.patch.set_facecolor('#f8f9fa')
                    st.pyplot(fig)
                else:
                    st.info("No frame-wise data available for graph.")

                st.markdown("---")

                # 4. Face Reconstruction Visualization
                st.subheader("👁️ Face Reconstruction Visuals (Original vs Reconstructed)")
                st.caption("Visual proof of analysis. Fakes often show blurriness or artifacts in reconstruction.")
                
                visuals = results["visuals"]
                if visuals:
                    for i, item in enumerate(visuals[:5]):
                        c1, c2 = st.columns([1, 1])
                        error = item.get('error', 0.0)
                        orig = item['orig']
                        recon = item['recon']
                        
                        start_caption = f"Frame Sample {i+1} | Error: {error:.5f}"
                        
                        with c1:
                            st.image(orig, caption="Original Detected Face", use_column_width=True)
                        with c2:
                            st.image(recon, caption="Autoencoder Reconstruction", use_column_width=True)
                        
                        st.markdown("<hr style='margin: 10px 0; opacity: 0.1;'>", unsafe_allow_html=True)
                else:
                    st.warning("No faces detected to visualize.")
                
                # 5. Video Processing Summary
                st.markdown("### 📋 Processing Summary")
                stats = results.get('stats', {})
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Frames Extracted", stats.get('frames_analyzed', 'N/A'))
                s2.metric("Faces Detected", stats.get('faces_detected', 'N/A'))
                s3.metric("Inference Time", f"{stats.get('time', 0):.2f}s")
                s4.metric("Device", results.get('device', 'CPU'))

        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            try: os.unlink(file_path)
            except: pass

# --- Footer ---
st.markdown("""
<div style="text-align: center; margin-top: 50px; font-family: 'Arial'; color: #6c757d; font-size: 0.8em;">
    <hr style="border-top: 1px solid #333; margin-bottom: 20px;">
    <p>Developed with ❤️ by Deepfake Defender Team</p>
    <p style="opacity: 0.6;">Powered by EfficientNet & Streamlit</p>
</div>
""", unsafe_allow_html=True)
