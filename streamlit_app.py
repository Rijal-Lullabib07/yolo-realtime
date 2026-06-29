import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="FaceVision AI",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "FaceVision AI - Face Detection System v2.0"}
)

# Custom CSS untuk UI yang elegant
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    :root {
        --primary: #10b981;
        --primary-dark: #059669;
        --primary-light: #6ee7b7;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --bg-input: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --border-color: #475569;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }

    body, .main {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%);
        color: var(--text-primary);
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%);
    }

    /* Header & Title */
    .header-container {
        text-align: center;
        padding: 2rem 0 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid var(--border-color);
    }

    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981, #6ee7b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 300;
        letter-spacing: 1px;
    }

    /* Cards */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .card:hover {
        border-color: var(--primary);
        box-shadow: 0 15px 40px rgba(16, 185, 129, 0.1);
        transform: translateY(-2px);
    }

    .card-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Sliders */
    .stSlider {
        padding: 1rem 0;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
    }

    /* Radio buttons */
    .stRadio > label > div:first-child {
        background-color: var(--bg-input) !important;
        border: 1.5px solid var(--border-color) !important;
    }

    .stRadio > label > div:first-child:has(input:checked) {
        background-color: var(--primary) !important;
        border-color: var(--primary) !important;
    }

    /* Detection Stats */
    .detection-stat {
        background: var(--bg-input);
        border-left: 3px solid var(--primary);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .stat-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
    }

    /* Confidence bars */
    .confidence-container {
        margin: 1.5rem 0;
    }

    .confidence-item {
        background: var(--bg-input);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    .confidence-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }

    .confidence-bar {
        background: var(--border-color);
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }

    .conf-high .confidence-fill {
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
        width: var(--conf-width);
    }

    .conf-medium .confidence-fill {
        background: linear-gradient(90deg, var(--warning), #fbbf24);
        width: var(--conf-width);
    }

    .conf-low .confidence-fill {
        background: linear-gradient(90deg, var(--danger), #f87171);
        width: var(--conf-width);
    }

    /* Grid untuk multiple faces */
    .faces-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .face-card {
        background: var(--bg-input);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .face-card:hover {
        border-color: var(--primary);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
        transform: scale(1.05);
    }

    .face-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }

    .face-conf {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    .face-conf-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--primary-light);
    }

    /* Progress animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    .shimmer-loading {
        background: linear-gradient(90deg, var(--bg-input) 25%, var(--border-color) 50%, var(--bg-input) 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f35 0%, #0f172a 100%);
        border-right: 1px solid var(--border-color);
    }

    /* Info boxes */
    .stInfo, .stSuccess, .stWarning {
        background: var(--bg-input) !important;
        border-left: 3px solid var(--primary) !important;
        color: var(--text-primary) !important;
    }

    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border-left: 3px solid var(--danger) !important;
        color: var(--text-primary) !important;
    }

    /* Markdown styling */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--primary);
    }

    /* Spinners & Loading */
    .stSpinner {
        color: var(--primary) !important;
    }

    /* Image containers */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }

    .image-container:hover {
        border-color: var(--primary);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }

    .image-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* JSON results */
    .stJson {
        background: var(--bg-input) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCTIONS
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load YOLO model dengan error handling"""
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)


def parse_results(result):
    """Parse hasil deteksi YOLO"""
    dets = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append((cls_id, conf, x1, y1, x2, y2))
    return dets


def draw_boxes(image_bgr: np.ndarray, detections: list, names: dict) -> np.ndarray:
    """Draw detection boxes dengan color coding berdasarkan confidence"""
    img = image_bgr.copy()
    for idx, (cls_id, conf, x1, y1, x2, y2) in enumerate(detections):
        name = names.get(cls_id, str(cls_id))
        label = f"{name} • {conf * 100:.1f}%"

        # Color based on confidence
        if conf >= 0.75:
            color = (16, 185, 129)  # Green
            thickness = 3
        elif conf >= 0.50:
            color = (245, 158, 11)  # Orange
            thickness = 2
        else:
            color = (239, 68, 68)  # Red
            thickness = 2

        # Draw rectangle dengan glow effect
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label dengan background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = max(y1 - 10, 20)
        label_x = x1
        
        cv2.rectangle(
            img,
            (label_x - 5, label_y - label_size[1] - 8),
            (label_x + label_size[0] + 5, label_y + 5),
            color,
            -1
        )
        
        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def create_confidence_html(conf_val: float, max_width: str = "100%"):
    """Create HTML untuk confidence bar"""
    conf_pct = min(conf_val * 100, 100)
    
    if conf_val >= 0.75:
        level_class = "conf-high"
        color_hex = "#10b981"
    elif conf_val >= 0.50:
        level_class = "conf-medium"
        color_hex = "#f59e0b"
    else:
        level_class = "conf-low"
        color_hex = "#ef4444"
    
    return f"""
    <div class="confidence-item {level_class}">
        <div class="confidence-label">
            <span>Face Confidence</span>
            <span style="color: {color_hex}; font-weight: 600;">{conf_pct:.1f}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="--conf-width: {conf_pct}%"></div>
        </div>
    </div>
    """


# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🔍 FaceVision AI</div>
    <div class="header-subtitle">Advanced Face Detection & Recognition System</div>
</div>
""", unsafe_allow_html=True)

# Load model
model_path = "best.pt"
model = load_model(model_path)

if model is None:
    st.error(f"❌ Model tidak ditemukan: {model_path}")
    st.info("Pastikan file `best.pt` ada di direktori yang sama dengan script ini.")
    st.stop()

names = model.names

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Deteksi")
    conf = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.45,
        step=0.01,
        help="Minimum confidence untuk sebuah deteksi dianggap valid"
    )
    iou = st.slider(
        "IoU (NMS)",
        min_value=0.05,
        max_value=1.0,
        value=0.45,
        step=0.01,
        help="Intersection over Union untuk Non-Maximum Suppression"
    )

# Mode selection
mode = st.radio(
    "Pilih Mode",
    ["📸 Upload Foto", "🎥 Realtime Kamera"],
    horizontal=True,
    label_visibility="collapsed"
)

# ============================================================================
# MODE: UPLOAD FOTO (MULTI-IMAGE)
# ============================================================================

if mode == "📸 Upload Foto":
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📤 Upload Gambar</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem;">✨ Bisa upload hingga 10 gambar sekaligus!</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Pilih file gambar",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
            accept_multiple_files=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if not uploaded_files:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
                <div style="color: var(--text-secondary); font-size: 1rem;">
                    Silakan upload gambar untuk memulai deteksi wajah
                </div>
                <div style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 1rem;">
                    Format: JPG, PNG, BMP, WEBP<br/>
                    ✓ Upload 1 hingga 10 gambar sekaligus
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

    # Detection button
    if st.button("🚀 Mulai Deteksi", type="primary", use_container_width=True, key="detect_btn"):
        total_faces = 0
        all_results = []
        
        with st.spinner(f"⏳ Memproses {len(uploaded_files)} gambar..."):
            for file_idx, uploaded in enumerate(uploaded_files, 1):
                st.markdown("---")
                
                # Process image
                image = Image.open(uploaded).convert("RGB")
                img_rgb = np.array(image)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # Deteksi
                t0 = time.perf_counter()
                results = model.predict(img_bgr, conf=conf, iou=iou, verbose=False)
                elapsed = (time.perf_counter() - t0) * 1000

                dets = parse_results(results[0])
                annotated_bgr = draw_boxes(img_bgr, dets, names)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                # Display untuk setiap gambar
                st.markdown(f'<div class="card-header">📷 Gambar {file_idx}/{len(uploaded_files)}: {uploaded.name}</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="image-label">📸 Input</div>', unsafe_allow_html=True)
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(img_rgb)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="image-label">🎯 Hasil Deteksi</div>', unsafe_allow_html=True)
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(annotated_rgb)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Results section per gambar
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                if dets:
                    st.success(f"✅ **{len(dets)} wajah** terdeteksi • {elapsed:.1f}ms")
                    total_faces += len(dets)
                    
                    # Multiple faces visualization
                    if len(dets) > 1:
                        st.markdown(f'<div class="card-header">👥 {len(dets)} Wajah Terdeteksi</div>', unsafe_allow_html=True)
                        
                        faces_html = '<div class="faces-grid">'
                        for idx, (cls_id, conf, x1, y1, x2, y2) in enumerate(dets):
                            conf_pct = conf * 100
                            person_name = names.get(cls_id, f"Unknown {cls_id}")
                            color = "#10b981" if conf >= 0.75 else "#f59e0b" if conf >= 0.50 else "#ef4444"
                            faces_html += f"""
                            <div class="face-card">
                                <div class="face-number" style="color: {color}; font-size: 1.5rem;">{person_name}</div>
                                <div class="face-conf">
                                    <div style="font-size: 0.8rem; color: var(--text-secondary);">Confidence</div>
                                    <div class="face-conf-value">{conf_pct:.1f}%</div>
                                </div>
                            </div>
                            """
                        faces_html += '</div>'
                        st.markdown(faces_html, unsafe_allow_html=True)
                    else:
                        person_name = names.get(dets[0][0], f"Unknown {dets[0][0]}")
                        st.markdown(f'<div class="card-header">👤 1 Wajah: {person_name}</div>', unsafe_allow_html=True)

                    # Confidence bars
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="card-header">📊 Detail Confidence</div>', unsafe_allow_html=True)
                    
                    for idx, (cls_id, conf, x1, y1, x2, y2) in enumerate(dets):
                        st.markdown(create_confidence_html(conf), unsafe_allow_html=True)

                    # JSON results
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander(f"📋 Data JSON - Gambar {file_idx}", expanded=False):
                        st.json(
                            [
                                {
                                    "image": uploaded.name,
                                    "face_id": idx + 1,
                                    "person": names.get(c, f"Unknown {c}"),
                                    "class_id": int(c),
                                    "confidence": round(float(cf), 4),
                                    "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                                    "size": {"width": int(x2 - x1), "height": int(y2 - y1)}
                                }
                                for idx, (c, cf, x1, y1, x2, y2) in enumerate(dets)
                            ]
                        )
                    
                    all_results.append({
                        "image": uploaded.name,
                        "detections": len(dets),
                        "processing_time_ms": round(elapsed, 2),
                        "faces": [
                            {
                                "face_id": idx + 1,
                                "person": names.get(c, f"Unknown {c}"),
                                "class_id": int(c),
                                "confidence": round(float(cf), 4),
                                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                            }
                            for idx, (c, cf, x1, y1, x2, y2) in enumerate(dets)
                        ]
                    })
                else:
                    st.info(f"⚠️ Tidak ada wajah yang terdeteksi • {elapsed:.1f}ms")
                    all_results.append({
                        "image": uploaded.name,
                        "detections": 0,
                        "processing_time_ms": round(elapsed, 2),
                        "faces": []
                    })

                st.markdown('</div>', unsafe_allow_html=True)

        # Summary
        st.markdown("---")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-header">📊 Ringkasan Deteksi</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Total Gambar", len(uploaded_files))
        with col2:
            st.metric("👤 Total Wajah", total_faces)
        with col3:
            avg_faces = total_faces / len(uploaded_files) if uploaded_files else 0
            st.metric("📈 Rata-rata", f"{avg_faces:.1f} wajah/gambar")
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Semua Hasil Deteksi (JSON)", expanded=False):
            st.json(all_results)
        
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# MODE: REALTIME KAMERA
# ============================================================================

elif mode == "🎥 Realtime Kamera":
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div style="font-size: 1.1rem; margin-bottom: 1rem;">
            🎥 Mode kamera memerlukan akses ke webcam Anda
        </div>
        <div style="color: var(--text-secondary); font-size: 0.9rem;">
            Pastikan webcam sudah aktif dan diizinkan mengakses browser
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        run = st.button("▶️ Mulai", key="start_cam", use_container_width=True)
    with col2:
        st.empty()
    with col3:
        stop = st.button("⏹️ Stop", disabled=not run, key="stop_cam", use_container_width=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Tidak dapat mengakses webcam. Pastikan webcam tersedia dan diizinkan.")
    else:
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        stats_placeholder = st.empty()

        frame_count = 0
        skip_frames = 3

        while run and not stop:
            ret, frame_bgr = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame_count += 1
            dets = []  # Initialize dets

            # Skip frames untuk performa
            if frame_count % skip_frames == 0:
                t0 = time.perf_counter()
                results = model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)
                elapsed = (time.perf_counter() - t0) * 1000

                dets = parse_results(results[0])
                annotated_bgr = draw_boxes(frame_bgr, dets, names)
            else:
                annotated_bgr = frame_bgr
                elapsed = 0

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels="RGB")

            # Info
            if dets:
                mean_conf = float(np.mean([cf for (_c, cf, *_rest) in dets]))
                info_placeholder.success(
                    f"✅ **{len(dets)} wajah** · Confidence: **{mean_conf*100:.1f}%** · {elapsed:.1f}ms"
                )
                
                # Stats grid
                faces_html = '<div class="faces-grid">'
                for idx, (cls_id, conf, *_) in enumerate(dets):
                    conf_pct = conf * 100
                    person_name = names.get(cls_id, f"Unknown {cls_id}")
                    color = "#10b981" if conf >= 0.75 else "#f59e0b" if conf >= 0.50 else "#ef4444"
                    faces_html += f"""
                    <div class="face-card">
                        <div class="face-number" style="color: {color}; font-size: 1.5rem;">{person_name}</div>
                        <div class="face-conf-value">{conf_pct:.0f}%</div>
                    </div>
                    """
                faces_html += '</div>'
                stats_placeholder.markdown(faces_html, unsafe_allow_html=True)
            else:
                info_placeholder.info(f"📷 Tidak ada wajah · {elapsed:.1f}ms")
                stats_placeholder.empty()

            time.sleep(0.01)

        cap.release()
        st.info("✋ Kamera dihentikan")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); font-size: 0.85rem; margin-top: 2rem;">
    <div>FaceVision AI v2.1 • Powered by YOLOv11 • Multi-Image Support ✨</div>
    <div style="margin-top: 0.5rem; opacity: 0.7;">For face detection and recognition tasks</div>
</div>
""", unsafe_allow_html=True)