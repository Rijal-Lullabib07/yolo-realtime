import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="FaceVision AI",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "FaceVision AI - Face Detection System v2.0"}
)

st.markdown("""
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }

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

    body, .main { background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%); color: var(--text-primary); }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%); }

    .header-container {
        text-align: center; padding: 2rem 0 1rem;
        margin-bottom: 1.5rem; border-bottom: 2px solid var(--border-color);
    }
    .header-title {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(135deg, #10b981, #6ee7b7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; letter-spacing: -0.5px; margin-bottom: 0.5rem;
    }
    .header-subtitle { font-size: 1rem; color: var(--text-secondary); font-weight: 300; letter-spacing: 1px; }

    .card {
        background: var(--bg-card); border: 1px solid var(--border-color);
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3); transition: all 0.3s ease;
    }
    .card:hover { border-color: var(--primary); box-shadow: 0 15px 40px rgba(16,185,129,0.1); transform: translateY(-2px); }
    .card-header { font-size: 1.2rem; font-weight: 600; color: var(--primary); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white; border: none; border-radius: 8px; padding: 0.75rem 2rem;
        font-weight: 600; font-size: 1rem; cursor: pointer; transition: all 0.3s ease;
        text-transform: uppercase; letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(16,185,129,0.2);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(16,185,129,0.3); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f35 0%, #0f172a 100%);
        border-right: 1px solid var(--border-color);
    }

    .image-label {
        font-size: 0.95rem; font-weight: 600; color: var(--text-secondary);
        margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .image-container {
        border-radius: 12px; overflow: hidden; border: 1px solid var(--border-color); transition: all 0.3s ease;
    }
    .image-container:hover { border-color: var(--primary); box-shadow: 0 0 20px rgba(16,185,129,0.15); }

    .confidence-item { background: var(--bg-input); border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; }
    .confidence-label { display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.9rem; }
    .confidence-bar { background: var(--border-color); border-radius: 4px; height: 6px; overflow: hidden; }
    .confidence-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
    .conf-high .confidence-fill { background: linear-gradient(90deg, var(--primary), var(--primary-light)); width: var(--conf-width); }
    .conf-medium .confidence-fill { background: linear-gradient(90deg, var(--warning), #fbbf24); width: var(--conf-width); }
    .conf-low .confidence-fill { background: linear-gradient(90deg, var(--danger), #f87171); width: var(--conf-width); }

    .faces-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 1rem; margin-top: 1rem; }
    .face-card {
        background: var(--bg-input); border: 1px solid var(--border-color);
        border-radius: 8px; padding: 1rem; text-align: center; transition: all 0.3s ease;
    }
    .face-card:hover { border-color: var(--primary); box-shadow: 0 0 20px rgba(16,185,129,0.2); transform: scale(1.05); }
    .face-conf-value { font-size: 1.3rem; font-weight: 600; color: var(--primary-light); }

    /* Anti-spoof badges */
    .badge-real  { display:inline-block; background:rgba(16,185,129,.15); color:#10b981; border:1px solid #10b981; border-radius:20px; padding:.2rem .7rem; font-size:.75rem; font-weight:700; letter-spacing:.5px; text-transform:uppercase; margin-left:.5rem; }
    .badge-spoof { display:inline-block; background:rgba(239,68,68,.15);  color:#ef4444; border:1px solid #ef4444; border-radius:20px; padding:.2rem .7rem; font-size:.75rem; font-weight:700; letter-spacing:.5px; text-transform:uppercase; margin-left:.5rem; }
    .badge-unk   { display:inline-block; background:rgba(245,158,11,.15); color:#f59e0b; border:1px solid #f59e0b; border-radius:20px; padding:.2rem .7rem; font-size:.75rem; font-weight:700; letter-spacing:.5px; text-transform:uppercase; margin-left:.5rem; }

    .stInfo, .stSuccess, .stWarning { background: var(--bg-input) !important; border-left: 3px solid var(--primary) !important; color: var(--text-primary) !important; }
    .stError { background: rgba(239,68,68,0.1) !important; border-left: 3px solid var(--danger) !important; color: var(--text-primary) !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: var(--primary); }
    .stJson { background: var(--bg-input) !important; border-radius: 8px !important; padding: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ANTI-SPOOFING ENGINE
# ============================================================================

class AntiSpoofing:
    """
    Passive anti-spoofing tanpa model tambahan → tidak ada overhead berat.
    Tiga sinyal dianalisis dari ROI wajah 64×64 px:
      1. Laplacian variance  – foto/layar cenderung flat/blur
      2. LBP texture entropy – wajah asli punya pola tekstur lebih kaya
      3. Glare ratio         – layar HP sering punya over-exposure
    Jika 2 dari 3 sinyal menunjukkan anomali → SPOOFED
    """
    LAP_THRESH   = 60.0    # di bawah ini = flat/blur
    LBP_THRESH   = 4.0     # di bawah ini = tekstur seragam
    GLARE_THRESH = 0.015   # di atas ini  = glare layar

    @staticmethod
    def _laplacian_var(gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _lbp_entropy(gray: np.ndarray) -> float:
        h, w = gray.shape
        neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        center = gray[1:-1, 1:-1].astype(np.int32)
        code   = np.zeros_like(center, dtype=np.uint8)
        for bit, (dy, dx) in enumerate(neighbors):
            nb    = gray[1+dy:h-1+dy, 1+dx:w-1+dx].astype(np.int32)
            code |= ((nb >= center).astype(np.uint8) << bit)
        lbp      = np.zeros_like(gray, dtype=np.uint8)
        lbp[1:-1, 1:-1] = code
        hist, _  = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        hist     = hist[hist > 0].astype(np.float32)
        hist    /= hist.sum()
        return float(-(hist * np.log2(hist)).sum())

    @staticmethod
    def _glare_ratio(gray: np.ndarray) -> float:
        return float((gray > 240).sum()) / gray.size

    @classmethod
    def check(cls, face_bgr: np.ndarray) -> dict:
        if face_bgr is None or face_bgr.size == 0:
            return {"result": "UNKNOWN", "lap": 0, "lbp": 0, "glare": 0}
        face_sm  = cv2.resize(face_bgr, (64, 64))
        gray     = cv2.cvtColor(face_sm, cv2.COLOR_BGR2GRAY)
        lap      = cls._laplacian_var(gray)
        lbp      = cls._lbp_entropy(gray)
        glare    = cls._glare_ratio(gray)
        flags    = int(lap < cls.LAP_THRESH) + int(lbp < cls.LBP_THRESH) + int(glare > cls.GLARE_THRESH)
        result   = "SPOOFED" if flags >= 2 else ("REAL" if flags == 0 else "UNKNOWN")
        return {"result": result, "lap": round(lap, 1), "lbp": round(lbp, 3), "glare": round(glare, 4)}

# ============================================================================
# FUNCTIONS
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)


def parse_results(result):
    dets = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append((cls_id, conf, x1, y1, x2, y2))
    return dets


def draw_boxes(image_bgr: np.ndarray, detections: list, names: dict,
               spoof_map: dict = None) -> np.ndarray:
    img = image_bgr.copy()
    for cls_id, conf, x1, y1, x2, y2 in detections:
        name     = names.get(cls_id, str(cls_id))
        sp_res   = (spoof_map or {}).get(name, "UNKNOWN")

        if sp_res == "REAL":
            color = (16, 185, 129)   # hijau
        elif sp_res == "SPOOFED":
            color = (50, 50, 239)    # merah (BGR)
        else:
            color = (245, 158, 11)   # oranye

        label = f"{name} • {conf*100:.1f}% [{sp_res}]"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        ly  = max(y1 - 10, 22)
        cv2.rectangle(img, (x1-5, ly-lsz[1]-8), (x1+lsz[0]+6, ly+5), color, -1)
        cv2.putText(img, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

    return img


def spoof_badge_html(result: str) -> str:
    if result == "REAL":
        return '<span class="badge-real">✓ Wajah Asli</span>'
    elif result == "SPOOFED":
        return '<span class="badge-spoof">✗ Foto/Layar</span>'
    return '<span class="badge-unk">? Tidak Diketahui</span>'


def create_confidence_html(conf_val: float):
    pct = min(conf_val * 100, 100)
    if conf_val >= 0.75:
        cls, col = "conf-high", "#10b981"
    elif conf_val >= 0.50:
        cls, col = "conf-medium", "#f59e0b"
    else:
        cls, col = "conf-low", "#ef4444"
    return f"""
    <div class="confidence-item {cls}">
        <div class="confidence-label">
            <span>Confidence</span>
            <span style="color:{col};font-weight:600">{pct:.1f}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="--conf-width:{pct}%"></div>
        </div>
    </div>"""

# ============================================================================
# VIDEO PROCESSOR — OPTIMIZED (no lag)
# ============================================================================

class YOLOVideoProcessor(VideoProcessorBase):
    """
    Optimasi performa realtime:
    • Frame di-resize ke 640px sebelum inference → lebih cepat
    • YOLO hanya jalan setiap SKIP frame (default 3) → ~10 fps inference
    • Hasil bbox di-scale balik ke resolusi asli → akurat
    • Frame yang di-skip pakai hasil cache → gambar tetap smooth tanpa flicker
    • Anti-spoof dijalankan hanya saat inference (bukan tiap frame)
    """
    SKIP       = 3
    INFER_W    = 640

    def __init__(self):
        self._count      = 0
        self._last_dets  = []
        self._last_spoof = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img    = frame.to_ndarray(format="bgr24")
        h, w   = img.shape[:2]
        self._count += 1

        if self._count % self.SKIP == 0:
            scale  = min(1.0, self.INFER_W / w)
            small  = cv2.resize(img, (int(w*scale), int(h*scale))) if scale < 1 else img
            res    = model.predict(small, conf=conf_thresh, iou=iou_thresh, verbose=False)
            raw    = parse_results(res[0])

            inv  = 1.0 / scale if scale < 1 else 1.0
            dets = [(c, cf, int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv))
                    for c, cf, x1, y1, x2, y2 in raw]
            self._last_dets = dets

            spoof_map = {}
            for cls_id, _, x1, y1, x2, y2 in dets:
                name = names.get(cls_id, str(cls_id))
                face = img[y1:y2, x1:x2]
                spoof_map[name] = AntiSpoofing.check(face)["result"]
            self._last_spoof = spoof_map

        annotated = draw_boxes(img, self._last_dets, names, self._last_spoof)

        # watermark
        cv2.putText(annotated, "FaceVision AI  |  Anti-Spoof ON",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (16,185,129), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ============================================================================
# MAIN APP
# ============================================================================

st.markdown("""
<div class="header-container">
    <div class="header-title">🔍 FaceVision AI</div>
    <div class="header-subtitle">Advanced Face Detection & Recognition System</div>
</div>
""", unsafe_allow_html=True)

model_path = "best.pt"
model      = load_model(model_path)

if model is None:
    st.error(f"❌ Model tidak ditemukan: {model_path}")
    st.info("Pastikan file `best.pt` ada di direktori yang sama dengan script ini.")
    st.stop()

names = model.names

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Deteksi")
    conf_thresh = st.slider("Confidence Threshold", 0.05, 1.0, 0.45, 0.01,
                            help="Minimum confidence untuk deteksi dianggap valid")
    iou_thresh  = st.slider("IoU (NMS)", 0.05, 1.0, 0.45, 0.01,
                            help="Intersection over Union untuk Non-Maximum Suppression")

    st.markdown("---")
    st.markdown("### 🛡️ Anti-Spoofing")
    spoof_enabled = st.toggle("Aktifkan Anti-Spoofing", value=True,
                              help="Deteksi apakah wajah asli atau foto dari HP/layar")
    if spoof_enabled:
        st.success("🔒 Anti-spoofing AKTIF")
    else:
        st.warning("⚠️ Anti-spoofing NONAKTIF")

    st.markdown("---")
    st.markdown("### ⚡ Performa Kamera")
    frame_skip = st.slider("Frame Skip", 1, 6, 3,
                           help="Makin tinggi = makin ringan, makin rendah = makin responsif")

# Mode selection
mode = st.radio(
    "Pilih Mode",
    ["📸 Upload Foto", "🎥 Realtime Kamera"],
    horizontal=True,
    label_visibility="collapsed"
)

# ============================================================================
# MODE: UPLOAD FOTO
# ============================================================================

if mode == "📸 Upload Foto":
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📤 Upload Gambar</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:1rem;">✨ Bisa upload hingga 10 gambar sekaligus!</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Pilih file gambar",
            type=["jpg","jpeg","png","bmp","webp"],
            label_visibility="collapsed",
            accept_multiple_files=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if not uploaded_files:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem;">
                <div style="font-size:3rem;margin-bottom:1rem;">📁</div>
                <div style="color:var(--text-secondary);font-size:1rem;">Silakan upload gambar untuk memulai deteksi</div>
                <div style="color:var(--text-secondary);font-size:0.85rem;margin-top:1rem;">Format: JPG, PNG, BMP, WEBP<br/>✓ Upload 1 hingga 10 gambar sekaligus</div>
            </div>""", unsafe_allow_html=True)
            st.stop()

    if st.button("🚀 Mulai Deteksi", type="primary", use_container_width=True, key="detect_btn"):
        total_faces = 0
        all_results = []

        with st.spinner(f"⏳ Memproses {len(uploaded_files)} gambar..."):
            for file_idx, uploaded in enumerate(uploaded_files, 1):
                st.markdown("---")

                image        = Image.open(uploaded).convert("RGB")
                img_rgb      = np.array(image)
                img_bgr      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                t0      = time.perf_counter()
                results = model.predict(img_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)
                elapsed = (time.perf_counter() - t0) * 1000

                dets = parse_results(results[0])

                # Anti-spoof setiap wajah
                spoof_map = {}
                if spoof_enabled:
                    for cls_id, _, x1, y1, x2, y2 in dets:
                        name = names.get(cls_id, str(cls_id))
                        face = img_bgr[y1:y2, x1:x2]
                        spoof_map[name] = AntiSpoofing.check(face)["result"]

                annotated_bgr = draw_boxes(img_bgr, dets, names, spoof_map if spoof_enabled else None)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                st.markdown(f'<div class="card-header">📷 Gambar {file_idx}/{len(uploaded_files)}: {uploaded.name}</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="image-label">📸 Input</div>', unsafe_allow_html=True)
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(img_rgb)
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="image-label">🎯 Hasil Deteksi</div>', unsafe_allow_html=True)
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(annotated_rgb)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                if dets:
                    st.success(f"✅ **{len(dets)} wajah** terdeteksi • {elapsed:.1f}ms")
                    total_faces += len(dets)

                    if len(dets) > 1:
                        st.markdown(f'<div class="card-header">👥 {len(dets)} Wajah Terdeteksi</div>', unsafe_allow_html=True)
                        html = '<div class="faces-grid">'
                        for cls_id, cf, x1, y1, x2, y2 in dets:
                            pname  = names.get(cls_id, f"Unknown {cls_id}")
                            pct    = cf * 100
                            color  = "#10b981" if cf >= 0.75 else "#f59e0b" if cf >= 0.50 else "#ef4444"
                            sp_res = spoof_map.get(pname, "")
                            sp_lbl = (f'<div style="font-size:.7rem;margin-top:.3rem;color:{"#10b981" if sp_res=="REAL" else "#ef4444" if sp_res=="SPOOFED" else "#f59e0b"}">'
                                      f'{"✓ ASLI" if sp_res=="REAL" else "✗ SPOOF" if sp_res=="SPOOFED" else "? UNK"}</div>') if spoof_enabled else ""
                            html += f"""<div class="face-card">
                                <div style="font-size:1.4rem;font-weight:700;color:{color}">{pname}</div>
                                <div style="font-size:.8rem;color:var(--text-secondary)">Confidence</div>
                                <div class="face-conf-value">{pct:.1f}%</div>
                                {sp_lbl}
                            </div>"""
                        html += '</div>'
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        pname  = names.get(dets[0][0], f"Unknown {dets[0][0]}")
                        sp_res = spoof_map.get(pname, "")
                        badge  = spoof_badge_html(sp_res) if spoof_enabled else ""
                        st.markdown(f'<div class="card-header">👤 1 Wajah: {pname} {badge}</div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="card-header">📊 Detail Confidence</div>', unsafe_allow_html=True)
                    for cls_id, cf, *_ in dets:
                        pname  = names.get(cls_id, str(cls_id))
                        sp_res = spoof_map.get(pname, "")
                        badge  = spoof_badge_html(sp_res) if spoof_enabled else ""
                        st.markdown(f'<div style="font-size:.85rem;color:var(--text-secondary);margin:.4rem 0 .2rem">{pname} {badge}</div>', unsafe_allow_html=True)
                        st.markdown(create_confidence_html(cf), unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander(f"📋 Data JSON – Gambar {file_idx}", expanded=False):
                        st.json([{
                            "image": uploaded.name,
                            "face_id": idx + 1,
                            "person": names.get(c, f"Unknown {c}"),
                            "class_id": int(c),
                            "confidence": round(float(cf), 4),
                            "anti_spoof": spoof_map.get(names.get(c, str(c)), "disabled"),
                            "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                            "size": {"width": int(x2-x1), "height": int(y2-y1)},
                        } for idx, (c, cf, x1, y1, x2, y2) in enumerate(dets)])

                    all_results.append({
                        "image": uploaded.name,
                        "detections": len(dets),
                        "processing_time_ms": round(elapsed, 2),
                        "faces": [{"face_id": i+1, "person": names.get(c, f"Unknown {c}"),
                                   "confidence": round(float(cf), 4),
                                   "anti_spoof": spoof_map.get(names.get(c, str(c)), "disabled")}
                                  for i, (c, cf, *_) in enumerate(dets)]
                    })
                else:
                    st.info(f"⚠️ Tidak ada wajah yang terdeteksi • {elapsed:.1f}ms")
                    all_results.append({"image": uploaded.name, "detections": 0,
                                        "processing_time_ms": round(elapsed, 2), "faces": []})
                st.markdown('</div>', unsafe_allow_html=True)

        # Summary
        st.markdown("---")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📊 Ringkasan Deteksi</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("📁 Total Gambar", len(uploaded_files))
        with c2: st.metric("👤 Total Wajah",  total_faces)
        with c3:
            avg = total_faces / len(uploaded_files) if uploaded_files else 0
            st.metric("📈 Rata-rata", f"{avg:.1f} wajah/gambar")
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Semua Hasil Deteksi (JSON)", expanded=False):
            st.json(all_results)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MODE: REALTIME KAMERA
# ============================================================================

elif mode == "🎥 Realtime Kamera":
    st.markdown("""
    <div class="card" style="text-align:center;">
        <div style="font-size:1.2rem;font-weight:700;color:var(--primary);margin-bottom:.5rem;">
            🎥 Realtime Face Detection
        </div>
        <div style="font-size:.9rem;color:var(--text-secondary)">
            Kamera berjalan di browser · Anti-Spoofing aktif · Foto dari HP/layar akan ditandai <b style="color:#ef4444">SPOOFED</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Update SKIP dari sidebar slider
    YOLOVideoProcessor.SKIP = frame_skip

    webrtc_streamer(
        key="facevision",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

st.markdown("""
<div style="text-align:center;color:var(--text-secondary);font-size:0.85rem;margin-top:2rem;">
    <div>FaceVision AI v2.1 • Powered by YOLOv11 • Anti-Spoofing Engine ✨</div>
    <div style="margin-top:.5rem;opacity:.7;">For face detection and recognition tasks</div>
</div>
""", unsafe_allow_html=True)
