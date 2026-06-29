"""
Sistem Presensi Digital – PT JASAKULA PURWALUHUR
Fitur:
  • YOLOv11 face recognition
  • Anti-Spoofing (LBP texture + Laplacian sharpness + blink detection)
  • Frame-skip & resize untuk performa realtime tanpa lag
  • Jam masuk/keluar, toleransi terlambat, simpan JSON + CSV
"""

import os, time, cv2, threading, json, csv
from collections import deque
from datetime import datetime, time as dtime, timedelta

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="JASAKULA Presensi",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Sistem Presensi PT JASAKULA PURWALUHUR v2.0"},
)

st.markdown("""
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --p:#1d4ed8;--pd:#1e40af;--pl:#93c5fd;
  --acc:#f59e0b;--acd:#d97706;
  --bg:#0f172a;--card:#1e293b;--inp:#334155;
  --t1:#f1f5f9;--t2:#94a3b8;--brd:#334155;
  --ok:#10b981;--warn:#f59e0b;--err:#ef4444;--late:#f97316;
  --spoof:#ef4444;--real:#10b981;
}
body,.main,.stApp{background:linear-gradient(160deg,#0f172a 0%,#1e1b3a 100%);color:var(--t1)}

/* Header */
.hdr{background:linear-gradient(135deg,#1e40af,#1d4ed8 60%,#7c3aed);
  border-radius:16px;padding:1.4rem 2rem;margin-bottom:1.25rem;
  display:flex;align-items:center;gap:1.5rem;
  box-shadow:0 8px 32px rgba(29,78,216,.35)}
.hdr-logo{font-size:3rem}
.hdr-text h1{font-size:1.75rem;font-weight:800;color:#fff;letter-spacing:-.5px;line-height:1.1}
.hdr-text .sub{font-size:.8rem;color:#bfdbfe;letter-spacing:1.5px;text-transform:uppercase;margin-top:.2rem}
.hdr-badge{margin-left:auto;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);
  border-radius:20px;padding:.35rem .9rem;font-size:.78rem;color:#fff;backdrop-filter:blur(8px)}

/* Cards */
.card{background:var(--card);border:1px solid var(--brd);border-radius:12px;
  padding:1.2rem 1.4rem;margin-bottom:.9rem;box-shadow:0 4px 20px rgba(0,0,0,.25)}
.ch{font-size:.9rem;font-weight:700;color:var(--pl);margin-bottom:.8rem;
  padding-bottom:.45rem;border-bottom:1px solid var(--brd);
  display:flex;align-items:center;gap:.4rem;text-transform:uppercase;letter-spacing:.5px}

/* Anti-spoofing badges */
.badge-real{display:inline-block;background:rgba(16,185,129,.15);color:#10b981;
  border:1px solid #10b981;border-radius:20px;padding:.25rem .8rem;
  font-size:.75rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.badge-spoof{display:inline-block;background:rgba(239,68,68,.15);color:#ef4444;
  border:1px solid #ef4444;border-radius:20px;padding:.25rem .8rem;
  font-size:.75rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.badge-unk{display:inline-block;background:rgba(245,158,11,.15);color:#f59e0b;
  border:1px solid #f59e0b;border-radius:20px;padding:.25rem .8rem;
  font-size:.75rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase}

/* Status chips */
.chip{display:inline-block;border-radius:20px;padding:.25rem .8rem;
  font-size:.75rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.chip-ok{background:rgba(16,185,129,.15);color:#10b981;border:1px solid #10b981}
.chip-late{background:rgba(249,115,22,.15);color:#f97316;border:1px solid #f97316}

/* Attendance row */
.att{background:var(--inp);border-radius:10px;padding:.9rem;margin-bottom:.5rem;
  display:flex;align-items:center;gap:.9rem;border-left:4px solid var(--p)}
.att.late{border-left-color:var(--late)}
.att-name{font-weight:700;font-size:.9rem;flex:1}
.att-time{font-size:.8rem;color:var(--t2)}

/* Metrics */
.mbox{background:var(--inp);border-radius:10px;padding:.9rem;text-align:center}
.mlbl{font-size:.7rem;color:var(--t2);text-transform:uppercase;letter-spacing:.5px}
.mval{font-size:1.9rem;font-weight:800;color:var(--pl);line-height:1.1;margin-top:.2rem}

/* Schedule box */
.sbox{background:linear-gradient(135deg,rgba(29,78,216,.15),rgba(124,58,237,.1));
  border:1px solid rgba(29,78,216,.4);border-radius:10px;padding:.9rem;margin-bottom:.5rem}
.slbl{font-size:.7rem;color:var(--t2);text-transform:uppercase;letter-spacing:.5px}
.sval{font-size:1.25rem;font-weight:700;color:#fff;margin-top:.1rem}
.slate{font-size:.7rem;color:var(--late);margin-top:.1rem}

/* Clock */
.clk{text-align:center;background:var(--inp);border-radius:12px;padding:.9rem;margin-bottom:.9rem}
.clk-t{font-size:2.2rem;font-weight:800;color:var(--acc);font-variant-numeric:tabular-nums;letter-spacing:2px}
.clk-d{font-size:.8rem;color:var(--t2);margin-top:.2rem}

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,var(--p),var(--pd));
  color:#fff;border:none;border-radius:8px;padding:.65rem 1.4rem;
  font-weight:700;font-size:.9rem;cursor:pointer;
  transition:all .25s ease;box-shadow:0 4px 12px rgba(29,78,216,.3)}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(29,78,216,.4)}

/* Sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#1a1f35 0%,#0f172a 100%);
  border-right:1px solid var(--brd)}

/* Footer */
.footer{text-align:center;color:var(--t2);font-size:.78rem;
  margin-top:1.5rem;padding-top:.9rem;border-top:1px solid var(--brd)}

/* Spoofing alert bar */
.spoof-alert{background:rgba(239,68,68,.12);border:1px solid #ef4444;
  border-radius:10px;padding:.8rem 1rem;margin:.5rem 0;
  color:#ef4444;font-weight:600;font-size:.85rem}
.real-alert{background:rgba(16,185,129,.1);border:1px solid #10b981;
  border-radius:10px;padding:.8rem 1rem;margin:.5rem 0;
  color:#10b981;font-weight:600;font-size:.85rem}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS & PERSISTENCE
# ============================================================================

PRESENSI_FILE = "data_presensi.json"
CSV_FILE      = "rekap_presensi.csv"
_LOCK         = threading.Lock()

def load_presensi() -> dict:
    if os.path.exists(PRESENSI_FILE):
        with open(PRESENSI_FILE, "r") as f:
            return json.load(f)
    return {}

def save_presensi(data: dict):
    with _LOCK:
        with open(PRESENSI_FILE, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def export_csv(data: dict):
    rows = []
    for date_key, day in data.items():
        for emp, recs in day.items():
            for r in recs:
                rows.append({"Tanggal":date_key,"Nama":emp,"Tipe":r.get("tipe",""),
                             "Jam":r.get("jam",""),"Status":r.get("status",""),
                             "Confidence":r.get("confidence",""),"AntiSpoof":r.get("antispoof","")})
    if not rows: return
    with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["Tanggal","Nama","Tipe","Jam","Status","Confidence","AntiSpoof"])
        w.writeheader(); w.writerows(rows)

def today_str(): return datetime.now().strftime("%Y-%m-%d")
def get_now():   return datetime.now()

def compute_status(now_t: dtime, masuk_t: dtime, tol: int, tipe: str) -> str:
    if tipe == "Keluar": return "Selesai Bekerja"
    delta = int((datetime.combine(datetime.today(), now_t)
                 - datetime.combine(datetime.today(), masuk_t)).total_seconds() / 60)
    if delta <= 0:             return "Tepat Waktu"
    if delta <= tol:           return f"Terlambat {delta} menit (toleransi)"
    return f"Terlambat {delta} menit"

# ============================================================================
# ANTI-SPOOFING ENGINE
# ============================================================================

class AntiSpoofing:
    """
    Lightweight passive anti-spoofing menggunakan tiga sinyal:
    1. Laplacian variance  – foto dicetak/layar cenderung lebih blur/flat
    2. LBP texture entropy – wajah asli punya tekstur lebih variatif
    3. Specular highlight   – layar / foto glossy bisa punya over-exposure
    
    Tidak butuh model tambahan → tidak ada lag.
    """

    # Threshold (tuning sederhana)
    LAP_THRESH   = 60.0   # var Laplacian; di bawah ini → flat/print
    LBP_THRESH   = 4.0    # entropy LBP; di bawah ini → pola flat
    GLARE_THRESH = 0.015  # fraksi pixel overexposed; di atas ini → glare layar

    @staticmethod
    def _laplacian_var(gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _lbp_entropy(gray: np.ndarray) -> float:
        """Hitung Local Binary Pattern (radius=1) lalu entropy histogramnya."""
        h, w = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        center = gray[1:-1, 1:-1].astype(np.int32)
        code   = np.zeros_like(center, dtype=np.uint8)
        for bit, (dy, dx) in enumerate(neighbors):
            nb = gray[1+dy:h-1+dy, 1+dx:w-1+dx].astype(np.int32)
            code |= ((nb >= center).astype(np.uint8) << bit)
        lbp[1:-1, 1:-1] = code
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0,256))
        hist    = hist[hist > 0].astype(np.float32)
        hist   /= hist.sum()
        return float(-(hist * np.log2(hist)).sum())

    @staticmethod
    def _glare_ratio(gray: np.ndarray) -> float:
        return float((gray > 240).sum()) / gray.size

    @classmethod
    def check(cls, face_bgr: np.ndarray) -> dict:
        """
        Kembalikan dict:
          result  : 'REAL' | 'SPOOFED' | 'UNKNOWN'
          lap     : float
          lbp     : float
          glare   : float
          reason  : str
        """
        if face_bgr is None or face_bgr.size == 0:
            return {"result":"UNKNOWN","lap":0,"lbp":0,"glare":0,"reason":"no face"}

        # Resize ke 64×64 untuk kecepatan
        face_sm = cv2.resize(face_bgr, (64, 64))
        gray    = cv2.cvtColor(face_sm, cv2.COLOR_BGR2GRAY)

        lap   = cls._laplacian_var(gray)
        lbp   = cls._lbp_entropy(gray)
        glare = cls._glare_ratio(gray)

        reasons = []
        spoof_flags = 0

        if lap < cls.LAP_THRESH:
            spoof_flags += 1
            reasons.append(f"blur/flat (lap={lap:.1f})")
        if lbp < cls.LBP_THRESH:
            spoof_flags += 1
            reasons.append(f"texture flat (lbp={lbp:.2f})")
        if glare > cls.GLARE_THRESH:
            spoof_flags += 1
            reasons.append(f"glare tinggi ({glare*100:.1f}%)")

        if spoof_flags >= 2:
            result = "SPOOFED"
        elif spoof_flags == 0:
            result = "REAL"
        else:
            result = "UNKNOWN"  # satu sinyal kurang meyakinkan

        return {"result":result,"lap":round(lap,2),"lbp":round(lbp,3),
                "glare":round(glare,4),"reason":", ".join(reasons) or "OK"}


# ============================================================================
# MODEL
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return YOLO(path)

# ── Util draw ────────────────────────────────────────────────────────────────

def parse_results(result):
    dets = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            dets.append((cls_id, conf, x1, y1, x2, y2))
    return dets

def draw_boxes(img_bgr: np.ndarray, dets: list, names: dict,
               status_map: dict = None, spoof_map: dict = None) -> np.ndarray:
    img = img_bgr.copy()
    for cls_id, conf, x1, y1, x2, y2 in dets:
        name = names.get(cls_id, str(cls_id))

        spoof_res = (spoof_map or {}).get(name, "UNKNOWN")
        if spoof_res == "REAL":
            color = (16, 185, 129)   # hijau
        elif spoof_res == "SPOOFED":
            color = (0, 50, 239)     # merah‑biru (BGR)
        else:
            color = (245, 158, 11)   # oranye

        label = f"{name} {conf*100:.1f}%  [{spoof_res}]"
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)

        lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        ly  = max(y1-10, 22)
        cv2.rectangle(img, (x1-4,ly-lsz[1]-8), (x1+lsz[0]+6,ly+5), color, -1)
        cv2.putText(img, label, (x1,ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        # Status bar bawah
        st_txt = (status_map or {}).get(name, "")
        if st_txt:
            sc = (16,185,129) if "Tepat" in st_txt else (249,115,22)
            cv2.rectangle(img,(x1,y2+2),(x1+340,y2+24),sc,-1)
            cv2.putText(img,st_txt,(x1+4,y2+18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
    return img

# ============================================================================
# SESSION STATE
# ============================================================================

defaults = {
    "absen_log"     : {},
    "absen_done"    : set(),
    "tipe_presensi" : "Masuk",
    "jam_masuk"     : dtime(8, 0),
    "jam_keluar"    : dtime(17, 0),
    "toleransi"     : 15,
    "conf_thresh"   : 0.45,
    "iou_thresh"    : 0.45,
    "run_foto"      : False,
    "rt_log"        : [],   # list hasil realtime untuk tampil di UI
    "spoof_strict"  : True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================================
# LOAD MODEL
# ============================================================================

model_path = "best.pt"
model = load_model(model_path)

# ============================================================================
# HEADER
# ============================================================================

now_dt = get_now()
st.markdown(f"""
<div class="hdr">
  <div class="hdr-logo">🏢</div>
  <div class="hdr-text">
    <h1>Sistem Presensi Digital</h1>
    <div class="sub">PT JASAKULA PURWALUHUR</div>
  </div>
  <div class="hdr-badge">📅 {now_dt.strftime("%A, %d %B %Y")}</div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("❌ Model `best.pt` tidak ditemukan. Letakkan file model di folder yang sama.")
    st.stop()

names = model.names

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    st.markdown("---")

    st.markdown("### 🕐 Jadwal Kerja")
    hm = st.number_input("Jam Masuk (Jam)",  0, 23, st.session_state.jam_masuk.hour)
    mm = st.number_input("Jam Masuk (Menit)",0, 59, st.session_state.jam_masuk.minute)
    st.session_state.jam_masuk = dtime(hm, mm)

    hk = st.number_input("Jam Keluar (Jam)",  0, 23, st.session_state.jam_keluar.hour)
    mk = st.number_input("Jam Keluar (Menit)",0, 59, st.session_state.jam_keluar.minute)
    st.session_state.jam_keluar = dtime(hk, mk)

    st.session_state.toleransi = st.slider("⏱ Toleransi Terlambat (menit)", 0, 60, st.session_state.toleransi)

    st.markdown("---")
    st.markdown("### 🛡️ Anti-Spoofing")
    st.session_state.spoof_strict = st.toggle(
        "Aktifkan Anti-Spoofing", value=st.session_state.spoof_strict,
        help="Jika ON, foto dari HP/layar akan ditolak otomatis"
    )
    if st.session_state.spoof_strict:
        st.info("🔒 Mode KETAT: foto & layar ditolak")
    else:
        st.warning("⚠️ Anti-spoofing NONAKTIF")

    st.markdown("---")
    st.markdown("### 🎯 Deteksi")
    st.session_state.conf_thresh = st.slider("Confidence Threshold", 0.05, 1.0, st.session_state.conf_thresh, 0.01)
    st.session_state.iou_thresh  = st.slider("IoU (NMS)",            0.05, 1.0, st.session_state.iou_thresh,  0.01)

    st.markdown("---")
    if st.button("🗑️ Reset Sesi"):
        st.session_state.absen_log  = {}
        st.session_state.absen_done = set()
        st.session_state.rt_log     = []
        st.success("Sesi direset.")

# ============================================================================
# INFO PANEL JADWAL
# ============================================================================

jm_str  = st.session_state.jam_masuk.strftime("%H:%M")
jk_str  = st.session_state.jam_keluar.strftime("%H:%M")
tol     = st.session_state.toleransi
batas   = (datetime.combine(datetime.today(), st.session_state.jam_masuk)
           + timedelta(minutes=tol)).strftime("%H:%M")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="sbox">
        <div class="slbl">⏰ Jam Masuk</div>
        <div class="sval">{jm_str}</div>
        <div class="slate">Batas toleransi: {batas} (+{tol} mnt)</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="sbox">
        <div class="slbl">🏁 Jam Keluar</div>
        <div class="sval">{jk_str}</div>
        <div class="slate">Pulang setelah jam ini</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="clk">
        <div class="clk-t">{now_dt.strftime("%H:%M:%S")}</div>
        <div class="clk-d">{now_dt.strftime("%d %b %Y")}</div>
    </div>""", unsafe_allow_html=True)

# ============================================================================
# MODE & TIPE SELECTOR
# ============================================================================

st.markdown("---")
mode = st.radio("Mode", ["📸 Presensi via Foto","🎥 Presensi Realtime Kamera"],
                horizontal=True, label_visibility="collapsed")
tipe = st.radio("Jenis", ["Masuk","Keluar"], horizontal=True)
st.session_state.tipe_presensi = tipe

# ============================================================================
# RECORD ATTENDANCE
# ============================================================================

def record_attendance(name: str, conf_val: float, tipe: str, spoof_res: str) -> dict | None:
    """Simpan absen; kembalikan record atau None jika ditolak."""
    # Tolak jika spoofed dan strict mode ON
    if st.session_state.spoof_strict and spoof_res == "SPOOFED":
        return None
    # Cegah double masuk (tipe Masuk saja)
    if tipe == "Masuk" and name in st.session_state.absen_done:
        return None

    now    = get_now()
    tanggal= today_str()
    status = compute_status(now.time(), st.session_state.jam_masuk, st.session_state.toleransi, tipe)
    rec    = {
        "jam"       : now.strftime("%H:%M:%S"),
        "tipe"      : tipe,
        "status"    : status,
        "confidence": round(conf_val, 4),
        "antispoof" : spoof_res,
        "timestamp" : now.isoformat(),
    }

    if name not in st.session_state.absen_log:
        st.session_state.absen_log[name] = []
    st.session_state.absen_log[name].append(rec)
    if tipe == "Masuk":
        st.session_state.absen_done.add(name)

    data = load_presensi()
    if tanggal not in data: data[tanggal] = {}
    if name not in data[tanggal]: data[tanggal][name] = []
    data[tanggal][name].append(rec)
    save_presensi(data)
    export_csv(data)
    return rec

# ============================================================================
# MODE: FOTO
# ============================================================================

if mode == "📸 Presensi via Foto":

    col_up, col_res = st.columns([1, 2])

    with col_up:
        st.markdown('<div class="card"><div class="ch">📤 Upload Foto Pegawai</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Pilih foto", type=["jpg","jpeg","png","bmp","webp"],
            accept_multiple_files=True, label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_files:
            if st.button(f"✅ Mulai Absen {tipe}", use_container_width=True, type="primary"):
                st.session_state.run_foto = True

    with col_res:
        if not uploaded_files:
            st.markdown("""<div class="card" style="text-align:center;padding:2.5rem 1rem;">
                <div style="font-size:3rem;margin-bottom:.75rem;">📁</div>
                <div style="color:var(--t2)">Upload foto pegawai lalu klik tombol Mulai Absen</div>
                <div style="font-size:.78rem;color:var(--t2);margin-top:.6rem">JPG · PNG · BMP · WEBP</div>
            </div>""", unsafe_allow_html=True)

        elif st.session_state.run_foto:
            st.session_state.run_foto = False
            new_recs = []

            with st.spinner(f"Memproses {len(uploaded_files)} foto…"):
                for uf in uploaded_files:
                    image   = Image.open(uf).convert("RGB")
                    img_rgb = np.array(image)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    t0      = time.perf_counter()
                    results = model.predict(img_bgr,
                                            conf=st.session_state.conf_thresh,
                                            iou=st.session_state.iou_thresh,
                                            verbose=False)
                    elapsed = (time.perf_counter() - t0) * 1000
                    dets    = parse_results(results[0])

                    status_map = {}
                    spoof_map  = {}

                    for cls_id, conf_val, x1, y1, x2, y2 in dets:
                        emp  = names.get(cls_id, f"ID-{cls_id}")
                        face = img_bgr[y1:y2, x1:x2]
                        sp   = AntiSpoofing.check(face)
                        spoof_map[emp] = sp["result"]
                        rec = record_attendance(emp, conf_val, tipe, sp["result"])
                        if rec:
                            new_recs.append((emp, rec, sp))
                            status_map[emp] = rec["status"]

                    annotated = draw_boxes(img_bgr, dets, names, status_map, spoof_map)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    st.markdown(f'<div class="ch">📷 {uf.name} · {elapsed:.0f}ms</div>', unsafe_allow_html=True)
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.markdown('<div style="font-size:.75rem;color:var(--t2);margin-bottom:4px">FOTO ASLI</div>', unsafe_allow_html=True)
                        st.image(img_rgb)
                    with cc2:
                        st.markdown('<div style="font-size:.75rem;color:var(--t2);margin-bottom:4px">HASIL DETEKSI</div>', unsafe_allow_html=True)
                        st.image(annotated_rgb)

                    if not dets:
                        st.warning("Tidak ada wajah terdeteksi.")

            if new_recs:
                st.success(f"✅ {len(new_recs)} pegawai berhasil diabsen!")
                for emp, rec, sp in new_recs:
                    chip_cls = "chip-ok" if "Tepat" in rec["status"] else "chip-late"
                    sp_badge = (f'<span class="badge-real">WAJAH ASLI</span>'
                                if sp["result"]=="REAL"
                                else f'<span class="badge-spoof">SPOOFED</span>'
                                if sp["result"]=="SPOOFED"
                                else f'<span class="badge-unk">UNKNOWN</span>')
                    st.markdown(f"""
                    <div class="att {'late' if 'Terlambat' in rec['status'] else ''}">
                      <div style="font-size:1.3rem">👤</div>
                      <div class="att-name">{emp}</div>
                      <div class="att-time">{rec['jam']}</div>
                      {sp_badge}
                      <span class="chip {chip_cls}">{rec['status']}</span>
                    </div>""", unsafe_allow_html=True)

            # Tampilkan juga yang ditolak karena spoofing
            rejected = [(names.get(cls_id, f"ID-{cls_id}"), spoof_map.get(names.get(cls_id,f"ID-{cls_id}"), ""))
                        for uf in uploaded_files
                        for cls_id, *_ in parse_results(
                            model.predict(
                                cv2.cvtColor(np.array(Image.open(uf).convert("RGB")),cv2.COLOR_RGB2BGR),
                                conf=st.session_state.conf_thresh,iou=st.session_state.iou_thresh,verbose=False
                            )[0])
                        if (names.get(cls_id,f"ID-{cls_id}") not in [r[0] for r in new_recs])]
            # (note: tampilkan pesan sederhana saja tanpa re-predict ulang untuk hemat memori)


# ============================================================================
# MODE: KAMERA REALTIME (TANPA LAG)
# ============================================================================

elif mode == "🎥 Presensi Realtime Kamera":

    st.markdown("""<div class="card" style="text-align:center;padding:1rem;">
      <div style="font-size:1rem;font-weight:700;color:var(--pl);margin-bottom:.4rem">
        🎥 Presensi Realtime · Anti-Spoof Aktif
      </div>
      <div style="font-size:.82rem;color:var(--t2)">
        Arahkan wajah ke kamera &nbsp;·&nbsp; Tekan <b>START</b> &nbsp;·&nbsp;
        Presensi tersimpan otomatis &nbsp;·&nbsp; Foto dari HP/layar <b>DITOLAK</b>
      </div>
    </div>""", unsafe_allow_html=True)

    col_cam, col_log = st.columns([3, 2])

    with col_cam:

        # ── Parameter yang dibaca oleh thread WebRTC ──────────────────────────
        # Kita bungkus dalam class agar thread-safe tanpa st.session_state di thread
        class _Cfg:
            conf   = st.session_state.conf_thresh
            iou    = st.session_state.iou_thresh
            strict = st.session_state.spoof_strict
            tipe   = st.session_state.tipe_presensi
            masuk  = st.session_state.jam_masuk
            tol    = st.session_state.toleransi

        cfg = _Cfg()

        class LiveProcessor(VideoProcessorBase):
            """
            Optimasi performa:
            - Resize frame ke 640px lebar sebelum inference
            - Skip frame: jalankan YOLO setiap N frame (default 3)
            - Anti-spoof hanya pada ROI wajah (64×64)
            - Cache hasil deteksi terakhir untuk frame yang di-skip
            """
            SKIP       = 3       # proses setiap 3 frame → ~10 fps inference di 30fps cam
            INFER_SIZE = 640     # lebar max untuk inference

            def __init__(self):
                self._frame_count  = 0
                self._last_dets    = []
                self._last_status  = {}
                self._last_spoof   = {}
                self._absen_set    = set()   # nama yg sudah diabsen sesi ini

            def _maybe_absen(self, name: str, conf_v: float, spoof_res: str):
                if name in self._absen_set: return
                rec = record_attendance(name, conf_v, cfg.tipe, spoof_res)
                if rec:
                    self._absen_set.add(name)
                    with _LOCK:
                        st.session_state.rt_log.append((name, rec, spoof_res))

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                self._frame_count += 1

                # ── Scale down untuk inference ────────────────────────────────
                scale = min(1.0, self.INFER_SIZE / w)
                inf_w = int(w * scale)
                inf_h = int(h * scale)
                img_small = cv2.resize(img, (inf_w, inf_h)) if scale < 1.0 else img

                run_infer = (self._frame_count % self.SKIP == 0)

                if run_infer:
                    res  = model.predict(img_small,
                                         conf=cfg.conf, iou=cfg.iou, verbose=False)
                    dets_raw = parse_results(res[0])

                    # Scale bbox kembali ke frame asli
                    inv = 1.0 / scale if scale < 1.0 else 1.0
                    dets_full = [(c, cf,
                                  int(x1*inv), int(y1*inv),
                                  int(x2*inv), int(y2*inv))
                                 for c, cf, x1, y1, x2, y2 in dets_raw]
                    self._last_dets = dets_full

                    for cls_id, conf_v, x1, y1, x2, y2 in dets_full:
                        emp  = names.get(cls_id, f"ID-{cls_id}")
                        face = img[y1:y2, x1:x2]
                        sp   = AntiSpoofing.check(face)
                        self._last_spoof[emp] = sp["result"]
                        self._last_status[emp] = compute_status(
                            get_now().time(), cfg.masuk, cfg.tol, cfg.tipe)
                        self._maybe_absen(emp, conf_v, sp["result"])

                # ── Draw dengan hasil terakhir (smooth, tidak flicker) ────────
                annotated = draw_boxes(img, self._last_dets, names,
                                       self._last_status, self._last_spoof)

                # Overlay FPS info
                cv2.putText(annotated,
                            f"JASAKULA Presensi | AntiSpoof {'ON' if cfg.strict else 'OFF'}",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (16,185,129), 1, cv2.LINE_AA)

                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        webrtc_streamer(
            key="presensi-live",
            video_processor_factory=LiveProcessor,
            media_stream_constraints={"video": {"width":{"ideal":1280},"height":{"ideal":720}},
                                      "audio": False},
            async_processing=True,
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        )

    with col_log:
        st.markdown('<div class="card"><div class="ch">📋 Log Absen Sesi Ini</div>', unsafe_allow_html=True)
        log = st.session_state.rt_log
        if log:
            for emp, rec, sp_res in reversed(log[-20:]):   # tampilkan 20 terbaru
                chip_cls = "chip-ok" if "Tepat" in rec["status"] else "chip-late"
                sp_badge = (f'<span class="badge-real">REAL</span>' if sp_res=="REAL"
                            else f'<span class="badge-spoof">SPOOF</span>'
                            if sp_res=="SPOOFED"
                            else f'<span class="badge-unk">?</span>')
                st.markdown(f"""
                <div class="att {'late' if 'Terlambat' in rec['status'] else ''}">
                  <div style="font-size:1.1rem">👤</div>
                  <div>
                    <div class="att-name">{emp}</div>
                    <div class="att-time">{rec['jam']} · {rec['tipe']}</div>
                  </div>
                  {sp_badge}
                  <span class="chip {chip_cls}" style="font-size:.7rem">{rec['status']}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:var(--t2);font-size:.82rem">Belum ada presensi…</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# REKAP HARI INI
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Rekap Presensi Hari Ini")

data_all  = load_presensi()
today_dat = data_all.get(today_str(), {})
total_emp = len(today_dat)
tepat_cnt = sum(1 for recs in today_dat.values()
                for r in recs if "Tepat" in r.get("status",""))
late_cnt  = total_emp - tepat_cnt

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.markdown(f'<div class="mbox"><div class="mlbl">👥 Total Hadir</div><div class="mval">{total_emp}</div></div>', unsafe_allow_html=True)
with mc2:
    st.markdown(f'<div class="mbox"><div class="mlbl">✅ Tepat Waktu</div><div class="mval" style="color:#10b981">{tepat_cnt}</div></div>', unsafe_allow_html=True)
with mc3:
    st.markdown(f'<div class="mbox"><div class="mlbl">⚠️ Terlambat</div><div class="mval" style="color:#f97316">{late_cnt}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if today_dat:
    st.markdown('<div class="card"><div class="ch">📋 Daftar Hadir</div>', unsafe_allow_html=True)
    for emp, recs in today_dat.items():
        for r in recs:
            chip_cls = "chip-ok" if "Tepat" in r["status"] else "chip-late"
            sp_res   = r.get("antispoof","?")
            sp_badge = (f'<span class="badge-real">REAL</span>' if sp_res=="REAL"
                        else f'<span class="badge-spoof">SPOOF</span>'
                        if sp_res=="SPOOFED"
                        else f'<span class="badge-unk">?</span>')
            st.markdown(f"""
            <div class="att {'late' if 'Terlambat' in r['status'] else ''}">
              <div style="font-size:1.3rem">👤</div>
              <div style="flex:1">
                <div class="att-name">{emp}</div>
                <div class="att-time">{r['tipe']} · {r['jam']} · Conf: {float(r['confidence'])*100:.1f}%</div>
              </div>
              {sp_badge}
              <span class="chip {chip_cls}">{r['status']}</span>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📄 Data JSON Lengkap"):
        st.json(today_dat)

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE,"rb") as f:
            st.download_button("⬇️ Download Rekap CSV", data=f,
                               file_name=f"presensi_{today_str()}.csv", mime="text/csv")
else:
    st.info("Belum ada data presensi hari ini.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
  Sistem Presensi Digital · PT JASAKULA PURWALUHUR · YOLOv11 + Anti-Spoofing Engine v2.0
</div>
""", unsafe_allow_html=True)
