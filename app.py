"""
╔══════════════════════════════════════════════════════════════════╗
║           FaceVision AI  –  Pure Python + OpenCV + Tkinter      ║
║   Upload Foto  |  Realtime Kamera  |  YOLOv8 best.pt            ║
║                                                                  ║
║  FIXED VERSION - Solid Tkinter Pattern (No PhotoImage Magic)    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import os
import time
from datetime import datetime


# ─── Warna & Font ───────────────────────────────────────────────
BG_DARK        = "#0a0a0f"
BG_CARD        = "#0d1117"
BG_PANEL       = "#161b22"
ACCENT_PURPLE  = "#7c3aed"
ACCENT_GREEN   = "#10b981"
ACCENT_YELLOW  = "#f59e0b"
ACCENT_RED     = "#ef4444"
TEXT_PRIMARY   = "#e2e8f0"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED     = "#475569"
BORDER_COLOR   = "#1e293b"
HOVER_BG       = "#1e293b"

FONT_HEADER = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_BADGE  = ("Segoe UI", 9, "bold")
FONT_MONO   = ("Consolas", 9)


# ════════════════════════════════════════════════════════════════
#  HELPER
# ════════════════════════════════════════════════════════════════

def make_btn(parent, text, command, bg=ACCENT_PURPLE, fg=TEXT_PRIMARY,
             font=FONT_BADGE, padx=18, pady=8):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, font=font,
        activebackground=_darken(bg), activeforeground=fg,
        relief="flat", bd=0, cursor="hand2",
        padx=padx, pady=pady,
    )
    btn.bind("<Enter>", lambda _: btn.config(bg=_darken(bg)))
    btn.bind("<Leave>", lambda _: btn.config(bg=bg))
    return btn


def _darken(hex_color: str) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    f = 0.72
    return f"#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}"


def draw_boxes(image_np: np.ndarray, detections: list, names: dict) -> np.ndarray:
    img = image_np.copy()
    for (cls_id, conf, x1, y1, x2, y2) in detections:
        name = names.get(cls_id, str(cls_id))
        label = f"{name}  {conf * 100:.1f}%"
        if conf >= 0.75:
            color = (16, 185, 129)
        elif conf >= 0.50:
            color = (245, 158, 11)
        else:
            color = (239, 68, 68)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        corner, thick = 14, 3
        for (cx, cy, dx, dy) in [
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(img, (cx, cy), (cx + dx * corner, cy), color, thick)
            cv2.line(img, (cx, cy), (cx, cy + dy * corner), color, thick)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx1, ly1 = x1, max(y1 - th - 10, 0)
        cv2.rectangle(img, (lx1, ly1), (lx1 + tw + 12, ly1 + th + 8), color, -1)
        cv2.putText(img, label, (lx1 + 6, ly1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def parse_results(result) -> list:
    dets = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append((cls_id, conf, x1, y1, x2, y2))
    return dets


def resize_keep_ratio(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    ratio = min(max_w / w, max_h / h, 1.0)
    return img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


# ════════════════════════════════════════════════════════════════
#  PANEL – UPLOAD FOTO
# ════════════════════════════════════════════════════════════════

class UploadPanel(tk.Frame):
    def __init__(self, parent, model, names, get_conf, get_iou):
        super().__init__(parent, bg=BG_DARK)
        self._model = model
        self._names = names
        self._get_conf = get_conf
        self._get_iou = get_iou
        self._orig_np = None
        self._result_np = None

        self._canvas = None
        self._img_item = None
        self._current_photo = None  # SOLID: Simpan photo reference DI SINI

        self._det_inner = None
        self._status_var = None

        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", padx=24, pady=(18, 4))
        tk.Label(hdr, text="📁  Upload & Deteksi Foto",
                 bg=BG_DARK, fg=TEXT_PRIMARY, font=FONT_HEADER).pack(side="left")
        tk.Label(hdr, text="Klik drop zone atau Pilih Foto",
                 bg=BG_DARK, fg=TEXT_MUTED, font=FONT_SMALL).pack(side="left", padx=14)

        outer = tk.Frame(self, bg=BG_PANEL,
                          highlightbackground=ACCENT_PURPLE, highlightthickness=1)
        outer.pack(fill="both", expand=True, padx=24, pady=8)
        self._canvas = tk.Canvas(outer, bg=BG_PANEL, highlightthickness=0, cursor="hand2")
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Button-1>", lambda _: self._open_file())

        self._placeholder()

        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.pack(fill="x", padx=24, pady=(0, 8))
        make_btn(btn_row, "📂  Pilih Foto", self._open_file).pack(side="left", padx=(0, 8))
        make_btn(btn_row, "🔍  Deteksi", self._run_detect, bg="#1d4ed8").pack(side="left", padx=(0, 8))
        make_btn(btn_row, "💾  Simpan Hasil", self._save_result, bg="#065f46").pack(side="left", padx=(0, 8))
        make_btn(btn_row, "↺  Reset", self._reset, bg="#374151").pack(side="left")

        sf = tk.Frame(self, bg=BG_PANEL,
                      highlightbackground=BORDER_COLOR, highlightthickness=1)
        sf.pack(fill="x", padx=24, pady=(0, 8))
        self._status_var = tk.StringVar(value="Belum ada gambar — klik drop zone atau Pilih Foto")
        tk.Label(sf, textvariable=self._status_var,
                 bg=BG_PANEL, fg=TEXT_SECONDARY, font=FONT_MONO,
                 anchor="w", padx=12, pady=8).pack(fill="x")

        det_f = tk.Frame(self, bg=BG_DARK)
        det_f.pack(fill="x", padx=24, pady=(0, 16))
        tk.Label(det_f, text="🎯  Hasil Deteksi",
                 bg=BG_DARK, fg=ACCENT_PURPLE, font=FONT_BADGE, anchor="w").pack(fill="x", pady=(0, 5))
        self._det_inner = tk.Frame(det_f, bg=BG_DARK)
        self._det_inner.pack(fill="x")

    def _placeholder(self):
        if self._canvas is None:
            return
        self._canvas.delete("all")
        self._img_item = None
        w = self._canvas.winfo_width() or 600
        h = self._canvas.winfo_height() or 360
        cx, cy = w // 2, h // 2
        self._canvas.create_text(cx, cy - 28, text="🖼️",
                                  font=("Segoe UI", 46), fill=TEXT_MUTED)
        self._canvas.create_text(cx, cy + 25,
                                  text="Klik di sini atau Pilih Foto",
                                  font=FONT_BODY, fill=TEXT_MUTED)
        self._canvas.create_text(cx, cy + 48,
                                  text="JPG  ·  PNG  ·  BMP  ·  WEBP",
                                  font=FONT_SMALL, fill=TEXT_MUTED)

    def _show_np(self, img_np):
        """Update gambar di Canvas dengan aman (tanpa state image hilang)."""
        w = self._canvas.winfo_width() or 600
        h = self._canvas.winfo_height() or 360

        pil = Image.fromarray(img_np)
        pil = resize_keep_ratio(pil, w - 4, h - 4)

        # FOTO DISPLAY: gunakan PIL.ImageTk.PhotoImage, tapi setel ulang item selalu.
        # Pada lingkungan kamu, registry image 'pyimageX' mudah invalid, jadi kita recreate item.
        self._current_photo = ImageTk.PhotoImage(pil)

        cx, cy = w // 2, h // 2

        # Recreate item image setiap kali agar tidak mengacu ke image item yang invalid
        self._canvas.delete("all")
        self._img_item = self._canvas.create_image(
            cx, cy, anchor="center", image=self._current_photo
        )

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Pilih Foto",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Semua", "*.*")])
        if not path:
            return
        pil = Image.open(path).convert("RGB")
        self._orig_np = np.array(pil)
        self._result_np = None
        self._show_np(self._orig_np)
        self._status_var.set(f"✅  {os.path.basename(path)}  ({pil.width}×{pil.height})")
        self._clear_det()

    def _run_detect(self):
        if self._orig_np is None:
            messagebox.showwarning("Peringatan", "Belum ada gambar!")
            return
        self._status_var.set("⏳  Mendeteksi...")
        self.update()

        t0 = time.perf_counter()
        results = self._model.predict(
            self._orig_np,
            conf=self._get_conf(),
            iou=self._get_iou(),
            verbose=False
        )
        elapsed = (time.perf_counter() - t0) * 1000
        dets = parse_results(results[0])
        self._result_np = draw_boxes(self._orig_np, dets, self._names)
        self._show_np(self._result_np)

        self._status_var.set(
            f"🎯  {len(dets)} wajah terdeteksi  ·  ⏱ {elapsed:.1f} ms  ·  conf ≥ {self._get_conf():.2f}")
        self._render_det(dets)

    def _render_det(self, dets):
        self._clear_det()
        if not dets:
            tk.Label(self._det_inner, text="Tidak ada wajah terdeteksi.",
                     bg=BG_DARK, fg=TEXT_MUTED, font=FONT_SMALL).pack(side="left")
            return

        for i, (cls_id, conf, *_rest) in enumerate(dets, 1):
            name = self._names.get(cls_id, str(cls_id))
            color = ACCENT_GREEN if conf >= .75 else (ACCENT_YELLOW if conf >= .5 else ACCENT_RED)
            tk.Label(self._det_inner,
                     text=f"  #{i}  {name}  {conf * 100:.1f}%  ",
                     bg=BG_PANEL, fg=color, font=FONT_BADGE,
                     padx=8, pady=5, relief="flat",
                     highlightbackground=color, highlightthickness=1
                     ).pack(side="left", padx=4, pady=2)

    def _clear_det(self):
        for w in self._det_inner.winfo_children():
            w.destroy()

    def _save_result(self):
        if self._result_np is None:
            messagebox.showwarning("Peringatan", "Jalankan deteksi dulu!")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            title="Simpan Hasil")
        if path:
            Image.fromarray(self._result_np).save(path)
            messagebox.showinfo("Berhasil", f"Disimpan:\n{path}")

    def _reset(self):
        self._orig_np = self._result_np = None
        self._current_photo = None  # Clear photo reference
        self._canvas.delete("all")
        self._img_item = None
        self._placeholder()
        self._status_var.set("Belum ada gambar — klik drop zone atau Pilih Foto")
        self._clear_det()


# ════════════════════════════════════════════════════════════════
#  PANEL – REALTIME KAMERA
# ════════════════════════════════════════════════════════════════

class CameraPanel(tk.Frame):
    def __init__(self, parent, model, names, get_conf, get_iou):
        super().__init__(parent, bg=BG_DARK)
        self._model = model
        self._names = names
        self._get_conf = get_conf
        self._get_iou = get_iou

        self._cap = None
        self._running = False
        self._fps_list = []
        self._frame_count = 0
        self._last_frame = None

        self._conf_cache = float(self._get_conf())
        self._iou_cache = float(self._get_iou())

        self._canvas = None
        self._img_item = None
        self._current_photo = None  # SOLID: Simpan photo reference DI SINI

        self._stat_vars = {}
        self._det_inner = None
        self._time_lbl = None
        self._live_lbl = None

        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", padx=24, pady=(18, 4))
        tk.Label(hdr, text="📹  Deteksi Wajah Realtime",
                 bg=BG_DARK, fg=TEXT_PRIMARY, font=FONT_HEADER).pack(side="left")
        self._live_lbl = tk.Label(hdr, text="●  OFFLINE",
                                   bg=BG_DARK, fg=TEXT_MUTED, font=FONT_SMALL)
        self._live_lbl.pack(side="right")

        outer = tk.Frame(self, bg=BG_PANEL,
                          highlightbackground=ACCENT_PURPLE, highlightthickness=1)
        outer.pack(fill="both", expand=True, padx=24, pady=8)
        self._canvas = tk.Canvas(outer, bg=BG_PANEL, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)

        self._draw_placeholder()

        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.pack(fill="x", padx=24, pady=(0, 8))
        make_btn(btn_row, "▶  Mulai Kamera", self._start).pack(side="left", padx=(0, 8))
        make_btn(btn_row, "⏹  Stop Kamera", self._stop, bg=ACCENT_RED).pack(side="left", padx=(0, 8))
        make_btn(btn_row, "📸  Screenshot", self._screenshot, bg="#065f46").pack(side="left")

        sf = tk.Frame(self, bg=BG_PANEL,
                      highlightbackground=BORDER_COLOR, highlightthickness=1)
        sf.pack(fill="x", padx=24, pady=(0, 8))

        self._stat_vars = {}
        for label in ("FPS", "Frame", "Wajah"):
            col = tk.Frame(sf, bg=BG_PANEL)
            col.pack(side="left", padx=20, pady=8)
            v = tk.StringVar(value="—")
            self._stat_vars[label] = v
            tk.Label(col, textvariable=v, bg=BG_PANEL, fg=ACCENT_PURPLE,
                     font=("Segoe UI", 18, "bold")).pack()
            tk.Label(col, text=label, bg=BG_PANEL, fg=TEXT_MUTED, font=FONT_SMALL).pack()

        tk.Frame(sf, bg=BORDER_COLOR, width=1).pack(side="left", fill="y", padx=10, pady=6)
        self._time_lbl = tk.Label(sf, text="--:--:--",
                                   bg=BG_PANEL, fg=TEXT_MUTED, font=FONT_MONO)
        self._time_lbl.pack(side="right", padx=16)

        det_f = tk.Frame(self, bg=BG_DARK)
        det_f.pack(fill="x", padx=24, pady=(0, 16))
        tk.Label(det_f, text="🎯  Deteksi Frame Terakhir",
                 bg=BG_DARK, fg=ACCENT_PURPLE, font=FONT_BADGE, anchor="w").pack(fill="x", pady=(0, 5))
        self._det_inner = tk.Frame(det_f, bg=BG_DARK)
        self._det_inner.pack(fill="x")

    def _draw_placeholder(self):
        if self._canvas is None:
            return
        self._canvas.delete("all")
        self._img_item = None

        w = self._canvas.winfo_width() or 720
        h = self._canvas.winfo_height() or 400
        cx, cy = w // 2, h // 2
        self._canvas.create_text(cx, cy - 28, text="📹",
                                  font=("Segoe UI", 48), fill=TEXT_MUTED)
        self._canvas.create_text(cx, cy + 25,
                                  text="Klik ▶ Mulai Kamera untuk memulai",
                                  font=FONT_BODY, fill=TEXT_MUTED)

    def _start(self):
        if self._running:
            return

        self._conf_cache = float(self._get_conf())
        self._iou_cache = float(self._get_iou())

        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Error", "Kamera tidak dapat dibuka!\nPeriksa koneksi kamera.")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        self._running = True
        self._frame_count = 0
        self._fps_list = []
        self._live_lbl.config(text="● LIVE", fg=ACCENT_GREEN)

        threading.Thread(target=self._loop, daemon=True).start()

    def _stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._live_lbl.config(text="●  OFFLINE", fg=TEXT_MUTED)

        self._canvas.delete("all")
        self._img_item = None
        self._current_photo = None
        self._draw_placeholder()

        for v in self._stat_vars.values():
            v.set("—")
        self._clear_det()

    def _loop(self):
        while self._running:
            if not self._cap or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            results = self._model.predict(
                frame_rgb,
                conf=self._conf_cache,
                iou=self._iou_cache,
                verbose=False,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            fps = 1000 / elapsed if elapsed > 0 else 0

            self._fps_list.append(fps)
            if len(self._fps_list) > 20:
                self._fps_list.pop(0)
            avg_fps = sum(self._fps_list) / len(self._fps_list)

            dets = parse_results(results[0])
            ann = draw_boxes(frame_rgb, dets, self._names)

            self._frame_count += 1
            self._last_frame = ann

            self.after(0, self._update_ui, ann, dets, avg_fps)

    def _update_ui(self, frame_np, dets, avg_fps):
        """SOLID PATTERN: Simpan photo di instance, update canvas"""
        if not self._running:
            return

        w = self._canvas.winfo_width() or 720
        h = self._canvas.winfo_height() or 400

        pil = resize_keep_ratio(Image.fromarray(frame_np), w, h)
        
        # KEY: Simpan PhotoImage di instance variable — jangan let it GC!
        self._current_photo = ImageTk.PhotoImage(pil)

        cx, cy = w // 2, h // 2

        if self._img_item is None:
            # First frame: create image
            self._canvas.delete("all")
            self._img_item = self._canvas.create_image(cx, cy, anchor="center", image=self._current_photo)
        else:
            # Subsequent frames: update only
            self._canvas.itemconfig(self._img_item, image=self._current_photo)

        self._stat_vars["FPS"].set(f"{avg_fps:.1f}")
        self._stat_vars["Frame"].set(str(self._frame_count))
        self._stat_vars["Wajah"].set(str(len(dets)))
        self._time_lbl.config(text=datetime.now().strftime("%H:%M:%S"))

        self._clear_det()
        if not dets:
            tk.Label(self._det_inner, text="Tidak ada wajah terdeteksi.",
                     bg=BG_DARK, fg=TEXT_MUTED, font=FONT_SMALL).pack(side="left")
            return

        for i, (cls_id, conf, *_rest) in enumerate(dets, 1):
            name = self._names.get(cls_id, str(cls_id))
            color = ACCENT_GREEN if conf >= .75 else (ACCENT_YELLOW if conf >= .5 else ACCENT_RED)
            tk.Label(self._det_inner,
                     text=f"  #{i}  {name}  {conf * 100:.1f}%  ",
                     bg=BG_PANEL, fg=color, font=FONT_BADGE,
                     padx=8, pady=5, relief="flat",
                     highlightbackground=color, highlightthickness=1
                     ).pack(side="left", padx=4, pady=2)

    def _clear_det(self):
        for w in self._det_inner.winfo_children():
            w.destroy()

    def _screenshot(self):
        if self._last_frame is None:
            messagebox.showwarning("Peringatan", "Belum ada frame!")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"screenshot_{ts}.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            title="Simpan Screenshot")
        if path:
            Image.fromarray(self._last_frame).save(path)
            messagebox.showinfo("Berhasil", f"Tersimpan:\n{path}")

    def on_close(self):
        self._stop()


# ════════════════════════════════════════════════════════════════
#  APLIKASI UTAMA
# ════════════════════════════════════════════════════════════════

class FaceVisionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FaceVision AI  –  Face Recognition")
        self.geometry("1280x820")
        self.minsize(960, 660)
        self.configure(bg=BG_DARK)
        try:
            self.iconbitmap("icon.ico")
        except:
            pass

        self._conf_var = tk.DoubleVar(value=0.45)
        self._iou_var = tk.DoubleVar(value=0.45)

        self._load_model()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_model(self):
        if not os.path.exists("best.pt"):
            messagebox.showerror(
                "Model Tidak Ditemukan",
                "File 'best.pt' tidak ditemukan!\n\n"
                "Letakkan best.pt di folder yang sama dengan app.py.\n"
                f"Folder saat ini: {os.getcwd()}")
            self.destroy()
            return

        try:
            self._model = YOLO("best.pt")
            self._names = self._model.names
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat model:\n{e}")
            self.destroy()

    def _build_ui(self):
        top = tk.Frame(self, bg=BG_CARD, height=54)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="🎭  FaceVision AI",
                 bg=BG_CARD, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 15, "bold")).pack(side="left", padx=20, pady=12)
        tk.Label(top, text=f"Model: best.pt  ·  {len(self._names)} kelas  ·  YOLOv8",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(side="right", padx=20)
        tk.Frame(self, bg=ACCENT_PURPLE, height=2).pack(fill="x")

        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True)

        sb = tk.Frame(main, bg=BG_CARD, width=220)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        self._build_sidebar(sb)
        tk.Frame(main, bg=BORDER_COLOR, width=1).pack(side="left", fill="y")

        self._content = tk.Frame(main, bg=BG_DARK)
        self._content.pack(side="left", fill="both", expand=True)

        self._panel_upload = UploadPanel(
            self._content, self._model, self._names,
            lambda: self._conf_var.get(),
            lambda: self._iou_var.get(),
        )
        self._panel_camera = CameraPanel(
            self._content, self._model, self._names,
            lambda: self._conf_var.get(),
            lambda: self._iou_var.get(),
        )

        self._show("upload")

    def _build_sidebar(self, sb):
        tk.Label(sb, text="🎭", bg=BG_CARD, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 32)).pack(pady=(22, 2))
        tk.Label(sb, text="FaceVision AI", bg=BG_CARD, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 11, "bold")).pack()
        tk.Label(sb, text="YOLOv8  ·  best.pt", bg=BG_CARD, fg=TEXT_MUTED,
                 font=FONT_SMALL).pack()
        tk.Frame(sb, bg=BORDER_COLOR, height=1).pack(fill="x", padx=14, pady=12)

        tk.Label(sb, text="MODE", bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=16, pady=(0, 4))

        self._nav_frames = {}
        for key, icon, label in [
            ("upload", "📁", "Upload Foto"),
            ("camera", "📹", "Realtime Kamera"),
        ]:
            btn = tk.Frame(sb, bg=BG_CARD, cursor="hand2")
            btn.pack(fill="x", padx=8, pady=2)
            inner = tk.Frame(btn, bg=BG_CARD, padx=10, pady=10)
            inner.pack(fill="x")
            ico = tk.Label(inner, text=icon, bg=BG_CARD, fg=TEXT_PRIMARY, font=("Segoe UI", 13))
            ico.pack(side="left")
            lbl = tk.Label(inner, text=label, bg=BG_CARD, fg=TEXT_SECONDARY, font=FONT_BODY)
            lbl.pack(side="left", padx=8)

            widgets = (btn, inner, ico, lbl)

            def make_cmd(k=key, ws=widgets):
                def cmd(_=None):
                    self._show(k)
                for w in ws:
                    w.bind("<Button-1>", cmd)
            make_cmd()

            self._nav_frames[key] = widgets

        tk.Frame(sb, bg=BORDER_COLOR, height=1).pack(fill="x", padx=14, pady=12)

        tk.Label(sb, text="PARAMETER", bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=16, pady=(0, 6))
        prm = tk.Frame(sb, bg=BG_CARD)
        prm.pack(fill="x", padx=14)

        for label, var in [("Confidence", self._conf_var), ("IoU (NMS)", self._iou_var)]:
            row = tk.Frame(prm, bg=BG_CARD)
            row.pack(fill="x", pady=5)
            tk.Label(row, text=label, bg=BG_CARD, fg=TEXT_SECONDARY, font=FONT_SMALL).pack(anchor="w")
            val_lbl = tk.Label(row, text=f"{var.get():.2f}",
                                bg=BG_CARD, fg=ACCENT_PURPLE, font=FONT_BADGE)
            val_lbl.pack(anchor="e")

            def make_trace(v=var, lv=val_lbl):
                def cb(*_):
                    lv.config(text=f"{v.get():.2f}")
                return cb
            var.trace_add("write", make_trace())

            ttk.Scale(row, from_=0.05, to=1.0, orient="horizontal", variable=var).pack(fill="x")

        tk.Frame(sb, bg=BORDER_COLOR, height=1).pack(fill="x", padx=14, pady=12)

        tk.Label(sb, text="KELAS DIKENALI", bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=16, pady=(0, 4))
        cf = tk.Frame(sb, bg=BG_CARD)
        cf.pack(fill="x", padx=14)
        for name in list(self._names.values())[:14]:
            tk.Label(cf, text=f"👤 {name}", bg=BG_CARD, fg=TEXT_SECONDARY,
                     font=FONT_SMALL, anchor="w").pack(fill="x")
        if len(self._names) > 14:
            tk.Label(cf, text=f"   … +{len(self._names) - 14} lainnya",
                     bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w")

    def _show(self, mode: str):
        self._panel_upload.pack_forget()
        self._panel_camera.pack_forget()

        for key, widgets in self._nav_frames.items():
            c = HOVER_BG if key == mode else BG_CARD
            for w in widgets:
                w.config(bg=c)

        if mode == "upload":
            self._panel_upload.pack(fill="both", expand=True)
        else:
            self._panel_camera.pack(fill="both", expand=True)

    def _on_close(self):
        self._panel_camera.on_close()
        self.destroy()


if __name__ == "__main__":
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except:
        pass
    style.configure(
        "TScale",
        background=BG_CARD,
        troughcolor=BORDER_COLOR,
        sliderlength=16,
        sliderrelief="flat"
    )
    style.map("TScale", background=[("active", ACCENT_PURPLE)])

    app = FaceVisionApp()
    app.mainloop()