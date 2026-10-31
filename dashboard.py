import io, json, time, zipfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "best.pt"
DATA_YAML = APP_DIR / "data" / "data.yaml"
HISTORY_PATH = APP_DIR / "data" / "history.json"

st.set_page_config(
    page_title="VisionAIRIEZ — AI Vision Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling inspired by the lab assistant's Streamlit reference: welcome page,
# short registration, sidebar mode selection, image upload, and paired previews.
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #b3c7ff 0%, #8e9fff 100%);
  color: #f5f7ff;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px;}
.main-title {
  background: linear-gradient(145deg,#5368c9,#3549a6);
  border: 1px solid rgba(255,255,255,.18); border-radius: 18px;
  color: #fff; text-align:center; padding: 20px 16px; font-size: 27px;
  font-weight: 800; margin: 8px 0 18px; box-shadow: 0 8px 22px rgba(30,45,110,.16);
}
.section-title {
  background: rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.16);
  border-radius:12px; padding:10px 14px; color:#fff; text-align:center;
  font-size:19px; font-weight:750; margin: 12px 0;
}
.explain-box {
  background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.18);
  border-radius:12px; padding:15px 18px; color:#fff; line-height:1.6;
}
.metric-card {
  background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.22);
  border-radius:14px; padding:14px; color:#fff;
}
div[data-testid="stMetric"] {
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.18);
  padding:12px 14px; border-radius:12px;
}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#fff;}
.stButton>button, .stDownloadButton>button {
  border-radius:10px; border:0; background:#3449a8; color:white; font-weight:700;
}
.stButton>button:hover, .stDownloadButton>button:hover {background:#263987;color:white;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#303f91,#242f72);}
[data-testid="stSidebar"] * {color:#fff !important;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

def read_history():
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8")) if HISTORY_PATH.exists() else []
    except Exception:
        return []

def save_history(rows):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(rows[-100:], ensure_ascii=False, indent=2), encoding="utf-8")

@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def load_model(path_str):
    return YOLO(path_str)

def get_model():
    if not MODEL_PATH.exists():
        return None, f"File model tidak ditemukan: {MODEL_PATH}"
    try:
        return load_model(str(MODEL_PATH)), None
    except Exception as exc:
        return None, f"Model gagal dimuat. Periksa kompatibilitas file bobot dan versi Ultralytics. Detail: {exc}"

def color_name(hex_code):
    # Approximate descriptive color names, not scientific labels.
    r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (1, 3, 5))
    if max(r,g,b)-min(r,g,b) < 22:
        return "Abu-abu" if 65 < (r+g+b)/3 < 205 else ("Putih" if r > 205 else "Hitam")
    if r > 1.18*g and r > 1.18*b: return "Merah"
    if g > 1.15*r and g > 1.1*b: return "Hijau"
    if b > 1.12*r and b > 1.08*g: return "Biru"
    if r > 180 and g > 130 and b < 100: return "Oranye/Kuning"
    if r > 130 and b > 120 and g < 130: return "Ungu/Magenta"
    return "Warna campuran"

def analyze_colors(image_rgb, k=5):
    small = cv2.resize(image_rgb, (min(180, image_rgb.shape[1]), min(180, image_rgb.shape[0])))
    pixels = small.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    k = min(k, len(pixels))
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(counts)[::-1]
    palette = []
    for idx in order:
        rgb = np.clip(centers[idx], 0, 255).astype(int)
        hx = '#%02X%02X%02X' % tuple(rgb)
        palette.append({"hex": hx, "share": float(counts[idx]/len(labels)*100), "name": color_name(hx)})
    return palette

def image_analysis(img):
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    palette = analyze_colors(arr)
    return {
        "width": w, "height": h, "pixels": int(w*h),
        "mode": img.mode, "palette": palette, "rgb": arr
    }

def plot_rgb(arr):
    rows = []
    for channel, name in enumerate(["Red", "Green", "Blue"]):
        hist = cv2.calcHist([arr], [channel], None, [32], [0,256]).flatten()
        for i, val in enumerate(hist):
            rows.append({"Intensitas": int(i*8), "Jumlah piksel": int(val), "Kanal": name})
    return px.line(pd.DataFrame(rows), x="Intensitas", y="Jumlah piksel", color="Kanal",
                   title="Histogram kanal RGB", template="plotly_white")

def run_detection(model, img, conf):
    start = time.perf_counter()
    results = model.predict(source=np.array(img.convert("RGB")), conf=conf, verbose=False)
    elapsed = time.perf_counter() - start
    result = results[0]
    annotated = result.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    detections = []
    boxes = result.boxes
    if boxes is not None:
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            names = result.names
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            detections.append({
                "No.": i+1, "Kelas": str(label), "Confidence": confidence*100,
                "X1": round(xyxy[0], 1), "Y1": round(xyxy[1], 1),
                "X2": round(xyxy[2], 1), "Y2": round(xyxy[3], 1)
            })
    return annotated, detections, elapsed

def page_home():
    st.markdown('<div class="main-title">🔬 SELAMAT DATANG DI VISIONAIRIEZ</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Image Detection & Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    VisionAIRIEZ merupakan aplikasi analisis citra berbasis AI Computer Vision.
    Aplikasi menggunakan model YOLOv8 yang kamu sediakan untuk mendeteksi kelas
    <b>BENIGNA</b> dan <b>MALIGNA</b> (nama kelas aktual mengikuti metadata model).
    Selain deteksi objek, tersedia analisis visual warna dan riwayat analisis.
    Silakan lanjutkan ke registrasi singkat sebelum masuk ke dashboard.
    </div>""", unsafe_allow_html=True)
    st.write("")
    if st.button("Lanjut ke Registrasi →", use_container_width=True):
        st.session_state.page = "register"
        st.rerun()

def page_register():
    st.markdown('<div class="main-title">📝 Registrasi Singkat</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Isi data pengguna sebelum masuk dashboard</div>', unsafe_allow_html=True)
    with st.form("registration"):
        name = st.text_input("Nama Lengkap", value=st.session_state.get("user_name", ""))
        campus = st.text_input("Nama Kampus / Institusi", value=st.session_state.get("user_campus", ""))
        submit = st.form_submit_button("Masuk ke Dashboard Utama", use_container_width=True)
        if submit:
            if not name.strip() or not campus.strip():
                st.warning("Mohon isi nama dan kampus/institusi.")
            else:
                st.session_state.user_name = name.strip()
                st.session_state.user_campus = campus.strip()
                st.session_state.page = "main"
                st.rerun()
    if st.button("⬅️ Kembali ke Halaman Awal"):
        st.session_state.page = "home"
        st.rerun()

def page_main():
    model, model_error = get_model()
    with st.sidebar:
        st.markdown("## 🔬 VisionAIRIEZ")
        st.caption("AI Image Detection & Analytics")
        st.write(f"Pengguna: **{st.session_state.user_name}**")
        st.caption(st.session_state.user_campus)
        menu = st.radio("🧭 Pilih Menu:", [
            "Dashboard", "Deteksi Objek (YOLO)", "Analisis Warna", "Riwayat Analisis"
        ])
        st.divider()
        st.caption("Status Model")
        if model is None: st.error("Model tidak siap")
        else: st.success("YOLOv8 siap digunakan")
        st.caption("Model: model/best.pt")
        if st.button("Keluar / Logout", use_container_width=True):
            for key in ["user_name", "user_campus", "page", "last_result"]:
                st.session_state.pop(key, None)
            st.session_state.page = "home"
            st.rerun()

    st.markdown('<div class="main-title">🔎 VisionAIRIEZ — Dashboard Analisis Citra</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="explain-box">Halo <b>{st.session_state.user_name}</b> dari <b>{st.session_state.user_campus}</b>. Pilih menu di sidebar untuk memulai analisis.</div>', unsafe_allow_html=True)

    history = read_history()
    if menu == "Dashboard":
        total_images = len(history)
        total_objects = sum(x.get("total_objects", 0) for x in history)
        confs = [d.get("Confidence", 0) for x in history for d in x.get("detections", [])]
        avg_conf = float(np.mean(confs)) if confs else 0
        a,b,c,d = st.columns(4)
        a.metric("Total Gambar Dianalisis", total_images)
        b.metric("Total Objek Terdeteksi", total_objects)
        c.metric("Rata-rata Confidence", f"{avg_conf:.1f}%")
        classes = sorted({d.get("Kelas","") for x in history for d in x.get("detections",[])})
        d.metric("Jumlah Kelas Ditemukan", len(classes))
        st.markdown('<div class="section-title">📊 Ringkasan Aktivitas</div>', unsafe_allow_html=True)
        if history:
            rows = []
            for item in history:
                rows.append({"Waktu": item["timestamp"], "Gambar": item["filename"], "Objek": item["total_objects"]})
            df = pd.DataFrame(rows)
            st.plotly_chart(px.bar(df, x="Waktu", y="Objek", hover_name="Gambar",
                                   title="Jumlah objek per analisis", template="plotly_white"),
                            use_container_width=True)
            class_counts = {}
            for item in history:
                for det in item.get("detections", []):
                    class_counts[det["Kelas"]] = class_counts.get(det["Kelas"], 0) + 1
            if class_counts:
                col1,col2 = st.columns(2)
                with col1:
                    st.plotly_chart(px.bar(x=list(class_counts), y=list(class_counts.values()),
                                           labels={"x":"Kelas","y":"Jumlah"},
                                           title="Distribusi objek per kelas", template="plotly_white"),
                                    use_container_width=True)
                with col2:
                    st.plotly_chart(px.pie(names=list(class_counts), values=list(class_counts.values()),
                                           hole=.55, title="Proporsi kelas", template="plotly_white"),
                                    use_container_width=True)
            st.subheader("Analisis Terbaru")
            st.dataframe(pd.DataFrame(rows[::-1]), use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada riwayat analisis. Buka menu Deteksi Objek untuk memulai.")

    elif menu == "Deteksi Objek (YOLO)":
        st.markdown('<div class="section-title">📦 Deteksi Objek Menggunakan YOLOv8</div>', unsafe_allow_html=True)
        if model is None:
            st.error(model_error or "Model belum tersedia.")
            st.info("Pastikan file bobot tersedia di folder model dengan nama best.pt.")
        else:
            uploaded = st.file_uploader("📂 Unggah gambar untuk dideteksi", type=["jpg","jpeg","png","webp"], key="detect_upload")
            conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
            if uploaded:
                try:
                    img = Image.open(uploaded).convert("RGB")
                    st.caption(f"File: {uploaded.name} • {img.width} × {img.height} px • {len(uploaded.getvalue())/1024:.1f} KB")
                    left, right = st.columns(2)
                    with left: st.image(img, caption="🖼️ Gambar Asli", use_container_width=True)
                    if st.button("🚀 Mulai Deteksi", type="primary", use_container_width=True):
                        with st.spinner("AI sedang mendeteksi objek..."):
                            annotated, detections, elapsed = run_detection(model, img, conf)
                        with right:
                            st.image(annotated, caption="📦 Hasil Deteksi YOLOv8", use_container_width=True)
                        if detections:
                            st.success(f"{len(detections)} objek berhasil terdeteksi.")
                        else:
                            st.warning("Tidak ada objek yang memenuhi confidence threshold.")
                        df = pd.DataFrame(detections, columns=["No.","Kelas","Confidence","X1","Y1","X2","Y2"])
                        if not df.empty:
                            df["Confidence"] = df["Confidence"].map(lambda v: f"{v:.2f}%")
                            st.subheader("📋 Rincian Hasil Deteksi")
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        avg = float(np.mean([x["Confidence"] for x in detections])) if detections else 0
                        st.caption(f"Waktu inferensi terukur: {elapsed:.3f} detik")
                        analysis = image_analysis(img)
                        record = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "filename": uploaded.name, "total_objects": len(detections),
                            "average_confidence": avg, "detections": detections,
                            "width": img.width, "height": img.height, "inference_seconds": elapsed,
                            "palette": analysis["palette"]
                        }
                        all_history = read_history()
                        all_history.append(record)
                        save_history(all_history)
                        st.session_state.last_result = {
                            "annotated": annotated, "detections": detections, "filename": uploaded.name
                        }
                        st.download_button("⬇️ Unduh Gambar Hasil Deteksi",
                            data=Image.fromarray(annotated).convert("RGB").tobytes()
                            if False else _png_bytes(annotated),
                            file_name=f"hasil_{Path(uploaded.name).stem}.png", mime="image/png")
                except Exception as exc:
                    st.error(f"Gambar gagal diproses: {exc}")

    elif menu == "Analisis Warna":
        st.markdown('<div class="section-title">🎨 Analisis Distribusi dan Pola Warna</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Unggah gambar untuk analisis warna", type=["jpg","jpeg","png","webp"], key="color_upload")
        if uploaded:
            try:
                img = Image.open(uploaded).convert("RGB")
                analysis = image_analysis(img)
                arr = analysis.pop("rgb")
                c1,c2 = st.columns(2)
                with c1: st.image(img, caption="Gambar yang dianalisis", use_container_width=True)
                with c2:
                    st.metric("Dimensi", f'{analysis["width"]} × {analysis["height"]} px')
                    st.metric("Jumlah piksel", f'{analysis["pixels"]:,}')
                    st.metric("Mode warna sumber", Image.open(io.BytesIO(uploaded.getvalue())).mode)
                st.markdown("#### Palet 5 Warna Dominan")
                cols = st.columns(5)
                for col, item in zip(cols, analysis["palette"]):
                    with col:
                        st.markdown(f'<div style="height:65px;background:{item["hex"]};border-radius:10px;border:1px solid #fff"></div>', unsafe_allow_html=True)
                        st.markdown(f'**{item["hex"]}**')
                        st.caption(f'{item["name"]} • {item["share"]:.1f}%')
                st.plotly_chart(plot_rgb(arr), use_container_width=True)
                st.info("Analisis warna dihitung dari keseluruhan gambar, bukan khusus area objek. Nama warna hanya perkiraan deskriptif.")
            except Exception as exc:
                st.error(f"Gagal menganalisis gambar: {exc}")

    elif menu == "Riwayat Analisis":
        st.markdown('<div class="section-title">🕘 Riwayat Deteksi Sebelumnya</div>', unsafe_allow_html=True)
        if not history:
            st.info("Belum ada riwayat yang tersimpan.")
        else:
            rows = [{
                "Waktu": x.get("timestamp"), "Nama File": x.get("filename"),
                "Total Objek": x.get("total_objects"), "Rata-rata Confidence": f'{x.get("average_confidence",0):.2f}%',
                "Waktu Inferensi": f'{x.get("inference_seconds",0):.3f} s'
            } for x in history[::-1]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if st.button("Hapus seluruh riwayat", type="secondary"):
                save_history([])
                st.success("Riwayat dihapus.")
                st.rerun()

def _png_bytes(arr):
    out = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(out, format="PNG")
    return out.getvalue()

if "page" not in st.session_state:
    st.session_state.page = "home"
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "register":
    page_register()
else:
    page_main()
