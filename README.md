# VisionAIRIEZ — Image Detection & Analytics Dashboard

Aplikasi Streamlit dengan alur antarmuka yang mengikuti contoh proyek asisten laboratorium:
halaman sambutan → registrasi singkat → dashboard dengan navigasi sidebar → unggah gambar
→ preview asli dan hasil → tabel deteksi. Implementasi menggunakan aset yang disediakan:
model YOLOv8 `model/best.pt`, dataset ZIP, dan `data/data.yaml`.

## Fitur
- Registrasi singkat pengguna (nama dan kampus/institusi) untuk sesi aplikasi.
- Deteksi gambar sungguhan menggunakan Ultralytics YOLOv8 dari bobot yang disediakan.
- Preview gambar asli dan gambar dengan bounding box.
- Tabel kelas, confidence, dan koordinat bounding box `xyxy` dalam piksel.
- Confidence threshold yang dapat diatur.
- Unduh gambar hasil deteksi.
- Analisis warna keseluruhan gambar: lima warna dominan (K-Means), HEX, persentase, histogram RGB.
- Dashboard statistik dan grafik distribusi kelas/aktivitas.
- Riwayat analisis disimpan lokal di `data/history.json`.

## Struktur
```text
VisionAIRIEZ_Streamlit/
├── dashboard.py
├── requirements.txt
├── README.md
├── model/
│   └── best.pt
├── data/
│   ├── data.yaml
│   ├── FARIZ3.v1i.yolov8.zip
│   └── history.json (dibuat otomatis)
└── assets/
```

## Menjalankan di Windows
Disarankan Python 3.10 atau 3.11.

```bash
cd VisionAIRIEZ_Streamlit
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run dashboard.py
```

Aplikasi akan terbuka di browser pada alamat lokal yang ditampilkan Streamlit.

## Menjalankan di Linux/macOS
```bash
cd VisionAIRIEZ_Streamlit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run dashboard.py
```

## Catatan penting
1. Model yang dipakai adalah file asli yang kamu unggah, bukan model dummy. Jika file bobot tidak kompatibel dengan versi Ultralytics atau dependensi lokal, aplikasi menampilkan pesan error yang bisa digunakan untuk troubleshooting.
2. Nama kelas deteksi dibaca dari metadata model YOLO. Konfigurasi dataset menyebut `BENIGNA` dan `MALIGNA`; nama yang muncul saat inferensi mengikuti metadata bobot yang benar-benar dimuat.
3. Contoh aslab mempunyai model klasifikasi CNN `.h5`. File tersebut tidak tersedia di aset kamu, sehingga proyek ini tidak mengada-adakan model CNN atau fitur klasifikasi gambar terpisah. Mode utama adalah deteksi YOLOv8.
4. Analisis warna adalah deskripsi statistik pada seluruh gambar, bukan diagnosis atau karakteristik biologis kelas.
5. Registrasi ini hanya untuk identitas sesi antarmuka, bukan sistem akun/password. Contoh aslab menggunakan registrasi singkat, bukan autentikasi penuh.
6. Riwayat disimpan lokal di `data/history.json`. Untuk penggunaan multi-user/produksi, pindahkan penyimpanan ke database dan tambahkan autentikasi yang sesuai.
7. Dataset ZIP disertakan sebagai aset sumber. Aplikasi inferensi tidak perlu mengekstrak dataset untuk menjalankan model.
