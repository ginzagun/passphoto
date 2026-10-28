# 📸 Pasfoto Batch Studio (macOS)
*Otomatisasi Pasfoto Ijazah & Dokumen Resmi Berbasis AI*

**Developed by gnzgn ([@ginzagun](https://github.com/ginzagun)) • GINZAGUN PRODUCTION**

---

## ✨ Fitur Unggulan

- ✂️ **AI Background Removal**: Otomatis potong latar belakang menjadi merah (#DB1515), biru (#0072CE), putih, atau custom warna.
- 🎨 **Skin Tone Normalization**: Menyamakan kecerahan dan warna kulit wajah siswa/siswi secara merata tanpa mengubah kepekatan rambut hitam atau warna baju putih.
- 📐 **Standar Pasfoto Ijazah**: Pilihan ukuran 2x3 cm, 3x4 cm, 4x6 cm (300 DPI siap cetak) dengan bingkai putih dan teks nama sekolah di bawah foto.
- 🛠️ **Edit Manual Interaktif (Live Preview ~12ms)**:
  - 🔍 **Zoom**: Menyesuaikan proporsi subjek atau memotong baju dalaman hitam siswi berhijab.
  - ↔️ **Geser Horizontal & ↕️ Vertikal**: Menggeser posisi jika kurang ke tengah.
  - 🔄 **Rotasi & Pelurusan**: Meluruskan posisi kepala yang miring atau putar 90°.
  - ☀️ **Pencahayaan & Warna**: Atur kecerahan per-foto tanpa memengaruhi warna background.
- 📦 **Download Sekali Klik**: Otomatis mengemas seluruh foto hasil ke dalam file `pasfoto_hasil.zip`.

---

## 🚀 Cara Instalasi di Mac (Pengguna Biasa)

1. Unduh file installer **`PassPhoto-v1.0-macOS-arm64.dmg`** dari tab [Releases](https://github.com/ginzagun/passphoto/releases).
2. Buka file `.dmg` yang telah diunduh.
3. Tarik (*drag & drop*) icon **Pass Photo** ke dalam folder **Applications**.
4. Buka aplikasi **Pass Photo** dari Launchpad atau folder Applications.

### ⚠️ Catatan Keamanan macOS (Gatekeeper)
Karena aplikasi ini belum ditandatangani oleh sertifikat Apple Developer berbayar ($99/tahun), macOS akan memunculkan peringatan keamanan saat pertama kali dibuka dari internet:

> *"Pass Photo.app cannot be opened because Apple cannot check it for malicious software"*

**Cara mengatasinya (cukup 1 kali saja):**
- **Metode 1 (Paling Mudah)**: Buka Finder ➡️ masuk ke folder **Applications** ➡️ Klik kanan (atau `Control + Klik`) pada **Pass Photo** ➡️ Pilih **Open** ➡️ Klik **Open**.
- **Metode 2 (Lewat Terminal jika masih terblokir)**:
  ```bash
  xattr -cr "/Applications/Pass Photo.app"
  ```

---

## 💻 Menjalankan dari Source Code (Developer)

Jika ingin menjalankan atau mengembangkan script langsung:

```bash
# 1. Clone repository
git clone https://github.com/ginzagun/passphoto.git
cd passphoto

# 2. Buat virtual environment & install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Jalankan aplikasi
python3 pasfoto_app.py
```

Buka browser di **http://127.0.0.1:7860**.

---

## 🛠️ Cara Build Ulang Installer DMG

Untuk membuat file installer `.dmg` sendiri:

```bash
chmod +x build_dmg.sh
./build_dmg.sh
```

File `.dmg` siap pakai akan otomatis dibuat di dalam folder `dist/`.

---

## 📄 Lisensi & Hak Cipta
Copyright © 2026 **GINZAGUN PRODUCTION** by **gnzgn** ([@ginzagun](https://github.com/ginzagun)). All Rights Reserved.
