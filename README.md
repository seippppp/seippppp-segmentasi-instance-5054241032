# Segmentasi Instance Indoor — Training di Kaggle

**Seifeldein Akbar · NRP 5054241032 · variabel NRP (digit 2): jumlah data train 25% / 50% / 100%**

Folder ini adalah **repositori Git utama** untuk tugas segmentasi instance. Isinya:

| Berkas | Fungsi |
|---|---|
| `segmentasi_instance_kaggle.ipynb` | Satu notebook untuk semua tahap: instalasi → dataset v5 → audit → smoke test → 9 training → evaluasi → tabel/grafik → test sendiri → analisis kegagalan → demo |
| `LOGBOOK.md` | Logbook mingguan + **hipotesis bertanggal** (ditulis sebelum training) |
| `laporan_awal.md` | Laporan awal: tujuan, dataset, model, metodologi, hipotesis, rencana |
| `hasil/` | Diisi dari `hasil_ringan.zip` setelah training (log, metrik, grafik, overlay; tanpa bobot) |
| `.gitignore` | Mengecualikan data, bobot `.pt/.pth/.ckpt`, zip, dan folder `kerja/` |

## Status (24 November 2026)

- Notebook sudah **diuji end-to-end di laptop (RTX 4060) dengan dataset sintetis** untuk ketiga model: audit, subset,
  training, seleksi checkpoint di validasi, evaluasi test, test sendiri, kondisi sulit, AP per kelas, overlay, EXIF,
  demo, dan ekspor.
- **Belum diuji:** dataset v5 asli dan lingkungan Kaggle. Instalasi paket di Kaggle belum pernah dijalankan; semua wheel
  Linux yang dibutuhkan RF-DETR sudah dicek tersedia di PyPI.
- **Belum ada training pada dataset v5.** Hipotesis sudah ditulis di `LOGBOOK.md`.

## Langkah 0 — Git (lakukan sebelum apa pun)

```powershell
cd "C:\Users\asuss\Documents\visi komputer\segmentasi_kaggle"
git init
git add .
git commit -m "Notebook Kaggle, laporan awal, dan hipotesis sebelum training"
# buat repo kosong di GitHub, lalu:
git remote add origin https://github.com/<username>/segmentasi-instance-5054241032.git
git push -u origin main
```

Commit setiap langkah di bawah secara terpisah. Jangan menunggu satu commit besar di akhir.

## Langkah 1 — Siapkan dataset v5 untuk Kaggle

Pilih salah satu:

- **A. Unggah sebagai Kaggle Dataset (disarankan).** Login Roboflow → buka
  [dataset v5](https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5) → *Download Dataset*
  → format **COCO Segmentation** → unduh zip. Di Kaggle: *Datasets → New Dataset* → unggah zip → **Private**.
- **B. API key Roboflow.** Di notebook Kaggle: *Add-ons → Secrets → Add* dengan nama `ROBOFLOW_API_KEY`.
  Notebook akan mengunduh sendiri. Jangan menulis API key di notebook.

## Langkah 2 — Buat notebook di Kaggle

1. Akun Kaggle harus sudah **verifikasi nomor HP** (syarat GPU dan internet).
2. *Create → New Notebook → File → Import Notebook* → pilih `segmentasi_instance_kaggle.ipynb`.
3. *Settings*: **Accelerator = GPU T4 x2** atau **GPU P100** (notebook memakai 1 GPU), **Internet = On**.
4. *Add Input* → dataset dari Langkah 1 (jika memakai pilihan A).

## Langkah 3 + 4 — Run All pertama: audit dan smoke test (status sekarang)

Dataset sudah diunggah ke Kaggle di `/kaggle/input/datasets/seifeldeinakbar/panoptic-indoor` (berisi `train/`,
`valid/`, `test/`), dan path ini sudah diisi di `DATA_SOURCE`. Audit sudah diverifikasi di laptop dan lampiran kelas
sudah masuk `LOGBOOK.md` (commit `fe16b12`). Sakelar saat ini: `RUN_INSTALL`, `RUN_PREPARE`, `RUN_SMOKE` = `True`,
sisanya `False`. Jadi **Run All** = instalasi → audit → smoke test 2 epoch per model pada 25% data.

1. Klik *Run All* (atau *Save Version → Save & Run All*).
2. Periksa output audit: harus 397 / 166 / 359 gambar, 74 kelas, 4 gambar diperbaiki orientasinya, `Masalah: []`.
3. Di akhir smoke test tercetak waktu per model dan **perkiraan total 9 run**. Unduh `hasil_smoke.zip` dari *Output*,
   ekstrak ke repo, lalu commit.

Catatan smoke test:

- Catat waktu per epoch dan VRAM puncak (`hasil/catatan_waktu_gpu.csv`).
- Jika OOM, kecilkan `batch` di `MODEL_CFG` (sel 1) dan ulangi. Gunakan batch yang sama untuk ketiga proporsi.
- Perkirakan waktu total: `menit per epoch × 50 × (0,25 + 0,5 + 1)`. Jika melebihi kuota GPU mingguan Kaggle (lihat
  profil Kaggle), turunkan `EPOCHS` **untuk semua model**, lalu catat keputusannya di logbook.
- Commit catatan smoke test.

## Langkah 5 — Training utama (dijalankan di latar belakang)

Untuk tiap model (agar muat dalam batas sesi Kaggle 12 jam):

1. `RUN_SMOKE = False`, `RUN_TRAIN = True`. Jika perkiraan total dari smoke test < 10 jam, ketiga model boleh dalam
   satu sesi; jika lebih, isi `TRAIN_MODELS = ['yolo']` (lalu `['maskrcnn']`, lalu `['rfdetr']`).
2. *Save Version → Save & Run All (Commit)*. Laptop boleh dimatikan.
3. Setelah selesai, buka versi tersebut → *Output* → unduh `hasil/` dan file `runs/*/run.json` serta log.
4. Isi logbook (waktu, kendala), lalu commit log tersebut.

Setiap run otomatis: memilih checkpoint dengan mask AP **validation**, mengevaluasi validation dan test, mencatat waktu
GPU, dan menghapus checkpoint yang tidak terpilih (`HAPUS_CHECKPOINT_LAIN`).

## Langkah 6 — Rangkuman, tabel, dan grafik

1. Di notebook Kaggle: *Add Input → Your Work → Notebooks* → tambahkan output versi training ketiga model.
2. `RUN_TRAIN = False`, `RUN_SUMMARY = True` → jalankan. Notebook menyalin run dari input, lalu membuat:
   `tabel_hasil_nrp.csv`, `grafik_nrp.png`, `ap_per_kelas_p100.csv`, `hasil_ringan.zip`, `bobot_terpilih.zip`.
3. Ekstrak `hasil_ringan.zip` ke folder repo (`hasil/` dan `runs/`) → commit.
4. Ekstrak `bobot_terpilih.zip` ke `segmentasi_kaggle/kerja/` di laptop (tidak masuk Git) untuk demo.
5. Bandingkan hasil dengan H1–H3 di logbook. Tulis penjelasan jika hipotesis salah.

## Langkah 7 — Test set sendiri, analisis kegagalan, demo

1. Ambil ≥ 40 foto (≥ 10 kondisi sulit). Simpan file asli (kirim lewat kabel/Drive, bukan WhatsApp).
2. Jalankan `export_exif('folder/foto_asli')` di laptop → `hasil/exif_test_sendiri.csv`.
3. Anotasi di Roboflow dengan nama kelas yang sama persis, screenshot proses, ekspor COCO Segmentation.
4. Isi `OWN_TEST_DIR` dan `KONDISI_SULIT`, `RUN_OWN_TEST = True` → `hasil/domain_shift.csv`.
5. Pilih 5 kegagalan → `tampilkan_kasus(...)` → eksperimen kecil dengan `uji_variasi(...)`
   (resolusi inferensi YOLO, gamma/kecerahan, dll.).
6. Demo: isi `DEMO_FOTO`, ubah `DEMO_CONF`, panggil `ap_kelas('nama_kelas')`. Latih minimal sekali.

## Menjalankan di laptop (alternatif)

Per 24 Nov, PyTorch 2.12 + CUDA sudah berjalan di RTX 4060 8 GB. Notebook mendeteksi VRAM < 14 GB dan memakai batch
lebih kecil (Mask R-CNN 2, YOLO 8, RF-DETR 2 × akumulasi 8). Folder kerja lokal adalah `segmentasi_kaggle/kerja/`.
Pasang paket dalam virtual environment sendiri agar `transformers` versi 5 dari RF-DETR tidak mengganggu proyek lain:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install "ultralytics==8.4.66" "rfdetr[train]==1.12.1" pycocotools roboflow pandas matplotlib ipykernel
```

Jika training dilakukan di laptop, batch berbeda dari Kaggle. Catat di logbook dan jangan mencampur run dari dua
konfigurasi batch dalam satu tabel.

## Catatan teknis

- **Evaluator bersama** (pycocotools `segm`, maxDets 100, conf ≥ 0,001, threshold mask 0,5). Prediksi dicocokkan ke
  ground truth berdasarkan **nama kelas**, sehingga dapat dipakai juga untuk test set sendiri yang ID kelasnya berbeda.
- Angka evaluator bersama dapat berbeda dari metrik bawaan framework. Laporan memakai evaluator bersama.
- RF-DETR mengeluarkan slot `__background__`; slot ini dibuang sebelum evaluasi.
- YOLO: checkpoint dipilih dengan mask mAP50-95 validasi Ultralytics (callback `best_mask.pt`), bukan *fitness*
  gabungan box+mask. Mask R-CNN dan RF-DETR: dipilih dengan evaluator bersama pada validasi.
- Statistik bentuk kelas (luas, *fill ratio*) dihitung dari train + valid saja.
- `elapsed_seconds` mencakup pemuatan model, training, validasi per epoch, dan penyimpanan checkpoint.
