# Logbook — Segmentasi Instance Indoor

**Seifeldein Akbar · NRP 5054241032 · variabel NRP: jumlah data train 25% / 50% / 100%**

Logbook ditulis per minggu. Setiap entri memuat apa yang dikerjakan, kendala, keputusan, dan alasannya.
Entri hipotesis bertanggal ditulis **sebelum** log training pertama pada dataset v5.

---

## Minggu 1 — 18 s.d. 24 November 2026

### Dikerjakan

- 18 Nov: membaca ketentuan tugas, memilih tiga model (Mask R-CNN R50-FPN, YOLO26s-seg, RF-DETR-Seg Medium),
  menentukan variabel NRP (digit 2: jumlah data train 25/50/100%).
- 18 Nov: membuat kode awal di `segmentasi_instance/` dan notebook di `cpmk 3/`. Keduanya belum pernah dijalankan.
- 24 Nov: aturan tugas diperbarui (hipotesis bertanggal, riwayat Git, demo langsung). Rangkuman tugas ditulis ulang.
- 24 Nov: membuat folder baru `segmentasi_kaggle/` berisi satu notebook untuk semua tahap, dijalankan di Kaggle.
- 24 Nov, 18:20–18:35: **uji pipeline pada dataset sintetis buatan sendiri** (28 gambar bentuk geometris, 4 kelas), bukan
  dataset v5. Tujuannya hanya memastikan kode ketiga model berjalan dari audit sampai demo. Angka dari uji ini tidak
  dipakai di laporan dan lognya tidak disimpan di repo.

### Kendala

- 18 Nov: `import torch` gagal (`WinError 4551`, Application Control memblokir `caffe2_nvrtc.dll`); pip gagal
  mengakses PyPI (DNS). Pada 24 Nov keduanya sudah berjalan: torch 2.12.0+cu126, CUDA aktif di RTX 4060 Laptop 8 GB.
- Halaman Roboflow Universe menolak akses otomatis (HTTP 403), sehingga daftar kelas v5 belum diketahui sebelum dataset
  diunduh. Hipotesis per kelas ditulis berdasarkan sifat kelas (ukuran, bentuk, frekuensi), lalu dipetakan ke nama kelas
  setelah audit (lihat "Lampiran kelas").
- Ukuran checkpoint RF-DETR sekitar 1,5 GB per run (checkpoint Lightning + EMA). Sembilan run bisa mendekati batas disk
  Kaggle 20 GB.

### Keputusan

| Keputusan | Alasan |
|---|---|
| Training di Kaggle (GPU T4/P100 16 GB) | VRAM laptop 8 GB; sesi Kaggle bisa berjalan di latar belakang (*Save & Run All*) |
| Satu notebook untuk semua tahap | Satu sumber kode untuk tiga model, satu evaluator, satu alur bukti |
| Evaluator COCO mask bersama untuk ketiga model | Metrik bawaan framework berbeda cara hitungnya; uji sintetis: YOLO 0,73 (evaluator bersama) vs 0,81 (Ultralytics) |
| Versi paket dikunci: `ultralytics==8.4.66`, `rfdetr[train]==1.12.1` | Paket tidak boleh berubah di tengah rangkaian eksperimen |
| Epoch tetap 50 untuk semua proporsi data | Variabel NRP adalah jumlah data; konsekuensinya jumlah langkah optimizer ikut berubah (dicatat di hipotesis H3) |
| Hanya checkpoint terpilih yang disimpan | Batas disk Kaggle; checkpoint terpilih tetap bisa dimuat ulang (sudah diuji) |

### Kasus kode AI yang salah dan perbaikannya (untuk lampiran deklarasi AI)

1. **Pemetaan kelas RF-DETR.** Kode awal dari AI memakai `class_names[class_id]`. Saat diuji, muncul `IndexError`.
   Pemeriksaan menunjukkan RF-DETR 1.12.1 bisa mengeluarkan `class_id = jumlah_kelas` dengan nama `__background__`
   (slot "bukan objek"). Perbaikan: memakai `class_name` dari hasil prediksi dan membuang `__background__`.
2. **Deteksi lingkungan Kaggle.** Kode awal menganggap notebook berjalan di Kaggle jika folder `/kaggle/working` ada.
   Di laptop ini ternyata ada folder `C:\kaggle\working` dari proyek lain, sehingga uji pertama menulis data ke folder
   tersebut. File uji itu dihapus (hanya yang dibuat hari ini), dan deteksi diganti dengan variabel lingkungan
   `KAGGLE_KERNEL_RUN_TYPE` yang hanya ada di Kaggle.
3. **Penyimpanan prediksi.** `pycocotools.loadRes` menambahkan field bertipe numpy ke daftar prediksi, sehingga
   `json.dump` gagal (`uint32 is not JSON serializable`). Diperbaiki dengan konversi tipe saat menyimpan.
4. **Luas objek dari field `area`.** Kode awal memakai field `area` sebagai luas mask. Audit dataset asli menunjukkan
   *fill ratio* semua kelas tepat 1,0 — ternyata Roboflow mengisi `area` dengan luas bounding box. Diperbaiki dengan
   menghitung luas dari polygon (`pycocotools`), dan field `area` di anotasi yang disiapkan diganti dengan luas mask.

---

## Hipotesis — Senin, 24 November 2026, 18:40 WIB

**Ditulis sebelum log training pertama pada dataset Panoptic Indoor Segmentation v5.** Belum ada satu pun training,
validasi, atau evaluasi pada dataset v5. Draf disusun dengan bantuan AI atas permintaan saya; penalarannya saya periksa
dan saya pertanggungjawabkan.

### Dasar arsitektur yang dipakai

| Model | Fakta yang relevan (dicek dari kode/dokumentasi) |
|---|---|
| Mask R-CNN R50-FPN | Two-stage: RPN → RoIAlign 14×14 → mask **28×28 per objek**, lalu diregangkan ke ukuran box. NMS per kelas. Backbone ResNet-50. |
| YOLO26s-seg | One-stage, end-to-end tanpa NMS. Mask = kombinasi **32 prototype 160×160** (1/4 dari input 640) yang dipotong box. |
| RF-DETR-Seg Medium | Backbone **DINOv2** (ViT, patch 12), decoder DETR tanpa NMS, input **432×432**, peta embedding mask 1/4 resolusi (**108×108**). |

Konsekuensi resolusi mask terhadap foto 640 px:

- Mask R-CNN: objek besar (misal 400 px) hanya diwakili 28 sel → satu sel ≈ 14 px → tepi kasar. Objek kecil (30 px)
  tetap mendapat 28 sel → tepi relatif halus.
- YOLO26: satu sel prototype = 4 px untuk semua objek → tepi objek besar halus, tetapi objek 20 px hanya ≈ 5×5 sel.
- RF-DETR: gambar diperkecil ke 432 lalu 1/4 → satu sel ≈ 6 px pada skala 640; objek kecil kehilangan detail paling banyak.

Jumlah langkah optimizer (50 epoch; batch rencana di Kaggle 16 GB):

| Model | Batch efektif | 25% (99 gambar) | 50% (198) | 100% (397) |
|---|---|---:|---:|---:|
| Mask R-CNN | 4 | 1.250 | 2.500 | 5.000 |
| YOLO26s-seg | 16, diakumulasi ke 64 (`nbs`), warmup ≥ 100 iterasi | ≈ 100 | ≈ 180 | ≈ 330 |
| RF-DETR-Seg M | 4 × akumulasi 4 = 16 | ≈ 310 | ≈ 620 | ≈ 1.250 |

### H1 — Model terbaik keseluruhan (100% data, mask mAP@50–95, test asli)

**RF-DETR-Seg Medium > YOLO26s-seg > Mask R-CNN.**
Alasan: fitur DINOv2 hasil pretraining skala besar paling mudah dipindahkan ke dataset kecil (397 gambar) dengan banyak
kelas; prediksi berbasis query tanpa NMS tidak menghapus objek sekelas yang saling menutupi. Mask R-CNN tertinggal di
mAP@50–95 karena mask 28×28 menurunkan IoU pada ambang ketat (0,75–0,95).
**Hipotesis salah jika:** urutan berbeda, atau selisih antar-model < 1 poin (dianggap setara).

### H2 — Kelas mana yang unggul pada model mana

Kelompok kelas ditentukan dari `statistik_kelas.csv` (train + valid, tanpa test):
kecil = median luas < 1% gambar; besar = median luas > 10%; tipis/berlubang = *fill ratio* (luas mask / luas box) < 0,5;
langka = < 10 instance train.

| Kelompok | Prediksi | Mekanisme |
|---|---|---|
| H2a. Besar dan padat (contoh yang diharapkan: kasur, lemari, meja) | YOLO26 dan RF-DETR > Mask R-CNN; selisih terbesar di AP75 | Mask 28×28 diregangkan → tepi bergerigi pada objek besar |
| H2b. Kecil (contoh: `palte`/piring, gelas, botol) | **Mask R-CNN ≥ YOLO26 > RF-DETR** | RoIAlign memberi objek kecil grid 28×28 sendiri; RF-DETR bekerja di 432 px sehingga objek kecil makin kecil |
| H2c. Tipis/berlubang (contoh: kursi, `staris`/tangga, lampu) | AP terendah untuk ketiga model; RF-DETR paling tidak buruk | Kaki kursi dan anak tangga lebih tipis dari satu sel mask (≈ 4–14 px), sehingga mask "bocor" ke celah |
| H2d. `persob`/person | AP tinggi untuk ketiga model, selisih kecil | Kelas ini ada di pretraining COCO ketiga model |
| H2e. Kelas langka | AP ≈ 0 untuk semua model; urutan tidak stabil | Contoh terlalu sedikit; nilainya didominasi kebetulan |

**Hipotesis salah jika:** pada kelompok kecil Mask R-CNN berada di bawah kedua model lain, atau pada kelompok besar
Mask R-CNN tidak tertinggal di AP75.

### H3 — Pengaruh jumlah data train (variabel NRP)

1. Ketiga model naik monoton dari 25% → 50% → 100%.
2. **Penurunan relatif terbesar dari 100% ke 25% terjadi pada YOLO26s-seg**, karena pada 25% hanya mendapat ≈ 100
   langkah optimizer (akumulasi ke batch 64 dan warmup 100 iterasi menghabiskan sebagian besar training).
3. **Penurunan relatif terkecil terjadi pada RF-DETR-Seg Medium**, karena fitur DINOv2 sudah kuat sebelum fine-tuning.
4. Sebagian penurunan pada 25% bersifat mekanis: kelas yang tidak muncul di subset train pasti ber-AP ≈ 0.
   Jumlahnya dicatat di `audit.json` (`kelas_tanpa_instance_train_p25`) dan dibahas terpisah dari efek arsitektur.

**Hipotesis salah jika:** YOLO26 bukan yang turun paling besar. Jika itu terjadi dan Mask R-CNN yang turun paling
besar, penjelasan "jumlah langkah optimizer" gugur dan perlu diganti.

### H4 — Domain shift ke foto HP sendiri

Ketiga model turun pada test sendiri. Penurunan terbesar terjadi pada subset kondisi sulit (redup, backlight).
RF-DETR turun paling sedikit karena DINOv2 dilatih dengan data yang sangat beragam; Mask R-CNN turun paling banyak.

### Cara menguji

- H1, H3: `tabel_hasil_nrp.csv` dan `grafik_nrp.png` (mask mAP@50–95, mAP50, mAP75 dari evaluator bersama).
- H2: `ap_per_kelas_p100.csv` + `mAP_small/medium/large` + pengelompokan dari `statistik_kelas.csv`.
- H4: `domain_shift.csv` (test asli vs test sendiri vs kondisi sulit).
- Jika hipotesis salah, penjelasannya ditulis di laporan dengan bukti (per kelas, kurva loss, jumlah langkah, overlay).

### Lampiran kelas — diisi 24 November 2026, ±18:55 WIB, setelah audit dan sebelum training (commit `fe16b12`, 18:58)

Sumber: audit notebook pada ekspor COCO Segmentation yang diunggah ke Kaggle (folder lokal
`panoptic-indoor-segmentation.v1i.coco-segmentation`). Jumlah split 397 / 166 / 359 sama dengan halaman v5; label
versi "1" di file ekspor kemungkinan karena dataset diekspor melalui workspace Roboflow sendiri. **Belum ada training.**
Kelompok dihitung dari train + valid (luas mask median relatif terhadap gambar, *fill ratio* = luas mask / luas box).

| Kelompok | Nama kelas pada dataset |
|---|---|
| Besar (luas > 10%) | `trash-bin`, `toilet`, `paper-merged`, `wall`, `st-`, `tree-merged` (tiga terakhir ≤ 2 instance train) |
| Kecil (luas < 1%) | `bottle`, `cup`, `knife`, `spoon`, `light`, `toy`, `book`, `chair`, `bowl`, `paper`, `potted plant`, `remote`, `clock`, `fork`, `microwave`, `toaster`, `tv`, `vase` |
| Tipis/berlubang (fill < 0,5) | `dining table` (0,35), `spoon` (0,41), `fork` (0,45), `knife` (0,48) |
| Person | `person` (192 train / 413 test) |
| Langka (< 10 instance train) | 33 kelas, a.l. `handbag`, `dining table`, `curtain`, `laptop`, `cell phone`, `tv`, `vase`, `sink` |
| Tidak ada sama sekali di train | `hair drier`, `keyboard`, `mouse`, `flower`, `scissor`, `scissors`, `toaster`, `teddy bear`, `toothbrush` (AP pasti ≈ 0) |
| Tidak ada di subset train 25% | 21 kelas (9 di atas + `backpack`, `cardboard`, `clock`, `dining table`, `fork`, `microwave`, `st-`, `tree-merged`, `tv`, `vase`, `wall`, `wall-stone`) |

**Koreksi contoh nama kelas di H2** (tanpa mengubah prediksinya): ekspor ini memakai `plate`, `person`, `stairs`,
bukan `palte`, `persob`, `staris`. `chair` di dataset ini tergolong **kecil dan padat** (fill 0,70) karena polygon
kursi digambar sebagai satu blok, sehingga uji H2c (tipis/berlubang) memakai `dining table`, `spoon`, `fork`, `knife`.
Kelompok besar dan padat untuk H2a: `trash-bin` dan `toilet`.

**Catatan audit yang memengaruhi interpretasi:**

- **Distribusi kelas train dan test sangat berbeda.** Contoh: `handbag` 5 train / 125 test, `dining table` 4 / 112,
  `curtain` 3 / 107, `light` 41 / 295, sebaliknya `wallet` 197 / 5, `trash-bin` 153 / 4, `box` 95 / 1, `stairs` 175 / 6.
  mAP test (rata-rata per kelas) akan banyak ditentukan oleh kelas yang jarang dilihat saat training.
- 9 kelas tidak punya contoh train tetapi ada di test, sehingga batas atas mAP test di bawah 1 untuk semua model.
- 4 gambar (`182169107`, `182169108`, `182169109`, `182169121`) tersimpan potret tanpa EXIF, sedangkan anotasinya
  landscape. Dicek visual: cocok setelah diputar 90° berlawanan jarum jam; notebook memutarnya saat menyalin.
- Field `area` ekspor Roboflow berisi luas bounding box. Notebook menggantinya dengan luas mask agar kategori
  small/medium/large di COCOeval benar.
- 1 gambar validation tidak memiliki anotasi; tetap dipakai (prediksi di gambar itu dihitung sebagai false positive).

---

## Minggu 2 — 25 November s.d. 1 Desember 2026

_(isi: hasil audit, smoke test, waktu per epoch, keputusan epoch/batch, kendala Kaggle, run yang selesai)_
