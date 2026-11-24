# Laporan Awal — Perbandingan CNN Two-Stage, CNN One-Stage, dan Transformer untuk Segmentasi Instance Ruangan Indoor

**Nama:** Seifeldein Akbar
**NRP:** 5054241032
**Mata kuliah:** Visi Komputer — Evaluasi CPMK 3
**Tanggal:** 24 November 2026
**Status:** rencana dan hipotesis. **Belum ada hasil training pada dataset v5**; semua tabel hasil masih kosong.

---

## 1. Tujuan

Membandingkan tiga model segmentasi instance dari dua keluarga arsitektur pada citra ruangan indoor, lalu menguji
seberapa baik model yang dilatih pada dataset publik bekerja pada foto ruangan nyata yang diambil sendiri.

Pertanyaan yang dijawab:

1. Model mana yang menghasilkan mask paling akurat pada test set asli (mask mAP@50–95)?
2. Kelas seperti apa (besar, kecil, tipis, langka) yang menjadi kekuatan atau kelemahan tiap arsitektur, dan mengapa?
3. **(Variabel NRP, digit 2)** Bagaimana performa tiap model berubah jika data train dikurangi menjadi 50% dan 25%?
4. Seberapa besar penurunan performa pada foto HP sendiri (domain shift), terutama pada kondisi sulit?

## 2. Dataset

### 2.1 Data latih: Panoptic Indoor Segmentation v5

| Split | Gambar | Fungsi |
|---|---:|---|
| Train | 397 | Melatih model (dipotong menjadi 25/50/100% untuk eksperimen NRP) |
| Validation | 166 | Memilih checkpoint; tidak pernah diperkecil |
| Test asli | 359 | Evaluasi akhir, satu kali per run; tidak dipakai untuk memilih apa pun |

Sumber: https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5, format COCO Segmentation.
Jumlah di atas berasal dari halaman v5 dan akan diverifikasi oleh audit di notebook.

### 2.2 Audit dan subset

- Validasi file gambar, ukuran, polygon, kategori, dan duplikasi identik antar-split (hash MD5).
- Kategori root sintetis Roboflow dibuang. Kelas bernama salah ketik (`palte`, `persob`, `staris`) dipertahankan apa
  adanya, karena mengganti nama tanpa memeriksa anotasi bisa menggabungkan kelas yang berbeda.
- Subset bersarang dengan seed 42: 25% (99 gambar) ⊂ 50% (198) ⊂ 100% (397), sama untuk ketiga model.
- Statistik kelas (luas relatif median, *fill ratio* = luas mask / luas box) dihitung dari train + valid saja,
  untuk mengelompokkan kelas menjadi kecil/besar/tipis/langka tanpa melihat test.

### 2.3 Test set sendiri (menyusul)

≥ 40 foto ruangan nyata dari kamera HP sendiri, ≥ 10 di antaranya kondisi sulit (redup, backlight, objek saling
menutupi, sudut ekstrem). Dianotasi di Roboflow dengan nama kelas yang sama. EXIF diekstrak dari file asli.

## 3. Model

| Model | Keluarga | Bobot awal | Mekanisme mask |
|---|---|---|---|
| Mask R-CNN ResNet-50 FPN (He et al., 2017) | CNN, two-stage | COCO (torchvision) | RPN → RoIAlign → mask 28×28 per objek |
| YOLO26s-seg (Ultralytics) | CNN, one-stage, tanpa NMS | COCO | 32 prototype mask 160×160 dikombinasikan per objek |
| RF-DETR-Seg Medium (Roboflow) | Transformer, backbone DINOv2 | COCO | Query DETR × peta embedding piksel 108×108 (input 432) |

Ketiganya memenuhi syarat **3 model dari 2 keluarga**: CNN vs transformer, sekaligus two-stage vs one-stage.

## 4. Metodologi

### 4.1 Lingkungan

Kaggle Notebook, GPU NVIDIA T4 atau P100 (16 GB). Versi paket dikunci: `ultralytics==8.4.66`,
`rfdetr[train]==1.12.1`; versi PyTorch, CUDA, dan GPU dicatat otomatis di setiap `run.json`.

### 4.2 Konfigurasi training

| Pengaturan | Mask R-CNN | YOLO26s-seg | RF-DETR-Seg Medium |
|---|---|---|---|
| Resolusi | sisi terpanjang 640 | 640 | 432 (bawaan Medium) |
| Batch (GPU 16 GB) | 4 | 16 (akumulasi ke 64) | 4 × akumulasi 4 |
| Optimizer | SGD 0,005, momentum 0,9, wd 1e-4, warmup + cosine | `auto` bawaan Ultralytics | AdamW 1e-4 bawaan RF-DETR |
| Augmentasi | flip horizontal 0,5 | bawaan Ultralytics (mosaic, HSV, dll.) | bawaan RF-DETR |
| Epoch | 50 | 50 | 50 |
| Seleksi checkpoint | mask AP validasi (evaluator bersama) | mask mAP50-95 validasi Ultralytics | mask AP validasi (evaluator bersama) |

Resolusi, augmentasi, dan batch efektif berbeda antar-model karena mengikuti resep bawaan masing-masing. Karena itu
perbandingan ini adalah perbandingan **konfigurasi praktis**, bukan isolasi murni efek arsitektur.

### 4.3 Eksperimen NRP: jumlah data train

Tiga model × tiga proporsi = **9 run**, masing-masing dari bobot pretrained baru, seed 42, epoch tetap 50.
Dengan epoch tetap, jumlah langkah optimizer ikut naik bersama jumlah data (lihat tabel di hipotesis). Hal ini menjadi
bagian dari penjelasan, bukan disembunyikan.

### 4.4 Protokol evaluasi

- Evaluator COCO mask yang sama untuk ketiga model (pycocotools `segm`, maxDets 100, conf ≥ 0,001, mask > 0,5).
- Metrik utama mask mAP@50–95; pelengkap mAP@50, mAP@75, mAP small/medium/large, AP per kelas.
- Waktu inferensi: median ms/gambar setelah 3 warm-up, termasuk pra- dan pascaproses, pada GPU yang sama.
- Rata-rata hanya atas kelas yang memiliki ground truth pada set yang dievaluasi.

### 4.5 Domain shift dan analisis kegagalan

Checkpoint p100 yang sama dievaluasi pada test asli, test sendiri, dan subset kondisi sulit. Lima foto gagal
dianalisis dengan overlay GT vs prediksi ketiga model. Tiap dugaan penyebab diuji dengan satu eksperimen kecil
(checkpoint tetap, satu variabel diubah: resolusi inferensi, gamma/kecerahan, atau threshold).

## 5. Hipotesis

Hipotesis lengkap dan bertanggal ada di `LOGBOOK.md` (entri **24 November 2026, 18:40 WIB**, sebelum training pertama).
Ringkasannya:

| Kode | Hipotesis | Mekanisme |
|---|---|---|
| H1 | Pada 100% data: RF-DETR-Seg M > YOLO26s-seg > Mask R-CNN (mask mAP@50–95) | Fitur DINOv2; tanpa NMS; mask 28×28 Mask R-CNN lemah di IoU ketat |
| H2a | Kelas besar/padat: YOLO26 dan RF-DETR > Mask R-CNN, selisih terbesar di AP75 | 28×28 diregangkan ke objek besar → tepi kasar |
| H2b | Kelas kecil: Mask R-CNN ≥ YOLO26 > RF-DETR | RoIAlign memberi objek kecil grid sendiri; RF-DETR bekerja di 432 px |
| H2c | Kelas tipis/berlubang: AP terendah untuk semua; RF-DETR paling tidak buruk | Struktur lebih tipis dari satu sel mask |
| H3 | Semua naik monoton 25→100%; YOLO26 turun paling besar di 25%, RF-DETR paling kecil | YOLO hanya ≈ 100 langkah optimizer di 25% (akumulasi ke 64 + warmup); DINOv2 kuat tanpa banyak data |
| H4 | Semua turun di foto HP sendiri, terbesar pada kondisi sulit; RF-DETR turun paling sedikit | Keragaman data pretraining DINOv2 |

## 6. Ancaman terhadap validitas

- **Satu seed per run.** Selisih kecil (< 1 poin) tidak ditafsirkan sebagai perbedaan nyata.
- **Konfigurasi berbeda antar-model** (resolusi, augmentasi, batch efektif): disebut eksplisit di setiap kesimpulan.
- **Jumlah langkah optimizer berbeda** antar-model dan antar-proporsi karena epoch tetap.
- **Cakupan kelas pada subset kecil:** kelas yang hilang dari train 25% pasti ber-AP ≈ 0. Dilaporkan terpisah.
- **Test set sendiri kecil (≥ 40 foto)** dan hanya memuat sebagian kelas; mAP-nya dirata-rata atas kelas yang berbeda
  dari test asli, sehingga dibandingkan juga pada kelas yang sama-sama muncul.
- **Metrik framework vs evaluator bersama** berbeda; hanya evaluator bersama yang dipakai untuk perbandingan.

## 7. Rencana kerja

| Minggu | Kegiatan | Luaran |
|---|---|---|
| 1 (18–24 Nov) | Pemilihan model, notebook, uji pipeline sintetis, hipotesis | Notebook, logbook, laporan awal |
| 2 (25 Nov–1 Des) | Audit v5, lampiran kelas, smoke test, 9 training, evaluasi test asli; mulai ambil foto | `tabel_hasil_nrp.csv`, `grafik_nrp.png`, log training |
| 3 (2–8 Des) | Anotasi foto sendiri, EXIF, domain shift, 5 analisis kegagalan + eksperimen kecil | `domain_shift.csv`, studi kasus |
| 4 | Laporan akhir, deklarasi AI, latihan demo | Laporan, repo final |

Tenggat resmi belum diketahui; jadwal disesuaikan setelah dikonfirmasi ke dosen.

## 8. Tabel hasil (belum diukur)

**Eksperimen NRP — mask mAP@50–95 pada test asli**

| Model | 25% | 50% | 100% | Waktu GPU total | ms/gambar (100%) |
|---|---:|---:|---:|---:|---:|
| Mask R-CNN R50-FPN | — | — | — | — | — |
| YOLO26s-seg | — | — | — | — | — |
| RF-DETR-Seg Medium | — | — | — | — | — |

**Domain shift — checkpoint 100%**

| Model | Test asli | Test sendiri | Selisih (poin) | Kondisi sulit |
|---|---:|---:|---:|---:|
| Mask R-CNN R50-FPN | — | — | — | — |
| YOLO26s-seg | — | — | — | — |
| RF-DETR-Seg Medium | — | — | — | — |

Tanda "—" berarti belum diukur.

## 9. Deklarasi penggunaan AI (sementara)

AI (Claude) dipakai untuk: merangkum ketentuan tugas, memilih dan membandingkan model, menulis notebook (persiapan data,
training, evaluator, demo), menyusun draf hipotesis berdasarkan arsitektur, dan menyusun laporan awal ini. Setiap fakta
arsitektur yang dipakai di hipotesis dicek terhadap kode sumber paket (resolusi prototype YOLO26, resolusi dan
`__background__` RF-DETR, akumulasi gradien Ultralytics). Tiga kasus nyata kode AI yang salah beserta perbaikannya
tercatat di `LOGBOOK.md` (pemetaan kelas RF-DETR, deteksi lingkungan Kaggle, serialisasi JSON). Prompt utama akan
dilampirkan dari riwayat percakapan.

## 10. Referensi

1. He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. *ICCV*. https://arxiv.org/abs/1703.06870
2. Torchvision `maskrcnn_resnet50_fpn`. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html
3. Ultralytics YOLO26. https://docs.ultralytics.com/models/yolo26/
4. Ultralytics Instance Segmentation. https://docs.ultralytics.com/tasks/segment/
5. RF-DETR Segmentation. https://rfdetr.roboflow.com/latest/learn/run/segmentation/
6. Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. https://arxiv.org/abs/2304.07193
7. Panoptic Indoor Segmentation v5. https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5
