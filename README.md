# Segmentasi instance indoor di Kaggle T4

Seifeldein Akbar · NRP 5054241032. Eksperimen membandingkan Mask R-CNN R50-FPN, YOLO26s-seg, dan RF-DETR-Seg Medium dengan 25%, 50%, dan 100% data latih. Hipotesis dan riwayat proyek ada di [LOGBOOK.md](LOGBOOK.md).

## 1. Jalankan smoke test di Kaggle

1. Impor `segmentasi_instance_kaggle.ipynb` ke Kaggle. Pilih **GPU T4 x2** dan **Internet On**. Notebook memakai satu T4 untuk setiap run dan menghentikan tahap eksperimen jika GPU yang cocok tidak terdeteksi.
2. Tambahkan dataset **Panoptic Indoor Segmentation v5**, ekspor COCO Segmentation, lewat **Add Input**. Path yang disiapkan adalah `/kaggle/input/datasets/seifeldeinakbar/panoptic-indoor`. Jika Kaggle menampilkan path lain, salin path tersebut ke `DATA_SOURCE`.
3. Jalankan **Save Version → Save & Run All** dengan pengaturan bawaan. Notebook menjalankan instalasi, audit dan smoke test dua epoch per model pada 25% data. Periksa GPU pada `hasil/lingkungan.json`. Audit memeriksa 397/166/359 gambar, 74 kelas, dan kebocoran identik antarsplit. Pastikan sumber unggahan memang v5 dari Roboflow, karena nama folder ekspor dapat berakhiran v1.
4. Unduh `hasil_smoke_t4.zip` dari tab **Output**. Catat waktu, GPU, VRAM, dan masalah di logbook. Jika perlu mengubah batch atau jumlah epoch, putuskan setelah smoke test dan sebelum training utama. Gunakan epoch yang sama untuk seluruh eksperimen dan batch yang sama antarsubset untuk tiap model.

## 2. Jalankan sembilan eksperimen utama

Set `RUN_SMOKE = False` dan `RUN_TRAIN = True`. Biarkan `RUN_INSTALL` dan `RUN_PREPARE` aktif pada sesi baru. Biarkan `RUN_SUMMARY = False` saat training.

Untuk menghemat risiko sesi terputus, simpan satu versi **Save & Run All** per model. Isi `TRAIN_MODELS = ['yolo']`, lalu `['maskrcnn']`, lalu `['rfdetr']`. Gunakan `TRAIN_PERCENTS = [25, 50, 100]`. Jika satu model perlu dipecah lagi, pilih subset yang akan dikerjakan pada tiap versi. Catat ID versi Kaggle di logbook.

Setiap versi yang berhasil mengevaluasi run pada validation dan test asli, menyimpan checkpoint terpilih berdasarkan validation, dan membuat dua ZIP di **Output**. Pada sesi yang menghasilkan sebagian dari sembilan run, nama ZIP berakhiran `_parsial`. Simpan tab Output dari tiap versi karena folder `runs/` pada versi tersebut diperlukan untuk penggabungan.

## 3. Gabungkan hasil dan unduh

1. Buat versi rangkuman. Tambahkan output tiap versi training melalui **Add Input → Your Work → Notebooks**.
2. Set `RUN_SMOKE = False`, `RUN_TRAIN = False`, `RUN_PREPARE = False`, dan `RUN_SUMMARY = True`. Jalankan pada T4. Jika ada dua run berbeda untuk kombinasi yang sama, pilih sumber yang benar melalui `PREVIOUS_RUN_INPUTS`.
3. Unduh berkas berikut di **Output** setelah sembilan run terdeteksi.

| Berkas | Isi |
|---|---|
| `bobot_t4_v5.zip` | Sembilan checkpoint, satu per model dan proporsi, dengan konfigurasi dan metrik terkait. Ekstrak ke `segmentasi_kaggle/kerja/` untuk demo lokal. |
| `hasil_t4_v5.zip` | Tabel test asli dan validasi, grafik NRP PNG/PDF, AP per kelas pada 100%, prediksi, overlay, log training asli, catatan waktu GPU, dan catatan sumber dataset di metadata run. |
| `hasil_test_pribadi_t4_v5.zip` | Dibuat setelah evaluasi foto HP. Isinya mencakup tabel domain shift dan bukti evaluasi foto pribadi. |

Gunakan mask mAP@50–95 sebagai metrik utama. Laporkan juga mask mAP@50, mAP@75, AP menurut ukuran objek dan kelas, serta latensi median dalam ms/gambar. Angka laporan berasal dari evaluator COCO bersama. Catatan waktu GPU mengukur durasi run pada GPU termasuk pemuatan model dan validasi.

## 4. Lengkapi test pribadi dan bukti tugas

- Ambil minimal 40 foto ruangan dengan kamera HP sendiri, termasuk minimal 10 kondisi sulit. Simpan foto asli dan ekspor EXIF dengan `export_exif()`. Anotasi mask di Roboflow memakai nama kelas dataset v5 dan ambil tangkapan layar prosesnya.
- Tambahkan ekspor COCO foto pribadi dan output training ke Kaggle. Matikan smoke, training, summary dan prepare. Isi `OWN_TEST_DIR` dan `KONDISI_SULIT` di bagian 10, lalu aktifkan `RUN_OWN_TEST`. Bandingkan skornya dengan test asli untuk ketiga checkpoint 100%. Jelaskan perbedaan kelas yang terwakili dan kondisi pengambilan foto saat membahas domain shift.
- Pilih lima foto pribadi yang gagal dan gunakan `tampilkan_kasus()` serta `uji_variasi()`. Sertakan gambar, dugaan berdasarkan arsitektur, dan hasil uji kecil dalam laporan.
- Perbarui logbook mingguan dan commit bertahap. Pertahankan tanggal hipotesis asli dan catat apa yang dijalankan di laptop serta waktunya jika diketahui. Sertakan link repo, log asli, waktu GPU, prompt utama, dan satu contoh kode atau saran AI yang keliru beserta perbaikannya.
- Untuk demo lokal 10 menit, jalankan notebook dengan sakelar eksperimen dan `RUN_PREPARE` dimatikan. Ekstrak bobot ke `kerja/` dan isi `DEMO_FOTO` serta `DEMO_CONF`. Pakai `ap_kelas()` untuk menunjukkan AP kelas yang ditanya dan siapkan penjelasan untuk satu angka di tabel.

Notebook ini memakai satu T4 per run dan menyimpan metadata perangkat di setiap `run.json`. Gunakan hasil Kaggle untuk tabel dan grafik laporan. Simpan log run laptop sebagai catatan pengembangan.
