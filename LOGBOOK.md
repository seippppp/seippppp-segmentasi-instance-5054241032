# Logbook — Segmentasi Instance Indoor

**Seifeldein Akbar · NRP 5054241032**  
**Variabel NRP:** proporsi data latih 25%, 50%, dan 100%.

| Tanggal | Kegiatan | Detail / hasil |
|---|---|---|
| 18 Nov 2026 | Persiapan tugas | Memilih Mask R-CNN, YOLO26s-seg, dan RF-DETR-Seg serta menyiapkan kode awal. |
| 19 Nov 2026 | Penyiapan lingkungan | Mengatasi kendala instalasi hingga PyTorch dan CUDA berjalan. |
| 19 Nov 2026 | Penyusunan notebook | Menyiapkan notebook Kaggle dan evaluasi COCO mask. |
| 20 Nov 2026 | Uji alur | Memeriksa kode menggunakan 28 gambar sintetis. |
| 20 Nov 2026 | Penyusunan hipotesis | Menulis dugaan performa model, pengaruh jumlah data, dan kondisi foto sebelum training. |
| 21 Nov 2026 | Audit dataset | Memeriksa pembagian data, orientasi foto, anotasi, dan distribusi kelas Panoptic. |
| 21 Nov 2026 | Perbaikan kode | Memperbaiki pemetaan kelas RF-DETR dan evaluasi dengan bantuan AI. |
| 22–23 Nov 2026 | Training model | Menjalankan uji awal dan sembilan eksperimen, masing-masing 50 epoch. |
| 24 Nov 2026 | Audit foto mandiri | Memeriksa 36 foto dengan 301 mask pada 38 kelas, termasuk 10 foto sulit. |
| 24 Nov 2026 | Rekap hasil Panoptic | Merangkum metrik dan grafik; RF-DETR memperoleh skor tertinggi. |
| 24 Nov 2026 | Evaluasi foto mandiri | Menguji ketiga model pada foto mandiri; RF-DETR memperoleh skor tertinggi. |
| 25 Nov 2026 | Analisis kegagalan | Mengkaji lima kasus dengan membandingkan ambang confidence 0,25 dan 0,10. |
| 25 Nov 2026 | Laporan dan demo | Menyusun laporan akhir serta notebook visualisasi hasil. |
