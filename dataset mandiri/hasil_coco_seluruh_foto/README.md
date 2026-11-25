# Seluruh foto — COCO Instance Segmentation

Folder hasil lengkap untuk **36 dari 36 foto HEIC** pada folder sumber. Total **301 anotasi**. Gunakan `_annotations.coco.json` untuk seluruh foto, bukan file subset hard10.

## Kondisi sulit

Terdapat **31 foto bertanda sulit** berdasarkan penilaian visual (occlusion, objek terpotong/kecil, refleksi, backlight, atau pencahayaan rendah). Sepuluh contoh khusus untuk memenuhi permintaan dicatat dalam `metadata/hard10.csv` dan ditandai `selected_hard10` pada JSON utama. Foto lain yang juga sulit tetap ditandai sesuai kondisinya.

`_annotations.hard10.coco.json` adalah subset opsional dari dataset utama, bukan 10 foto tambahan. Tiga foto (IMG_5065, IMG_5136, IMG_5141) memakai brightness 50% melalui perkalian nilai RGB. Total dataset tetap 36 foto. Foto HEIC sumber tidak diubah.

## Isi

- `images/`: 36 JPG dataset dengan lebar 1600 piksel, rasio aspek dan orientasi dipertahankan.
- `_annotations.coco.json`: anotasi untuk seluruh 36 foto.
- `originals_jpg/`: JPG sebelum penggelapan pada ukuran keluaran, bukan HEIC resolusi asli.
- `previews/`: mask dan ID kelas per foto.
- `overview_01.jpg` hingga `overview_06.jpg`: pratinjau per kelompok 6 foto.
- `metadata/difficulty.csv`: alasan kondisi sulit untuk semua foto.
- `metadata/categories.json`: pemetaan kategori persis dari referensi lokal versi v1.
- `metadata/validation.json`: hasil validasi struktur, mask, ukuran, area, bbox dan brightness.
- `metadata/coverage.json`: pemeriksaan tidak ada foto sumber yang terlewat.
- `metadata/manual_polygons_800x600.json`: tracing manual untuk koreksi lebih lanjut.

## Status dan batasan anotasi

**Semua foto sudah memiliki anotasi, tetapi anotasi masih draft manual, belum ground truth final yang diverifikasi manusia.** Jumlah foto yang tercakup tidak berarti seluruh objek/detail di tiap foto sudah lengkap. Objek utama ditelusuri secara visual pada pratinjau 800×600. Tepi objek tipis, kaki kursi, celah rak, tanaman, pegangan tangga, APAR di balik kaca dan objek kecil perlu diperbaiki atau dilengkapi melalui review. Beberapa objek latar belum diberi mask. Kotak komputer, kabel, pakaian tersampir dan objek tanpa kelas cocok tidak dipaksakan menjadi kelas lain. Mask permukaan pada sebagian foto masih dapat mencakup detail kecil yang belum dipisahkan.

Monitor memakai `tv`, bangku memakai `chair`, boneka non-beruang memakai `toy`. Pemetaan semantik tersebut perlu dikonfirmasi terhadap kebijakan dataset referensi. Pantulan benda pada layar/kaca tidak diberi instance tambahan. Polygon bagian yang terlihat digunakan tanpa merekonstruksi bagian tersembunyi; draft tangga khususnya perlu review batas railing yang menutup anak tangga.

Kategori mengikuti file lokal `panoptic-indoor-segmentation.v1i.coco-segmentation/train/_annotations.coco.json`: 74 kelas ditambah kategori induk ID 0 yang tidak dipakai sebagai label. Sebelum pelaporan hasil model, selesaikan review ketelitian dan kelengkapan label. Validasi otomatis hanya menjamin struktur data, bukan ketepatan visual.

## Pemakaian

Image root adalah folder ini; `file_name` di JSON sudah memiliki awalan `images/`. Tidak dibuat split train/valid/test. Salinan normal dan gelap dari foto sama jangan dipisahkan ke split berbeda.

Untuk reproduksi, jalankan `python metadata/extend_all.py` di folder ini. Memerlukan HEIC sumber, dataset referensi lokal dan paket Pillow, pillow-heif, numpy, opencv-python, pycocotools. `extend_all.py` mengekspor seluruh foto; `build_dataset.py` sendiri hanya berisi definisi 10 foto awal.

Hasil lama `hasil_coco_10foto` tetap ada sebagai versi sebelumnya. Hasil yang memenuhi cakupan seluruh foto berada di folder **hasil_coco_seluruh_foto** ini.
