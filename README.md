# Paket akhir tugas segmentasi instance

Mulai dari [laporan akhir](laporan_akhir_segmentasi_instance.md). Buka laporan dengan folder `hasil/` di lokasi yang sama agar grafik dan gambar studi kasus tampil.

| Lokasi | Isi dan fungsi |
|---|---|
| [Laporan akhir](laporan_akhir_segmentasi_instance.md) | Naskah utama dengan latar belakang, alasan memilih model, metode, perbandingan, lima studi kegagalan, kesimpulan, dan deklarasi AI. |
| [Logbook](LOGBOOK.md), [laporan awal](laporan_awal.md), dan [audit dataset](laporan_audit_dataset_mandiri.md) | Hipotesis bertanggal, progres mingguan, dan pemeriksaan dataset mandiri. |
| [hasil/](hasil/) | Tabel eksperimen NRP, grafik PNG/PDF, AP per kelas, domain shift, rekap EXIF, waktu GPU, dan lima studi kasus. |
| [bukti_proses/](bukti_proses/) | Log dan metadata sembilan run, serta tempat menambahkan screenshot Roboflow. |
| [dataset mandiri/](dataset%20mandiri/hasil_coco_seluruh_foto/) | 36 foto evaluasi dalam JPEG, anotasi COCO, dan daftar 10 foto sulit. |
| [Notebook Kaggle](segmentasi_instance_kaggle.ipynb) | Kode training, evaluasi, dan demo. |
| [arsip_kaggle/](arsip_kaggle/) | Empat ZIP asli untuk bukti hasil dan pemulihan checkpoint. |

## Empat arsip dan kapan dipakai

- `hasil_t4_v5.zip` memuat hasil evaluasi Panoptic, grafik, dan log training asli.
- `hasil_test_pribadi_t4_v5.zip` memuat hasil evaluasi foto mandiri, prediksi dan overlay.
- `bobot_t4_v5.zip` memuat sembilan checkpoint. Untuk demo, muat checkpoint 100% dari tiap model.
- `hasil_smoke_t4.zip` memuat bukti uji awal T4.

## Checklist sebelum dikumpulkan

- [x] Laporan lengkap, tabel, grafik, lima contoh kegagalan, dan hasil uji ambang.
- [x] Logbook, waktu GPU, log training, EXIF, anotasi COCO, serta arsip bobot.
- [ ] Tambahkan screenshot anotasi ke `bukti_proses/screenshot_roboflow/` dan tautkan di laporan.
- [ ] Cocokkan deklarasi AI dengan proses yang kamu lakukan.
- [ ] Commit laporan, notebook, logbook, dan bukti yang sesuai ke [GitHub](https://github.com/seippppp/seippppp-segmentasi-instance-5054241032). Pilih file secara eksplisit. Simpan arsip besar, bobot, dan dataset pribadi di penyimpanan privat.
- [ ] Siapkan demo dengan foto baru, perubahan threshold, dan contoh AP per kelas. Ekspor laporan ke PDF jika format pengumpulan memintanya.

Paket ini mencakup foto indoor pribadi. Periksa izin akses sebelum membagikan dataset atau seluruh folder.
