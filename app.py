import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
@st.cache_data(show_spinner=False)
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
st.set_page_config(page_title="Prediksi Jumlah Kunjungan Wisatawan Nusantara Kabupaten Lamongan", layout="wide")
st.sidebar.markdown("## 🌟 Dashboard Prediksi Jumlah Kunjungan Wisatawan Nusantara")
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    try:
        st.image("kabupaten-lamongan-logo.png", width=120)
    except Exception:
        st.write("")

menu = st.sidebar.radio(
    "📌 PILIH MENU TAHAPAN:",
    [
        "🏠 Home",
        "📂 Upload Data",
        "🧹 Preprocessing",
        "📉 Dekomposisi CEEMDAN",
        "⚙️ Normalisasi",
        "✂️ Split Data",
        "📊 Modelling",
        "📈 Prediksi"
    ],
    index=0,
)
if menu != "🏠 Home":
    st.markdown(
        """
        <h1 style='text-align: center; color: #333333; font-size: 32px;'>
            📊 Dashboard Prediksi Jumlah Kunjungan Wisatawan Nusantara Kabupaten Lamongan
        </h1>
        """,
        unsafe_allow_html=True
    )
st.sidebar.info("🔍 Pilih tahap untuk menampilkan proses prediksi.")
if menu == "🏠 Home":
    st.markdown(
        """
        <style>
        /* HERO */
        .hero {
            background: linear-gradient(90deg, rgba(11,92,255,0.18), rgba(0,183,255,0.12));
            padding: 35px;
            border-radius: 14px;
            border: 1px solid rgba(11,92,255,0.25);
            margin-bottom: 22px;
            text-align: center;
        }

        .hero h1 {
            margin: 0;
            color: #053474;
            font-size: 35px;
            font-weight: 750;
            line-height: 1.25;
        }

        .hero p {
            margin: 10px 0 0 0;
            color: #1d1d1d;
            font-size: 17px;
            font-weight: 450;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
            <h1>🏠 SELAMAT DATANG DI APLIKASI PREDIKSI JUMLAH KUNJUNGAN WISATAWAN<br>KABUPATEN LAMONGAN</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📘 Tentang Aplikasi")
    st.write("""
    Aplikasi ini digunakan untuk melakukan prediksi jumlah kunjungan wisatawan nusantara 
    di Kabupaten Lamongan menggunakan metode *CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise)* 
    dan *ELM (Extreme Learning Machine)* yang dioptimasi dengan algoritma *PSO (Particle Swarm Optimization)*.
    """)
    st.markdown("### ⚙️ Fitur Utama:")
    st.markdown("""
        - **📂 Upload Data**  
      Mengunggah file CSV/Excelberisi data kunjungan wisata yang akan digunakan untuk analisis.  

    - **🧹 Preprocessing**  
      Menampilkan hasil *preprocessing* seperti mengecek *missing value*, mengecek jumlah data yang 0, penyesuaian format waktu dan imputasi median.  

    - **📊 Dekomposisi CEEMDAN**  
      Melakukan dekomposisi sinyal data menjadi beberapa komponen IMF dan residu menggunakan metode CEEMDAN.  

    - **⚙️ Normalisasi**  
      Melakukan normalisasi setiap komponen hasil dekomposisi agar siap digunakan dalam proses pelatihan model.  

    - **✂️ Split Data**  
      Memisahkan data menjadi data latih (*training set*) dan data uji (*testing set*).  

    - **🤖 Modelling**  
      Melatih model ELM standar dan ELM dengan optimasi PSO pada setiap komponen hasil dekomposisi.  

    - **📈 Prediksi**  
      Menampilkan hasil prediksi gabungan (rekonstruksi) untuk periode tertentu, seperti 1 bulan berikutnya.  
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("📚 Panduan singkat — langkah demi langkah", expanded=False):
        st.markdown(
            """
            <div class="howto-list">
            1. <strong>Upload Data</strong> — Pastikan kolom <code>no, bulan, tahun, jumlah</code>. Periksa preview dan koreksi langsung bila perlu.<br>
            2. <strong>Preprocessing</strong> — Muat artifact pickle preprocessing dari Colab agar data tersimpan di aplikasi.<br>
            3. <strong>CEEMDAN → Normalisasi → Split</strong> — Muat file pickle masing-masing untuk menampilkan IMF, scaler, dan hasil split.<br>
            4. <strong>Modelling</strong> — Muat file modelling untuk membandingkan ELM biasa dan ELM+PSO.<br>
            5. <strong>Prediksi</strong> — Tampilkan grafik prediksi yang digambar ulang dengan Matplotlib dan bisa diunduh PNG bila ingin laporan.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown("#### ℹ️ Catatan Penting")
        st.markdown(
            "- Pastikan semua file pickle yang dihasilkan di Google Colab diletakkan di folder aplikasi (sesuai path yang dipetakan di tiap menu).  \n"
            "- Nama kolom dan tipe data harus konsisten agar pipeline berjalan lancar.  \n"
            "- Bila grafik tidak muncul: periksa isi file pickle (pastikan `original_series`, `components`, atau `forecast_series` tersedia)."
        )
    with c2:
        st.markdown("#### ✉️ Kontak")
        st.markdown("Jika perlu bantuan, tulis ringkasan masalah dan kirimkan file pickle terkait ke: **devidwi1809@gmail.com**")
    st.markdown("")
    st.stop()

elif menu == "📂 Upload Data":
    st.markdown("---")
    st.markdown(
        """
        <style>
            .upload-title-center {
                text-align: left; 
                font-size: 26px; 
                font-weight: 700;
                color: #2b2b2b;
                margin-top: -10px;
                margin-bottom: 18px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .upload-title-center span.icon {
                font-size: 30px;
            }
        </style>

        <div class="upload-title-center">
            <span class="icon">📂</span>
            <span>Menu Upload Data</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    def to_excel_bytes(dataframe: pd.DataFrame) -> bytes:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            dataframe.to_excel(writer, index=False, sheet_name="DataWisata")
        buffer.seek(0)
        return buffer.getvalue()

    def safe_save_backup(df):
        st.session_state["_df_backup"] = df.copy()

    def restore_backup():
        if "_df_backup" in st.session_state:
            st.session_state.df = st.session_state["_df_backup"].copy()
            st.success("🔁 Perubahan dibatalkan — data telah dikembalikan dari backup.")
        else:
            st.warning("❗ Tidak ada backup tersedia untuk dikembalikan.")

    uploaded_file = st.file_uploader(
        "Silakan Upload File CSV atau Excel",
        type=["csv", "xlsx", "xls"],
        help="Pastikan kolom: no, bulan, tahun, jumlah."
    )

    col_reset, col_undo, _ = st.columns([1,1,3])
    with col_reset:
        if st.button("🔄 Reset Data di Memori", use_container_width=True):
            if "df" in st.session_state:
                del st.session_state["df"]
            st.success("Data di memori aplikasi berhasil di-reset. Silakan unggah ulang file.")
    with col_undo:
        if st.button("↩️ Undo Perubahan Terakhir", use_container_width=True):
            restore_backup()

    if uploaded_file is not None and "df" not in st.session_state:
        try:
            name = uploaded_file.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            df.columns = df.columns.str.strip().str.lower()

            required_cols = ["no", "bulan", "tahun", "jumlah"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                st.error(f"❌ Format kolom tidak sesuai. Kolom yang hilang: {', '.join(missing_cols)}. Pastikan ada kolom: no, bulan, tahun, jumlah.")
            else:

                df["no"] = pd.to_numeric(df["no"], errors="coerce").astype("Int64")
                df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce").astype("Int64")
                df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce")

                safe_save_backup(df)
                st.session_state.df = df.copy()
                st.success("✅ Data berhasil dimuat dan disimpan ke memori aplikasi.")
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")

    if "df" not in st.session_state:
        st.info("📥 Silakan unggah file terlebih dahulu untuk mulai mengelola data.")
        st.stop()

    df = st.session_state.df.copy()

    st.markdown("### 🔎 Ringkasan Dataset")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.markdown('<div class="metric-card"><div style="font-size:1.05rem;font-weight:600">{}</div><div class="small-muted">Jumlah Baris</div></div>'.format(df.shape[0]), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div style="font-size:1.05rem;font-weight:600">{}</div><div class="small-muted">Jumlah Kolom</div></div>'.format(df.shape[1]), unsafe_allow_html=True)
    with c3:
        if "tahun" in df.columns and df["tahun"].notna().any():
            try:
                tahun_min = int(df["tahun"].min())
                tahun_max = int(df["tahun"].max())
                rentang = f"{tahun_min} – {tahun_max}"
            except Exception:
                rentang = "-"
        else:
            rentang = "-"
        st.markdown('<div class="metric-card"><div style="font-size:1.05rem;font-weight:600">{}</div><div class="small-muted">Rentang Tahun</div></div>'.format(rentang), unsafe_allow_html=True)

    with st.expander("ℹ️ Tipe Data Kolom", expanded=False):
        st.code(df.dtypes.to_string())

    st.markdown("### 👀 Preview 5 Baris Pertama")
    st.dataframe(df.head(), use_container_width=True, height=200)

    csv_preview = df.head().to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Preview (CSV)", data=csv_preview, file_name="preview_data.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### 📌 Data Terakhir yang Tersimpan")
    if df.shape[0] == 0:
        st.warning("Data kosong — belum ada baris tersimpan.")
        last_no, last_bulan, last_tahun, last_jumlah = 0, "", datetime.now().year, 0
    else:
        last_row = df.tail(1).iloc[0]
        last_no = int(last_row.get("no", df["no"].dropna().max() or 0)) if not pd.isna(last_row.get("no")) else int(df["no"].dropna().max() or 0)
        last_bulan = str(last_row.get("bulan", "")) if not pd.isna(last_row.get("bulan")) else ""
        last_tahun = int(last_row.get("tahun", datetime.now().year)) if not pd.isna(last_row.get("tahun")) else datetime.now().year
        last_jumlah = int(last_row.get("jumlah", 0)) if not pd.isna(last_row.get("jumlah")) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nomor Terakhir", last_no)
    c2.metric("Bulan Terakhir", last_bulan if last_bulan else "-")
    c3.metric("Tahun Terakhir", last_tahun)
    c4.metric("Jumlah Wisatawan", f"{last_jumlah:,}".replace(",", "."))

    if last_bulan:
        st.info(f"📅 Data terakhir: **{last_bulan} {last_tahun}** — **{last_jumlah:,}** wisatawan.".replace(",", "."))
    else:
        st.info(f"📅 Data terakhir pada tahun **{last_tahun}** — **{last_jumlah:,}** wisatawan.".replace(",", "."))

    st.markdown("---")

    st.markdown("### ➕ Tambah Data Baru (Manual)")
    if df.shape[0] > 0:
        next_no_default = int(df["no"].dropna().max() or 0) + 1
        default_year = int(df["tahun"].dropna().max() or datetime.now().year)
    else:
        next_no_default = 1
        default_year = datetime.now().year

    with st.form("form_tambah_data", clear_on_submit=True):
        col_a, col_b, col_c = st.columns([2,1,1])
        with col_a:
            bulan_baru = st.text_input("Bulan (nama)", placeholder="Contoh: Januari")
        with col_b:
            tahun_baru = st.number_input("Tahun", min_value=1900, max_value=9999, step=1, value=default_year)
        with col_c:
            jumlah_baru = st.number_input("Jumlah Wisatawan", min_value=0, step=1, value=0)

        submitted = st.form_submit_button("✅ Tambah ke Tabel")

    if submitted:
        if not bulan_baru.strip():
            st.warning("⚠️ Nama bulan tidak boleh kosong.")
        else:
            safe_save_backup(df)
            next_no = next_no_default
            new_row = pd.DataFrame({
                "no": [next_no],
                "bulan": [bulan_baru.strip()],
                "tahun": [int(tahun_baru)],
                "jumlah": [int(jumlah_baru)]
            })
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True).reset_index(drop=True)
            df = st.session_state.df.copy()
            st.success(f"Data baru untuk **{bulan_baru.strip()} {tahun_baru}** berhasil ditambahkan.")

    st.markdown("---")
    st.markdown("### 📝 Edit & Hapus Data")
    st.caption("Klik sel untuk mengedit. Untuk menghapus, hapus isi baris atau gunakan opsi di bawah. Setelah selesai, perubahan otomatis tersimpan ke memori aplikasi.")
    safe_save_backup(df)

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_upload",
    )

    if not edited_df.equals(st.session_state.df):
        expected = set(["no", "bulan", "tahun", "jumlah"])
        if not expected.issubset(set(edited_df.columns)):
            st.error("⚠️ Kolom wajib hilang setelah edit. Perubahan tidak disimpan.")
            restore_backup()
        else:
            try:
                edited_df["no"] = pd.to_numeric(edited_df["no"], errors="coerce").astype("Int64")
                edited_df["tahun"] = pd.to_numeric(edited_df["tahun"], errors="coerce").astype("Int64")
                edited_df["jumlah"] = pd.to_numeric(edited_df["jumlah"], errors="coerce")
                st.session_state.df = edited_df.copy()
                df = st.session_state.df.copy()
                st.success("✅ Perubahan pada tabel berhasil disimpan ke memori aplikasi.")
            except Exception as e:
                st.error(f"⚠️ Gagal menyimpan perubahan: {e}")
                restore_backup()

    st.markdown("---")
    st.markdown("### ❌ Hapus Baris Tertentu")
    st.caption("Masukkan nomor baris (kolom 'no') yang ingin dihapus, lalu tekan Hapus.")
    col_del1, col_del2 = st.columns([2,1])
    with col_del1:
        hapus_no = st.number_input("No (hapus berdasarkan kolom 'no')", min_value=0, step=1, value=0)
    with col_del2:
        if st.button("Hapus Baris"):
            if hapus_no and (hapus_no in df["no"].tolist()):
                safe_save_backup(df)
                st.session_state.df = df[df["no"] != hapus_no].reset_index(drop=True)
                df = st.session_state.df.copy()
                st.success(f"Baris dengan no = {hapus_no} berhasil dihapus.")
            else:
                st.warning("No yang dimasukkan tidak ditemukan di tabel.")

    st.markdown("---")
    st.markdown("### 💾 Unduh Data yang Sudah Diperbarui")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    excel_bytes = to_excel_bytes(df)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇️ Download sebagai CSV", data=csv_bytes, file_name="data_wisata_diperbarui.csv", mime="text/csv", use_container_width=True)
    with col_dl2:
        st.download_button("⬇️ Download sebagai Excel", data=excel_bytes, file_name="data_wisata_diperbarui.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Preview Data Saat Ini")
    st.dataframe(df, use_container_width=True, height=350)

elif menu == "🧹 Preprocessing":
    st.markdown("---")
    st.markdown(
        """
        <style>
            .upload-title-center {
                text-align: left; 
                font-size: 26px;
                font-weight: 700;
                color: #2b2b2b;
                margin-top: -10px;
                margin-bottom: 18px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .upload-title-center span.icon {
                font-size: 30px;
            }
        </style>

        <div class="upload-title-center">
            <span class="icon">🧹</span>
            <span>Menu Preprocessing</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    WISATA_PICKLE_MAP = {
        "Wisata Brumbun":              "wisata_brumbun/preprocessing_brumbun.pkl",
        "Wisata Goa Maharani":         "goa_maharani/preprocessing_goa_maharani.pkl",
        "Wisata Makam Sendang Duwur":  "makam_sendang_duwur/preprocessing_makam_sendang_duwur.pkl",
        "Wisata Makam Sunan Drajat":   "makam_sunan_drajat/preprocessing_makam_sunan_drajat.pkl",
        "Wisata Monumen Van Der Wijk": "monumen_van_der_wijk/preprocessing_monumen_van_der_wijk.pkl",
        "Wisata Museum Sunan Drajat":  "museum_sunan_drajat/preprocessing_museum_sunan_drajat.pkl",
        "Wisata Waduk Gondang":        "waduk_gondang/preprocessing_waduk_gondang.pkl",
        "Wisata Bahari Lamongan":      "wisata_bahari_lamongan/preprocessing_wbl.pkl",
    }

    wisata_choice = st.selectbox(
        "Pilihlah Objek Wisata:",
        list(WISATA_PICKLE_MAP.keys()),
        index=0,
    )
    pickle_path = WISATA_PICKLE_MAP[wisata_choice]

    if not os.path.exists(pickle_path):
        st.error(
            f"❌ File pickle untuk **{wisata_choice}** tidak ditemukan.\n\n"
            f"Pastikan file bernama **`{pickle_path}`** berada di folder yang sama dengan aplikasi Streamlit."
        )
        st.stop()

    try:
        preproc_artifact = load_pickle(pickle_path)
    except Exception as e:
        st.error(f"❌ Gagal membaca file pickle `{pickle_path}`: {e}")
        st.stop()

    df = preproc_artifact.get("df_preprocessed")
    median_jumlah = preproc_artifact.get("median_jumlah", None)
    jumlah_col = preproc_artifact.get("jumlah_col", "jumlah")
    missing_info = preproc_artifact.get("missing_info", None)
    total_missing = preproc_artifact.get("total_missing", None)
    n_zero = preproc_artifact.get("n_zero", None)
    baris_imputasi = preproc_artifact.get("baris_imputasi", None)
    n_before = preproc_artifact.get("n_before", None)
    n_after = preproc_artifact.get("n_after", None)
    n_dup = preproc_artifact.get("n_dup", None)

    if df is None:
        st.error("❌ `df_preprocessed` tidak ditemukan di dalam pickle. Cek kembali isi file preprocessing di Colab.")
        st.stop()

    try:
        df = df.copy()
        df.columns = [c.strip().lower() if isinstance(c, str) else c for c in df.columns]
        if "jumlah" not in df.columns:
            if jumlah_col and jumlah_col in df.columns:
                df = df.rename(columns={jumlah_col: "jumlah"})
            else:
                st.error("❌ Kolom target 'jumlah' tidak ditemukan di data hasil preprocessing.")
                st.stop()

        if "tahun" not in df.columns:
            possible = [c for c in df.columns if "tahun" in c]
            if possible:
                df = df.rename(columns={possible[0]: "tahun"})
            else:
                st.error("❌ Kolom 'tahun' tidak ditemukan di data hasil preprocessing.")
                st.stop()

        df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce")
        df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce").astype("Int64")
    except Exception as e:
        st.error(f"❌ Gagal memproses dataframe hasil preprocessing: {e}")
        st.stop()
    try:
        st.session_state["df"] = df.copy()
        st.info("ℹ️ Data hasil preprocessing telah disimpan ke `st.session_state['df']` dan siap dipakai di menu Prediksi.")
    except Exception as e:
        st.warning(f"⚠️ Gagal menyimpan ke session_state: {e}")

    st.markdown("### 📊 Ringkasan Data – " + wisata_choice)
    with st.container():
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.markdown('<div class="metric-card"><div style="font-size:1.05rem;font-weight:600">{}</div><div class="small-muted">Jumlah Baris (akhir)</div></div>'.format(df.shape[0]), unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card"><div style="font-size:1.05rem;font-weight:600">{}</div><div class="small-muted">Jumlah Kolom</div></div>'.format(df.shape[1]), unsafe_allow_html=True)
        with c3:
            if "tahun" in df.columns:
                try:
                    tahun_min = int(df["tahun"].min())
                    tahun_max = int(df["tahun"].max())
                    rentang = f"{tahun_min}–{tahun_max}"
                except Exception:
                    rentang = "-"
            else:
                rentang = "-"
            st.markdown('<div class="metric-card"><div style="font-size:1.05rem;font-weight:600">{}</div><div class="small-muted">Rentang Tahun</div></div>'.format(rentang), unsafe_allow_html=True)

    cols_info = st.columns(2)
    with cols_info[0]:
        if (n_before is not None) and (n_after is not None) and (n_dup is not None):
            st.info(
                f"🧾 **Duplikat data**  \n"
                f"- Sebelum: **{n_before}**  \n"
                f"- Setelah: **{n_after}**  \n"
                f"- Dihapus: **{n_dup}**"
            )
        else:
            st.write("")
    with cols_info[1]:
        if median_jumlah is not None:
            teks_median = f"📌 Median untuk imputasi kolom **`{jumlah_col}`**: **{median_jumlah:,.0f}**".replace(",", ".")
            if n_zero is not None:
                teks_median += f"  \nJumlah baris bernilai 0 yang diimputasi: **{n_zero}**"
            st.success(teks_median)
        else:
            st.write("")

    with st.expander("🔎 Detail Missing Value & Tipe Kolom", expanded=False):
        if missing_info is not None:
            if total_missing is not None:
                st.caption(f"Total missing value di seluruh kolom: **{total_missing}**.")
            try:
                st.dataframe(missing_info.to_frame("Jumlah Missing"), use_container_width=True, height=220)
            except Exception:
                st.write(missing_info)
        else:
            st.info("Tidak ada informasi missing value yang disimpan di artifact.")

        st.markdown("**Tipe data kolom**")
        st.code(df.dtypes.to_string())

    st.markdown("### 👀 Preview Data (5 baris pertama)")
    st.dataframe(df.head(), use_container_width=True, height=200)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh CSV Preview", data=csv_bytes, file_name=f"{wisata_choice.replace(' ','_')}_preview.csv", mime="text/csv")

    if isinstance(baris_imputasi, pd.DataFrame) and not baris_imputasi.empty:
        with st.expander("🧮 Lihat Baris yang Diimputasi Median (Sebelumnya 0)", expanded=False):
            st.dataframe(baris_imputasi, use_container_width=True, height=300)
            csv_imp = baris_imputasi.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Unduh Baris Imputasi (CSV)", data=csv_imp, file_name=f"{wisata_choice.replace(' ','_')}_imputasi.csv", mime="text/csv")

    bulan_col = None
    tahun_col = None
    for c in df.columns:
        if str(c).lower() == "bulan":
            bulan_col = c
        if str(c).lower() == "tahun":
            tahun_col = c

    if bulan_col is None or tahun_col is None:
        st.error("❌ Kolom 'bulan' atau 'tahun' tidak ditemukan di data hasil preprocessing.")
        st.stop()

    if "bulan_num" not in df.columns or "tanggal" not in df.columns:
        bulan_mapping = {
            'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
            'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
            'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
        }
        df["bulan_num"] = df[bulan_col].map(bulan_mapping)
        df["tanggal"] = pd.to_datetime(dict(
            year=df[tahun_col].astype(int),
            month=df["bulan_num"].astype(int),
            day=1
        ))

    df = df.sort_values("tanggal").reset_index(drop=True)

    st.markdown("### 📈 Visualisasi Jumlah Wisatawan per Bulan")
    try:
        fig, ax = plt.subplots(figsize=(12,4))

        ax.plot(df["tanggal"], df["jumlah"], marker="o", linewidth=2)
        ax.set_title("Jumlah Kunjungan Wisatawan", fontsize=13, fontweight="bold")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Jumlah Wisatawan")

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", "."))
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=45)
        ax.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Gagal membuat grafik interaktif: {e}")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["tanggal"], df["jumlah"], marker="o", linewidth=2)
        ax.set_title(f"Jumlah Kunjungan Wisatawan\n{wisata_choice}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Bulan", fontsize=12)
        ax.set_ylabel("Jumlah Wisatawan", fontsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, loc: f"{int(x):,}".replace(",", ".")))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)
    st.success("✅ Hasil lengkap preprocessing berhasil ditampilkan.")

elif menu == "📉 Dekomposisi CEEMDAN":
    from PyEMD import CEEMDAN
    st.markdown("---")
    st.markdown(
        """
        <style>
            .upload-title-center {
                text-align: left; 
                font-size: 26px;
                font-weight: 700;
                color: #2b2b2b;
                margin-top: -10px;
                margin-bottom: 18px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .upload-title-center span.icon {
                font-size: 30px;
            }
        </style>

        <div class="upload-title-center">
            <span class="icon">📉</span>
            <span>Menu Dekomposisi CEEMDAN</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    wisata_options = {
        "Wisata Brumbun": "wisata_brumbun/ceemdan_brumbun.pkl",
        "Wisata Goa Maharani": "goa_maharani/ceemdan_goa_maharani.pkl",
        "Wisata Makam Sendang Duwur": "makam_sendang_duwur/ceemdan_sendang_duwur.pkl",
        "Wisata Makam Sunan Drajat": "makam_sunan_drajat/ceemdan_sunan_drajat.pkl",
        "Wisata Monumen Van Der Wijk": "monumen_van_der_wijk/ceemdan_vdw.pkl",
        "Wisata Museum Sunan Drajat": "museum_sunan_drajat/ceemdan_museum_sunan.pkl",
        "Wisata Waduk Gondang": "waduk_gondang/ceemdan_gondang.pkl",
        "Wisata Bahari Lamongan": "wisata_bahari_lamongan/ceemdan_wbl.pkl"
    }

    wisata_choice = st.selectbox("🗺 Silahkan Pilih Data Wisata :", list(wisata_options.keys()))
    pickle_path = wisata_options[wisata_choice]

    if not os.path.exists(pickle_path):
        st.error(f"❌ File `{pickle_path}` tidak ditemukan. Pastikan file pickle CEEMDAN ada di folder aplikasi.")
        st.stop()

    try:
        ceemdan_artifact = load_pickle(pickle_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat `{pickle_path}`: {e}")
        st.stop()

    components = ceemdan_artifact.get("components", {})
    original_series = np.asarray(ceemdan_artifact.get("original_series", []), dtype=float)
    jumlah_col = ceemdan_artifact.get("jumlah_col", "jumlah")
    seed_used = ceemdan_artifact.get("seed", None)
    imf_descriptions = ceemdan_artifact.get("imf_descriptions", {}) or {}
    imf_energy = ceemdan_artifact.get("imf_energy", {}) or {}

    if len(original_series) == 0 or not components:
        st.error("❌ Artifact CEEMDAN tidak berisi 'original_series' atau 'components'. Periksa hasil Colab.")
        st.stop()

    n_imf = len([k for k in components.keys() if k.lower() != "residual"])
    series_length = len(original_series)

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        st.markdown('<div class="metric-card"><div style="font-size:1.02rem;font-weight:600">{}</div><div class="small-muted">Wisata</div></div>'.format(wisata_choice), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div style="font-size:1.02rem;font-weight:600">{}</div><div class="small-muted">Jumlah IMF</div></div>'.format(n_imf), unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div style="font-size:1.02rem;font-weight:600">{}</div><div class="small-muted">Panjang Sinyal</div></div>'.format(series_length), unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div style="font-size:1.02rem;font-weight:600">{}</div><div class="small-muted">Seed CEEMDAN</div></div>'.format(seed_used if seed_used is not None else "-"), unsafe_allow_html=True)

    if imf_energy:
        st.markdown("### ⚡ Ringkasan Perkiraan Energi per Komponen")
        df_energy = pd.DataFrame(
            [{"Komponen": k, "Energi (%)": imf_energy.get(k, np.nan)} for k in components.keys()]
        ).sort_values("Komponen")
        st.dataframe(df_energy.style.format({"Energi (%)":"{:.2f}"}), use_container_width=True)

    st.markdown("---")

    all_components = list(components.keys())
    def sort_key(name):
        if name.lower() == "residual":
            return 999
        try:
            return int("".join(filter(str.isdigit, name)) or 0)
        except Exception:
            return 0
    all_components = sorted(all_components, key=sort_key)

    with st.expander("🔧 Opsi Tampilan Komponen (Pilih komponen untuk ditampilkan)", expanded=True):
        default_selection = [c for c in all_components if c.lower() != "residual"][:min(6, len(all_components))]
        sel_components = st.multiselect("Pilih komponen (IMF) yang ingin divisualisasikan:", all_components, default=default_selection)

        col_download1, col_download2 = st.columns([1,1])
        comp_df = pd.DataFrame({name: components[name] for name in all_components})
        csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
        excel_buf = BytesIO()
        with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
            comp_df.to_excel(writer, index=False, sheet_name="components")
        excel_buf.seek(0)
        excel_bytes = excel_buf.read()

        with col_download1:
            st.download_button("⬇️ Unduh Komponen (CSV)", data=csv_bytes, file_name=f"{wisata_choice.replace(' ','_')}_components.csv", mime="text/csv", use_container_width=True)
        with col_download2:
            st.download_button("⬇️ Unduh Komponen (Excel)", data=excel_bytes, file_name=f"{wisata_choice.replace(' ','_')}_components.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    st.markdown("---")

    st.markdown("### 📈 Sinyal Asli ")
    try:
        df_orig = pd.DataFrame({"value": original_series})
        fig, ax = plt.subplots(figsize=(12,4))

        ax.plot(df_orig.index, df_orig["value"], linewidth=2)
        ax.set_title(f"Sinyal Asli — {jumlah_col} — {wisata_choice}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Index Waktu")
        ax.set_ylabel("Jumlah Wisatawan")

        max_idx = int(df_orig["value"].idxmax())
        max_val = df_orig.loc[max_idx, "value"]

        ax.scatter(max_idx, max_val, s=80, zorder=3)
        ax.text(
            max_idx, max_val,
            f"{int(max_val):,}".replace(",", "."),
            ha="center", va="bottom", fontsize=9
        )

        ax.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Gagal membuat grafik sinyal asli: {e}")

    st.markdown("---")

    st.markdown("### 📉 Visualisasi IMF yang Dipilih")
    if not sel_components:
        st.warning("Pilih minimal 1 komponen IMF pada opsi tampilan di atas.")
    else:
        try:
            n_sel = len(sel_components)
            max_plots = min(n_sel, 12)
            rows = max_plots
            n = len(sel_components)
            fig, axes = plt.subplots(n, 1, figsize=(12, 2.5*n), sharex=True)

            if n == 1:
                axes = [axes]

            for ax, name in zip(axes, sel_components):
                y = components[name]
                ax.plot(y, linewidth=1.5)
                ax.set_title(name, fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.4)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Gagal membuat subplot IMF: {e}")

    st.markdown("---")

    st.markdown("### 🔍 Nilai & Penjelasan Komponen")
    for name in all_components:
        with st.expander(f"{name} — lihat penjelasan", expanded=False):
            arr = np.asarray(components[name], dtype=float)
            preview_vals = np.round(arr[:8], 3).tolist()
            st.write(f"Contoh 8 nilai pertama: `{preview_vals}`")
            desc = imf_descriptions.get(name, None)
            if desc:
                st.caption(desc)
            else:
                if name.lower() == "residual":
                    st.caption("Residual: komponen yang mewakili trend atau sisa setelah semua IMF diambil.")
                else:
                    st.caption("IMF: komponen osilasi pada frekuensi tertentu. IMF dengan indeks kecil biasanya frekuensi lebih tinggi.")

    st.success("🎉 Dekomposisi CEEMDAN beserta penjelasan tiap komponen berhasil ditampilkan!")

elif menu == "⚙️ Normalisasi":
    from sklearn.preprocessing import MinMaxScaler
    st.markdown("---")
    st.markdown(
        """
        <style>
            .upload-title-center {
                text-align: left; 
                font-size: 26px;
                font-weight: 700;
                color: #2b2b2b;
                margin-top: -10px;
                margin-bottom: 18px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .upload-title-center span.icon {
                font-size: 30px;
            }
        </style>

        <div class="upload-title-center">
            <span class="icon">⚙️</span>
            <span>Menu Normalisasi</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    normalisasi_files = {
        "Wisata Brumbun": "wisata_brumbun/normalisasi_brumbun.pkl",
        "Wisata Goa Maharani": "goa_maharani/normalisasi_goa_maharani.pkl",
        "Wisata Makam Sendang Duwur": "makam_sendang_duwur/normalisasi_sendang_duwur.pkl",
        "Wisata Makam Sunan Drajat": "makam_sunan_drajat/normalisasi_sunan_drajat.pkl",
        "Wisata Monumen Van Der Wijk": "monumen_van_der_wijk/normalisasi_vdw.pkl",
        "Wisata Museum Sunan Drajat": "museum_sunan_drajat/normalisasi_museum_sunan.pkl",
        "Wisata Waduk Gondang": "waduk_gondang/normalisasi_gondang.pkl",
        "Wisata Bahari Lamongan": "wisata_bahari_lamongan/normalisasi_wbl.pkl",
    }

    wisata_choice = st.selectbox("🗺 Silahkan Pilih Data Wisata:", list(normalisasi_files.keys()))
    pickle_path = normalisasi_files[wisata_choice]

    if not os.path.exists(pickle_path):
        st.error(f"❌ File `{pickle_path}` tidak ditemukan. Pastikan file pickle normalisasi berada di folder aplikasi.")
        st.stop()

    try:
        norm_artifact = load_pickle(pickle_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat file normalisasi `{pickle_path}`: {e}")
        st.stop()

    norm_components = norm_artifact.get("normalized_components", {}) or {}
    scalers = norm_artifact.get("scalers", {}) or {}
    original_series = np.asarray(norm_artifact.get("original_series", []), dtype=float)
    jumlah_col = norm_artifact.get("jumlah_col", "jumlah")

    if not norm_components:
        st.error("❌ Artifact normalisasi tidak berisi 'normalized_components'. Periksa hasil Colab.")
        st.stop()

    comp_names = list(norm_components.keys())
    n_comp = len(comp_names)
    len_series = len(original_series)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.markdown('<div class="metric-card"><div style="font-size:1.02rem;font-weight:600">{}</div><div class="small-muted">Komponen Tersedia</div></div>'.format(n_comp), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div style="font-size:1.02rem;font-weight:600">{}</div><div class="small-muted">Panjang Sinyal</div></div>'.format(len_series), unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div style="font-size:1.02rem;font-weight:600">{}</div><div class="small-muted">Kolom Target</div></div>'.format(jumlah_col), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔎 Pilih Komponen untuk Dibandingkan")
    sel_components = st.multiselect("Pilih 1 atau lebih komponen (IMF / residual):", comp_names, default=comp_names[:min(4, len(comp_names))])

    comp_df = pd.DataFrame({name: np.asarray(norm_components[name]).flatten() for name in comp_names})
    csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
    excel_buf = BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        comp_df.to_excel(writer, index=False, sheet_name="normalized_components")
    excel_buf.seek(0)
    excel_bytes = excel_buf.read()

    dl_col1, dl_col2 = st.columns([1,1])
    with dl_col1:
        st.download_button("⬇️ Unduh Semua Komponen (CSV)", data=csv_bytes, file_name=f"{wisata_choice.replace(' ','_')}_normalized_components.csv", mime="text/csv", use_container_width=True)
    with dl_col2:
        st.download_button("⬇️ Unduh Semua Komponen (Excel)", data=excel_bytes, file_name=f"{wisata_choice.replace(' ','_')}_normalized_components.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔢 Preview Nilai & Statistik Singkat")
    preview_rows = []
    for name in comp_names:
        arr = np.asarray(norm_components[name]).flatten()
        preview_rows.append({
            "Komponen": name,
            "Contoh (5 nilai pertama)": ", ".join([f"{v:.3f}" for v in arr[:5]]),
            "Min": np.min(arr),
            "Max": np.max(arr),
            "Mean": np.mean(arr),
            "Std": np.std(arr)
        })
    df_preview = pd.DataFrame(preview_rows)
    st.dataframe(df_preview[["Komponen", "Contoh (5 nilai pertama)", "Min", "Max", "Mean", "Std"]], use_container_width=True, height=260)

    st.markdown("---")

    if not sel_components:
        st.warning("Pilih minimal 1 komponen untuk divisualisasikan.")
    else:
        for comp in sel_components:
            st.markdown(f"#### 🔸 {comp}")

            arr_norm = np.asarray(norm_components[comp]).reshape(-1, 1)
            scaler = scalers.get(comp, None)

            arr_orig = None
            if scaler is not None:
                try:
                    arr_orig = scaler.inverse_transform(arr_norm).flatten()
                except Exception:
                    arr_orig = None

            df_plot = pd.DataFrame({
                "index": np.arange(len(arr_norm)),
                "normalized": arr_norm.flatten()
            })
            if arr_orig is not None:
                df_plot["inverse"] = arr_orig

            fig, ax = plt.subplots(figsize=(12,4))

            ax.plot(
                df_plot["index"],
                df_plot["normalized"],
                marker="o",
                linewidth=2,
                label=f"{comp} (Normalized)"
            )

            if "inverse" in df_plot:
                ax.plot(
                    df_plot["index"],
                    df_plot["inverse"],
                    linestyle="--",
                    linewidth=2,
                    label=f"{comp} (Inverse → Skala Asli)"
                )

            ax.set_title(f"{comp} — Normalized vs Inverse", fontsize=11)
            ax.set_xlabel("Index")
            ax.set_ylabel("Nilai")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)

            st.pyplot(fig)
            plt.close(fig)


            if arr_orig is not None:
                try:
                    st.markdown(
                        f"- Skala asli (inverse) → Min: **{arr_orig.min():,.0f}**, Max: **{arr_orig.max():,.0f}**, Mean: **{arr_orig.mean():,.1f}**".replace(",", ".")
                    )
                except Exception:
                    st.write("Ringkasan skala asli tidak tersedia (format numeric tidak sesuai).")

            if scaler is not None:
                with st.expander("ℹ️ Lihat parameter scaler (MinMaxScaler)"):
                    try:
                        data_min = getattr(scaler, "data_min_", None)
                        data_max = getattr(scaler, "data_max_", None)
                        feature_range = getattr(scaler, "feature_range", None)
                        st.write(f"- feature_range: {feature_range}")
                        if data_min is not None and data_max is not None:
                            st.write(f"- data_min_: {np.round(data_min, 6).tolist()}")
                            st.write(f"- data_max_: {np.round(data_max, 6).tolist()}")
                        else:
                            st.write("Informasi `data_min_` / `data_max_` tidak tersedia pada scaler.")
                    except Exception as e:
                        st.write(f"Gagal menampilkan parameter scaler: {e}")
            else:
                st.info("⚠️ Scaler untuk komponen ini tidak ditemukan di artifact. Hanya tampilan normalisasi yang tersedia.")
            st.markdown("---")
    st.success("✅ Normalisasi komponen berhasil ditampilkan.")

elif menu == "✂️ Split Data":
    st.markdown("---")
    st.markdown(
        """
        <style>
            .upload-title-center {
                text-align: left; 
                font-size: 26px;
                font-weight: 700;
                color: #2b2b2b;
                margin-top: -10px;
                margin-bottom: 18px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .upload-title-center span.icon {
                font-size: 30px;
            }
        </style>

        <div class="upload-title-center">
            <span class="icon">✂️</span>
            <span>Menu Split Data</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    split_files = {
        "Wisata Brumbun": {
            "80_20": "wisata_brumbun/split_brumbun_80_20.pkl",
            "90_10": "wisata_brumbun/split_brumbun_90_10.pkl",
        },
        "Wisata Goa Maharani": {
            "80_20": "goa_maharani/split_goa_maharani_80_20.pkl",
            "90_10": "goa_maharani/split_goa_maharani_90_10.pkl",
        },
        "Wisata Makam Sendang Duwur": {
            "80_20": "makam_sendang_duwur/split_sendang_duwur_80_20.pkl",
            "90_10": "makam_sendang_duwur/split_sendang_duwur_90_10.pkl",
        },
        "Wisata Makam Sunan Drajat": {
            "80_20": "makam_sunan_drajat/split_sunan_drajat_80_20.pkl",
            "90_10": "makam_sunan_drajat/split_sunan_drajat_90_10.pkl",
        },
        "Wisata Monumen Van Der Wijk": {
            "80_20": "monumen_van_der_wijk/split_vdw_80_20.pkl",
            "90_10": "monumen_van_der_wijk/split_vdw_90_10.pkl",
        },
        "Wisata Museum Sunan Drajat": {
            "80_20": "museum_sunan_drajat/split_museum_sunan_80_20.pkl",
            "90_10": "museum_sunan_drajat/split_museum_sunan_90_10.pkl",
        },
        "Wisata Waduk Gondang": {
            "80_20": "waduk_gondang/split_gondang_80_20.pkl",
            "90_10": "waduk_gondang/split_gondang_90_10.pkl",
        },
        "Wisata Bahari Lamongan": {
            "80_20": "wisata_bahari_lamongan/split_wbl_80_20.pkl",
            "90_10": "wisata_bahari_lamongan/split_wbl_90_10.pkl",
        },
    }

    col_wisata, col_split = st.columns([2, 1])
    with col_wisata:
        wisata_choice = st.selectbox("🗺 Silahkan Pilih Data Wisata:", list(split_files.keys()))
    with col_split:
        split_label_ui = st.radio("Pilih rasio split:", ["80% / 20%", "90% / 10%"], index=0, horizontal=True)

    split_key = "80_20" if "80%" in split_label_ui else "90_10"
    pickle_path = split_files[wisata_choice][split_key]

    st.info(f"📂 File split yang akan dimuat: `{pickle_path}`")

    if not os.path.exists(pickle_path):
        st.error(f"❌ File `{pickle_path}` tidak ditemukan. Pastikan file pickle split ada di folder aplikasi.")
        st.stop()

    try:
        split_artifact = load_pickle(pickle_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat `{pickle_path}`: {e}")
        st.stop()

    train_components = split_artifact.get("train_components", {}) or {}
    test_components = split_artifact.get("test_components", {}) or {}
    y_train = np.asarray(split_artifact.get("y_train", []), dtype=float)
    y_test = np.asarray(split_artifact.get("y_test", []), dtype=float)
    train_ratio = split_artifact.get("train_ratio", None)
    split_index = split_artifact.get("split_index", None)
    N = split_artifact.get("N", len(y_train) + len(y_test))     
    label = split_artifact.get("label", split_label_ui)

    st.session_state["current_split_artifact"] = split_artifact
    st.session_state["current_split_info"] = {
        "wisata": wisata_choice,
        "pickle_path": pickle_path,
        "split_key": split_key,
        "label": label,
    }

    st.success("✅ File split berhasil dimuat.")

    st.subheader("🧾 Ringkasan Split Data")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.markdown('<div class="metric-card"><div style="font-size:1.02rem;font-weight:600">{}</div><div class="small-muted">Total (N)</div></div>'.format(N), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div style="font-size:1.02rem;font-weight:600">{}</div><div class="small-muted">Panjang Latih</div></div>'.format(len(y_train)), unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div style="font-size:1.02rem;font-weight:600">{}</div><div class="small-muted">Panjang Uji</div></div>'.format(len(y_test)), unsafe_allow_html=True)

    if train_ratio is not None:
        st.caption(f"Rasio latih ≈ {train_ratio*100:.1f}%  |  uji ≈ {(1-train_ratio)*100:.1f}%")
    st.write(f"Label split : **{label}**")
    st.markdown("---")

    st.subheader("🧩 Panjang Komponen IMF (Latih / Uji)")
    comp_names = list(train_components.keys())
    st.write(f"Jumlah komponen: **{len(comp_names)}**")

    preview_comps = comp_names[:6]
    rows = []
    for name in preview_comps:
        rows.append({"Komponen": name, "Latih": len(train_components.get(name, [])), "Test": len(test_components.get(name, []))})
    df_comp_preview = pd.DataFrame(rows)
    st.dataframe(df_comp_preview, use_container_width=True, height=160)
    if len(comp_names) > len(preview_comps):
        st.caption(f"... dan {len(comp_names) - len(preview_comps)} komponen lainnya.")

    combined_comp_df = pd.DataFrame({k: np.concatenate([train_components.get(k, []), test_components.get(k, [])]) for k in comp_names})
    csv_comp_bytes = combined_comp_df.to_csv(index=False).encode("utf-8")
    excel_buf = BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        combined_comp_df.to_excel(writer, index=False, sheet_name="components")
    excel_buf.seek(0)
    excel_bytes = excel_buf.read()

    dlc1, dlc2 = st.columns([1,1])
    with dlc1:
        st.download_button("⬇️ Unduh Komponen (CSV)", data=csv_comp_bytes, file_name=f"{wisata_choice.replace(' ','_')}_components.csv", mime="text/csv", use_container_width=True)
    with dlc2:
        st.download_button("⬇️ Unduh Komponen (Excel)", data=excel_bytes, file_name=f"{wisata_choice.replace(' ','_')}_components.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Visualisasi Sinyal Asli (Latih + Uji)")
    full_series = np.concatenate([y_train, y_test])
    idx = np.arange(len(full_series))
    df_plot = pd.DataFrame({
        "index": idx,
        "value": full_series,
        "set": ["latih"] * len(y_train) + ["uji"] * len(y_test)
    })

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(df_plot[df_plot["set"]=="latih"]["index"],
            df_plot[df_plot["set"]=="latih"]["value"],
            label="Latih", linewidth=2)

    ax.plot(df_plot[df_plot["set"]=="uji"]["index"],
            df_plot[df_plot["set"]=="uji"]["value"],
            label="Uji", linewidth=2)

    ax.axvline(split_index, linestyle="--", color="red", label="Batas Latih/Uji")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)
    plt.close(fig)


    st.markdown("---")
    st.markdown("---")
    st.markdown("### ✅ Ringkasan Akhir")
    st.write(
        f"- Wisata dipilih: **{wisata_choice}**  \n"
        f"- File pickle split: `{pickle_path}`  \n"
        f"- Label / Rasio: **{label}**  \n"
        f"- Data latih: **{len(y_train)}** titik  \n"
        f"- Data uji: **{len(y_test)}** titik"
    )
    st.success("🎉 Artefak split berhasil dimuat dan ditampilkan. Artefak tersimpan di `st.session_state['current_split_artifact']` untuk langkah berikutnya.")

elif menu == "📊 Modelling":
    import matplotlib.pyplot as plt
    st.markdown("---")
    st.markdown(
        """
        <style>
            .upload-title-center {
                text-align: left; 
                font-size: 26px;
                font-weight: 700;
                color: #2b2b2b;
                margin-top: -10px;
                margin-bottom: 18px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .upload-title-center span.icon {
                font-size: 30px;
            }
        </style>

        <div class="upload-title-center">
            <span class="icon">📊</span>
            <span>Menu Modelling</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("""
    Pada tahap ini ditampilkan **ringkasan hasil modelling**:
    - **ELM standar** (tanpa optimasi *Particle Swarm Optimization* / PSO)
    - **CEEMDAN–ELM–PSO** (ELM yang sudah dioptimasi PSO)

    Untuk setiap objek wisata, terdapat **4 kombinasi**:
    - Split **80% / 20%**, Konfigurasi **1** : **Partikel = 20**, **Iterasi = 200**
    - Split **80% / 20%**, Konfigurasi **2** : **Partikel = 30**, **Iterasi = 300**
    - Split **90% / 10%**, Konfigurasi **1** : **Partikel = 20**, **Iterasi = 200**
    - Split **90% / 10%**, Konfigurasi **2** : **Partikel = 30**, **Iterasi = 300**
    """)

    model_files = {
        "Wisata Brumbun": {
            ("80_20", "cfg1"): "wisata_brumbun/modelling_brumbun_split_80_20_cfg1.pkl",
            ("80_20", "cfg2"): "wisata_brumbun/modelling_brumbun_split_80_20_cfg2.pkl",
            ("90_10", "cfg1"): "wisata_brumbun/modelling_brumbun_split_90_10_cfg1.pkl",
            ("90_10", "cfg2"): "wisata_brumbun/modelling_brumbun_split_90_10_cfg2.pkl",
        },
        "Wisata Goa Maharani": {
            ("80_20", "cfg1"): "goa_maharani/modelling_goa_maharani_split_80_20_cfg1.pkl",
            ("80_20", "cfg2"): "goa_maharani/modelling_goa_maharani_split_80_20_cfg2.pkl",
            ("90_10", "cfg1"): "goa_maharani/modelling_goa_maharani_split_90_10_cfg1.pkl",
            ("90_10", "cfg2"): "goa_maharani/modelling_goa_maharani_split_90_10_cfg2.pkl",
        },
        "Wisata Makam Sendang Duwur": {
            ("80_20", "cfg1"): "makam_sendang_duwur/modelling_sendang_duwur_split_80_20_cfg1.pkl",
            ("80_20", "cfg2"): "makam_sendang_duwur/modelling_sendang_duwur_split_80_20_cfg2.pkl",
            ("90_10", "cfg1"): "makam_sendang_duwur/modelling_sendang_duwur_split_90_10_cfg1.pkl",
            ("90_10", "cfg2"): "makam_sendang_duwur/modelling_sendang_duwur_split_90_10_cfg2.pkl",
        },
        "Wisata Makam Sunan Drajat": {
            ("80_20", "cfg1"): "makam_sunan_drajat/modelling_sunan_drajat_split_80_20_cfg1.pkl",
            ("80_20", "cfg2"): "makam_sunan_drajat/modelling_sunan_drajat_split_80_20_cfg2.pkl",
            ("90_10", "cfg1"): "makam_sunan_drajat/modelling_sunan_drajat_split_90_10_cfg1.pkl",
            ("90_10", "cfg2"): "makam_sunan_drajat/modelling_sunan_drajat_split_90_10_cfg2.pkl",
        },
        "Wisata Monumen Van Der Wijk": {
            ("80_20", "cfg1"): "monumen_van_der_wijk/modelling_vdw_split_80_20_cfg1.pkl",
            ("80_20", "cfg2"): "monumen_van_der_wijk/modelling_vdw_split_80_20_cfg2.pkl",
            ("90_10", "cfg1"): "monumen_van_der_wijk/modelling_vdw_split_90_10_cfg1.pkl",
            ("90_10", "cfg2"): "monumen_van_der_wijk/modelling_vdw_split_90_10_cfg2.pkl",
        },
        "Wisata Museum Sunan Drajat": {
            ("80_20", "cfg1"): "museum_sunan_drajat/modelling_museum_sunan_split_80_20_cfg1.pkl",
            ("80_20", "cfg2"): "museum_sunan_drajat/modelling_museum_sunan_split_80_20_cfg2.pkl",
            ("90_10", "cfg1"): "museum_sunan_drajat/modelling_museum_sunan_split_90_10_cfg1.pkl",
            ("90_10", "cfg2"): "museum_sunan_drajat/modelling_museum_sunan_split_90_10_cfg2.pkl",
        },
        "Wisata Waduk Gondang": {
            ("80_20", "cfg1"): "waduk_gondang/modelling_gondang_split_80_20_cfg1.pkl",
            ("80_20", "cfg2"): "waduk_gondang/modelling_gondang_split_80_20_cfg2.pkl",
            ("90_10", "cfg1"): "waduk_gondang/modelling_gondang_split_90_10_cfg1.pkl",
            ("90_10", "cfg2"): "waduk_gondang/modelling_gondang_split_90_10_cfg2.pkl",
        },
        "Wisata Bahari Lamongan": {
            ("80_20", "cfg1"): "wisata_bahari_lamongan/modelling_wbl_split_80_20_cfg1.pkl",
            ("80_20", "cfg2"): "wisata_bahari_lamongan/modelling_wbl_split_80_20_cfg2.pkl",
            ("90_10", "cfg1"): "wisata_bahari_lamongan/modelling_wbl_split_90_10_cfg1.pkl",
            ("90_10", "cfg2"): "wisata_bahari_lamongan/modelling_wbl_split_90_10_cfg2.pkl",
        },
    }

    wisata_choice = st.selectbox("🗺 Silahkan Pilih Data Wisata:", list(model_files.keys()))
    st.markdown(f"**Wisata terpilih:** {wisata_choice}")

    rows_elm = []
    rows_pso = []

    for (split_key, cfg_label), file_path in model_files[wisata_choice].items():
        try:
            artifact = load_pickle(file_path)

            elm_std = artifact.get("elm_standard", {})
            elm_pso = artifact.get("elm_pso", {})

            split_label = elm_pso.get("split_label", split_key)
            config = elm_pso.get("config", cfg_label)

            mt_train_std = elm_std.get("metrics_train", {})
            mt_test_std = elm_std.get("metrics_test", {})

            mape_train_std_pct = mt_train_std.get("MAPE", np.nan) * 100.0 if mt_train_std.get("MAPE") is not None else np.nan
            mape_test_std_pct = mt_test_std.get("MAPE", np.nan) * 100.0 if mt_test_std.get("MAPE") is not None else np.nan

            row_std = {
                "Split": split_label.replace("_", "/"),
                "MAPE Train (%)": mape_train_std_pct,
                "MAPE Test (%)": mape_test_std_pct,
                "MAE Train": mt_train_std.get("MAE", np.nan),
                "RMSE Train": mt_train_std.get("RMSE", np.nan),
                "R² Train": mt_train_std.get("R2", np.nan),
                "MAE Test": mt_test_std.get("MAE", np.nan),
                "RMSE Test": mt_test_std.get("RMSE", np.nan),
                "R² Test": mt_test_std.get("R2", np.nan),
            }
            rows_elm.append(row_std)

            mt_train_pso = elm_pso.get("metrics_train", {})
            mt_test_pso = elm_pso.get("metrics_test", {})

            mape_train_pso_pct = mt_train_pso.get("MAPE", np.nan) * 100.0 if mt_train_pso.get("MAPE") is not None else np.nan
            mape_test_pso_pct = mt_test_pso.get("MAPE", np.nan) * 100.0 if mt_test_pso.get("MAPE") is not None else np.nan

            row_pso = {
                "File Pickle": file_path,
                "Split": split_label.replace("_", "/"),
                "Konfigurasi": config,
                "MAPE Train (%)": mape_train_pso_pct,
                "MAPE Test (%)": mape_test_pso_pct,
                "MAE Train": mt_train_pso.get("MAE", np.nan),
                "RMSE Train": mt_train_pso.get("RMSE", np.nan),
                "R² Train": mt_train_pso.get("R2", np.nan),
                "MAE Test": mt_test_pso.get("MAE", np.nan),
                "RMSE Test": mt_test_pso.get("RMSE", np.nan),
                "R² Test": mt_test_pso.get("R2", np.nan),
            }
            rows_pso.append(row_pso)

        except FileNotFoundError:
            st.warning(f"⚠️ File tidak ditemukan: {file_path}")
        except Exception as e:
            st.error(f"❌ Gagal membaca file `{file_path}`: {e}")

    if not rows_pso:
        st.error("Tidak ada file modelling yang berhasil dimuat untuk wisata ini.")
        st.stop()

    df_elm = pd.DataFrame(rows_elm)
    df_pso = pd.DataFrame(rows_pso)

    if not df_elm.empty:
        df_elm = (
            df_elm
            .groupby("Split", as_index=False)
            .agg({
                "MAPE Train (%)": "mean",
                "MAPE Test (%)": "mean",
                "MAE Train": "mean",
                "RMSE Train": "mean",
                "R² Train": "mean",
                "MAE Test": "mean",
                "RMSE Test": "mean",
                "R² Test": "mean",
            })
        )

    df_pso["Terbaik?"] = ""
    try:
        if "MAPE Test (%)" in df_pso.columns and df_pso["MAPE Test (%)"].dropna().shape[0] > 0:
            best_idx = df_pso["MAPE Test (%)"].idxmin()
            df_pso.loc[best_idx, "Terbaik?"] = "✅"
        else:
            best_idx = None
    except Exception:
        best_idx = None

    tab1, tab2 = st.tabs(["📄 ELM Standar (tanpa PSO)", "🚀 CEEMDAN–ELM–PSO"])

    with tab1:
        st.subheader("📊 Ringkasan Hasil ELM Standar")
        st.write(
            "Tabel berikut menampilkan metrik **ELM tanpa optimasi PSO** untuk setiap kombinasi split data "
            "(satu baris per split)."
        )

        st.dataframe(
            df_elm.style.format({
                "MAPE Train (%)": "{:.2f}",
                "MAPE Test (%)": "{:.2f}",
                "MAE Train": "{:.4f}",
                "RMSE Train": "{:.4f}",
                "R² Train": "{:.4f}",
                "MAE Test": "{:.4f}",
                "RMSE Test": "{:.4f}",
                "R² Test": "{:.4f}",
            }),
            use_container_width=True
        )

    with tab2:
        st.subheader("📊 Ringkasan Hasil CEEMDAN–ELM–PSO (4 Kombinasi)")
        st.write(
            "Nilai **MAPE** dalam persen, semakin kecil semakin baik. "
            "Kolom **Terbaik?** menandai model dengan MAPE Test paling rendah."
        )
        st.dataframe(
            df_pso.style.format({
                "MAPE Train (%)": "{:.2f}",
                "MAPE Test (%)": "{:.2f}",
                "MAE Train": "{:.4f}",
                "RMSE Train": "{:.4f}",
                "R² Train": "{:.4f}",
                "MAE Test": "{:.4f}",
                "RMSE Test": "{:.4f}",
                "R² Test": "{:.4f}",
            }),
            use_container_width=True
        )

    if best_idx is not None:
        best_row = df_pso.loc[best_idx]

        st.markdown("---")
        st.subheader("🏆 Model Terbaik untuk Wisata Ini (berdasarkan CEEMDAN–ELM–PSO)")

        colA, colB = st.columns(2)
        with colA:
            st.metric(
                "MAPE Test Terbaik (%)",
                f"{best_row['MAPE Test (%)']:.2f}"
            )
            st.metric(
                "Split Data",
                best_row["Split"]
            )
        with colB:
            st.metric(
                "Konfigurasi",
                best_row["Konfigurasi"]
            )
            st.caption(f"File: `{best_row['File Pickle']}`")

        st.success(f"""
        Model terbaik (CEEMDAN–ELM–PSO) diperoleh dari:
        - **Split**: {best_row['Split']}
        - **Konfigurasi**: {best_row['Konfigurasi']}
        - **MAPE Test**: {best_row['MAPE Test (%)']:.2f}%
        """)

        st.session_state["best_model_info"] = {
            "wisata": wisata_choice,
            "pickle_path": best_row["File Pickle"],
            "split": best_row["Split"],
            "config": best_row["Konfigurasi"],
            "mape_test_pct": float(best_row["MAPE Test (%)"]),
        }
        st.caption(
            "ℹ️ Informasi model terbaik (CEEMDAN–ELM–PSO) sudah disimpan dan dapat dipakai "
            "di menu berikutnya (visualisasi & forecasting)."
        )
    else:
        st.info("Tidak ditemukan model PSO yang valid untuk dijadikan patokan terbaik (MAPE Test tidak tersedia).")

    comparison_files = {
        "Wisata Brumbun": "wisata_brumbun/comparison_brumbun.pkl",
        "Wisata Goa Maharani": "goa_maharani/comparison_goa_maharani.pkl",
        "Wisata Makam Sendang Duwur": "makam_sendang_duwur/comparison_sendang_duwur.pkl",
        "Wisata Makam Sunan Drajat": "makam_sunan_drajat/comparison_makam_sunan_drajat.pkl",
        "Wisata Monumen Van Der Wijk": "monumen_van_der_wijk/comparison_vdw.pkl",
        "Wisata Museum Sunan Drajat": "museum_sunan_drajat/comparison_museum_sunan_drajat.pkl",
        "Wisata Waduk Gondang": "waduk_gondang/comparison_gondang.pkl",
        "Wisata Bahari Lamongan": "wisata_bahari_lamongan/comparison_wbl.pkl",
    }

    st.markdown("---")
    st.subheader("📉 Perbandingan Data Aktual vs CEEMDAN–ELM vs CEEMDAN–ELM–PSO")

    st.write("""
    Bagian ini menampilkan **grafik perbandingan** antara:
    - Data aktual,
    - Prediksi **CEEMDAN–ELM**,
    - Prediksi **CEEMDAN–ELM–PSO**.

    Grafik dihasilkan dari **file pickle perbandingan** yang sudah dibuat di Colab.
    """)

    comp_path = comparison_files.get(wisata_choice, None)

    if comp_path is None:
        st.warning("Belum ada file perbandingan yang dipetakan untuk wisata ini.")
    elif not os.path.exists(comp_path):
        st.warning(f"File perbandingan tidak ditemukan: `{comp_path}`")
    else:
        if st.button("Tampilkan Grafik Perbandingan (dari Pickle)"):
            comp = load_pickle(comp_path)

            actual = np.asarray(comp["actual"], dtype=float)
            pred_elm_visual = np.asarray(comp["pred_elm_visual"], dtype=float)
            pred_pso = np.asarray(comp["pred_pso"], dtype=float)

            split_label = comp.get("split_label", "")
            config = comp.get("config", "")
            mape_test = comp.get("mape_test", None)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(actual, label="Data Aktual", linewidth=2)
            ax.plot(pred_elm_visual, label="CEEMDAN–ELM", linestyle="-", linewidth=2)
            ax.plot(pred_pso, label="CEEMDAN–ELM–PSO", linestyle="-", linewidth=2)

            title = f"Perbandingan Data Aktual vs CEEMDAN–ELM vs CEEMDAN–ELM–PSO\n(Split {split_label.replace('_','/')} | {config})"
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Index Waktu")
            ax.set_ylabel("Jumlah Wisatawan")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            st.pyplot(fig)

            if mape_test is not None:
                st.caption(f"ℹ️ MAPE Test (PSO) model terbaik: {mape_test:.2%}")

elif menu == "📈 Prediksi":
    import matplotlib.pyplot as plt
    st.markdown("---")
    st.markdown(
        """
        <style>
            .upload-title-center {
                text-align: left; 
                font-size: 26px;
                font-weight: 700;
                color: #2b2b2b;
                margin-top: -10px;
                margin-bottom: 18px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .upload-title-center span.icon {
                font-size: 30px;
            }
        </style>

        <div class="upload-title-center">
            <span class="icon">📈</span>
            <span>Menu Prediksi 1 Bulan Berikutnya</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    forecast_files = {
        "Wisata Brumbun": "wisata_brumbun/forecast_brumbun_next_1month_best.pkl",
        "Wisata Goa Maharani": "goa_maharani/forecast_goa_maharani_next_1month_best.pkl",
        "Wisata Makam Sendang Duwur": "makam_sendang_duwur/forecast_sendang_duwur_next_1month_best.pkl",
        "Wisata Makam Sunan Drajat": "makam_sunan_drajat/forecast_makam_sunan_drajat_next_1month_best.pkl",
        "Wisata Monumen Van Der Wijk": "monumen_van_der_wijk/forecast_vdw_next_1month_best.pkl",
        "Wisata Museum Sunan Drajat": "museum_sunan_drajat/forecast_museum_sunan_drat_next_1month_best.pkl",
        "Wisata Waduk Gondang": "waduk_gondang/forecast_gondang_next_1month_best.pkl",
        "Wisata Bahari Lamongan": "wisata_bahari_lamongan/forecast_wbl_next_1month_best.pkl",
    }

    wisata_choice = st.selectbox("🗺 Silahkan Pilih Data Wisata Untuk Diprediksi:", list(forecast_files.keys()))
    st.markdown(f"**Wisata terpilih :** {wisata_choice}")

    if st.button("🔍 Tampilkan Hasil Prediksi 1 Bulan Berikutnya"):
        forecast_path = forecast_files.get(wisata_choice)
        if not forecast_path or not os.path.exists(forecast_path):
            st.error(f"❌ File pickle tidak ditemukan: `{forecast_path}`. Pastikan path benar.")
            st.stop()

        try:
            artifact = load_pickle(forecast_path)
        except Exception as e:
            st.error(f"❌ Gagal memuat pickle: {e}")
            st.stop()

        st.session_state["last_forecast_artifact"] = artifact

        y_next_orig = artifact.get("y_next_orig", None)
        try:
            if y_next_orig is not None:
                st.metric("🔮 Prediksi 1 Bulan ", f"{int(round(float(y_next_orig))):,}".replace(",", "."))
            else:
                st.metric("🔮 Prediksi 1 Bulan ", "Tidak tersedia")
        except Exception:
            st.metric("🔮 Prediksi 1 Bulan ", str(y_next_orig))

        st.markdown("---")
        st.subheader("📷 Visualisasi Prediksi ")

        plotted = False

        if "forecast_series" in artifact:
            try:
                fs = np.asarray(artifact["forecast_series"], dtype=float).flatten()
                if fs.size >= 2:
                    hist = fs[:-1]
                    pred_val = float(fs[-1])
                    idx = np.arange(len(hist))

                    fig, ax = plt.subplots(figsize=(12,5))
                    ax.plot(idx, hist, label="Data Aktual", linewidth=2)
                    last_idx = len(hist) - 1
                    next_idx = last_idx + 1
                    ax.plot([last_idx, next_idx], [hist[-1], pred_val],
                            linestyle="--", marker="o", linewidth=2, label="Prediksi 1 Bulan")
                    ax.scatter([next_idx], [pred_val], s=80, zorder=3)
                    ax.set_title(f"Prediksi 1 Bulan — {wisata_choice}")
                    ax.set_xlabel("Index Waktu (Bulan)")
                    ax.set_ylabel("Jumlah Wisatawan")
                    ax.legend()
                    ax.grid(True, linestyle="--", alpha=0.5)
                    fig.tight_layout()

                    st.pyplot(fig)                       
                    buf = BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
                    st.download_button("⬇️ Download Plot ", data=buf.getvalue(),
                                       file_name=f"{wisata_choice.replace(' ','_')}_plot_generated.png", mime="image/png")
                    plt.close(fig)
                    plotted = True
            except Exception as e:
                st.warning("⚠️ Gagal menggambar dari `forecast_series`: " + str(e))
                plotted = False

        if (not plotted) and "original_series" in artifact and artifact.get("y_next_orig", None) is not None:
            try:
                orig = np.asarray(artifact["original_series"], dtype=float).flatten()
                if orig.size >= 1:
                    pred_val = float(artifact["y_next_orig"])
                    idx = np.arange(len(orig))
                    fig, ax = plt.subplots(figsize=(12,5))
                    ax.plot(idx, orig, label="Data Aktual", linewidth=2)
                    last_idx = len(orig)-1
                    next_idx = last_idx + 1
                    ax.plot([last_idx, next_idx], [orig[-1], pred_val],
                            linestyle="--", marker="o", linewidth=2, label="Prediksi 1 Bulan")
                    ax.scatter([next_idx], [pred_val], s=80, zorder=3)
                    ax.set_title(f"Prediksi 1 Bulan — {wisata_choice}")
                    ax.set_xlabel("Index Waktu (Bulan)")
                    ax.set_ylabel("Jumlah Wisatawan")
                    ax.legend()
                    ax.grid(True, linestyle="--", alpha=0.5)
                    fig.tight_layout()

                    st.pyplot(fig)
                    buf = BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
                    st.download_button("⬇️ Download Plot ", data=buf.getvalue(),
                                       file_name=f"{wisata_choice.replace(' ','_')}_plot_generated.png", mime="image/png")
                    plt.close(fig)
                    plotted = True
            except Exception as e:
                st.warning("⚠️ Gagal menggambar dari `original_series`: " + str(e))

        if not plotted:
            st.info("📉 Tidak cukup data untuk menggambar plot (tidak ada forecast_series maupun original_series dengan prediksi).")
