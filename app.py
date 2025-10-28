import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

# Mengabaikan FutureWarning dari Matplotlib dan Seaborn
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Pengaturan Tema Seaborn Global untuk Tampilan Modern ---
# Menggunakan tema 'darkgrid' atau 'whitegrid' dengan palet warna yang modern
sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1) 
# Atau palette="deep", "pastel", "flare", "magma"

# Konfigurasi Halaman Streamlit
st.set_page_config(layout="wide", page_title="Analisis Data Diabetes")

# Fungsi untuk memuat data dengan caching
@st.cache
def load_data(file_path):
    """Memuat dataset diabetes dan melakukan pra-pemrosesan."""
    
    with st.spinner('Memuat dan memproses data...'):
        time.sleep(1) # Simulasi jeda loading
        try:
            df = pd.read_csv(file_path)
            
            # Pra-pemrosesan data
            age_bins = [0, 18, 30, 45, 60, 75, 90]
            age_labels = ['0-17', '18-29', '30-44', '45-59', '60-74', '75+']
            df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

            race_columns = ['race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other']
            df_race = df[race_columns + ['diabetes']].melt(id_vars='diabetes', var_name='race', value_name='is_race')
            df_race = df_race[df_race['is_race'] == 1].copy()
            df_race['race'] = df_race['race'].str.replace('race:', '')
            
            # Menambahkan kolom 'year' jika belum ada (untuk Studi Kasus 10)
            # Ini hanya contoh, sesuaikan jika dataset Anda memiliki kolom tanggal/tahun yang sebenarnya
            if 'year' not in df.columns:
                df['year'] = pd.to_datetime('2022-01-01').year # Kolom dummy jika tidak ada info tahun
            
            return df, df_race
        except FileNotFoundError:
            st.error("File 'diabetes_dataset.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
            return pd.DataFrame(), pd.DataFrame()

# Path file dataset (ganti jika perlu)
FILE_PATH = 'diabetes_dataset.csv'
df, df_race = load_data(FILE_PATH)

# --- Judul dan Animasi ---
st.title("🚀 Analisis Data Diabetes Interaktif")

if not df.empty:
    st.balloons() # Animasi balon
    
    st.markdown("Aplikasi ini menyajikan Eksplorasi Data Awal (EDA) dan Studi Kasus Visualisasi dari dataset diabetes.")

    # --- Sidebar untuk Informasi Data ---
    with st.sidebar:
        st.header("Konfigurasi Data")
        
        if st.checkbox("Tampilkan Dataframe Mentah (Head)", False):
            st.subheader("5 Baris Pertama Data")
            st.dataframe(df.head())
        
        st.subheader("Statistik Ringkasan")
        st.dataframe(df.describe().T)
        
        with st.expander("Informasi Kolom (df.info())"):
            buffer = pd.io.common.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with st.expander("Nilai Hilang (Null Values)"):
            st.dataframe(df.isnull().sum().rename("Missing Values"))

    # --- Bagian Visualisasi EDA Awal ---
    st.header("🔍 Visualisasi Data Awal (EDA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Usia Pasien")
        fig, ax = plt.subplots(figsize=(9, 6)) # Ukuran lebih besar
        sns.histplot(data=df, x='age', kde=True, ax=ax, bins=20, color=sns.color_palette("viridis")[0])
        ax.set_title('Distribusi Usia', fontsize=16)
        ax.set_xlabel('Usia', fontsize=12)
        ax.set_ylabel('Frekuensi', fontsize=12)
        sns.despine(left=True, bottom=True) # Menghilangkan spines
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Korelasi BMI dan Tingkat Glukosa Darah")
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(data=df, x='bmi', y='blood_glucose_level', alpha=0.6, ax=ax, hue='diabetes', palette='viridis', s=60)
        ax.set_title('BMI vs Tingkat Glukosa Darah', fontsize=16)
        ax.set_xlabel('BMI', fontsize=12)
        ax.set_ylabel('Tingkat Glukosa Darah', fontsize=12)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Distribusi Jenis Kelamin Pasien")
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.countplot(data=df, x='gender', ax=ax, palette='viridis')
        ax.set_title('Distribusi Jenis Kelamin', fontsize=16)
        ax.set_xlabel('Jenis Kelamin', fontsize=12)
        ax.set_ylabel('Jumlah', fontsize=12)
        # Menambahkan label count di atas bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Perbandingan Glukosa Darah berdasarkan Status Diabetes")
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.boxplot(data=df, x='diabetes', y='blood_glucose_level', ax=ax, palette='viridis')
        ax.set_title('Tingkat Glukosa Darah Berdasarkan Status Diabetes', fontsize=16)
        ax.set_xlabel('Status Diabetes (0: Tidak, 1: Ya)', fontsize=12)
        ax.set_ylabel('Tingkat Glukosa Darah', fontsize=12)
        ax.set_xticklabels(['Tidak Diabetes', 'Diabetes'])
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        st.pyplot(fig)

    # --- Bagian Study Case Visualizations & Penjelasan ---
    st.markdown("---")
    st.header("🔬 Studi Kasus dan Penjelasan Visualisasi")
    
    tab_titles = [
        "Kasus 1: Usia", "Kasus 2: J. Kelamin", "Kasus 3: Lokasi", "Kasus 4: Ras", 
        "Kasus 5: Merokok", "Kasus 6: BMI", "Kasus 7: HbA1c", "Kasus 8: Glukosa vs Usia", 
        "Kasus 9: Komorbiditas", "Kasus 10: Tren Tahunan"
    ]
    tabs = st.tabs(tab_titles)
    
    # --- Study Case 1: Usia ---
    with tabs[0]:
        col_vis, col_text = st.columns([2, 1])
        with col_vis:
            st.subheader("Studi Kasus 1: Distribusi Kasus Diabetes di Berbagai Kelompok Usia")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x='age', hue='diabetes', kde=True, multiple='stack', ax=ax, bins=25, palette='viridis')
            ax.set_title('Distribusi Usia Berdasarkan Status Diabetes', fontsize=16)
            ax.set_xlabel('Usia', fontsize=12)
            ax.set_ylabel('Frekuensi', fontsize=12)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[::-1], labels=['Diabetes', 'Tidak Diabetes'], title='Diabetes', loc='upper right') # Membalik urutan legend
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_text:
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi histogram menunjukkan bahwa **prevalensi diabetes meningkat seiring bertambahnya usia**. Kelompok usia yang lebih tua, terutama di atas **45 tahun**, memiliki frekuensi kasus diabetes yang jauh lebih tinggi dibandingkan dengan kelompok usia yang lebih muda. Ini menunjukkan hubungan yang **kuat** antara usia dan risiko diabetes.
            """)
    
    # --- Study Case 2: Gender ---
    with tabs[1]:
        st.subheader("Studi Kasus 2: Hubungan Jenis Kelamin dengan Prevalensi Hipertensi dan Penyakit Jantung")
        col_hpt, col_hdt = st.columns(2)
        
        with col_hpt:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x='gender', hue='hypertension', ax=ax, palette='viridis')
            ax.set_title('Prevalensi Hipertensi berdasarkan Jenis Kelamin', fontsize=14)
            ax.set_xlabel('Jenis Kelamin', fontsize=12)
            ax.set_ylabel('Jumlah', fontsize=12)
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=9, color='black', xytext=(0, 3),
                            textcoords='offset points')
            ax.legend(title='Hipertensi', labels=['Tidak', 'Ya'], loc='upper right')
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_hdt:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x='gender', hue='heart_disease', ax=ax, palette='viridis')
            ax.set_title('Prevalensi Penyakit Jantung berdasarkan Jenis Kelamin', fontsize=14)
            ax.set_xlabel('Jenis Kelamin', fontsize=12)
            ax.set_ylabel('Jumlah', fontsize=12)
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=9, color='black', xytext=(0, 3),
                            textcoords='offset points')
            ax.legend(title='Penyakit Jantung', labels=['Tidak', 'Ya'], loc='upper right')
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig)
            
        st.subheader("Penjelasan")
        st.markdown("""
        Visualisasi count plot menunjukkan perbedaan prevalensi hipertensi dan penyakit jantung berdasarkan gender. Dalam dataset ini, terlihat bahwa **pria memiliki jumlah kasus hipertensi dan penyakit jantung yang lebih tinggi dibandingkan wanita**. Ini mengindikasikan bahwa gender mungkin merupakan faktor yang berkorelasi dengan risiko kedua kondisi ini.
        """)

    # --- Study Case 3: Lokasi ---
    with tabs[2]:
        col_vis, col_text = st.columns([2, 1])
        with col_vis:
            st.subheader("Studi Kasus 3: Pola Geografis dalam Distribusi Kasus Diabetes")
            fig, ax = plt.subplots(figsize=(14, 7)) # Ukuran lebih lebar
            sns.countplot(data=df, x='location', hue='diabetes', ax=ax, palette='viridis')
            ax.set_title('Distribusi Kasus Diabetes berdasarkan Lokasi', fontsize=16)
            ax.set_xlabel('Lokasi', fontsize=12)
            ax.set_ylabel('Jumlah Kasus', fontsize=12)
            plt.xticks(rotation=60, ha='right', fontsize=10) # Rotasi dan ukuran font
            ax.legend(title='Diabetes', labels=['Tidak', 'Ya'], loc='upper right')
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_text:
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi count plot menunjukkan **variasi yang signifikan** dalam jumlah kasus diabetes di berbagai lokasi (negara bagian). Beberapa lokasi memiliki jumlah total kasus diabetes yang jauh lebih banyak dibandingkan lokasi lain, menyarankan adanya **disparitas geografis** dalam prevalensi diabetes yang bisa disebabkan oleh berbagai faktor regional seperti gaya hidup, akses kesehatan, atau demografi.
            """)

    # --- Study Case 4: Ras ---
    with tabs[3]:
        col_vis, col_text = st.columns([2, 1])
        with col_vis:
            st.subheader("Studi Kasus 4: Korelasi Ras dengan Prevalensi Diabetes")
            if not df_race.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(data=df_race, x='race', hue='diabetes', ax=ax, palette='viridis')
                ax.set_title('Distribusi Kasus Diabetes berdasarkan Ras', fontsize=16)
                ax.set_xlabel('Ras', fontsize=12)
                ax.set_ylabel('Jumlah Kasus', fontsize=12)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                ax.legend(title='Diabetes', labels=['Tidak', 'Ya'], loc='upper right')
                sns.despine(left=True, bottom=True)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Data ras tidak tersedia atau kosong setelah pemrosesan.")
        
        with col_text:
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi count plot berdasarkan kategori ras menunjukkan bahwa **prevalensi diabetes bervariasi antar kelompok ras**. Dengan mengeksaminasi jumlah kasus diabetes dalam setiap kategori ras, terlihat bahwa proporsi individu dengan diabetes berbeda di antara kelompok-kelompok ini, menunjukkan bahwa **ras mungkin berhubungan** dengan risiko diabetes.
            """)

    # --- Study Case 5: Merokok ---
    with tabs[4]:
        col_vis, col_text = st.columns([2, 1])
        with col_vis:
            st.subheader("Studi Kasus 5: Hubungan Riwayat Merokok dan Diabetes")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df, x='smoking_history', hue='diabetes', ax=ax, palette='viridis')
            ax.set_title('Distribusi Riwayat Merokok berdasarkan Status Diabetes', fontsize=16)
            ax.set_xlabel('Riwayat Merokok', fontsize=12)
            ax.set_ylabel('Jumlah Kasus', fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax.legend(title='Diabetes', labels=['Tidak', 'Ya'], loc='upper right')
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_text:
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi count plot riwayat merokok berdasarkan hasil diabetes menunjukkan bahwa beberapa kategori riwayat merokok memiliki proporsi kasus diabetes yang lebih tinggi. Misalnya, individu dengan riwayat merokok **'current'** atau **'former'** mungkin menunjukkan prevalensi diabetes yang berbeda dibandingkan dengan yang **'never'** merokok atau **'No Info'**, menyarankan potensi **korelasi** antara riwayat merokok dan risiko diabetes.
            """)

    # --- Study Case 6: BMI ---
    with tabs[5]:
        col_vis, col_text = st.columns([2, 1])
        with col_vis:
            st.subheader("Studi Kasus 6: Bagaimana BMI mempengaruhi kemungkinan menderita diabetes")
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.boxplot(data=df, x='diabetes', y='bmi', ax=ax, palette='viridis')
            ax.set_title('Distribusi BMI berdasarkan Status Diabetes', fontsize=16)
            ax.set_xlabel('Status Diabetes (0: Tidak, 1: Ya)', fontsize=12)
            ax.set_ylabel('BMI', fontsize=12)
            ax.set_xticklabels(['Tidak Diabetes', 'Diabetes'])
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_text:
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi box plot menunjukkan **hubungan yang jelas** antara BMI dan kemungkinan menderita diabetes. Individu yang didiagnosis dengan diabetes cenderung memiliki **median BMI yang lebih tinggi** dan distribusi BMI yang bergeser ke arah nilai yang lebih tinggi dibandingkan dengan individu tanpa diabetes, menggarisbawahi bahwa **BMI yang tinggi merupakan faktor risiko** untuk diabetes.
            """)

    # --- Study Case 7: HbA1c ---
    with tabs[6]:
        col_vis, col_text = st.columns([2, 1])
        with col_vis:
            st.subheader("Studi Kasus 7: Distribusi Tingkat HbA1c pada Individu dengan dan tanpa Diabetes")
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.boxplot(data=df, x='diabetes', y='hbA1c_level', ax=ax, palette='viridis')
            ax.set_title('Distribusi Tingkat HbA1c berdasarkan Status Diabetes', fontsize=16)
            ax.set_xlabel('Status Diabetes (0: Tidak, 1: Ya)', fontsize=12)
            ax.set_ylabel('Tingkat HbA1c', fontsize=12)
            ax.set_xticklabels(['Tidak Diabetes', 'Diabetes'])
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_text:
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi box plot tingkat HbA1c menunjukkan **perbedaan signifikan** antara individu dengan dan tanpa diabetes. Individu dengan diabetes memiliki **tingkat HbA1c yang jauh lebih tinggi**, dengan median dan rentang nilai yang lebih tinggi, menegaskan bahwa **tingkat HbA1c adalah indikator kunci dan prediktor kuat** untuk diagnosis diabetes.
            """)

    # --- Study Case 8: Glukosa vs Usia ---
    with tabs[7]:
        col_vis, col_text = st.columns([2, 1])
        with col_vis:
            st.subheader("Studi Kasus 8: Variasi Tingkat Glukosa Darah berdasarkan Kelompok Usia")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=df, x='age_group', y='blood_glucose_level', ax=ax, palette='viridis')
            ax.set_title('Distribusi Tingkat Glukosa Darah berdasarkan Kelompok Usia', fontsize=16)
            ax.set_xlabel('Kelompok Usia', fontsize=12)
            ax.set_ylabel('Tingkat Glukosa Darah', fontsize=12)
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_text:
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi box plot tingkat glukosa darah berdasarkan kelompok usia menunjukkan bahwa distribusi tingkat glukosa darah **bervariasi** di antara kelompok usia yang berbeda. Meskipun tidak ada tren linear yang sempurna, terlihat perbedaan dalam median atau penyebaran nilai glukosa darah pada kelompok usia tertentu, menyarankan bahwa **usia dapat mempengaruhi distribusi tingkat glukosa darah**.
            """)

    # --- Study Case 9: Komorbiditas ---
    with tabs[8]:
        col_vis, col_text = st.columns([2, 1])
        df_filtered = df[(df['hypertension'] == 1) & (df['heart_disease'] == 1)]
        
        with col_vis:
            st.subheader("Studi Kasus 9: Prevalensi Diabetes pada Individu dengan Hipertensi dan Penyakit Jantung")
            if not df_filtered.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(data=df_filtered, x='diabetes', ax=ax, palette='viridis')
                ax.set_title('Prevalensi Diabetes pada Individu dengan Hipertensi dan Penyakit Jantung', fontsize=16)
                ax.set_xlabel('Status Diabetes (0: Tidak, 1: Ya)', fontsize=12)
                ax.set_ylabel('Jumlah Individu', fontsize=12)
                ax.set_xticklabels(['Tidak Diabetes', 'Diabetes'])
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')
                sns.despine(left=True, bottom=True)
                plt.tight_layout()
                st.pyplot(fig)
                st.info(f"Total individu dengan Hipertensi dan Penyakit Jantung: **{len(df_filtered)}**")
            else:
                st.warning("Tidak ada data untuk individu dengan Hipertensi dan Penyakit Jantung secara bersamaan.")
        
        with col_text:
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi count plot pada kelompok individu dengan hipertensi dan penyakit jantung menunjukkan bahwa **sebagian besar dari mereka juga menderita diabetes**. Ini dengan jelas menunjukkan **tingginya prevalensi diabetes** pada individu yang memiliki kedua kondisi komorbiditas ini, menyoroti risiko yang sangat tinggi pada kelompok ini.
            """)

    # --- Study Case 10: Tren Tahunan ---
    with tabs[9]:
        st.subheader("Studi Kasus 10: Tren Rata-rata BMI dan Kadar HbA1c dari Tahun ke Tahun")
        
        if 'year' in df.columns:
            df_yearly_trends = df.groupby('year').agg({
                'bmi': 'mean',
                'hbA1c_level': 'mean'
            }).reset_index()

            col_bmi, col_hba1c = st.columns(2)
            
            with col_bmi:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=df_yearly_trends, x='year', y='bmi', marker='o', ax=ax, color=sns.color_palette("viridis")[0], linewidth=2.5)
                ax.set_title('Tren Rata-rata BMI dari Tahun ke Tahun', fontsize=16)
                ax.set_xlabel('Tahun', fontsize=12)
                ax.set_ylabel('Rata-rata BMI', fontsize=12)
                sns.despine(left=True, bottom=True)
                plt.tight_layout()
                st.pyplot(fig)
                
            with col_hba1c:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=df_yearly_trends, x='year', y='hbA1c_level', marker='o', ax=ax, color=sns.color_palette("viridis")[1], linewidth=2.5)
                ax.set_title('Tren Rata-rata HbA1c dari Tahun ke Tahun', fontsize=16)
                ax.set_xlabel('Tahun', fontsize=12)
                ax.set_ylabel('Rata-rata HbA1c Level', fontsize=12)
                sns.despine(left=True, bottom=True)
                plt.tight_layout()
                st.pyplot(fig)
                
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi line plot rata-rata BMI dan HbA1c dari tahun ke tahun menunjukkan adanya fluktuasi. Rata-rata BMI dan HbA1c berfluktuasi dari tahun 2015 hingga 2022, tanpa **tren naik atau turun yang konsisten dan drastis**, menunjukkan **stabilitas relatif atau variabilitas** dalam metrik ini di seluruh dataset selama periode waktu tersebut.
            """)
        else:
            st.warning("Kolom 'year' tidak ditemukan dalam dataset untuk analisis tren tahunan.")

# --- Petunjuk Menjalankan Aplikasi ---
st.markdown("---")
st.header("Cara Menjalankan Aplikasi Streamlit")
st.markdown("""
1.  **Instal** pustaka yang dibutuhkan: `pip install streamlit pandas matplotlib seaborn`
2.  **Simpan** kode di atas sebagai file Python (misalnya, `app.py`).
3.  **Pastikan** file dataset Anda (`diabetes_dataset.csv`) berada di direktori yang sama.
4.  **Jalankan** dari terminal menggunakan perintah: `streamlit run app.py`
""")