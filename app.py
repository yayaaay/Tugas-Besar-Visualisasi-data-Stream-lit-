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
st.title("🚀 Visualisasi Data Diabetes Interaktif")

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

    # --- Penjelasan Dataset Sebelum EDA ---
    st.header("📘 Tentang Dataset")

    st.markdown("""
    Dataset ini berisi data kesehatan dan demografi dari 100.000 individu yang dikumpulkan untuk mendukung penelitian serta pemodelan prediktif terkait penyakit diabetes. Setiap entri mencakup berbagai informasi penting seperti jenis kelamin, usia, lokasi, ras, riwayat hipertensi dan penyakit jantung, riwayat merokok, indeks massa tubuh (BMI), kadar HbA1c, dan kadar glukosa darah, serta status diabetes seseorang (apakah menderita atau tidak). Data ini memungkinkan analisis hubungan antara faktor gaya hidup dan kondisi medis terhadap risiko terjadinya diabetes, serta menjadi dasar dalam pengembangan model prediksi kesehatan berbasis data.
    """)

    # --- Bagian Visualisasi EDA Awal ---
    st.header("🔍 Visualisasi Data Awal (EDA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Usia ")
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
        st.subheader("Distribusi Jenis Kelamin ")
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
        "Kasus 9: Komorbiditas", "Kasus 10: Tren Tahunan", "Kasus 11: Distribusi Status Diabetes" 
    ]
    tabs = st.tabs(tab_titles)
    
    # --- Study Case 1: Usia ---
    with tabs[0]:
        col_vis, col_text = st.columns([2, 1])
        with col_vis:
            st.subheader("Studi Kasus 1: Distribusi Kasus Diabetes di Berbagai Kelompok Usia")
            fig, ax = plt.subplots(figsize=(10, 6))

            # Filter dataframe untuk hanya menyertakan kasus diabetes (diabetes = 1)
            df_diabetes = df[df['diabetes'] == 1]

            # plot utama - hanya kasus diabetes
            sns.histplot(
                data=df_diabetes, # Menggunakan dataframe yang sudah difilter
                x='age', 
                kde=True, # Biarkan kde=True untuk kurva kepadatan
                ax=ax, 
                bins=25, 
                color='red', # Menggunakan satu warna karena hanya ada satu kategori
                legend=False 
            )

            ax.set_title('Distribusi Usia Individu dengan Status Diabetes', fontsize=16)
            ax.set_xlabel('Usia', fontsize=12)
            ax.set_ylabel('Frekuensi', fontsize=12)

            # Hapus legend manual karena hanya ada satu kategori yang ditampilkan
            # from matplotlib.patches import Patch
            # legend_labels = ['Diabetes']
            # legend_colors = ['red'] 
            # patches = [Patch(color=legend_colors[0], label=legend_labels[0])]
            # ax.legend(handles=patches, title='Status Diabetes', loc='upper left')

            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig)

        with col_text:
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi histogram yang telah dimodifikasi ini **secara eksklusif menampilkan distribusi usia** dari individu yang didiagnosis **diabetes**.

            Visualisasi menunjukkan konsentrasi tertinggi kasus diabetes berada pada kelompok usia yang lebih tua, mengonfirmasi kembali bahwa **usia lanjut adalah faktor risiko yang dominan** dalam dataset ini. Kurva kepadatan (KDE) yang tumpang tindih memberikan estimasi bentuk distribusi probabilitas usia untuk populasi penderita diabetes.
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
        Visualisasi untuk Studi Kasus 2 yang menampilkan prevalensi hipertensi dan penyakit jantung berdasarkan gender menunjukkan bahwa dalam dataset ini, individu pria memiliki jumlah kasus hipertensi yang secara signifikan lebih tinggi dan juga jumlah kasus penyakit jantung yang lebih banyak dibandingkan dengan individu wanita; ini mengindikasikan adanya korelasi yang jelas antara gender pria dan peningkatan prevalensi kedua kondisi kesehatan ini dalam data yang diamati.
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
            Visualisasi Menunjukkan distribusi kasus diabetes di berbagai lokasi (negara bagian). Dari plot ini, jelas terlihat bahwa jumlah kasus diabetes sangat bervariasi di setiap lokasi, menunjukkan adanya pola geografis dalam prevalensi diabetes. Berdasarkan analisis data yang telah dilakukan, Delaware adalah lokasi dengan jumlah kasus diabetes terbanyak, yaitu sebanyak 200 kasus. Ini mengindikasikan bahwa prevalensi diabetes paling tinggi dalam dataset ini ditemukan di Delaware.
            """)

    # --- Study Case 4: Ras ---
    with tabs[3]:
        col_vis, col_text = st.columns([2, 1])
        with col_vis:
            st.subheader("Studi Kasus 4: Korelasi Ras dengan Prevalensi Diabetes")

            if not df_race.empty:
                # Hitung jumlah kasus per ras dan status diabetes
                race_counts = df_race.groupby(['race', 'diabetes']).size().reset_index(name='count')

                # Pastikan urutan ras tetap konsisten
                races = race_counts['race'].unique()

                # Buat subplot untuk tiap ras
                fig, axes = plt.subplots(
                    nrows=2, ncols=3, figsize=(12, 8),
                    subplot_kw=dict(aspect='equal')
                )
                axes = axes.flatten()

                colors = sns.color_palette('viridis', n_colors=2)
                labels = ['Tidak Diabetes', 'Diabetes']

                for i, race in enumerate(races):
                    data = race_counts[race_counts['race'] == race]
                    values = data['count'].values

                    if len(values) < 2:
                        # Jika data tidak lengkap (misal hanya 1 kategori)
                        values = [values[0], 0]

                    axes[i].pie(
                        values,
                        labels=labels,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors,
                        textprops={'fontsize': 10}
                    )
                    axes[i].set_title(race, fontsize=12)

                # Hapus subplot kosong kalau jumlah ras < jumlah subplot
                for j in range(len(races), len(axes)):
                    fig.delaxes(axes[j])

                fig.suptitle('Proporsi Diabetes vs Tidak Diabetes per Ras', fontsize=16)
                plt.tight_layout()
                st.pyplot(fig)

            else:
                st.warning("Data ras tidak tersedia atau kosong setelah pemrosesan.")

        with col_text:
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi ini menampilkan diagram lingkaran (pie chart) untuk masing-masing ras,
            membandingkan proporsi individu dengan diabetes dan tanpa diabetes.
            Setiap lingkaran merepresentasikan satu kelompok ras:
            - Warna biru mewakili individu tanpa diabetes  
            - Warna hijau menunjukkan individu dengan diabetes  

            Dari grafik terlihat bahwa proporsi penderita diabetes relatif kecil (<10%) di semua ras,
            dengan prevalensi tertinggi pada AfricanAmerican (8.74%) dan terendah pada Other (8.21%).
            Hal ini menunjukkan bahwa meskipun ada perbedaan antar ras, tingkat prevalensinya tetap cukup seragam.
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
            Visualisasi menunjukkan distribusi riwayat merokok berdasarkan status diabetes. Mayoritas individu, baik penderita maupun non-penderita diabetes, berada pada kategori never dan No Info. Sementara itu, kategori current dan former memiliki jumlah jauh lebih sedikit. Secara keseluruhan, perbedaan antar kategori kecil, sehingga riwayat merokok tidak tampak berpengaruh signifikan terhadap kejadian diabetes.
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
            Visualisasi box plot terlihat bahwa individu dengan diabetes cenderung memiliki BMI lebih tinggi dibandingkan dengan individu yang tidak menderita diabetes. Median BMI pada kelompok penderita diabetes juga lebih besar, menunjukkan bahwa berat badan berlebih atau obesitas mungkin berhubungan dengan meningkatnya risiko diabetes. Selain itu, kedua kelompok memiliki beberapa outlier dengan nilai BMI sangat tinggi, namun pola umumnya tetap menunjukkan bahwa semakin tinggi BMI, semakin besar kemungkinan seseorang memiliki diabetes.
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
            Visualisasi box plot tingkat HbA1c menunjukkan individu dengan diabetes memiliki nilai HbA1c yang secara signifikan lebih tinggi dibandingkan individu tanpa diabetes. Median HbA1c pada kelompok penderita diabetes berada di sekitar 6.5–7, sedangkan pada kelompok tanpa diabetes berada di bawah 6. Hal ini menunjukkan adanya perbedaan yang jelas antara kedua kelompok, di mana kenaikan kadar HbA1c berhubungan kuat dengan adanya diabetes.
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
            Visualisasi box plot tingkat glukosa darah berdasarkan kelompok usia menunjukkan bahwa rata-rata kadar glukosa darah cenderung meningkat seiring bertambahnya usia, terutama mulai pada kelompok umur 60–74 tahun. Meskipun median kadar glukosa antar kelompok tidak berbeda terlalu jauh, kelompok usia yang lebih tua menunjukkan peningkatan nilai minimum dan median serta lebih banyak outlier dengan kadar glukosa tinggi. Hal ini mengindikasikan bahwa risiko kadar gula darah tinggi (hiperglikemia) lebih umum terjadi pada usia lanjut dibandingkan usia muda.
            """)

    # --- Study Case 9: Komorbiditas ---
    import numpy as np
    with tabs[8]:
        col_vis, col_text = st.columns([2, 1])
        
        with col_vis:
            st.subheader("Studi Kasus 9: Analisis Komorbiditas pada Penderita Diabetes")

            # Filter hanya penderita diabetes
            diab = df[df['diabetes'] == 1]

            # Data untuk radial bar
            labels = ['Hipertensi', 'Penyakit Jantung', 'Perokok Aktif']
            sizes = [
                diab['hypertension'].mean() * 100,
                diab['heart_disease'].mean() * 100,
                diab['smoking_history'].value_counts(normalize=True).get('current', 0) * 100
            ]

            # Setup polar chart
            angles = np.linspace(0, 2 * np.pi, len(sizes), endpoint=False)
            fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(7, 6))
            bars = ax.bar(
                angles,
                sizes,
                width=0.8,
                color=sns.color_palette("viridis", len(sizes)),
                alpha=0.75,
                edgecolor='black'
            )

            # Label & styling
            ax.set_xticks(angles)
            ax.set_xticklabels(labels, fontsize=11)
            ax.set_yticklabels([])
            ax.set_title("Radial Bar Komorbiditas pada Penderita Diabetes", fontsize=15, pad=20)
            ax.set_ylim(0, max(sizes) + 10)

            # Tambahkan nilai persentase di atas bar
            for angle, size in zip(angles, sizes):
                ax.text(angle, size + 3, f"{size:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)

        with col_text:
            st.subheader("Penjelasan")
            st.markdown("""
            Visualisasi ini menggunakan *radial bar chart* untuk menggambarkan tiga komorbiditas utama pada individu penderita diabetes:

            - *Hipertensi:* ~{:.1f}% dari penderita diabetes juga memiliki hipertensi  
            - *Penyakit Jantung:* ~{:.1f}% memiliki penyakit jantung  
            - *Perokok Aktif:* ~{:.1f}% memiliki riwayat merokok aktif  

            Grafik ini menyoroti bagaimana ketiga faktor tersebut saling beririsan dengan kondisi diabetes. Terlihat bahwa hipertensi menjadi komorbiditas paling umum, diikuti oleh penyakit jantung, sedangkan proporsi perokok aktif relatif lebih kecil.
            """.format(sizes[0], sizes[1], sizes[2]))
            
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
            Visualisasi line plot rata-rata BMI dan HbA1c dari tahun ke tahun menunjukkan tren rata-rata BMI dan HbA1c level dari tahun 2015 hingga 2022. Grafik pertama memperlihatkan bahwa rata-rata BMI cenderung fluktuatif, mengalami sedikit peningkatan hingga 2021 sebelum menurun tajam pada 2022. Sementara itu, grafik kedua menunjukkan bahwa rata-rata HbA1c level relatif stabil dari 2015 sampai 2019, kemudian menurun pada 2020–2021, dan meningkat cukup signifikan pada 2022. Secara keseluruhan, kedua grafik memberikan gambaran mengenai perubahan pola kesehatan (berdasarkan BMI dan kadar gula darah) dari waktu ke waktu.
            """)
        else:
            st.warning("Kolom 'year' tidak ditemukan dalam dataset untuk analisis tren tahunan.")
    with tabs[10]:
        st.subheader("Studi Kasus 11: Hubungan BMI, Usia, dan Gula Darah")

        import plotly.express as px

        # Ambil sample agar render tidak terlalu berat
        df_sample = df.sample(3000, random_state=42)

        # Scatter Bubble Chart
        fig = px.scatter(
            df_sample,
            x="age",
            y="bmi",
            color="diabetes",
            size="blood_glucose_level",
            hover_data=["hbA1c_level"],
            color_discrete_sequence=["#5dade2", "#fd7e14"],
            labels={
                'diabetes': 'Status Diabetes',
                'age': 'Usia',
                'bmi': 'BMI'
            },
            title="Bubble Chart: Hubungan Usia, BMI, dan Kadar Gula Darah"
        )

        # Penyesuaian ukuran bubble & style
        fig.update_traces(
            marker=dict(
                sizeref=2. * df_sample['blood_glucose_level'].max() / (80. ** 2),
                sizemode='area',
                opacity=0.65,
                line=dict(width=1, color='DarkSlateGrey')
            )
        )

        # Layout
        fig.update_layout(
            legend_title_text='Status Diabetes',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        # Tampilkan chart di Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Penjelasan
        st.markdown("""
        *Interpretasi:*

        Visualisasi ini menampilkan hubungan antara *usia, **BMI (Body Mass Index), dan **kadar gula darah* pada sampel 3.000 individu.  
        Setiap titik mewakili satu individu:
        - *Sumbu X* menunjukkan usia pasien.  
        - *Sumbu Y* menunjukkan nilai BMI.  
        - *Ukuran lingkaran* merepresentasikan kadar *gula darah (blood_glucose_level)*.  
        - *Warna* menandakan status diabetes (biru = non-diabetes, oranye = diabetes).  

        Terlihat bahwa pasien dengan *usia lebih tua* dan *BMI lebih tinggi* cenderung memiliki *kadar gula darah yang lebih besar* serta lebih banyak berada pada kategori *penderita diabetes*.  
        Hal ini menunjukkan adanya hubungan yang kuat antara *faktor usia, obesitas, dan kadar gula darah* dalam menentukan risiko diabetes.
        """)
