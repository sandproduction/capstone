import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from googleapiclient.discovery import build
from deep_translator import GoogleTranslator

from pytube import YouTube

# ================== PREDICT SINGLE KOMEN ======================
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stpwds_id = list(set(stopwords.words('indonesian')))
stpwds_id.append('oh')

def text_proses(teks):
    if teks is None or not isinstance(teks, str):  # Periksa jika teks None atau bukan string
        teks = ""  # Berikan nilai default berupa string kosong
    teks = re.sub("@[A-Za-z0-9_]+"," ", teks)  # Menghilangkan mention
    teks = re.sub("#[A-Za-z0-9_]+"," ", teks)  # Menghilangkan hashtag
    teks = re.sub(r"\\n"," ",teks)             # Menghilangkan \n
    teks = teks.strip()                        # Menghilangkan whitespace
    teks = re.sub(r"http\S+", " ", teks)       # Menghilangkan link http
    teks = re.sub(r"www.\S+", " ", teks)       # Menghilangkan link www
    teks = re.sub("[^A-Za-z\s']"," ", teks)    # Menghilangkan yang bukan huruf
    tokens = word_tokenize(teks)               # Tokenisasi teks
    teks = ' '.join([word for word in tokens if word not in stpwds_id])  # Stopwords
    teks = stemmer.stem(teks)                  # Stemming
    return teks

def predict(processed_text):
    import requests

    API_KEY = ""
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token',data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    # Nama-nama fitur (opsional)
    fields = [f"feature_{i}" for i in range(len(processed_text[0]))]

    payload_scoring = {
        "input_data": [
            {
                "fields": fields, 
                "values": processed_text  # Data hasil vectorization
            }
        ]
    }

    # Kirim request ke endpoint IBM Watson ML
    response_scoring = requests.post(
        '',
        json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken}
    )

    return response_scoring

# Muat kembali TextVectorization layer
# @st.cache_resource  # Cache agar tidak perlu dimuat ulang setiap kali

def text_preprocessing(comment):
    loaded_model = tf.keras.models.load_model("vektorisasi_test_fix.keras")
    loaded_text_vectorization = loaded_model.layers[0]

    processed_text = text_proses(comment)
    processed_text_input = np.array([processed_text])
    vectorized_text = loaded_text_vectorization(processed_text_input)
    text_vectorized_list = vectorized_text.numpy().tolist()  # Konversi tensor ke NumPy, lalu ke list
    return text_vectorized_list

def print_result(response_scoring):
    label_mapping = {0: "non-bullying", 1: "bullying"}
    # Ekstrak prediksi
    predictions = response_scoring.json()['predictions'][0]['values']
    predicted_class = predictions[0][1]  # Kelas prediksi
    probability = predictions[0][0]  # Probabilitas masing-masing kelas

    # Pemetaan ke label deskriptif
    predicted_label = label_mapping[predicted_class]

    # Output lebih mudah dibaca
   # Tampilan Utama
    st.markdown("""
        <h1 style='text-align:center;'><b>Hasil Prediksi</b></h1>
    """, unsafe_allow_html=True)
    # st.markdown("---")
    if predicted_label == "bullying":
        st.markdown("""
            <h2 style='text-align:center;color:white;background-color:#d60000;'><b>BULLYING</b></h2>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <h2 style='text-align:center;color:white;background-color:#29c100;'><b>NON-BULLYING</b></h2>
        """, unsafe_allow_html=True)
        st.success("✅ Kalimat ini *bukan termasuk bullying*. Terima kasih telah menggunakan kata-kata positif! 😊")

    st.markdown(f"""
        <style>
            .progress-container {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-direction:column;
                width: 100%;
                margin: 10px 0px;
            }}
            .progress-bar {{
                display:flex;
                height: 30px;
                width: 100%;
                border-radius: 5px;
               
            }}
            .bar-fill-bully {{
                width: {probability[1]:.2%};
                background-color: #d60000;
                height: 100%;
                border-radius: 5px 0 0 5px;
                animation: bar-bully 2s ease-in-out;
        
            }}
            .bar-fill-nonbully {{
                width: {probability[0]:.2%};
                background-color: #29c100;
                height: 100%;
                border-radius: 0 5px 5px 0;
                animation: bar-nonbully 2s ease-in-out;
               
            }}
            .label {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-weight: bold;
                text-align: center;
                width: 90%;
            }}
            .label p{{
                display:flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }}

            @keyframe bar-bully{{
                from{{
                    width:0;
                }}
                to{{
                    width:{probability[1]:.2%};
                }}
            }}
            @keyframe bar-nonbully{{
                from{{
                    width:0;
                }}
                to{{
                    width:{probability[0]:.2%};
                }}
            }}
        </style>

        <div class="progress-container">
            <div class="label">
                <p><span>Bullying</span> <span style='color:red;'>{probability[1]:.2%}%</span></p>
                <p><span>Non-Bullying</span> <span style='color:green'>{probability[0]:.2%}%</span></p>
            </div>
            <div class="progress-bar">
                <div class="bar-fill-bully"></div>
                 <div class="bar-fill-nonbully"></div>
            </div>
           
        </div>
        """, unsafe_allow_html=True)

    if predicted_label == "bullying":
        feedback = ("🚨 *Nasihat:* Hindari menggunakan kata-kata negatif atau menyakitkan kepada orang lain. Bullying dapat merusak mental seseorang dan memiliki dampak jangka panjang.Sebagai gantinya, gunakan kata-kata yang membangun dan saling mendukung. 🌟")
        st.error(feedback)
        st.markdown("""
            💔 *Efek Membully:* Perilaku ini dapat mengakibatkan dampak serius, termasuk 
            menurunnya rasa percaya diri korban, gangguan emosional, atau bahkan isolasi sosial.
        """)
    else:
        st.success("✅ Kalimat ini *bukan termasuk bullying*. Terima kasih telah menggunakan kata-kata positif! 😊")

# ============= PREDICT KOMEN YT ===================

API_KEY = "AIzaSyDOOp3Bs4GXp2RQW5TyYvq4Ll6A7DJ0Y3Q"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

# Fungsi untuk mengambil komentar
def scrapping(video_id, max_results=100):
    comments = []
    next_page_token = None

    while True:
        # Panggil API untuk mengambil komentar
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            pageToken=next_page_token
        ).execute()

        # Ekstrak data komentar
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

            comments.append({
                "comment": comment,
            })

        # Cek apakah ada halaman berikutnya
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(comments)

def get_comments(id):
    video_id = re.search(r"v=([a-zA-Z0-9_-]+)", id)

    # Cek apakah match ditemukan
    if video_id:
        video_id = video_id.group(1)  # Ambil ID dari hasil regex
        st.write("ID Video YouTube:", video_id)
    else:
        st.write("Tidak ditemukan ID video pada URL.")

    comments_df = scrapping(video_id)

    # Simpan data ke CSV dengan delimiter yang sesuai
    comments_df.to_csv("youtube_comments.csv", index=False, encoding='utf-8', sep=',')

    # st.write("Data komentar berhasil disimpan ke 'youtube_comments.csv'")
    return comments_df

def process_data_comments(df):
    loaded_model = tf.keras.models.load_model("vektorisasi_test_fix.keras")
    loaded_text_vectorization = loaded_model.layers[0]

    # Preprocessing setiap komentar
    processed = df['comment'].apply(text_proses).tolist()
    df_clean = df
    df_clean['processed_text'] = processed

    # Batch input 2D (batch_size, sequence)
    batch_input = np.array(processed, dtype=object)
    vectorized_batch = loaded_text_vectorization(batch_input)
    vectorized_list = vectorized_batch.numpy().tolist()

    return [vectorized_list, df_clean]

def is_valid_youtube_url(url):
    youtube_regex = (
        r"(https?://)?(www\.)?"
        r"(youtube\.com|youtu\.?be)/.+$"
    )
    return re.match(youtube_regex, url) is not None

# ====================== TAMPILAN =============================

# ------ CSS -------

st.markdown(
    """
    <style>
    .centered-label {
        text-align: center;
    }
    .st-emotion-cache-1qg05tj{
        display:none;
    }

    </style>

    """,
    unsafe_allow_html=True
)

# --------------------

st.markdown("<h1 style='text-align:center'>SENTIMENT ANALISIS CYBERBULLYING</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'><b>By</b></p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Abyan Izzah | Ahmad Sappauni | Akmal Maulana | Rifqi Azmi | Sandika Suryananta</b></p>", unsafe_allow_html=True)

st.markdown("---")

with st.form("Single Predict"):
    st.markdown('<h3 class="centered-label">Masukkan Kalimat</h3>', unsafe_allow_html=True)
    comment = st.text_input("", key="single_predict")
    st.markdown('<p style="text-align:center"> kalimatnya jangan panjang-panjang ya maksimal 26 kata, dan juga model ini bisa menggunakan bahasa inggris tapi lebih bagus jika memakai bahasa indonesia karena model ini dilatih menggunakan bahasa indonesia</p>',unsafe_allow_html=True)
    state = st.form_submit_button("Predict")
    translator = GoogleTranslator(source='en', target='id')

    # Jika tombol ditekan dan input tidak kosong
    if state:
        if comment != '':
            comment_translate = translator.translate(comment)
            commentFix = comment_translate
            st.write(commentFix)
            processed_text=text_preprocessing(commentFix) 
            # st.write(processed_text)
            predict_text = predict(processed_text)
            print_result(predict_text)
        else:
            st.warning("Isi Kolom anda")

# Konversi hasil vektorisasi ke dalam list
with st.form("Link Predict"):

    # data = {'comments': ["hai cantik mau kemana hari ini sendiri aja", "muka kayak keset begitu sok-sokan pacaran ama bule", "selamat pagi bang keren banget konten lu lajutin gua suka bang", "anjing lu, mending uninstall aja tu youtube gk guna ada di tangan lu, merusak doang","gua suka video lu bang gameplay lu seru gak ngebosenin"]}
    # df = pd.DataFrame(data)
    st.markdown('<h3 class="centered-label">Masukkan Link Video Youtube</h3>', unsafe_allow_html=True)
    link = st.text_input("", key="link_predict")
    st.markdown('<p style="text-align:center">masukkan link dengan format "https://www.youtube.com/watch?v=VIDEO_ID"</p>', unsafe_allow_html=True)
    state_link = st.form_submit_button("Predict")
    translator = GoogleTranslator(source='en', target='id')

    # Jika tombol ditekan dan input tidak kosong
    if state_link:
        if link != '' and is_valid_youtube_url(link):
            label_mapping = {0: "non-bullying", 1: "bullying"}
            df=get_comments(link)

            df['comment'] = df['comment'].apply(lambda row: translator.translate(row) if isinstance(row, str) else row)

            data_jadi=process_data_comments(df)
            # st.dataframe(data)
            result=predict(data_jadi[0])
            data_view = data_jadi[1]

            predictions = result.json()['predictions'][0]['values']

            # Ambil nilai prediksi dan probabilitas
            pred_labels = [label_mapping[pred[1]] for pred in predictions]  # Kelas prediksi

            # Masukkan hasil ke DataFrame
            df['hasil'] = pred_labels
            
            st.markdown("<h2 style='text-align:center;background-color:#053d99;border-radius:20px;'>Hasil Predict</h2>",unsafe_allow_html=True)
            st.markdown('---')    
            link_yt_clean = link.split('&')[0]
            st.video(link_yt_clean)
            st.dataframe(data_view[['processed_text', 'hasil']])

            # ------------------------------ PRESENTASE --------------------------------

            persentase = df['hasil'].value_counts(normalize=True) * 100
            persentase_df = persentase.reset_index()
            persentase_df.columns = ['kategori', 'proportion']

             # Menampilkan hasil di Streamlit
            bullying_percentage = (
                persentase_df[persentase_df['kategori'] == 'bullying']['proportion'].values[0]
                if not persentase_df[persentase_df['kategori'] == 'bullying'].empty
                else 0
            )

            non_bullying_percentage = (
                persentase_df[persentase_df['kategori'] == 'non-bullying']['proportion'].values[0]
                if not persentase_df[persentase_df['kategori'] == 'non-bullying'].empty
                else 0
            )

            st.markdown(f"""
                <style>
                    .progress-container {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        flex-direction: column;
                        width: 100%;
                        margin: 10px 0px;
                    }}
                    .progress-bar {{
                        display: flex;
                        height: 30px;
                        width: 100%;
                        border-radius: 5px;
                    }}
                    .bar-fill-bully {{
                        width: {bullying_percentage}%;
                        background-color: #d60000;
                        height: 100%;
                        border-radius: 5px 0 0 5px;
                        animation: bar-bully 2s ease-in-out;
                    }}
                    .bar-fill-nonbully {{
                        width: {non_bullying_percentage}%;
                        background-color: #29c100;
                        height: 100%;
                        border-radius: 0 5px 5px 0;
                        animation: bar-nonbully 2s ease-in-out;
                    }}
                    .label {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        font-weight: bold;
                        text-align: center;
                        width: 90%;
                    }}
                    .label p {{
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        flex-direction: column;
                    }}
                </style>        
                
                <div class="progress-container">
                    <div class="label">
                        <p><span>Bullying</span> <span style='color:red;'>{bullying_percentage:.2f}%</span></p>
                        <p><span>Non-Bullying</span> <span style='color:green'>{non_bullying_percentage:.2f}%</span></p>
                    </div>
                    <div class="progress-bar">
                        <div class="bar-fill-bully"></div>
                        <div class="bar-fill-nonbully"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('---')

            # ------------------- BARPLOT ---------------------------

            st.markdown("<h3 style='text-align:center;background-color:#47484a;border-radius:10px;margin-bottom:10px;'>Barplot</h3>",unsafe_allow_html=True)
            
            grouped_data = df.groupby('hasil').size().reset_index(name='jumlah')

            fig = plt.figure(figsize=(8, 5))
            sns.barplot(data=grouped_data, x='hasil', y='jumlah', palette='Set2')
            plt.title('Distribusi Kategori Komentar')
            plt.xlabel('Kategori')
            plt.ylabel('Jumlah')

            # Menampilkan Grafik di Streamlit
            st.pyplot(fig)  
            st.markdown('---')

            # --------------------- WORDCLOUD ------------------------------

            # Menggabungkan komentar dengan default "none" jika kosong
            bullying_comments = ' '.join(df[df['hasil'] == 'bullying']['comment'].fillna('')) or 'none'
            non_bullying_comments = ' '.join(df[df['hasil'] == 'non-bullying']['comment'].fillna('')) or 'none'


            # Membuat Word Cloud
            bullying_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(bullying_comments)
            non_bullying_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_bullying_comments)
            
            col1,col2 = st.columns(2)

            # Menampilkan Word Cloud di Streamlit
            with col1:
                st.markdown("<h4 style='text-align:center;background-color:#9e0d08 ;border-radius:10px; margin-bottom:10px;'>Wordcloud Bullying</h4>",unsafe_allow_html=True)
                st.image(bullying_wordcloud.to_array())

            with col2:
                st.markdown("<h4 style='text-align:center;background-color:#04911c ;border-radius:10px; margin-bottom:10px;'>Wordcloud Non-Bullying</h4>",unsafe_allow_html=True)
                st.image(non_bullying_wordcloud.to_array())
        else:
            st.warning("Isi Kolom anda")
