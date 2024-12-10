import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import os

# Set page configuration
st.set_page_config(page_title="ગુજરાતી વ્યાકરણ", layout="wide")

# Add custom gradient background with new colors
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #3b82f6, #fbcfe8);
        background-size: auto;
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.8) !important;
        color: black !important;
        border: 1px solid #ffffff !important;
    }

    /* Button styling */
    .stButton button {
        background-color: #165BAA !important;
        color: black !important;
        border-radius: 10px;
        font-weight: bold;
    }

    /* Title and text color */
    h1, h2, h3, h4, h5, h5, p {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and preprocess dataset
@st.cache_data
def load_data():
    data = pd.read_excel("Samas_Final.xlsx")
    data.columns = data.columns.str.strip()
    data['Sentence'] = data[['Word', 'sangna1', 'Middle', 'sangna2']].apply(lambda x: ' '.join(x), axis=1)
    return data

@st.cache_data
def preprocess_data(data):
    X = data['Sentence']
    y = data['Label']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X_sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_sequences, padding='post')

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_padded, y_encoded, tokenizer, label_encoder

# Save and load model to avoid retraining
@st.cache_resource
def load_or_train_model():
    model_path = "bilstm_samas_model.h5"
    if not os.path.exists(model_path):
        data = load_data()
        X_padded, y_encoded, tokenizer, label_encoder = preprocess_data(data)

        X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

        class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
        class_weights_dict = dict(enumerate(class_weights))

        model = Sequential([
            Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X_padded.shape[1]),
            Bidirectional(LSTM(150, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(100)),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(len(np.unique(y_encoded)), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=16,
            validation_split=0.2,
            class_weight=class_weights_dict,
            callbacks=[early_stopping]
        )
        model.save(model_path)
    else:
        model = load_model(model_path)
    return model

# Prediction logic
def predict_compound(model, tokenizer, label_encoder, sentence, data, max_len):
    words = sentence.split()
    compound_word, meaning, samas_type = None, None, None
    for word in words:
        try:
            sequence = tokenizer.texts_to_sequences([word])
            padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
            prediction = model.predict(padded_sequence, verbose=0)
            predicted_label_index = np.argmax(prediction, axis=1)[0]
            predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

            if predicted_label in ['D', 'T', 'M']:
                row = data[data['Word'] == word]
                if not row.empty:
                    compound_word = word
                    meaning = f"{row.iloc[0]['sangna1']} {row.iloc[0]['Middle']} {row.iloc[0]['sangna2']}"
                    samas_type = {
                        'D': "દ્વંદ્વ સમાસ",
                        'T': "તત્પુરૂષ સમાસ",
                        'M': "મધ્યમપદલોપી સમાસ"
                    }[predicted_label]
                    break
        except Exception:
            continue
    return compound_word, meaning, samas_type

# Load and preprocess data
data = load_data()
X_padded, y_encoded, tokenizer, label_encoder = preprocess_data(data)
model = load_or_train_model()

# Streamlit UI
st.title("ગુજરાતી વ્યાકરણ")
st.subheader("સમાસ ચકાસો")

user_input = st.text_input("વાક્ય દાખલ કરો:")
if st.button("ચકાસો"):
    compound_word, meaning, samas_type = predict_compound(model, tokenizer, label_encoder, user_input, data, max_len=X_padded.shape[1])
    if compound_word:
        st.write(f"**સંયુક્ત શબ્દ**: {compound_word}")
        st.write(f"**અર્થ**: {meaning}")
        st.write(f"**સમાસ પ્રકાર**: {samas_type}")
    else:
        st.write("કોઈ સંયુક્ત શબ્દ શોધાઈ નથી.")
