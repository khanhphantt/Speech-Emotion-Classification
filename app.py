import os
import time
from datetime import datetime

import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from melspec import plot_colored_polar
from record_audio import sound

# load models
model = load_model("model3.h5")

# constants
starttime = datetime.now()
TEST_FILE = "examples//03-01-02-02-02-01-03.wav"

CAT6 = ["fear", "angry", "neutral", "happy", "sad", "surprise"]
CAT7 = ["fear", "disgust", "neutral", "happy", "sad", "surprise", "angry"]
CAT3 = ["positive", "neutral", "negative"]

COLOR_DICT = {
    "neutral": "grey",
    "positive": "green",
    "happy": "green",
    "surprise": "orange",
    "fear": "purple",
    "negative": "red",
    "angry": "red",
    "sad": "lightblue",
    "disgust": "brown",
}

TEST_CAT = ["fear", "disgust", "neutral", "happy", "sad", "surprise", "angry"]
TEST_PRED = np.array([0.3, 0.3, 0.4, 0.1, 0.6, 0.9, 0.1])

# page settings
st.set_page_config(
    page_title="SER web-app",
    page_icon=":speech_balloon:",
    layout="wide",
)

max_width = 1500
padding_top = 0
padding_right = "5%"
padding_left = "5%"
padding_bottom = 0
COLOR = "#1f1f2e"
BACKGROUND_COLOR = "#d1d1e0"
STYLE = f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def log_file(txt=None):
    with open("log.txt", "a") as f:
        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"{txt} - {datetoday};\n")


# @st.cache
def save_audio(file):
    if file.size > 4000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except FileNotFoundError:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


# @st.cache
def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb)


# @st.cache
def get_mfccs(audio, limit):
    y, sr = librosa.load(audio)
    a = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit))
        mfccs[:, : a.shape[1]] = a
    return mfccs


@st.cache
def get_title(predictions, categories=CAT6):
    title = f"Detected emotion: {categories[predictions.argmax()]} \
    - {predictions.max() * 100:.2f}%"
    return title


@st.cache
def color_dict(coldict=COLOR_DICT):
    return COLOR_DICT


@st.cache
def plot_polar(
    fig,
    predictions=TEST_PRED,
    categories=TEST_CAT,
    title="TEST",
    colors=COLOR_DICT,
):
    # color_sector = "grey"

    N = len(predictions)
    ind = predictions.argmax()

    COLOR = color_sector = colors[categories[ind]]
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = np.zeros_like(predictions)
    radii[predictions.argmax()] = predictions.max() * 10
    width = np.pi / 1.8 * predictions
    fig.set_facecolor("#d1d1e0")
    ax = plt.subplot(111, polar="True")
    ax.bar(
        theta,
        radii,
        width=width,
        bottom=0.0,
        color=color_sector,
        alpha=0.25,
    )

    angles = [i / float(N) * 2 * np.pi for i in range(N)]
    angles += angles[:1]

    data = list(predictions)
    data += data[:1]
    plt.polar(angles, data, color=COLOR, linewidth=2)
    plt.fill(angles, data, facecolor=COLOR, alpha=0.25)

    ax.spines["polar"].set_color("lightgrey")
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], color="grey", size=8)
    plt.suptitle(title, color="darkblue", size=12)
    plt.title(f"BIG {N}\n", color=COLOR)
    plt.ylim(0, 1)
    plt.subplots_adjust(top=0.75)


def main():
    side_img = Image.open("images/emotion3.jpg")
    with st.sidebar:
        st.image(side_img, width=300)
    st.sidebar.subheader("Menu")
    website_menu = st.sidebar.selectbox(
        "Detect emotion from:",
        (
            "An audio file",
            "Your recording",
        ),
    )
    st.set_option("deprecation.showfileUploaderEncoding", False)

    if website_menu == "An audio file":
        em3 = em6 = em7 = gender = False
        st.sidebar.subheader("Settings")

        st.markdown("## Upload the file")
        with st.container():
            upload_box, wave_form = st.columns(2)

            with upload_box:
                audio_file = st.file_uploader(
                    "Upload audio file",
                    type=["wav", "mp3", "ogg"],
                )
                if audio_file is not None:
                    if not os.path.exists("audio"):
                        os.makedirs("audio")
                    path = os.path.join("audio", audio_file.name)
                    if_save_audio = save_audio(audio_file)
                    if if_save_audio == 1:
                        st.warning("File size is too large. Try another file.")
                    elif if_save_audio == 0:
                        st.audio(audio_file, format="audio/wav", start_time=0)
                        try:
                            wav, sr = librosa.load(path, sr=44100)
                            Xdb = get_melspec(path)[1]
                            mfccs = librosa.feature.mfcc(wav, sr=sr)

                        except Exception as e:
                            audio_file = None
                            st.error(
                                f"Error {e} - wrong format of the file.",
                            )
                    else:
                        st.error("Unknown error")
                else:
                    if st.button("Try test file"):
                        wav, sr = librosa.load(TEST_FILE, sr=44100)
                        Xdb = get_melspec(TEST_FILE)[1]
                        mfccs = librosa.feature.mfcc(wav, sr=sr)
                        # display audio
                        st.audio(TEST_FILE, format="audio/wav", start_time=0)
                        path = TEST_FILE
                        audio_file = "test"
            with wave_form:
                if audio_file is not None:
                    display_wave_form(wav)
                else:
                    pass

        # model_type = "mfccs"
        em3 = st.sidebar.checkbox("3 emotions", True)
        em6 = st.sidebar.checkbox("6 emotions (recommended)", True)
        em7 = st.sidebar.checkbox("7 emotions")
        gender = st.sidebar.checkbox("gender", True)

        if audio_file is not None:
            st.markdown("## Analyzing...")
            if not audio_file == "test":
                st.sidebar.subheader("Audio file")
                file_details = {
                    "Filename": audio_file.name,
                    "FileSize": audio_file.size,
                }
                st.sidebar.write(file_details)
            with st.container():
                mfccs_features, mel_log_features = st.columns(2)
                with mfccs_features:
                    display_mfccs_features(mfccs, sr)

                with mel_log_features:
                    display_mel_log_features(Xdb, sr)

            st.markdown("## Predictions")
            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                mfccs = get_mfccs(path, model.input_shape[-1])
                mfccs = mfccs.reshape(1, *mfccs.shape)
                pred = model.predict(mfccs)[0]

                with col1:
                    if em3:
                        process_em3(pred)
                with col2:
                    if em6:
                        process_em6(pred)
                with col3:
                    if em7:
                        process_em7(pred, path)
                with col4:
                    if gender:
                        process_gender(path)

    elif website_menu == "Your recording":
        st.write("Record audio file")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path = "examples/" + timestr + ".wav"
        if st.button("Record 3 secs"):
            with st.spinner("Recording for 3 seconds ...."):
                sound.record(path)
            st.success("Recording completed")

        em3 = em6 = em7 = gender = False
        st.sidebar.subheader("Settings")
        # model_type = "mfccs"
        em3 = st.sidebar.checkbox("3 emotions", True)
        em6 = st.sidebar.checkbox("6 emotions (recommended)", True)
        em7 = st.sidebar.checkbox("7 emotions")
        gender = st.sidebar.checkbox("gender", True)

        audio_file = None
        try:
            audio_file = open(path, "rb")
        except FileNotFoundError:
            st.write("Please record sound first")

        if audio_file is not None:
            with st.container():
                recorded_file, wave_form = st.columns(2)

                with recorded_file:
                    wav, sr = librosa.load(path, sr=44100)
                    Xdb = get_melspec(path)[1]
                    mfccs = librosa.feature.mfcc(wav, sr=sr)
                    # display audio
                    st.audio(path, format="audio/wav", start_time=0)
                with wave_form:
                    display_wave_form(wav)

            st.markdown("## Analyzing...")
            st.sidebar.subheader("Audio file")
            file_details = {
                "Filename": audio_file.name,
                # "FileSize": audio_file.size,
            }
            st.sidebar.write(file_details)
            with st.container():
                mfccs_features, mel_log_features = st.columns(2)
                with mfccs_features:
                    display_mfccs_features(mfccs, sr)

                with mel_log_features:
                    display_mel_log_features(Xdb, sr)

            st.markdown("## Predictions")
            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                mfccs = get_mfccs(path, model.input_shape[-1])
                mfccs = mfccs.reshape(1, *mfccs.shape)
                pred = model.predict(mfccs)[0]

            with col1:
                if em3:
                    process_em3(pred)
            with col2:
                if em6:
                    process_em6(pred)
            with col3:
                if em7:
                    process_em7(pred, path)
            with col4:
                if gender:
                    process_gender(path)


def display_wave_form(wav):
    fig = plt.figure(figsize=(10, 2))
    fig.set_facecolor("#d1d1e0")
    plt.title("Wave-form")
    librosa.display.waveplot(wav, sr=44100)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.spines["right"].set_visible(False)
    plt.gca().axes.spines["left"].set_visible(False)
    plt.gca().axes.spines["top"].set_visible(False)
    plt.gca().axes.spines["bottom"].set_visible(False)
    plt.gca().axes.set_facecolor("#d1d1e0")
    st.write(fig)


def display_mfccs_features(mfccs, sr):
    fig = plt.figure(figsize=(10, 2))
    fig.set_facecolor("#d1d1e0")
    plt.title("MFCCs")
    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.spines["right"].set_visible(False)
    plt.gca().axes.spines["left"].set_visible(False)
    plt.gca().axes.spines["top"].set_visible(False)
    st.write(fig)


def display_mel_log_features(Xdb, sr):
    fig2 = plt.figure(figsize=(10, 2))
    fig2.set_facecolor("#d1d1e0")
    plt.title("Mel-log-spectrogram")
    librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.spines["right"].set_visible(False)
    plt.gca().axes.spines["left"].set_visible(False)
    plt.gca().axes.spines["top"].set_visible(False)
    st.write(fig2)


def process_em3(pred):
    pos = pred[3] + pred[5] * 0.5
    neu = pred[2] + pred[5] * 0.5 + pred[4] * 0.5
    neg = pred[0] + pred[1] + pred[4] * 0.5
    data3 = np.array([pos, neu, neg])
    txt = "MFCCs\n" + get_title(data3, CAT3)
    fig = plt.figure(figsize=(5, 5))
    COLORS = color_dict(COLOR_DICT)
    plot_colored_polar(
        fig,
        predictions=data3,
        categories=CAT3,
        title=txt,
        colors=COLORS,
    )
    st.write(fig)


def process_em6(pred):
    txt = "MFCCs\n" + get_title(pred, CAT6)
    fig2 = plt.figure(figsize=(5, 5))
    COLORS = color_dict(COLOR_DICT)
    plot_colored_polar(
        fig2,
        predictions=pred,
        categories=CAT6,
        title=txt,
        colors=COLORS,
    )
    st.write(fig2)


def process_em7(pred, path):
    model_ = load_model("model4.h5")
    mfccs_ = get_mfccs(path, model_.input_shape[-2])
    mfccs_ = mfccs_.T.reshape(1, *mfccs_.T.shape)
    pred_ = model_.predict(mfccs_)[0]
    txt = "MFCCs\n" + get_title(pred_, CAT7)
    fig3 = plt.figure(figsize=(5, 5))
    COLORS = color_dict(COLOR_DICT)
    plot_colored_polar(
        fig3,
        predictions=pred_,
        categories=CAT7,
        title=txt,
        colors=COLORS,
    )

    st.write(fig3)


def process_gender(path):
    with st.spinner("Wait for it..."):
        gmodel = load_model("model_mw.h5")
        gmfccs = get_mfccs(path, gmodel.input_shape[-1])
        gmfccs = gmfccs.reshape(1, *gmfccs.shape)
        gpred = gmodel.predict(gmfccs)[0]
        gdict = [["female", "woman.png"], ["male", "man.png"]]
        ind = gpred.argmax()
        txt = "Predicted gender: " + gdict[ind][0]
        img = Image.open("images/" + gdict[ind][1])

        fig4 = plt.figure(figsize=(3, 3))
        fig4.set_facecolor("#d1d1e0")
        plt.title(txt)
        plt.imshow(img)
        plt.axis("off")
        st.write(fig4)


if __name__ == "__main__":
    main()
