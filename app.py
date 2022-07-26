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
model_gender = load_model("models/model_gender.h5")
model_mfcc_BIG3 = load_model("models/model_mfcc_BIG3.h5")
model_mfcc_BIG7 = load_model("models/model_mfcc_BIG7.h5")
model_mel = load_model("models/model_mel.h5")

# constants
starttime = datetime.now()
TEST_FILE = "examples//03-02-05-02-02-02-02.wav"

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
    st.markdown(
        '<h6 align="center">TT monthly challenge 2nd - July 2022</h6>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 align="center">Speech Emotion Analysis</h1>',
        unsafe_allow_html=True,
    )

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

        [mfccs_fea, mel_fea, gender] = check_box_setting()

        if audio_file is not None:
            display_analysis(audio_file, mfccs, Xdb, sr)
            display_prediction(path, mfccs_fea, mel_fea, gender)

    elif website_menu == "Your recording":
        st.write("Record audio file")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path = "examples/" + timestr + ".wav"
        if st.button("Record 3 secs"):
            with st.spinner("Recording for 3 seconds ...."):
                sound.record(path)
            st.success("Recording completed")

        [mfccs_fea, mel_fea, gender] = check_box_setting()

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

            display_analysis(audio_file, mfccs, Xdb, sr)
            display_prediction(path, mfccs_fea, mel_fea, gender)


def check_box_setting():
    mfccs_fea = mel_fea = gender = False
    st.sidebar.subheader("Select features:")
    mfccs_fea = st.sidebar.checkbox("MFCCs", True)
    mel_fea = st.sidebar.checkbox("Mel-Spectrogram (take a while)", True)
    gender = st.sidebar.checkbox("Gender", True)
    return [mfccs_fea, mel_fea, gender]


def display_prediction(path, mfccs_fea, mel_fea, gender):
    st.markdown("## Predictions")
    with st.container():
        if mfccs_fea:
            mfccs = get_mfccs(path, model_mfcc_BIG3.input_shape[-1])
            mfccs = mfccs.reshape(1, *mfccs.shape)
            pred = model_mfcc_BIG3.predict(mfccs)[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                process_em3(pred)
            with col2:
                process_em7(path)
            with col3:
                if gender:
                    process_gender(path)

            if mel_fea:
                col1, col2 = st.columns([2, 1])
                with col1:
                    process_mel_log(path)

        elif mel_fea:
            col1, col2 = st.columns([2, 1])
            with col1:
                process_mel_log(path)
            with col2:
                if gender:
                    process_gender(path)


def display_analysis(audio_file, mfccs, Xdb, sr):
    st.markdown("## Analyzing...")
    if not audio_file == "test":
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


def process_em7(path):
    mfccs_ = get_mfccs(path, model_mfcc_BIG7.input_shape[-2])
    mfccs_ = mfccs_.T.reshape(1, *mfccs_.T.shape)
    pred_ = model_mfcc_BIG7.predict(mfccs_)[0]
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


def process_mel_log(path):

    mel = get_melspec(path)[0]
    mel = mel.reshape(1, *mel.shape)
    tpred = model_mel.predict(mel)[0]
    txt = "MEL-SPECTROGRAMS \n" + get_title(tpred)
    fig = plt.figure(figsize=(10, 4))
    print("plot mel-spectrogram")
    plot_emotions(data6=tpred, fig=fig, title=txt)
    st.write(fig)


def process_gender(path):
    with st.spinner("Wait for it..."):
        gmfccs = get_mfccs(path, model_gender.input_shape[-1])
        gmfccs = gmfccs.reshape(1, *gmfccs.shape)
        gpred = model_gender.predict(gmfccs)[0]
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


def plot_emotions(
    fig,
    data6,
    data3=None,
    title="Detected emotion",
    categories6=CAT6,
    categories3=CAT3,
    color_dict=COLOR_DICT,
):

    if data3 is None:
        pos = data6[3]
        neu = data6[2] + data6[5]
        neg = data6[0] + data6[1] + data6[4]
        data3 = np.array([pos, neu, neg])

    ind = categories6[data6.argmax()]
    color6 = color_dict[ind]

    # parameters for sector highlighting #6
    theta6 = np.linspace(0.0, 2 * np.pi, data6.shape[0], endpoint=False)
    radii6 = np.zeros_like(data6)
    radii6[data6.argmax()] = data6.max() * 10
    width6 = np.pi / 1.8 * data6

    data6 = list(data6)
    n = len(data6)
    data6 += data6[:1]
    angles6 = [i / float(n) * 2 * np.pi for i in range(n)]
    angles6 += angles6[:1]

    ind = categories3[data3.argmax()]
    color3 = color_dict[ind]

    # parameters for sector highlighting #3
    theta3 = np.linspace(0.0, 2 * np.pi, data3.shape[0], endpoint=False)
    radii3 = np.zeros_like(data3)
    radii3[data3.argmax()] = data3.max() * 10
    width3 = np.pi / 1.8 * data3

    data3 = list(data3)
    n = len(data3)
    data3 += data3[:1]
    angles3 = [i / float(n) * 2 * np.pi for i in range(n)]
    angles3 += angles3[:1]

    # fig = plt.figure(figsize=(10, 4))
    fig.set_facecolor("#d1d1e0")

    ax = plt.subplot(122, polar="True")
    plt.polar(angles6, data6, color=color6)
    plt.fill(angles6, data6, facecolor=color6, alpha=0.25)
    ax.bar(theta6, radii6, width=width6, bottom=0.0, color=color6, alpha=0.25)
    ax.spines["polar"].set_color("lightgrey")
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles6[:-1], categories6)
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], color="grey", size=8)
    plt.title("BIG 6", color=color6)
    plt.ylim(0, 1)

    ax = plt.subplot(121, polar="True")
    # ax.set_facecolor('#d1d1e0')
    plt.polar(
        angles3,
        data3,
        color=color3,
        linewidth=2,
        linestyle="--",
        alpha=0.8,
    )
    plt.fill(angles3, data3, facecolor=color3, alpha=0.25)
    ax.bar(theta3, radii3, width=width3, bottom=0.0, color=color3, alpha=0.25)
    ax.spines["polar"].set_color("lightgrey")
    ax.set_theta_offset(np.pi / 6)
    ax.set_theta_direction(-1)
    plt.xticks(angles3[:-1], categories3)
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], color="grey", size=8)
    plt.title("BIG 3", color=color3)
    plt.ylim(0, 1)
    plt.suptitle(title, color="darkblue", size=12)
    plt.subplots_adjust(top=0.75)


if __name__ == "__main__":
    main()
