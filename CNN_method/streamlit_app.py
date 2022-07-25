import queue
import time

import keras
import librosa
import numpy as np
import pydub
import streamlit as st
from config import MODEL_DIR_PATH
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
)

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(
    page_title="Emotion",
    page_icon=":-)",
    layout="centered",
    initial_sidebar_state="expanded",
)
example_audio = "examples/03-01-01-01-01-02-01.wav"
recorded_audio = "examples/record.wav"

# Load models
path = MODEL_DIR_PATH + "Emotion_Voice_Detection_Model.h5"
loaded_model = keras.models.load_model(path)
print("[INFO] loaded speech-emotion model")


def predict_emotion(audio_file):
    """
    Method to process the files and create your features.
    """
    print("audio_file:", audio_file)
    data, sampling_rate = librosa.load(audio_file)
    print("sampling rate:", sampling_rate)
    mfccs = np.mean(
        librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T,
        axis=0,
    )

    if mfccs.ndim == 1:
        mfccs = np.expand_dims(mfccs, axis=1)
    x = np.expand_dims(mfccs, axis=2)
    x = np.expand_dims(x, axis=0)
    predictions = loaded_model.predict(x)
    classes_x = np.argmax(predictions, axis=1)
    emotion = convert_class_to_emotion(classes_x)
    print("Prediction is:", emotion)
    return emotion


def convert_class_to_emotion(pred):
    """
    Method to convert the predictions (int) into human readable strings.
    """

    label_conversion = {
        "0": "neutral",
        "1": "calm",
        "2": "happy",
        "3": "sad",
        "4": "angry",
        "5": "fearful",
        "6": "disgust",
        "7": "surprised",
    }

    for key, value in label_conversion.items():
        if int(key) == pred:
            label = value
    return label


def audiosegment_to_librosawav(audiosegment):
    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    return fp_arr


def classify_live_emotion():
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        },
        media_stream_constraints={"video": False, "audio": True},
    )

    if not webrtc_ctx.state.playing:
        return

    print("Loading...")
    text_output = st.empty()

    while True:
        if webrtc_ctx.audio_receiver:

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                print("No frame arrived.")
                continue

            print("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 1000:
                audio = audiosegment_to_librosawav(sound_chunk)

                emotion = predict_emotion(audio)
                print("emotion: ", emotion)
                # Display result
                st.markdown(
                    '<h4 align="center">' + emotion + "</h4>",
                    unsafe_allow_html=True,
                )

                text_output.markdown(f"**Emotion:** {emotion}")
        else:
            print("AudioReceiver is not set. Abort.")
            break


def local_css(file_name):
    # Method for reading styles.css and applying necessary changes to HTML
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def create_UI():
    local_css("css/styles.css")
    st.markdown(
        '<h6 align="center">TT monthly challenge - July 2022</h6>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 align="center">Recognize Emotion From Speech</h1>',
        unsafe_allow_html=True,
    )
    st.set_option("deprecation.showfileUploaderEncoding", False)
    choice = st.sidebar.radio(
        "Select an input option:",
        ["Audio file", "Your microphone"],
    )
    if choice == "Audio file":
        # st.sidebar.markdown('Upload your image ⬇')
        audio_file = st.sidebar.file_uploader(
            "",
            type=[
                "m4a",
                "flac",
                "mp3",
                "mp4",
                "wav",
                "wma",
                "aac",
            ],
        )

        if not audio_file:
            text = """This is a detection example.
            Try your input from the left sidebar.
            """
            st.markdown(
                '<h6 align="center">' + text + "</h6>",
                unsafe_allow_html=True,
            )
            st.audio(example_audio)
        else:
            st.sidebar.markdown(
                "__Audio is uploaded successfully!__",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<h4 align="center">Detection result</h4>',
                unsafe_allow_html=True,
            )
            st.audio(audio_file)

            # Process audio
            emotion = predict_emotion(audio_file)
            print("emotion: ", emotion)
            # Display result
            st.markdown(
                '<h4 align="center">' + emotion + "</h4>",
                unsafe_allow_html=True,
            )

    if choice == "Your microphone":
        st.sidebar.markdown('Click "START" to connect this app to a server')
        st.sidebar.markdown("It may take a minute, please wait...")
        classify_live_emotion()
        # get and process audio


if __name__ == "__main__":
    create_UI()
