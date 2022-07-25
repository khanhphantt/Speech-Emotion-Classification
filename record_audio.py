import wave

import pyaudio

from settings import (
    CHUNK_SIZE,
    DEFAULT_SAMPLE_RATE,
    DURATION,
    INPUT_DEVICE,
    MAX_INPUT_CHANNELS,
)


class Sound:
    def __init__(self):
        # Set default configurations for recording device
        self.format = pyaudio.paInt16
        self.channels = MAX_INPUT_CHANNELS
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.chunk = CHUNK_SIZE
        self.duration = DURATION
        self.device = INPUT_DEVICE
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.device_info()
        print("Audio device configurations currently used")
        print(f"Default input device index = {self.device}")
        print(f"Max input channels = {self.channels}")
        print(f"Default samplerate = {self.sample_rate}")

    def device_info(self):
        num_devices = self.audio.get_device_count()
        keys = ["name", "index", "maxInputChannels", "defaultSampleRate"]
        print("List of System's Audio Devices configurations:")
        print(f"Number of audio devices: {num_devices}")
        for i in range(num_devices):
            info_dict = self.audio.get_device_info_by_index(i)
            print(
                [
                    (key, value)
                    for key, value in info_dict.items()
                    if key in keys
                ],
            )

    def record(self, path):
        # start Recording
        self.audio = pyaudio.PyAudio()
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device,
        )
        print(f"Recording started for {self.duration} seconds")
        self.frames = []
        for i in range(0, int(self.sample_rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            self.frames.append(data)
        print("Recording Completed")
        # stop Recording
        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        self.save(path)

    def save(self, path):
        waveFile = wave.open(path, "wb")
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.sample_rate)
        waveFile.writeframes(b"".join(self.frames))
        waveFile.close()
        print(f"Recording saved to {path}")

    # def play(self):
    #     print(f"Playing the recorded sound {self.path}")
    #     mixer.init(self.sample_rate)
    #     recording = mixer.Sound(self.path).play()
    #     while recording.get_busy():
    #         continue


sound = Sound()
