from transformers import pipeline
import numpy as np
import time
import whisperx
import torch
from scipy.signal import resample
import time
import os
import datetime
from operator import le
import time
import timeit
from transformers import AutoModelForMaskedLM, AutoTokenizer
from langchain_text_splitters import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import torch
from qdrant_client import http, models, QdrantClient
from transformers import T5Tokenizer, T5ForConditionalGeneration


class WhisperAutomaticSpeechRecognizer:
    device = "cuda"
    compute_type = "int8"  # change to "int8" if low on GPU mem (may reduce accuracy)
    batch_size = 4
    model = whisperx.load_model(
        "medium", device, language="en", compute_type=compute_type
    )
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token="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", device="cuda"
    )
    current_speaker = None

    @staticmethod
    def downsample_audio_scipy(audio: np.ndarray, original_rate, target_rate=16000):
        if original_rate == target_rate:
            return audio

        # Check if audio has one channel
        if len(audio.shape) != 1:
            raise ValueError("Input audio must have only one channel.")

        # Calculate the number of samples in the downsampled audio
        num_samples = int(len(audio) * target_rate / original_rate)
        downsampled_audio = resample(audio, num_samples)

        return downsampled_audio

    @staticmethod
    def transcribe_file(filepath: str):
        audio = whisperx.load_audio(filepath)
        return WhisperAutomaticSpeechRecognizer.transcribe((16000, audio), None, "")

    @staticmethod
    def transcribe(stream, full_stream, full_transcript):
        time.sleep(5)
        sr, y = stream
        y = WhisperAutomaticSpeechRecognizer.downsample_audio_scipy(y, sr)
        y = y.astype(np.float32)
        y /= 32768.0

        if full_transcript is None:
            full_transcript = ""
        transcribe_result = WhisperAutomaticSpeechRecognizer.model.transcribe(
            y, batch_size=WhisperAutomaticSpeechRecognizer.batch_size
        )
        new_transcript = ""
        for segment in transcribe_result["segments"]:
            new_transcript = new_transcript + segment["text"] + "\n"
        full_transcript = full_transcript + new_transcript
        return full_transcript, stream, full_transcript

    @staticmethod
    def transcribe_with_diarization_file(filepath: str):
        audio = whisperx.load_audio(filepath, 16000)
        return WhisperAutomaticSpeechRecognizer.transcribe_with_diarization(
            (16000, audio), None, "", False
        )

    @staticmethod
    def transcribe_with_diarization(
        stream, full_stream, full_transcript, streaming=True
    ):
        start_time = time.time()
        sr, y = stream
        if streaming:
            sr, y = stream
            y = WhisperAutomaticSpeechRecognizer.downsample_audio_scipy(y, sr)
            y = y.astype(np.float32)
            y /= 32768.0

        if full_transcript is None:
            full_transcript = ""
        transcribe_result = WhisperAutomaticSpeechRecognizer.model.transcribe(
            y, batch_size=WhisperAutomaticSpeechRecognizer.batch_size
        )
        diarize_segments = WhisperAutomaticSpeechRecognizer.diarize_model(y)

        diarize_result = whisperx.assign_word_speakers(
            diarize_segments, transcribe_result
        )

        new_transcript = ""
        for segment in diarize_result["segments"]:
            print(segment)
            if WhisperAutomaticSpeechRecognizer.current_speaker == None:
                try:
                    WhisperAutomaticSpeechRecognizer.current_speaker = segment["speaker"]
                except KeyError:
                    WhisperAutomaticSpeechRecognizer.current_speaker = "SPEAKER_00"
                new_transcript += f"\n {WhisperAutomaticSpeechRecognizer.current_speaker}  - "
            if segment["speaker"] != WhisperAutomaticSpeechRecognizer.current_speaker and segment["speaker"] is not "SPEAKER_00":
                WhisperAutomaticSpeechRecognizer.current_speaker = segment["speaker"]
                new_transcript += f"\n {WhisperAutomaticSpeechRecognizer.current_speaker}  - "
            new_transcript = new_transcript + segment["text"]
        full_transcript = full_transcript + new_transcript
        end_time = time.time()
        if streaming:
            time.sleep(10 - (end_time - start_time))
        return full_transcript, stream, full_transcript
