import numpy as np
import time
import whisperx
import torch


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
        current_speaker = None
        for segment in diarize_result["segments"]:
            if current_speaker == None:
                current_speaker = segment["speaker"]
            if segment["speaker"] != current_speaker:
                current_speaker = segment["speaker"]
                new_transcript += f"\n {current_speaker}  - "
            new_transcript = new_transcript + segment["text"]
        full_transcript = full_transcript + new_transcript
        end_time = time.time()
        if streaming:
            time.sleep(10 - (end_time - start_time))
        return full_transcript, stream, full_transcript


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


class HybridVectorSearch:
    cuda_device = torch.device("cpu")
    sparse_model = "naver/splade-v3"
    tokenizer = AutoTokenizer.from_pretrained(sparse_model)
    model = AutoModelForMaskedLM.from_pretrained(sparse_model).to(cuda_device)

    text_splitter = SpacyTextSplitter(chunk_size=1000)
    dense_encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    model_name_t5 = "Falconsai/text_summarization"  # "t5-small"
    tokenizer_t5 = T5Tokenizer.from_pretrained(model_name_t5)
    model_t5 = T5ForConditionalGeneration.from_pretrained(model_name_t5).to("cuda")

    client = QdrantClient(url="http://localhost:6333")
    earnings_collection = "earnings_calls"

    @staticmethod
    def reciprocal_rank_fusion(
        responses: List[List[http.models.ScoredPoint]], limit: int = 10
    ) -> List[http.models.ScoredPoint]:
        def compute_score(pos: int) -> float:
            ranking_constant = 2  # the constant mitigates the impact of high rankings by outlier systems
            return 1 / (ranking_constant + pos)

        scores: Dict[http.models.ExtendedPointId, float] = {}
        point_pile = {}
        for response in responses:
            for i, scored_point in enumerate(response):
                if scored_point.id in scores:
                    scores[scored_point.id] += compute_score(i)
                else:
                    point_pile[scored_point.id] = scored_point
                    scores[scored_point.id] = compute_score(i)

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        sorted_points = []
        for point_id, score in sorted_scores[:limit]:
            point = point_pile[point_id]
            point.score = score
            sorted_points.append(point)
        return sorted_points

    @staticmethod
    def summary(text: str):
        inputs = HybridVectorSearch.tokenizer_t5.encode(
            "summarize: " + text, return_tensors="pt", max_length=1024, truncation=True
        ).to("cuda")
        summary_ids = HybridVectorSearch.model_t5.generate(
            inputs,
            max_length=512,
            min_length=100,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        summary = HybridVectorSearch.tokenizer_t5.decode(
            summary_ids[0], skip_special_tokens=True
        )
        return summary

    @staticmethod
    def compute_vector(text):
        tokens = HybridVectorSearch.tokenizer(text, return_tensors="pt").to(
            HybridVectorSearch.cuda_device
        )
        split_texts = []
        if len(tokens["input_ids"][0]) >= 512:
            summary = HybridVectorSearch.summary(text)
            split_texts = HybridVectorSearch.text_splitter.split_text(text)
            tokens = HybridVectorSearch.tokenizer(summary, return_tensors="pt").to(
                HybridVectorSearch.cuda_device
            )

        output = HybridVectorSearch.model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vec = max_val.squeeze()

        return vec, tokens, split_texts

    @staticmethod
    def search(query_text: str, symbol="AMD"):
        vectors, tokens, split_texts = HybridVectorSearch.compute_vector(query_text)
        indices = vectors.cpu().nonzero().numpy().flatten()
        values = vectors.cpu().detach().numpy()[indices]

        sparse_query_vector = models.SparseVector(indices=indices, values=values)

        query_vector = HybridVectorSearch.dense_encoder.encode(query_text).tolist()
        limit = 3

        dense_request = models.SearchRequest(
            vector=models.NamedVector(name="dense_vector", vector=query_vector),
            # filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        sparse_request = models.SearchRequest(
            vector=models.NamedSparseVector(
                name="sparse_vector", vector=sparse_query_vector
            ),
            # filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        (dense_request_response, sparse_request_response) = (
            HybridVectorSearch.client.search_batch(
                collection_name=HybridVectorSearch.earnings_collection,
                requests=[dense_request, sparse_request],
            )
        )
        ranked_search_response = HybridVectorSearch.reciprocal_rank_fusion(
            [dense_request_response, sparse_request_response], limit=10
        )

        search_response = ""
        for search_result in ranked_search_response:
            search_response += search_result.payload["conversation"] + "\n"
        return ranked_search_response

    @staticmethod
    def chat_search(query: str, chat_history):
        result = HybridVectorSearch.search(query)
        chat_history.append((query, "Results"))
        for search_result in result[:3]:
            text = search_result.payload["conversation"]
            summary = HybridVectorSearch.summary(text)
            chat_history.append((None, summary))
        return "", chat_history
