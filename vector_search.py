import torch
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
            f"summarize: {text}", return_tensors="pt", max_length=1024, truncation=True
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
            limit=limit,
            with_payload=True,
        )
        sparse_request = models.SearchRequest(
            vector=models.NamedSparseVector(
                name="sparse_vector", vector=sparse_query_vector
            ),
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
        chat_history.append((query, "Search Results"))
        for search_result in result[:3]:
            text = search_result.payload["conversation"]
            summary = HybridVectorSearch.summary(text) + f'\n```\n{text} \n```'
            chat_history.append((None, summary))
        return "", chat_history
