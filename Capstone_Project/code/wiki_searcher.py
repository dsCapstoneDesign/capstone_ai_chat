import os
import json
import torch
from sentence_transformers import SentenceTransformer, util

class WikiSearcher:
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        # ✅ corpus.json 경로 설정
        base_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_path = os.path.abspath(os.path.join(base_dir, "../dataset/hotpot/corpus/corpus.json"))

        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"[❌] corpus.json 경로를 확인하세요: {corpus_path}")

        # ✅ corpus 로드 (이론 이름 → 설명 dict)
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        if not self.corpus:
            raise ValueError("corpus.json이 비어 있습니다.")

        # ✅ key: 이론명 / value: 설명
        self.keys = list(self.corpus.keys())
        self.entries = [f"{k}: {v}" for k, v in self.corpus.items()]

        # ✅ 모델 로드 및 임베딩 처리
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.model.eval()

        self.embeddings = self.model.encode(
            self.entries,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True  # ✅ 검색 정확도 향상
        ).to(self.device)

    def search(self, query: str, top_k: int = 3) -> list:
        """
        사용자 query와 유사한 상담 이론 설명 top-k 추출
        """
        if not self.entries or not self.embeddings.shape[0]:
            return ["[❌] corpus 내용이 비어 있어 검색이 불가능합니다."]

        formatted_query = f"query: {query.strip()}"
        query_embedding = self.model.encode(
            formatted_query,
            convert_to_tensor=True,
            normalize_embeddings=True  # ✅ 쿼리도 정규화
        ).to(self.device)

        similarity_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(similarity_scores, k=min(top_k, len(self.entries)))

        results = []
        for idx in top_results.indices:
            entry = self.entries[idx]
            if ": " in entry:
                _, desc = entry.split(": ", 1)
                results.append(desc.strip())

        return results
