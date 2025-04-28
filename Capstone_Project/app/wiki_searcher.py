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

        self.keys = list(self.corpus.keys())  # 이론명 리스트
        self.entries = [f"{k}: {v}" for k, v in self.corpus.items()]  # 이론명: 설명 형식

        # ✅ 모델 로드 및 임베딩 처리
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.model.eval()

        self.embeddings = self.model.encode(
            self.entries,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True
        ).to(self.device)

    def preprocess_query(self, query: str) -> str:
        """
        사용자가 너무 짧거나 단순한 감정 표현만 썼을 때, 검색용 문장 보정
        예: "불안해요" → "불안감을 완화할 수 있는 상담 이론"
        """
        query = query.strip()
        if len(query) < 6:  # 짧은 문장일 경우
            return f"'{query}'에 도움이 되는 심리 상담 이론"
        elif "?" not in query and not query.endswith("."):
            return f"{query}와 관련된 심리상담 이론"
        return query

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, str]]:
        """
        사용자 query와 유사한 상담 이론 설명 top-k 추출
        → 결과는 (이론명, 설명) 튜플 리스트로 반환
        """
        if not self.entries or not self.embeddings.shape[0]:
            return [("검색 실패", "[❌] corpus 내용이 비어 있어 검색이 불가능합니다.")]

        formatted_query = f"query: {self.preprocess_query(query)}"
        query_embedding = self.model.encode(
            formatted_query,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(self.device)

        similarity_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(similarity_scores, k=min(top_k, len(self.entries)))

        results = []
        for idx in top_results.indices:
            entry = self.entries[idx]
            if ": " in entry:
                theory, desc = entry.split(": ", 1)
                results.append((theory.strip(), desc.strip()))

        return results
