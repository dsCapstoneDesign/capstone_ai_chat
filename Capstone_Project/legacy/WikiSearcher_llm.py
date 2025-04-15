import json
import os
import torch
from sentence_transformers import SentenceTransformer, util

class WikiSearcher_llm:
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        # ✅ corpus.json 경로 설정
        base_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_path = os.path.join(base_dir, "../dataset/hotpot/corpus/corpus.json")

        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"corpus.json 파일을 찾을 수 없습니다: {corpus_path}")

        # ✅ corpus 로드 (key: 이론명, value: 설명)
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        # ✅ 검색 엔트리 구성
        self.theory_names = list(self.corpus.keys())
        self.entries = [f"{k}: {v}" for k, v in self.corpus.items()]

        # ✅ SBERT(E5) 모델 로드
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # ✅ corpus 임베딩
        self.embeddings = self.model.encode(
            self.entries,
            convert_to_tensor=True,
            show_progress_bar=False
        ).to(self.device)

    def search(self, query, top_k=3):
        formatted_query = f"query: {query}"
        query_embedding = self.model.encode(
            formatted_query,
            convert_to_tensor=True,
            show_progress_bar=False
        ).to(self.device)

        similarity_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(similarity_scores, k=top_k)

        results = []
        for idx in top_results.indices:
            entry = self.entries[idx]
            if ": " in entry:
                _, desc = entry.split(": ", 1)
                results.append(desc.strip())
            else:
                results.append(entry.strip())
        return results
