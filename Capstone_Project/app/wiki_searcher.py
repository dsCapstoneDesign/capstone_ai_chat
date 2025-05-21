import os
import json
import torch
from sentence_transformers import SentenceTransformer, util

class WikiSearcher:
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_path = os.path.abspath(os.path.join(base_dir, "../dataset/hotpot/corpus/corpus.json"))

        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"[❌] corpus.json 경로를 확인하세요: {corpus_path}")

        with open(corpus_path, "r", encoding="utf-8") as f:
            try:
                self.corpus = json.load(f)
            except json.JSONDecodeError:
                raise ValueError("[❌] corpus.json이 JSON 형식이 아닙니다.")

        if not isinstance(self.corpus, list):
            raise TypeError("[❌] corpus.json은 list 형식이어야 합니다 (각 이론이 dict로 구성된 구조).")

        if not self.corpus:
            raise ValueError("❌ corpus.json이 비어 있습니다.")

        self.entries = []
        self.theory_map = {}
        for entry in self.corpus:
            if "이론명" in entry and "설명" in entry:
                text = f"{entry['이론명']}: {entry['설명']}"
                self.entries.append(text)
                self.theory_map[entry['이론명']] = entry["설명"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.model.eval()

        print(f"📘 상담 이론 개수: {len(self.entries)}개 | 임베딩 중...")

        self.embeddings = self.model.encode(
            self.entries,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True
        ).to(self.device)

        print(f"✅ 상담 이론 임베딩 완료 ({self.embeddings.shape})")

    def preprocess_query(self, query: str) -> str:
        query = query.strip()
        if len(query) < 6:
            return f"'{query}'에 도움이 되는 심리 상담 이론"
        elif "?" not in query and not query.endswith("."):
            return f"{query}와 관련된 심리상담 이론"
        return query

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, str]]:
        if not self.entries or not self.embeddings.shape[0]:
            return [("검색 실패", "❌ corpus 내용이 비어 있어 검색할 수 없습니다.")]

        formatted_query = f"query: {self.preprocess_query(query)}"
        try:
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

            return results if results else [("검색 실패", "❌ 유사한 이론을 찾을 수 없습니다.")]

        except Exception as e:
            print(f"⚠️ 상담 이론 검색 중 오류: {e}")
            return [("검색 오류", str(e))]

    def keyword_search(self, keyword: str, top_k: int = 3) -> list[tuple[str, str]]:
        """
        '추천상황' 필드에 감정 키워드가 포함된 이론 반환
        """
        matches = [
            (entry["이론명"], entry["설명"])
            for entry in self.corpus
            if "추천상황" in entry and keyword in entry["추천상황"]
        ]
        return matches[:top_k] if matches else [("검색 실패", f"❌ '{keyword}'에 맞는 이론을 찾을 수 없습니다.")]
