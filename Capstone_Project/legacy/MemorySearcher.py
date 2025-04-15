import os
import json
import torch
from sentence_transformers import SentenceTransformer, util

class MemorySearcher:
    def __init__(self, model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        # ✅ memory.json 경로
        base_dir = os.path.dirname(os.path.abspath(__file__))
        memory_path = os.path.join(base_dir, "../data/memory.json")

        if not os.path.exists(memory_path):
            raise FileNotFoundError(f"memory.json 파일을 찾을 수 없습니다: {memory_path}")

        with open(memory_path, "r", encoding="utf-8") as f:
            self.memory_data = json.load(f)

        # ✅ 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ SBERT 모델 로딩
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.model.eval()

        # ✅ 과거 질문 임베딩
        self.questions = [item["question"] for item in self.memory_data]
        self.embeddings = self.model.encode(
            self.questions,
            convert_to_tensor=True,
            show_progress_bar=False  # 깔끔하게
        ).to(self.device)

    def search(self, query, top_k=3):
        # ✅ 쿼리 임베딩
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False
        ).to(self.device)

        # ✅ 유사도 계산
        scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(scores, k=top_k)

        retrieved_memories = []
        for idx in top_results.indices:
            memory_text = self.memory_data[idx]["memory"]
            if memory_text:  # 혹시라도 비어있을 경우
                retrieved_memories.append(memory_text.strip())

        return retrieved_memories
