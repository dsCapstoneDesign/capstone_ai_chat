import os
import time
from uuid import uuid4
from chromadb import PersistentClient
from .config.openai_client import client  # ✅ OpenAI API

# ✅ ChromaDB 저장 경로
CHROMA_DIR = "./chroma_db"

# ✅ 영구 클라이언트 초기화
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="chat_memory")


def get_embedding(text: str) -> list:
    """
    OpenAI 임베딩 생성
    - 빈 문자열: 0 벡터 반환
    - 실패: fallback 벡터
    """
    if not text.strip():
        return [0.0] * 1536

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ [임베딩 실패] {e}")
        return [0.0] * 1536


def add_chat_to_vector_db(member_id: str, user_input: str, bot_response: str,
                          persona: str = None, emotion: str = None, risk: str = None):
    """
    벡터 DB에 대화 기록 저장
    - 메타데이터로 사용자 정보와 감정 저장
    """
    combined_text = f"사용자: {user_input}\n상담사: {bot_response}"
    embedding = get_embedding(combined_text)

    metadata = {
        "member_id": member_id,
        "timestamp": time.time()
    }
    if persona:
        metadata["persona"] = persona
    if emotion:
        metadata["emotion"] = emotion
    if risk:
        metadata["risk"] = risk

    try:
        collection.add(
            documents=[combined_text],
            metadatas=[metadata],
            ids=[str(uuid4())],
            embeddings=[embedding]
        )
        print(f"✅ [벡터 저장 완료] member_id={member_id}, persona={persona}")
    except Exception as e:
        print(f"⚠️ [벡터 저장 실패] {e}")


def query_similar_chats(member_id: str, query: str, top_k: int = 3, return_all: bool = False) -> list:
    """
    벡터 DB에서 유사 대화 검색
    - member_id 기준으로 과거 대화 필터링
    - return_all=True 시 메타 포함
    """
    embedding = get_embedding(query)

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={"member_id": member_id}
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return []

        if return_all:
            return [{"text": doc, "meta": meta} for doc, meta in zip(documents, metadatas)]
        return documents

    except Exception as e:
        print(f"⚠️ [벡터 검색 실패] {e}")
        return []
