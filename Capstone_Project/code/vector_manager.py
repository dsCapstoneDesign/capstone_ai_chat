import os
import time
from uuid import uuid4
from chromadb import PersistentClient
from config.openai_client import client  # ✅ OpenAI client import

CHROMA_DIR = "./chroma_db"

# ✅ 최신 방식: PersistentClient 사용
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="chat_memory")

def get_embedding(text: str) -> list:
    """
    OpenAI text embedding 생성 함수
    - 실패 시 fallback 벡터 반환
    """
    if not text.strip():
        return [0.0] * 1536  # 빈 텍스트에 대한 fallback

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[⚠️ 임베딩 실패] {e}")
        return [0.0] * 1536

def add_chat_to_vector_db(member_id: str, user_input: str, bot_response: str,
                          persona: str = None, emotion: str = None, risk: str = None):
    """
    벡터 DB에 대화 저장 (기억용)
    - 추가적으로 persona, emotion, risk 메타데이터도 저장 가능
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

    collection.add(
        documents=[combined_text],
        metadatas=[metadata],
        ids=[str(uuid4())],
        embeddings=[embedding]
    )

def query_similar_chats(member_id: str, query: str, top_k: int = 3, return_all: bool = False) -> list:
    """
    벡터 DB에서 유사한 과거 대화 검색
    - member_id 필터링 포함
    - 결과는 대화 문장 리스트로 반환
    """
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"member_id": member_id}
    )

    documents = results.get("documents", [[]])[0]  # top_k 문장 리스트
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return []

    if return_all:
        return [{"text": doc, "meta": meta} for doc, meta in zip(documents, metadatas)]
    else:
        return documents
