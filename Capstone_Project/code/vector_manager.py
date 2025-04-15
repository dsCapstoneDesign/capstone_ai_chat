# ✅ 최신 Chroma 클라이언트 + 공통 OpenAI client 사용
from chromadb import PersistentClient
from uuid import uuid4
from config.openai_client import client  # ✅ 공통 client import

CHROMA_DIR = "./chroma_db"

# ✅ 최신 방식: PersistentClient 사용
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="chat_memory")

def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def add_chat_to_vector_db(member_id: str, user_input: str, bot_response: str):
    combined_text = f"사용자: {user_input}\n상담사: {bot_response}"
    embedding = get_embedding(combined_text)
    collection.add(
        documents=[combined_text],
        metadatas=[{"member_id": member_id}],
        ids=[str(uuid4())],
        embeddings=[embedding]
    )

def query_similar_chats(member_id: str, query: str, top_k: int = 3):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"member_id": member_id}
    )
    return results["documents"][0] if results["documents"] else []
