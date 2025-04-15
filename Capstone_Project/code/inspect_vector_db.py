from chromadb import PersistentClient
import os

# Chroma DB 저장 디렉토리 설정
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Chroma 클라이언트 초기화
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="chat_memory")

# 🔍 확인할 member_id 입력
target_member_id = input("확인할 member_id 입력: ").strip()

# ✅ 필터 조건으로 해당 사용자 대화 검색
results = collection.get(where={"member_id": target_member_id})

# ✅ 출력
print(f"\n🔎 member_id: {target_member_id}에 저장된 대화 수: {len(results['ids'])}")
print("=" * 50)

for idx, doc in enumerate(results["documents"]):
    print(f"[{idx+1}]")
    print(doc)
    print("-" * 50)