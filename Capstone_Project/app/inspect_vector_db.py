from chromadb import PersistentClient
import os
from datetime import datetime

# Chroma DB 저장 디렉토리 설정
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Chroma 클라이언트 초기화
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="chat_memory")

# ✅ 전체 저장된 사용자 목록 확인
all_data = collection.get()
all_member_ids = set(
    meta.get("member_id") for meta in all_data["metadatas"] if "member_id" in meta
)

print("\n📋 현재 저장된 member_id 목록:")
for idx, mid in enumerate(sorted(all_member_ids)):
    print(f"{idx+1}. {mid}")

# ✅ 사용자 선택
target_member_id = input("\n🔍 확인할 member_id 입력: ").strip()

# ✅ 필터 조건으로 해당 사용자 대화 검색
results = collection.get(where={"member_id": target_member_id})

# ✅ 문서 + 메타 정리
documents = results.get("documents", [])
metadatas = results.get("metadatas", [])
timestamps = [meta.get("timestamp", 0) for meta in metadatas]

# ✅ 정렬 기준 선택
sort_input = input("\n🗂 정렬 기준 선택 (1: 시간순, 2: 최근순, 기타: 그대로): ").strip()

if sort_input == "1":
    sorted_indices = sorted(range(len(documents)), key=lambda i: timestamps[i])
elif sort_input == "2":
    sorted_indices = sorted(range(len(documents)), key=lambda i: timestamps[i], reverse=True)
else:
    sorted_indices = list(range(len(documents)))

# ✅ 결과 출력
print(f"\n🔎 member_id: {target_member_id}에 저장된 대화 수: {len(documents)}")
print("=" * 60)

for idx in sorted_indices:
    doc = documents[idx]
    meta = metadatas[idx]
    ts = meta.get("timestamp", 0)
    readable_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "없음"

    print(f"[{idx+1}]")
    print(f"🗓 시간: {readable_time}")
    print(f"🗂 메타: {meta}")
    print(f"📝 대화 내용:\n{doc}")
    print("-" * 60)
