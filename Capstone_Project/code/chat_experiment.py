import os
from chat_agent import ChatAgent
from vector_manager import add_chat_to_vector_db, query_similar_chats
from wiki_searcher import WikiSearcher
from dotenv import load_dotenv

load_dotenv()

print("🔑 member_id 입력:")
member_id = input().strip()

print("🎭 페르소나 입력 (위로형, 논리분석형, 유쾌한친구형, 여자친구형, 남자친구형):")
persona = input().strip()

# ✅ 대화 기록 저장 디렉토리
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"{member_id}_session.txt")

print("=" * 60)
print(f"🤖 {persona} 상담자와의 대화를 시작합니다.")
print("고민이나 감정을 자유롭게 이야기해보세요.")
print("상담을 종료하려면 '종료', 'exit', '그만' 중 하나를 입력하세요.")
print("=" * 60)

agent = ChatAgent(persona=persona)
searcher = WikiSearcher()

while True:
    user_input = input("\n👤 사용자: ").strip()
    if user_input.lower() in ["종료", "exit", "그만"]:
        print("👋 오늘 상담은 여기까지 할게요. 언제든 다시 찾아주세요.")
        break

    # ✅ memory (벡터 DB에서 과거 대화 검색)
    similar_docs = query_similar_chats(member_id, user_input)
    memory_summary = "\n".join(similar_docs)

    # ✅ 이론 검색
    theory_pairs = searcher.search(user_input, top_k=2)

    # ✅ 챗봇 응답 생성
    response = agent.respond(user_input=user_input, memory=memory_summary, theory=theory_pairs)

    # ✅ 감정 및 위험도 출력
    print(f"\n📊 감정 상태: {agent.emotion or '분석 실패'} | 위험도: {agent.risk or '분석 실패'}")
    print(f"\n🧘 상담사 ({agent.persona}): {response}")

    # ✅ 대화 기록 로그 저장
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"👤 사용자: {user_input}\n")
        f.write(f"📊 감정: {agent.emotion or 'N/A'}, 위험도: {agent.risk or 'N/A'}\n")
        f.write(f"🧘 상담사: {response}\n")
        f.write("=" * 60 + "\n")

    # ✅ 벡터 DB 저장 (turn 기반 memory를 잘 활용하려면 저장도 중요)
    add_chat_to_vector_db(
        member_id=member_id,
        user_input=user_input,
        bot_response=response,
        persona=persona,
        emotion=agent.emotion,
        risk=agent.risk
    )
