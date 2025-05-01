from app.chat_agent import ChatAgent
from app.memory_manager import load_user_memory, summarize_memory
from app.wiki_searcher import WikiSearcher
from app.vector_manager import add_chat_to_vector_db  # ✅ 벡터 DB 저장 함수

# ✅ 테스트용 메모리 로그 (실제로는 DB에서 불러오게 될 예정)
message_log = [
    {"id": 1, "member_id": "1", "sender": "USER", "message": "요즘 너무 힘들어요.", "send_time": "2025-04-01 10:00:00", "type": "SEND"},
    {"id": 2, "member_id": "1", "sender": "BOT", "message": "많이 힘드셨겠어요. 무리하지 않아도 괜찮아요.", "send_time": "2025-04-01 10:01:00", "type": "RECEIVE"}
]

# ✅ 설정
MEMBER_ID = "2"  # 사용자가 로그인하면 백엔드에서 받아옴
PERSONA = "유쾌한친구형"

agent = ChatAgent(persona=PERSONA)
wiki_searcher = WikiSearcher()

print("=" * 60)
print(f"🤖 {PERSONA} 상담자와의 대화를 시작합니다.")
print("고민이나 감정을 자유롭게 이야기해보세요.")
print("상담을 종료하려면 '종료', 'exit', '그만' 중 하나를 입력하세요.")
print("=" * 60)

while True:
    user_input = input("👤 사용자: ").strip()

    if user_input.lower() in ["종료", "exit", "그만"]:
        print("👋 오늘 상담은 여기까지 할게요. 언제든 다시 찾아주세요.")
        break

    # ✅ memory 요약
    memory_list = load_user_memory(MEMBER_ID, message_log, max_turns=3)
    memory_summary = summarize_memory(memory_list)

    # ✅ 상담 이론 검색
    theory_refs = wiki_searcher.search(user_input, top_k=2)
    theory_text = "\n".join(theory_refs)

    # ✅ 응답 생성
    response = agent.respond(user_input=user_input, memory=memory_summary, theory=theory_text)

    # ✅ 콘솔 출력
    print(f"\n🧘 상담사: {response}\n")

    # ✅ 벡터 DB에 저장
    add_chat_to_vector_db(member_id=MEMBER_ID, user_input=user_input, bot_response=response)

    # ✅ message_log에 추가 (실제 서비스에서는 DB에도 반영될 로직)
    message_log.append({"id": len(message_log) + 1, "member_id": MEMBER_ID, "sender": "USER", "message": user_input, "send_time": "2025-04-14 18:00:00", "type": "SEND"})
    message_log.append({"id": len(message_log) + 1, "member_id": MEMBER_ID, "sender": "BOT", "message": response, "send_time": "2025-04-14 18:00:01", "type": "RECEIVE"})
