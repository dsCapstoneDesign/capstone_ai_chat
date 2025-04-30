import argparse
from .chat_agent import ChatAgent
from .wiki_searcher import WikiSearcher
from .vector_manager import query_similar_chats, add_chat_to_vector_db

# ✅ FastAPI에서 호출할 함수
def run_model(user_input: str, member_id: str = "1", persona: str = "위로형") -> str:
    if not user_input.strip():
        return "조금 더 구체적으로 말씀해주실 수 있을까요?"

    similar_chats = query_similar_chats(member_id, user_input, top_k=3)
    memory_summary = "\n".join(similar_chats)

    searcher = WikiSearcher()
    theory_pairs = searcher.search(user_input, top_k=2)

    agent = ChatAgent(persona=persona)
    response = agent.respond(user_input=user_input, memory=memory_summary, theory=theory_pairs)

    add_chat_to_vector_db(
        member_id=member_id,
        user_input=user_input,
        bot_response=response,
        persona=persona,
        emotion=agent.emotion,
        risk=agent.risk
    )

    return response

# ✅ CLI에서 실행할 함수
def run_chat(member_id: str, user_input: str, persona: str = "위로형"):
    if not user_input.strip():
        print("⚠️ 입력이 비어 있습니다. 고민이나 감정을 자유롭게 입력해 주세요.")
        return

    similar_chats = query_similar_chats(member_id, user_input, top_k=3)
    memory_summary = "\n".join(similar_chats)

    searcher = WikiSearcher()
    theory_pairs = searcher.search(user_input, top_k=2)

    agent = ChatAgent(persona=persona)
    response = agent.respond(user_input=user_input, memory=memory_summary, theory=theory_pairs)

    add_chat_to_vector_db(
        member_id=member_id,
        user_input=user_input,
        bot_response=response,
        persona=persona,
        emotion=agent.emotion,
        risk=agent.risk
    )

    print("=" * 70)
    print(f"👤 사용자: {user_input}")
    print(f"\n📜 Memory 요약:")
    print(memory_summary if memory_summary else "최근 대화 없음.")
    print(f"\n📚 관련 상담 이론:")
    for name, desc in theory_pairs:
        print(f"• [{name}] {desc}")
    print(f"\n📊 감정: {agent.emotion or '분석 실패'} | 위험도: {agent.risk or '분석 실패'}")
    print(f"\n🧘 상담사 응답:\n{response}")
    print("=" * 70)

# ✅ CLI 실행 진입점
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", type=str, required=True, help="사용자 입력 문장")
    parser.add_argument("--member_id", type=str, default="1", help="사용자 ID")
    parser.add_argument("--persona", type=str, default="위로형", help="상담 페르소나")
    args = parser.parse_args()

    run_chat(member_id=args.member_id, user_input=args.user_input, persona=args.persona)
