import argparse
from chat_agent import ChatAgent
from memory_manager import summarize_memory
from wiki_searcher import WikiSearcher
from vector_manager import query_similar_chats, add_chat_to_vector_db

# ✅ 실시간 챗봇 실행 함수
def run_chat(member_id: str, user_input: str, persona: str = "위로형"):
    # ✅ 벡터 DB에서 과거 대화 검색 후 요약 인사말 생성
    similar_chats = query_similar_chats(member_id, user_input, top_k=3)
    memory_summary = summarize_memory([{"message": chat} for chat in similar_chats])

    # ✅ 상담 이론 검색 (RAG)
    searcher = WikiSearcher()
    theory_summary = "\n".join(searcher.search(user_input, top_k=2))

    # ✅ 챗봇 응답 생성
    agent = ChatAgent(persona=persona)
    response = agent.respond(user_input=user_input, memory=memory_summary, theory=theory_summary)

    # ✅ 벡터 DB에 현재 대화 저장
    add_chat_to_vector_db(member_id=member_id, user_input=user_input, bot_response=response)

    # ✅ 출력 (테스트용 콘솔 확인용 표시)
    print("=" * 60)
    print(f"👤 사용자: {user_input}")
    if memory_summary:
        print(f"📜 상담 시작 멘트: {memory_summary}")
    print(f"📚 상담 이론 요약: {theory_summary}")
    print(f"🧘 상담사 응답: {response}")
    print("=" * 60)

# ✅ CLI 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", type=str, required=True, help="사용자 입력 문장")
    parser.add_argument("--member_id", type=str, default="1", help="사용자 ID")
    parser.add_argument("--persona", type=str, default="위로형", help="상담 페르소나")
    args = parser.parse_args()

    run_chat(member_id=args.member_id, user_input=args.user_input, persona=args.persona)
