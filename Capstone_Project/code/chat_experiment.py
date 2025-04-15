from chat_agent import ChatAgent
from vector_manager import add_chat_to_vector_db, query_similar_chats
from wiki_searcher import WikiSearcher

print("🔑 member_id 입력:")
member_id = input().strip()

print("🎭 페르소나 입력 (위로형, 논리분석형, 유쾌한친구형, 여자친구형, 남자친구형):")
persona = input().strip()

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
    memory_summary = "\n".join(similar_docs)  # summarize_memory() 대신 사용

    # ✅ 이론 검색
    theory = "\n".join(searcher.search(user_input, top_k=2))

    # ✅ 챗봇 응답 생성
    response = agent.respond(user_input=user_input, memory=memory_summary, theory=theory)
    print(f"\n🧘 상담사: {response}")

    # ✅ 벡터 DB 저장
    add_chat_to_vector_db(member_id, user_input, response)
