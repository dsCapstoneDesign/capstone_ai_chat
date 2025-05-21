import argparse
from app.chat_agent import ChatAgent
from app.vector_manager import add_chat_to_vector_db
from app.db_manager import fetch_recent_dialogue
from app.memory_manager import is_first_entry

# ✅ FastAPI에서 호출할 함수
def run_model(user_input: str, member_id: str = "1", persona: str = "위로형") -> str:
    if not user_input.strip():
        return "조금 더 구체적으로 말씀해주실 수 있을까요?"

    message_log = fetch_recent_dialogue(member_id, limit=100)
    agent = ChatAgent(persona=persona)

    if is_first_entry(member_id, message_log):
        return "안녕하세요! 처음 오셨군요. 편하게 이야기해 주세요. 😊"

    response = agent.respond(
        user_input=user_input,
        message_log=message_log,
        member_id=member_id
    )

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

    message_log = fetch_recent_dialogue(member_id, limit=100)
    agent = ChatAgent(persona=persona)

    if is_first_entry(member_id, message_log):
        print("🟡 첫 입장입니다.")
        print("🧘 상담사 응답:\n안녕하세요! 처음 오셨군요. 편하게 이야기해 주세요. 😊")
        return

    response = agent.respond(
        user_input=user_input,
        message_log=message_log,
        member_id=member_id,
        theory=None  # ✅ 상담 이론 적용은 ChatAgent 내부에서 판단
    )

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
