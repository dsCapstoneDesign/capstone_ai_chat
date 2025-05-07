from app.config.openai_client import client

def load_user_memory(member_id: str, message_log: list, max_turns: int = 3) -> list:
    """
    요청으로 받은 message_log 중 member_id에 해당하는 최근 max_turns 대화를 리스트로 반환
    """
    user_history = [
        row for row in message_log
        if str(row.get("member_id")) == str(member_id) and row.get("type") in ["SEND", "RECEIVE"]
    ]
    return user_history[-(max_turns * 2):]  # 1턴 = 사용자 + 챗봇 → 총 2개씩


def summarize_memory(memory_messages: list) -> str:
    """
    최근 memory_messages를 요약하여 자연스럽게 대화를 이어갈 수 있는 시작 멘트 생성
    ex: "저번엔 학업 문제로 힘드셨는데, 이번에도 비슷한 고민이신가요?"
    """
    if not memory_messages:
        return "최근에는 어떤 일이 있으셨나요? 편하게 이야기해 주세요."

    try:
        dialogue = "\n".join([
            f"{'사용자' if msg.get('type') == 'SEND' else '상담사'}: {msg.get('message', '').strip()}"
            for msg in memory_messages if isinstance(msg, dict) and msg.get("message", "").strip()
        ])

        # ✅ 너무 길 경우 뒤에서 1500자만 사용 (문맥 흐름은 유지)
        if len(dialogue) > 1500:
            dialogue = dialogue[-1500:]

        prompt = f"""다음은 사용자와 상담사의 과거 대화입니다. 이 내용을 상담 시작에 적절한 한두 문장으로 자연스럽게 요약해 주세요. 공감하는 말투로, 너무 기계적으로 반복하지 말고 따뜻하게 말해 주세요.

[대화 내역]
{dialogue}

[응답 예시]
- 지난번엔 감기 몸살로 많이 힘들어하셨는데, 요즘은 좀 어떠세요?
- 전엔 대인관계로 고민이 많으셨는데, 그 이후로 조금 나아지셨을까요?
- 그땐 감정이 자주 요동쳤다고 하셨는데, 지금은 좀 괜찮아지셨을까요?
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 사용자의 지난 상담 흐름을 자연스럽고 따뜻하게 이어주는 공감형 심리상담사야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=120
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Memory Summarization Error] {e}")
        return "지난 대화 내용을 불러오는 데 문제가 있었어요. 요즘은 기분이 좀 어떠세요?"
