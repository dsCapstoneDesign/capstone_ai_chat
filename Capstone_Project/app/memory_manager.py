from app.config.openai_client import client

def load_user_memory(member_id: str, message_log: list) -> list:
    """
    사용자 ID에 해당하는 전체 대화 중, 핵심 TURN들을 골라 정리
    - 사용자 메시지는 마지막 5개
    - 챗봇 응답은 마지막 10개
    """
    user_msgs = [m for m in message_log if str(m.get("member_id")) == str(member_id) and m.get("type") == "SEND"]
    bot_msgs = [m for m in message_log if str(m.get("member_id")) == str(member_id) and m.get("type") == "RECEIVE"]

    selected = user_msgs[-5:] + bot_msgs[-10:]
    selected.sort(key=lambda x: x.get("send_time", 0))  # 시간순 정렬 (옵션)

    print(f"🧩 load_user_memory: 선택된 메시지 {len(selected)}개")
    return selected


def summarize_memory(memory_messages: list) -> str:
    """
    대화 메시지를 바탕으로 상담 시작용 자연스러운 요약 멘트를 생성
    """
    if not memory_messages:
        print("⚠️ [summarize_memory] memory_messages가 비어 있음")
        return "최근에는 어떤 일이 있으셨나요? 편하게 이야기해 주세요."

    try:
        dialogue = "\n".join([
            f"{'사용자' if msg.get('type') == 'SEND' else '상담사'}: {msg.get('message', '').strip()}"
            for msg in memory_messages if msg.get("message", "").strip()
        ])

        print(f"🧠 [요약 대상 대화 - 총 {len(memory_messages)}개]\n{dialogue}\n---")

        # 길이 초과 방지 (최대 1500자)
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
        result = response.choices[0].message.content.strip()
        print(f"📝 [요약 결과]\n{result}\n---")

        if len(result) < 5:
            print("⚠️ 요약 실패 또는 응답 너무 짧음")

        return result

    except Exception as e:
        print(f"❌ [Memory Summarization Error] {e}")
        return "지난 대화 내용을 불러오는 데 문제가 있었어요. 요즘은 기분이 좀 어떠세요?"
