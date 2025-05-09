from app.config.openai_client import client

def load_user_memory(member_id: str, message_log: list) -> list:
    user_msgs = [m for m in message_log if str(m.get("member_id")) == str(member_id) and m.get("sender") == "USER"]
    bot_msgs = [m for m in message_log if str(m.get("member_id")) == str(member_id) and m.get("sender") == "BOT"]

    selected = user_msgs[-5:] + bot_msgs[-10:]
    selected.sort(key=lambda x: x.get("send_time", 0))
    print(f"🧩 load_user_memory: 선택된 메시지 {len(selected)}개")
    return selected

def is_first_entry(member_id: str, message_log: list) -> bool:
    has_history = any(m for m in message_log if str(m.get("member_id")) == str(member_id))
    if not has_history:
        print("NULL")
    return not has_history

def summarize_memory(memory_messages: list) -> str:
    if not memory_messages:
        print("⚠️ [summarize_memory] memory_messages가 비어 있음")
        return "최근에는 어떤 일이 있으셨나요? 편하게 이야기해 주세요."

    try:
        dialogue = "\n".join([
            f"{'사용자' if msg.get('sender') == 'USER' else '상담사'}: {msg.get('message', '').strip()}"
            for msg in memory_messages if msg.get("message", "").strip()
        ])

        print(f"🧠 [요약 대상 대화 - 총 {len(memory_messages)}개]\n{dialogue}\n---")

        if len(dialogue) > 1500:
            dialogue = dialogue[-1500:]

        prompt = f"""너는 공감 능력이 뛰어난 심리상담사야. 아래 대화를 참고해 사용자의 최근 감정과 흐름을 자연스럽게 이어주는 한두 문장 정도의 따뜻한 상담 시작 멘트를 만들어줘.
형식은 너무 기계적으로 반복하지 말고, 진짜 상담사처럼 공감하고 관심을 표현해줘.

[대화 내역]
{dialogue}

[상담 시작용 요약 멘트]
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 사용자의 지난 상담 흐름을 자연스럽고 따뜻하게 이어주는 공감형 심리상담사야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
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
