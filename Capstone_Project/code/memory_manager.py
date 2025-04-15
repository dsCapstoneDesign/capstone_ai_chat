from openai import OpenAI
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_user_memory(member_id: str, message_log: list, max_turns: int = 3) -> list:
    """
    요청으로 받은 message_log 중 member_id에 해당하는 최근 max_turns 대화를 리스트로 반환
    """
    user_history = [
        row for row in message_log
        if row.get("member_id") == member_id and row.get("type") in ["SEND", "RECEIVE"]
    ]
    return user_history[-(max_turns * 2):]  # 1턴 = 사용자 + 봇 → 총 2개씩

def summarize_memory(memory_messages: list) -> str:
    """
    최근 memory_messages를 요약하여 자연스럽게 대화를 이어갈 수 있는 시작 멘트 생성
    ex: "저번엔 학업 문제로 힘드셨는데, 이번에도 비슷한 고민이신가요?"
    """
    if not memory_messages:
        return "최근에는 어떤 일이 있으셨나요? 편하게 이야기해 주세요."

    try:
        dialogue = "\n".join([
            f"{'USER' if msg.get('type') == 'SEND' else 'BOT'}: {msg.get('message', '').strip()}"
            for msg in memory_messages if isinstance(msg, dict)
        ])

        prompt = f"""다음은 사용자의 과거 상담 대화입니다. 내용을 간단히 요약하여 상담 시작 멘트로 만들어 주세요.

[과거 대화]
{dialogue}

[응답 예시]
지난번엔 학업 문제로 많이 지쳐있었는데, 요즘은 조금 괜찮으신가요?
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 사용자의 지난 상담 흐름을 자연스럽게 이어주는 심리상담사야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return "지난 대화 요약에는 문제가 있었어요. 그래도 최근 마음은 어떤가요?"
