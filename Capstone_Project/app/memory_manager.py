from app.config.openai_client import client
from collections import Counter
import re
from datetime import datetime

# 캐시용 딕셔너리 (세션 유지 시 활용)
_summary_cache = {}

def load_user_memory(member_id: str, message_log: list) -> list:
    # 최근 메시지 더 많이 불러오기 (유저 중심)
    user_msgs = [m for m in message_log if str(m.get("member_id")) == str(member_id) and m.get("sender") == "USER"]
    bot_msgs = [m for m in message_log if str(m.get("member_id")) == str(member_id) and m.get("sender") == "BOT"]

    selected = user_msgs[-10:] + bot_msgs[-10:]  # 더 많은 문맥 확보
    selected.sort(key=lambda x: x.get("send_time", 0))
    print(f"🧩 load_user_memory: 선택된 메시지 {len(selected)}개")
    return selected

def is_first_entry(member_id: str, message_log: list) -> bool:
    has_history = any(m for m in message_log if str(m.get("member_id")) == str(member_id))
    if not has_history:
        print("NULL")
    return not has_history

def extract_keywords(dialogue_text: str, top_k: int = 3) -> list:
    tokens = re.findall(r"[가-힣]+", dialogue_text)
    counter = Counter(tokens)
    common = [word for word, _ in counter.most_common(top_k)]
    return common

def summarize_memory(memory_messages: list, persona: str = "위로형", member_id: str = "") -> str:
    if not memory_messages:
        print("⚠️ [summarize_memory] memory_messages가 비어 있음")
        return "최근에는 어떤 일이 있으셨나요? 편하게 이야기해 주세요."

    # 캐시 확인
    if member_id in _summary_cache:
        print("✅ [캐시 사용] 이전 요약 반환")
        return _summary_cache[member_id]

    try:
        dialogue = "\n".join([
            f"{'사용자' if msg.get('sender') == 'USER' else '상담사'}: {msg.get('message', '').strip()}"
            for msg in memory_messages if msg.get("message", "").strip()
        ])

        print(f"🧠 [요약 대상 대화 - 총 {len(memory_messages)}개]\n{dialogue}\n---")

        if len(dialogue) > 1500:
            dialogue = dialogue[-1500:]  # 토큰 절약

        keywords = extract_keywords(dialogue)

        system_prompt_map = {
            "위로형": (
                "너는 따뜻한 공감 능력을 가진 심리상담사야.\n"
                "사용자의 지난 대화를 바탕으로 최근 감정과 고민을 자연스럽게 이어주는 상담 시작 멘트를 1~2문장으로 생성해.\n"
                "✔️ 감정과 중심 주제를 직접 언급해주면 좋아.\n"
                "✔️ 반복되는 말투를 피하고 구체적인 단서로 이어줘."
            ),
            "논리형": (
                "너는 분석적인 심리상담사야.\n"
                "사용자의 대화를 바탕으로 핵심 고민을 간결하게 요약하고, 다시 대화를 이어가도록 1~2문장으로 시작 멘트를 만들어줘.\n"
                "✔️ 정확한 단서 중심.\n"
                "✔️ 공감보단 '정리하고 이어가기' 느낌이 좋음."
            ),
            "긍정형": (
                "너는 긍정 에너지를 전하는 상담사야.\n"
                "지난 대화의 키워드를 가볍게 언급하며 에너지 넘치고 밝은 말투로 오늘 대화를 이어줄 1~2문장을 제안해.\n"
                "✔️ 희망적인 언급 + 유쾌한 분위기 환기."
            )
        }

        system_prompt = system_prompt_map.get(persona, system_prompt_map["위로형"])

        prompt = f"""
아래는 사용자의 지난 대화 내용이야.
- 중심 주제를 자연스럽게 언급해 줘 (예: 공부, 인간관계, 감정 등)
- 감정 키워드만 나열하지 말고, 왜 그런 감정이었는지 연결해줘
- 표현은 자연스럽고 상담사가 다음 턴을 이끌 수 있게 해줘

[대화 내역]
{dialogue}

[핵심 키워드]
{', '.join(keywords)}

[예시]
- 공부가 많이 부담스러우셨던 것 같아요. 요즘은 좀 나아지셨을까요?
- 인간관계 때문에 힘들다고 하셨는데, 지금은 어떤가요?
- 무기력하다고 하셨던 게 기억나요. 오늘은 어떤 이야기부터 나눠볼까요?

[너의 응답 시작]
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=120
        )
        result = response.choices[0].message.content.strip()
        print(f"📝 [요약 결과]\n{result}\n---")

        if len(result) < 5:
            print("⚠️ 요약 실패 또는 응답 너무 짧음")

        # 캐싱 저장
        if member_id:
            _summary_cache[member_id] = result

        return result

    except Exception as e:
        print(f"❌ [Memory Summarization Error] {e}")
        return "지난 대화 내용을 불러오는 데 문제가 있었어요. 요즘은 기분이 좀 어떠세요?"
