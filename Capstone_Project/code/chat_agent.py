from config.openai_client import client

class ChatAgent:
    def __init__(self, persona="위로형"):
        self.turn = 0
        self.mode = "casual"
        self.intent = "상담 원함"
        self.emotion = ""
        self.risk = ""
        self.persona = persona

        self.persona_prompts = {
            "위로형": (
                "당신은 따뜻하고 부드러운 말투를 사용하는 상담자입니다.\n"
                "공감은 하되, 설명은 최소화하고 핵심만 말하세요.\n"
                "응답은 1~2문장 이내로 제한하고, 같은 말을 반복하지 마세요."
            ),
            "논리분석형": (
                "당신은 차분하고 객관적인 시선으로 상담하는 분석형 상담자입니다.\n"
                "상황 정리와 문제 해결 중심으로 이야기하지만, 말은 짧고 핵심적이어야 합니다.\n"
                "1~2문장 이내로 간결하게 말하고, 유사 문장 반복은 절대 하지 마세요."
            ),
            "유쾌한친구형": (
                "당신은 유쾌하고 친근한 말투를 사용하는 상담자입니다.\n"
                "이모지나 반말을 가볍게 섞을 수 있으나, 응답은 짧게!\n"
                "핵심만 담아 1~2문장 이내로 말하고 같은 유형의 말을 반복하지 마세요."
            ),
            "여자친구형": (
                "당신은 감성적이고 따뜻한 여자친구처럼 말하는 상담자입니다.\n"
                "공감하며 이야기하되, 응답은 짧고 핵심적이어야 합니다.\n"
                "반복되는 문장 금지. 응답은 1~2문장으로 제한하세요."
            ),
            "남자친구형": (
                "당신은 듬직하고 부드러운 남자친구처럼 말하는 상담자입니다.\n"
                "응답은 따뜻하지만 절대 길어지지 않게, 1~2문장 안에서 핵심을 담아 말하세요.\n"
                "비슷한 말 반복은 하지 말고 간결하게 마무리하세요."
            )
        }

    def get_persona_prompt(self):
        return self.persona_prompts.get(self.persona, self.persona_prompts["위로형"])

    def get_greeting(self) -> str:
        greetings = {
            "위로형": "안녕하세요. 오늘 기분은 어떠세요?",
            "논리분석형": "안녕하세요. 오늘 기분이 어땠는지 들어보고 싶어요.",
            "유쾌한친구형": "안녕~ 오늘 기분은 어때? 😊",
            "여자친구형": "안녕:) 오늘 하루 어땠어?",
            "남자친구형": "안녕, 오늘 하루는 어땠어?"
        }
        return greetings.get(self.persona, greetings["위로형"])

    def detect_mode_via_llm(self, user_input: str, memory: str = ""):
        prompt = f"""아래 사용자 입력과 과거 대화를 보고, 상담 단계(casual, explore, counseling), 감정 키워드, 위험도, 상담 의도를 판단해주세요.

[과거 대화]
{memory}

[사용자 입력]
{user_input}

[응답 형식 예시]
단계: counseling
감정: 무기력, 불안
위험도: 중간
의도: 상담 원함
"""
        try:
            result = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "너는 심리상담 대화 흐름을 분석하는 조력자야."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=150
            )
            content = result.choices[0].message.content.strip()
            for line in content.splitlines():
                if "단계:" in line:
                    self.mode = line.split(":")[-1].strip().lower()
                if "의도:" in line:
                    self.intent = line.split(":")[-1].strip()
                if "감정:" in line:
                    self.emotion = line.split(":")[-1].strip()
                if "위험도:" in line:
                    self.risk = line.split(":")[-1].strip()
        except Exception as e:
            print(f"[⚠️ 모드 예측 실패] {e} → 기존 모드 유지: {self.mode}, {self.intent}")

    def build_prompt(self, user_input: str, memory: str = "", theory: str = "") -> str:
        base_prompt = self.get_persona_prompt()

        if self.emotion:
            base_prompt += f"\n\n사용자의 감정은 '{self.emotion}'입니다. 이 감정에 맞춰 부드럽고 신중하게 반응해주세요."

        if self.risk.lower() in ["중간", "높음"]:
            base_prompt += "\n\n민감한 상태일 수 있으므로 조심스럽고 간결하게 표현해주세요."

        core_instruction = (
            "\n\n🧠 당신은 심리상담을 전문으로 하는 AI입니다.\n"
            "대화의 목적은 사용자의 감정을 이해하고, 안정적인 대화를 이어가는 것입니다.\n\n"
            "다음을 기준으로 응답을 구성하세요:\n"
            "- 공감 표현을 통해 사용자 감정을 수용하세요. (30~40%)\n"
            "- 대화를 이어가기 위해 질문을 유도하세요. (30~40%)\n"
            "- 필요한 경우 조심스럽게 현실적인 방향을 제시하세요. (20~30%)\n\n"
            "단, 공감 + 질문 + 방향 제시를 모두 포함하려고 하지 마세요.\n"
            "→ 상황에 맞게 2가지 정도만 섞되, 가장 자연스러운 흐름을 선택하세요.\n\n"
            "⛔ 유사한 말을 반복하지 마세요.\n"
            "⛔ '무엇이 그렇게 만들었나요? → 어떤 점이 그랬을까요?'처럼 구조만 다른 반복 질문은 피하세요.\n"
            "✅ 응답은 반드시 1~2문장 이내여야 하며, 사람처럼 자연스럽게 이어지도록 하세요.\n"
            "✅ 응답 후 무조건 질문으로 마무리할 필요는 없으며, 질문 없이 공감만 하는 응답도 허용됩니다."
        )

        return f"{base_prompt}\n{core_instruction}\n\n[과거 대화 요약]\n{memory}\n\n[상담 이론 요약]\n{theory}\n\n[사용자 입력]\n{user_input}\n\n[상담자 응답]"

    def respond(self, user_input: str, memory: str = "", theory: list = None, max_tokens: int = 150) -> str:
        if self.turn == 0:
            self.turn += 1
            return self.get_greeting()

        self.turn += 1
        self.detect_mode_via_llm(user_input, memory)

        if isinstance(theory, list) and theory and isinstance(theory[0], tuple):
            theory_text = "\n".join([f"[{name}] {desc}" for name, desc in theory])
        else:
            theory_text = theory or ""

        prompt = self.build_prompt(user_input, memory, theory_text)

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.get_persona_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4 if self.mode == "counseling" else 0.7,
                max_tokens=max_tokens
            )
            reply = response.choices[0].message.content.strip()

            if (
                len(reply) < 15 or
                any(x in reply.lower() for x in ["잘 모르겠어요", "죄송", "그건 어려워요", "확실하지 않아요"])
            ):
                return "조금 더 구체적으로 이야기해주실 수 있을까요?"

            return reply

        except Exception as e:
            return f"[⚠️ OpenAI 응답 실패] {e}"
