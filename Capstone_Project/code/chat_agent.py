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
                "엄마처럼 포근하게 사용자의 감정을 감싸줍니다.\n"
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
            base_prompt += f"\n\n사용자의 감정은 '{self.emotion}'입니다. 이 감정에 공감하되, 반복 없이 1~2문장으로 반응하세요."
        if self.risk.lower() in ["중간", "높음"]:
            base_prompt += "\n\n민감한 상태일 수 있으므로 따뜻하지만 짧고 신중하게 말해주세요."

        if memory and self.turn == 0:
            base_prompt += f"\n\n이전 대화 내용이 있습니다:\n{memory}\n\n👉 첫 응답은 특히 짧고 단정적으로 시작해주세요. 공감은 1문장으로, 질문은 간결하게 하세요."

        if self.turn >= 3:
            base_prompt += "\n\n지금은 상담이 어느 정도 진행된 상태입니다. 실천 전략이 필요하면 핵심만 1~2문장으로 제시하세요."

        theory_instruction = ""
        if self.intent == "상담 원함" and theory:
            theory_instruction = (
                "\n\n상담 이론을 설명하지 말고, 핵심 개념만 사용자 상황에 짧게 적용해주세요. 말은 2문장 이내로!"
            )

        closing_instruction = "\n\n⛔ 절대 유사 표현 반복하지 마세요. 응답은 반드시 1~2문장으로 제한하고, 질문은 한 문장만 덧붙이세요."

        return f"""{base_prompt}{theory_instruction}{closing_instruction}

[과거 대화 요약]
{memory}

[상담 이론 요약]
{theory}

[사용자 입력]
{user_input}

[상담자 응답]
""".strip()

    def respond(self, user_input: str, memory: str = "", theory: list = None, max_tokens: int = 150) -> str:
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
