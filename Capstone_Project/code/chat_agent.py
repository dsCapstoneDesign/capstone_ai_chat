from config.openai_client import client  # ✅ 공통 client import

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
                "기분이 어떤지 자연스럽게 물어보며 대화를 시작하고,\n"
                "힘든 이야기를 털어놓을 수 있도록 편안한 분위기를 만들어줍니다.\n"
                "응답은 2~3문장 이내로 짧고 핵심적으로 말해주세요. 반복하거나 장황하게 설명하지 마세요."
            ),
            "논리분석형": (
                "당신은 차분하고 객관적인 시선으로 상담하는 분석 중심 상담자입니다.\n"
                "사용자의 고민을 정리하고, 원인을 파악해 문제 해결을 도와줍니다.\n"
                "감정보다는 상황 분석, 행동 구조화에 집중하며,\n"
                "응답은 2~3문장 이내로 짧고 간결하게 말해주세요. 핵심만 전달하세요."
            ),
            "유쾌한친구형": (
                "당신은 친한 친구처럼 편하게 이야기하는 상담자입니다.\n"
                "라이트한 톤, 반말과 이모지를 사용해 유쾌하게 대화합니다.\n"
                "응답은 짧고 자연스러운 대화처럼 해주세요. 너무 길게 설명하지 마세요."
            ),
            "여자친구형": (
                "당신은 감성적이고 따뜻한 여자친구처럼 말하는 상담자입니다.\n"
                "응답은 다정하고 공감되지만, 너무 길지 않게 말해주세요. 핵심만 담아서 전달하세요."
            ),
            "남자친구형": (
                "당신은 듬직하고 부드러운 남자친구처럼 말하는 상담자입니다.\n"
                "과하지 않게 감정을 표현하며, 응답은 짧고 안정적으로 말해주세요. 문장이 길지 않도록 하세요."
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
            base_prompt += f"\n\n현재 사용자의 감정은 '{self.emotion}'입니다. 감정을 존중하면서 너무 길지 않게 말해주세요."
        if self.risk.lower() in ["중간", "높음"]:
            base_prompt += "\n\n상대방이 예민할 수 있으니 조심스럽게 표현하되, 말은 짧고 간결하게 해주세요."

        if memory and self.turn == 0:
            base_prompt += f"\n\n이전에 사용자와 다음과 같은 대화를 나눈 적이 있습니다:\n{memory}\n그 내용을 바탕으로 자연스럽게 이어서 짧고 핵심적으로 대답해주세요."

        if self.turn >= 3:
            base_prompt += "\n\n상담이 몇 차례 이어졌으므로 실천 전략이 필요한 경우 간결히 제시해주세요."

        if self.turn >= 6:
            base_prompt += "\n\n사용자가 신뢰를 보이고 있으므로 조금 더 명확한 조언을 제시해도 좋지만, 말은 짧게 하세요."

        theory_instruction = ""
        if self.intent == "상담 원함" and theory:
            theory_instruction = (
                "\n\n상담 이론을 적용하되, 설명은 최소화하고 사용자의 상황에 맞춰 핵심만 담아 간결히 말해주세요."
            )

        closing_instruction = "\n\n응답은 꼭 2~3문장 이내로 짧게 마무리해주세요. 다음 대화를 유도하되, 너무 많은 내용을 한꺼번에 말하지 마세요."

        return f"""{base_prompt}{theory_instruction}{closing_instruction}

[과거 대화 요약]
{memory}

[상담 이론 요약]
{theory}

[사용자 입력]
{user_input}

[상담자 응답]
""".strip()

    def respond(self, user_input: str, memory: str = "", theory: list = None, max_tokens: int = 200) -> str:
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
                len(reply) < 30 or
                any(x in reply.lower() for x in ["잘 모르겠어요", "죄송", "그건 어려워요", "확실하지 않아요"])
            ):
                return "조금 더 구체적으로 이야기해주실 수 있을까요? 당신의 이야기를 듣고 싶어요."

            return reply

        except Exception as e:
            return f"[⚠️ OpenAI 응답 실패] {e}"
