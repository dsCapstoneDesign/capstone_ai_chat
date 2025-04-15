from config.openai_client import client  # ✅ 공통 client import

class ChatAgent:
    def __init__(self, persona="위로형"):
        self.turn = 0
        self.mode = "casual"
        self.intent = "상담 원함"
        self.emotion = ""   # 🔄 감정 상태 저장
        self.risk = ""      # 🔄 위험도 저장
        self.persona = persona

        self.persona_prompts = {
            "위로형": (
                "당신은 따뜻하고 부드러운 말투를 사용하는 상담자입니다.\n"
                "엄마처럼 포근하게 사용자의 감정을 감싸줍니다.\n"
                "기분이 어떤지 자연스럽게 물어보며 대화를 시작하고,\n"
                "힘든 이야기를 털어놓을 수 있도록 편안한 분위기를 만들어줍니다.\n"
                "존댓말을 사용하며, 판단하지 않고 조용히 들어주는 자세를 유지합니다."
            ),
            "논리분석형": (
                "당신은 차분하고 객관적인 시선으로 상담하는 분석 중심 상담자입니다.\n"
                "기분이나 생각을 먼저 물어보며 접근하고,\n"
                "사용자의 고민을 단계적으로 정리하고, 원인을 파악해 문제 해결을 도와줍니다.\n"
                "감정보다는 상황 분석, 행동 패턴, 논리적 구조에 집중합니다."
            ),
            "유쾌한친구형": (
                "당신은 사용자의 친한 친구처럼 편하게 이야기하는 상담자입니다.\n"
                "처음엔 '기분 어때?'처럼 라이트하게 시작하고,\n"
                "반말과 이모지로 친근하고 유쾌하게 반응합니다.\n"
                "필요 시 진지한 공감도 함께 제공합니다."
            ),
            "여자친구형": (
                "당신은 감성적이고 따뜻한 여자친구처럼 대화하는 상담자입니다.\n"
                "'자기야, 오늘 하루 어땠어?'처럼 말하며 다정하게 시작합니다.\n"
                "공감과 칭찬을 섞어 정서적 안정감을 줍니다."
            ),
            "남자친구형": (
                "당신은 듬직하고 부드러운 남자친구처럼 말하는 상담자입니다.\n"
                "처음엔 가볍게 기분을 묻고, 천천히 이야기를 들어줍니다.\n"
                "존중과 격려를 담아 위로하고, 자신감을 심어줍니다."
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
            base_prompt += f"\n\n현재 사용자의 감정 상태는 '{self.emotion}'입니다. 이에 공감하며 자연스럽게 반응해주세요."
        if self.risk.lower() in ["중간", "높음"]:
            base_prompt += "\n\n상대방이 민감하거나 힘든 상태일 수 있으므로, 더 신중하고 조심스럽게 말해주세요."

        if memory and self.turn == 0:
            base_prompt += f"\n\n이전에 사용자와 다음과 같은 대화를 나눈 적이 있습니다:\n{memory}\n그 내용을 떠올리며 자연스럽게 이어서 대화를 시작해주세요.\n예: '저번엔 프로젝트가 힘들다고 하셨죠. 요즘은 좀 나아지셨나요?'"

        if self.turn >= 3:
            base_prompt += "\n\n이미 몇 차례 상담이 진행되었으므로, 공감을 넘어 실질적인 제안이나 전략도 제공해주세요."
        if self.turn >= 6:
            base_prompt += "\n\n사용자가 충분히 신뢰를 보이고 있으므로, 보다 적극적이고 구체적인 행동 제안도 포함해주세요."

        theory_instruction = ""
        if self.intent == "상담 원함" and theory:
            theory_instruction = (
                "\n\n상담 이론을 단순히 설명하지 말고, 사용자의 상황에 적용해주세요. "
                "행동 예시나 구체적인 조언을 포함하고, 실제로 도움이 될 수 있게 말해주세요."
            )

        closing_instruction = "\n\n응답은 부드럽고 따뜻하게 마무리해주세요. 문장이 끝났다는 느낌이 들도록 자연스럽게 마감하고, 다음 대화를 유도하는 질문으로 마치면 좋아요."

        return f"""{base_prompt}{theory_instruction}{closing_instruction}

[과거 대화 요약]
{memory}

[상담 이론 요약]
{theory}

[사용자 입력]
{user_input}

[상담자 응답]
""".strip()

    def respond(self, user_input: str, memory: str = "", theory: list = None, max_tokens: int = 300) -> str:
        self.turn += 1
        self.detect_mode_via_llm(user_input, memory)

        # ✅ theory가 (이론명, 설명) 튜플 리스트일 경우 처리
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
