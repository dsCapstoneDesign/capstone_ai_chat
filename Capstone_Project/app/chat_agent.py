from app.config.openai_client import client
from app.memory_manager import summarize_memory, load_user_memory, is_first_entry

with open("debug_log.txt", "a") as f:
    f.write("✅ chat_agent.py가 FastAPI에 로딩되었습니다!\n")

class ChatAgent:
    def __init__(self, persona="위로형"):
        self.mode = "casual"
        self.intent = "상담 원함"
        self.emotion = ""
        self.risk = ""
        self.persona = persona

        self.persona_prompts = {
            "위로형": (
                "[페르소나: 위로형]\n"
                "당신은 따뜻하고 부드러운 말투를 사용하는 상담자입니다.\n"
                "공감은 하되, 설명은 최소화하고 핵심만 말하세요.\n"
                "응답은 1~2문장 이내로 제한하고, 같은 말을 반복하지 마세요."
            ),
            "논리형": (
                "[페르소나: 논리형]\n"
                "당신은 차분하고 객관적인 시선으로 상담하는 분석형 상담자입니다.\n"
                "상황 정리와 문제 해결 중심으로 이야기하지만, 말은 짧고 핵심적이어야 합니다.\n"
                "1~2문장 이내로 간결하게 말하고, 유사 문장 반복은 절대 하지 마세요."
            ),
            "긍정형": (
                "[페르소나: 긍정형]\n"
                "당신은 유쾌하고 긍정적인 말투를 사용하는 상담자입니다.\n"
                "친근하고 명랑한 말투로 짧게 반응하되, 핵심을 놓치지 마세요.\n"
                "응답은 1~2문장 이내로 제한하고, 같은 유형의 말을 반복하지 마세요."
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
                model="gpt-4o",
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
                elif "의도:" in line:
                    self.intent = line.split(":")[-1].strip()
                elif "감정:" in line:
                    self.emotion = line.split(":")[-1].strip()
                elif "위험도:" in line:
                    self.risk = line.split(":")[-1].strip()

            with open("debug_log.txt", "a") as f:
                f.write(f"🔍 감정 분석 결과 - mode: {self.mode}, emotion: {self.emotion}, risk: {self.risk}\n")

        except Exception as e:
            with open("debug_log.txt", "a") as f:
                f.write(f"[⚠️ 모드 예측 실패] {e}\n")

    def build_prompt(self, user_input: str, memory: str = "", theory: str = "") -> str:
        base_prompt = self.get_persona_prompt()

        if self.emotion:
            base_prompt += f"\n\n[현재 감정 상태]\n사용자는 '{self.emotion}'라는 감정을 표현했습니다."

        if self.risk.lower() in ["중간", "높음"]:
            base_prompt += "\n위험도가 높으므로 조심스럽고 간결하게 표현해주세요."

        core_instruction = (
            "\n\n🧠 당신은 심리상담을 전문으로 하는 AI 상담자입니다.\n"
            "사용자와 대화를 이어가며 감정을 이해하고 공감하세요.\n"
            "각 상황에 따라 적절한 반응 유형(공감, 질문, 방향 제시)을 조합해 사용하세요.\n"
            "무조건 모두 포함하지 말고, 자연스러운 2가지를 선택하세요.\n"
            "응답은 1~2문장 이내로, 유사 문장 반복은 절대 금지합니다.\n"
            "페르소나에 맞는 말투를 유지하세요."
        )

        return f"{base_prompt}\n{core_instruction}\n\n[과거 대화 요약]\n{memory}\n\n[상담 이론 요약]\n{theory}\n\n[사용자 입력]\n{user_input}\n\n[상담자 응답]"

    def respond(self, user_input: str, message_log: list, member_id: str, theory: list = None, max_tokens: int = 150) -> str:
        with open("debug_log.txt", "a") as f:
            f.write(f"\n🧩 respond 진입 | user_input: {user_input}\n")

        if is_first_entry(member_id, message_log):
            return "안녕하세요! 처음 오셨군요. 편하게 이야기해 주세요. 😊"

        memory_raw = load_user_memory(member_id, message_log)
        memory = summarize_memory(memory_raw)

        self.detect_mode_via_llm(user_input, memory)

        theory_text = "\n".join([f"[{name}] {desc}" for name, desc in theory]) if isinstance(theory, list) else theory or ""
        system_prompt = self.build_prompt(user_input, memory, theory_text)

        with open("debug_log.txt", "a") as f:
            f.write("🧠 build_prompt 완료. GPT 호출 시작...\n")

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": memory},
                {"role": "user", "content": user_input}
            ]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.4 if self.mode == "counseling" else 0.7,
                max_tokens=max_tokens
            )
            reply = response.choices[0].message.content.strip().replace('\n', ' ')

            with open("debug_log.txt", "a") as f:
                f.write(f"✅ GPT 응답 수신: {reply}\n")

            if (
                len(reply) < 15 or
                any(x in reply.lower() for x in ["잘 모르겠어요", "죄송", "어려워요", "확실하지 않아요"])
            ):
                with open("debug_log.txt", "a") as f:
                    f.write("🧩 응답 품질 낮음 - fallback 문구 반환\n")
                return "조금 더 구체적으로 이야기해주실 수 있을까요?"

            return reply

        except Exception as e:
            with open("debug_log.txt", "a") as f:
                f.write(f"⚠️ GPT 호출 실패: {e}\n")
            return "조금 더 구체적으로 이야기해주실 수 있을까요?"
