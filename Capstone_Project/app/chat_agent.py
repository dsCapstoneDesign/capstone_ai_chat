from app.config.openai_client import client
from app.memory_manager import summarize_memory, load_user_memory, is_first_entry
import random

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
            base_prompt += (
                f"\n\n[현재 감정 상태]\n"
                f"사용자는 '{self.emotion}'라는 감정을 표현했습니다. 이 감정에 대해 진심 어린 공감 + 짧은 질문을 포함하세요."
            )

        if self.risk.lower() in ["중간", "높음"]:
            base_prompt += "\n위험도가 높으므로 조심스럽고 간결하게 표현해주세요."

        core_instruction = (
            "\n\n🧠 당신은 진짜 사람처럼 따뜻하고 공감력 있는 전문 심리상담자입니다.\n"
            "상담은 진심 어린 공감으로 시작되고, 감정에 정확히 반응하며, 다음 말을 자연스럽게 이어가야 합니다.\n"
            "사용자의 말 속에서 감정, 상황, 욕구를 파악하고, 다음 두 가지를 조합하여 응답하세요:\n"
            "- (1) 감정에 대한 공감 표현\n"
            "- (2) 감정을 유도하거나, 조금 더 깊이 물어볼 수 있는 짧은 질문\n"
            "절대 판단하지 말고, 설명도 최소화하며, 말은 짧고 진심 있게 해야 합니다.\n"
            "페르소나에 따라 말투만 달라지며, 진심과 흐름은 모두 동일합니다.\n"
            "반드시 2문장을 넘기지 마세요."
        )

        return f"{base_prompt}\n{core_instruction}\n\n[대화 흐름 요약]\n{memory}\n\n[상담 이론 요약]\n{theory}\n\n[사용자 말]\n{user_input}"

    def respond(self, user_input: str, message_log: list, member_id: str, theory: list = None, max_tokens: int = 400) -> str:
        with open("debug_log.txt", "a") as f:
            f.write(f"\n🧩 respond 진입 | user_input: {user_input}\n")

        if is_first_entry(member_id, message_log):
            return "안녕하세요! 처음 오셨군요. 편하게 이야기해 주세요. 😊"

        memory_raw = load_user_memory(member_id, message_log)
        memory = summarize_memory(memory_raw)

        if len(user_input) > 10:
            self.detect_mode_via_llm(user_input, memory)

        theory_text = "\n".join([f"[{name}] {desc}" for name, desc in theory]) if isinstance(theory, list) else theory or ""
        system_prompt = self.build_prompt(user_input, memory, theory_text)

        with open("debug_log.txt", "a") as f:
            f.write("🧠 build_prompt 완료. GPT 호출 시작...\n")

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"(과거 대화)\n{memory}"},
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

            with open("debug_log.txt", "a") as f:
                f.write(f"📌 사용된 모델: {response.model}\\n")

            fallback_candidates = [
                "지금 말해주신 것만으로도 충분히 소중해요. 혹시 더 나눠주실 수 있을까요?",
                "마음이 복잡하셨겠어요. 편하실 때 천천히 이어서 말해주셔도 괜찮아요.",
                "잘 전달되었어요. 어떤 부분부터 이야기하고 싶은지 알려주실래요?"
            ]

            if (
                len(reply) < 15 or
                any(x in reply.lower() for x in ["잘 모르겠어요", "죄송", "어려워요", "확실하지 않아요"])
            ):
                with open("debug_log.txt", "a") as f:
                    f.write("🧩 응답 품질 낮음 - fallback 문구 반환\n")
                return random.choice(fallback_candidates)

            return reply

        except Exception as e:
            with open("debug_log.txt", "a") as f:
                f.write(f"⚠️ GPT 호출 실패: {e}\n")
            return "조금 더 구체적으로 이야기해주실 수 있을까요?"