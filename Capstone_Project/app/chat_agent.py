import random
from app.config.openai_client import client
from app.memory_manager import summarize_memory, load_user_memory
from app.wiki_searcher import WikiSearcher
from datetime import datetime
import json

def is_first_entry(member_id, message_log):
    user_msgs = [m for m in message_log if m.get("sender") == "USER" and str(m.get("member_id")) == str(member_id)]
    return len(user_msgs) <= 1

class ChatAgent:
    def __init__(self, persona="위로형"):
        self.mode = "casual"
        self.intent = "상담 원함"
        self.emotion = ""
        self.risk = ""
        self.persona = persona
        self.searcher = WikiSearcher()
        self.theory_data = self.load_theories()
        self.theory = None

        self.persona_prompts = {
            "위로형": (
                "[페르소나: 위로형]\n"
                "당신은 감정을 우선시하며, 사용자가 충분히 위로받고 있다고 느끼도록 공감하는 따뜻한 상담자입니다.\n"
                "말투는 부드럽고 다정해야 하며, 감정을 알아주는 형태로 반응합니다.\n"
                "공감 50% + 질문 30% + 제안 20%"
            ),
            "논리형": (
                "[페르소나: 논리형]\n"
                "객관적으로 상황을 정리하고, 논리적 사고로 문제 해결을 돕는 상담자입니다.\n"
                "질문과 제안이 중심이며, 공감은 최소화하세요.\n"
                "질문 50% + 제안 40% + 공감 10%"
            ),
            "긍정형": (
                "[페르소나: 긍정형]\n"
                "분위기를 전환하고, 에너지를 회복시켜주는 유쾌한 상담자입니다.\n"
                "제안과 격려가 중심이며, 진정성을 유지하세요.\n"
                "제안 40% + 공감 40% + 질문 20%"
            )
        }

    def load_theories(self, path="dataset/hotpot/corpus/corpus.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def get_tone_example(self):
        examples = {
            "위로형": [
                {"role": "user", "content": "요즘 너무 지치고 외로워요."},
                {"role": "assistant", "content": "많이 힘드셨겠어요. 어떤 일이 있었는지 이야기해줄래요?"}
            ],
            "논리형": [
                {"role": "user", "content": "계속 실수하고 일이 꼬여요."},
                {"role": "assistant", "content": "어떤 상황에서 실수가 반복되고 있는지 함께 정리해볼까요?"}
            ],
            "긍정형": [
                {"role": "user", "content": "요즘 무기력하고 의욕이 없어요."},
                {"role": "assistant", "content": "그럴 땐 잠깐 쉬어가는 것도 괜찮아요. 다시 힘낼 준비가 됐을 때 뭐부터 하고 싶나요? 😊"}
            ]
        }
        return examples.get(self.persona, [])

    def match_theory(self, emotion):
        for theory in self.theory_data:
            if emotion in theory.get("추천상황", []):
                return theory
        return {}

    def get_persona_prompt(self):
        return self.persona_prompts.get(self.persona, self.persona_prompts["위로형"])

    def get_strategy_text(self, theory_dict):
        return (
            f"[상담 이론 적용]\n"
            f"이론: {theory_dict['이론명']}\n"
            f"핵심 개념: {', '.join(theory_dict['핵심개념'])}\n"
            f"대표 기법: {', '.join(theory_dict['대표기법'])}\n"
            f"예시: {theory_dict['적용사례'][0]}"
        )

    def merge_recent_user_inputs(self, message_log, member_id, max_gap_sec=30, max_merge_count=5):
        user_msgs = [m for m in message_log if m.get("sender") == "USER" and str(m.get("member_id")) == str(member_id)]
        if not user_msgs:
            return ""
        selected = sorted(user_msgs[-max_merge_count:], key=lambda x: x.get("send_time"))
        merged = [selected[-1]["message"]]
        for i in reversed(range(len(selected) - 1)):
            try:
                cur_time = datetime.fromisoformat(selected[i]["send_time"])
                next_time = datetime.fromisoformat(selected[i + 1]["send_time"])
                if (next_time - cur_time).total_seconds() <= max_gap_sec:
                    merged.insert(0, selected[i]["message"])
                else:
                    break
            except:
                break
        return " ".join(merged).strip()

    def detect_mode_via_llm(self, user_input, memory=""):
        keywords = ["불안", "우울", "외로움", "짜증", "슬픔", "무기력", "분노", "초조함", "혼란", "감정 없음"]
        prompt = f"""
아래 사용자 입력과 과거 대화를 보고, 상담 흐름을 판단해주세요.

- 현재 대화 단계 (casual, explore, counseling)
- 감정 키워드 (아래 목록 중 선택): {', '.join(keywords)}
- 위험도 (낮음/중간/높음)
- 상담 의도 (상담 원함/잡담/모름 등)

[과거 대화]
{memory}

[사용자 입력]
{user_input}

[출력 예시]
단계: counseling
감정: 무기력, 외로움
위험도: 중간
의도: 상담 원함
"""
        try:
            result = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "너는 사용자의 대화 흐름과 감정을 분석하는 심리상담 분석 도우미야."},
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
        except Exception as e:
            print(f"[⚠️ 감정 분석 실패] {e}")

    def build_prompt(self, user_input, memory="", theory_dict=None):
        system_behavior = (
            "너는 정서적 안정감을 주는 심리상담 전문가야.\n"
            "- 감정을 반영하며 공감하고, 필요할 경우 상담 이론을 바탕으로 조언해.\n"
            "- 질문을 반복하지 않고, 흐름에 따라 자연스럽게 진행해.\n"
            "- 같은 표현을 반복하지 말고, 다양한 어휘와 어조를 사용해.\n"
            "- 응답은 1~2문장 내외로 자연스럽고 따뜻하게 마무리해.\n"
        )
        dialogue_flow = (
            "[대화 흐름]\n"
            "인사 → 근황 → 감정 표현 → 문제 탐색 → 이론 기반 제안 → 감정 변화 확인 → 마무리"
        )
        persona_prompt = self.get_persona_prompt()
        if self.emotion:
            persona_prompt += f"\n[감정 감지] 현재 감정: {self.emotion}"
        if self.risk.lower() in ["중간", "높음"]:
            persona_prompt += "\n[주의] 민감한 상황입니다. 더 조심스럽게 반응하세요."
        if theory_dict:
            persona_prompt += f"\n\n{self.get_strategy_text(theory_dict)}"

        return f"{system_behavior}\n\n{persona_prompt}\n\n{dialogue_flow}\n\n[대화 요약]\n{memory}\n\n[사용자 발화]\n{user_input}\n\n[상담사 응답]"

    def respond(self, user_input, message_log, member_id, max_tokens=150):
        memory_raw = load_user_memory(member_id, message_log)
        memory = summarize_memory(memory_raw, self.persona)
        merged_input = self.merge_recent_user_inputs(message_log, member_id)

        if not merged_input:
            return "지금 어떤 생각이 드시나요? 편하게 이야기해 주세요. 😊"

        if is_first_entry(member_id, message_log):
            greetings = [
                "안녕하세요! 처음 오셨군요. 어떤 이야기가 마음에 남아 있는지 나눠주셔도 좋아요.",
                "처음 만나 반가워요. 요즘 어떤 고민이 있으신가요?",
                "처음이 가장 어려운 법이죠. 편하게 말씀하셔도 괜찮아요."
            ]
            return random.choice(greetings)

        self.detect_mode_via_llm(merged_input, memory)
        theory_dict = {}
        if self.mode in ["explore", "counseling"] and self.intent == "상담 원함":
            theory_dict = self.match_theory(self.emotion)

        prompt = self.build_prompt(merged_input, memory, theory_dict)

        try:
            messages = [
                {"role": "system", "content": prompt},
                *self.get_tone_example(),
                {"role": "assistant", "content": f"지금까지 이야기해 주신 걸 정리해 보면요:\n{memory}"},
                {"role": "user", "content": merged_input}
            ]
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.4 if self.mode == "counseling" else 0.7,
                max_tokens=max_tokens
            )
            reply = response.choices[0].message.content.strip().replace('\n', ' ')
            if any(x in merged_input.lower() for x in ["고마워", "도움 됐", "감사"]):
                reply += " 언제든지 또 이야기 나눠요. 당신의 마음을 응원해요. 😊"
            return reply
        except Exception as e:
            print(f"[⚠️ GPT 호출 실패] {e}")
            return "답변 중 문제가 발생했어요. 잠시 후 다시 시도해 주세요."
