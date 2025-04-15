import re
from openai import OpenAI

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def clean_repetitions(text):
    text = re.sub(r'(\S+)( \1)+', r'\1', text)
    text = re.sub(r'(\d{4}[년.]\d{2}[월.]\d{2}[일.]?)( \1)+', r'\1', text)
    return text.strip()

def get_openai_answer(user_prompt, system_prompt, max_tokens=300):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens
    )
    return clean_repetitions(response.choices[0].message.content.strip())

# ✅ 상담사 클래스
class assistant:
    def __init__(self, llm="openai", model=None, tokenizer=None):
        self.default_system_prompt = (
            "너는 따뜻하고 신뢰할 수 있는 심리상담사야.\n"
            "사용자의 말에 자연스럽게 공감하고, 감정을 부드럽게 언어로 표현해줘.\n"
            "가능하다면 감정이 시작된 원인을 함께 탐색할 수 있도록 도와줘.\n"
            "사용자가 스스로 말할 수 있도록 열린 질문을 하나 정도 던져줘.\n"
            "조언이 필요해보이면 조심스럽고 현실적인 제안을 짧게 건네줘도 좋아.\n"
            "응답은 2~4문장 이내로, 친구처럼 자연스럽고 따뜻하게 해줘.\n"
            "문장은 끊기지 않고 부드럽게 이어져야 해."
        )

        self.initial_questions = [
            "안녕하세요, 오늘 하루는 어떠셨나요?",
            "요즘은 어떻게 지내고 계세요?",
            "마음 편히 이야기할 수 있는 시간을 가져볼까요?",
            "혹시 최근에 감정이 요동친 일이 있었나요?"
        ]

    def greet_or_continue(self, turn_count: int = 0) -> str:
        if turn_count < len(self.initial_questions):
            return self.initial_questions[turn_count]
        else:
            return "편하게 말씀해 주세요. 계속 들어드릴게요."

    def generate_response(self, user_input):
        return get_openai_answer(
            user_prompt=user_input,
            system_prompt=self.default_system_prompt
        )

    def question_analysis(self, question):
        system_prompt = (
            "당신은 복잡한 질문을 분석해서 핵심적인 서브 질문들로 나누는 조력자입니다.\n"
            "주어진 질문을 1~3개의 간단한 질문으로 줄바꿈으로 나열하세요."
        )
        return get_openai_answer(
            user_prompt=question,
            system_prompt=system_prompt,
            max_tokens=200
        )

    def knowledge_extraction(self, question, reference_text):
        system_prompt = (
            "당신은 심리상담사이며, 주어진 외부 정보를 참고해 내담자의 고민에 적절한 내용을 간결하게 요약합니다.\n"
            "중복되거나 불필요한 정보는 제거하고 핵심만 정리하세요."
        )
        combined_prompt = f"[질문] {question}\n\n[참고 정보]\n{reference_text}"
        return get_openai_answer(
            user_prompt=combined_prompt,
            system_prompt=system_prompt,
            max_tokens=300
        )

    def build_conversational_prompt(self, user_input, memory_context="", wiki_context=""):
        return f"""
너는 따뜻하고 신뢰할 수 있는 심리상담사야.
사용자의 말에 자연스럽게 공감하고, 감정을 부드럽게 언어로 표현해줘.
가능하다면 감정이 시작된 원인을 함께 탐색할 수 있도록 도와줘.
사용자가 스스로 말할 수 있도록 열린 질문을 하나 정도 던져줘.
조언이 필요해보이면 조심스럽고 현실적인 제안을 짧게 건네줘도 좋아.
응답은 2~4문장 이내로, 친구처럼 자연스럽고 따뜻하게 해줘.
문장은 끊기지 않고 부드럽게 이어져야 해.

[기억된 대화]
{memory_context}

[상담 지식 요약]
{wiki_context}

[사용자 말]
{user_input}
"""

