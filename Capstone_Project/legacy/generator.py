import re
from openai import OpenAI

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def clean_repetitions(text):
    text = re.sub(r'(\S+)( \1)+', r'\1', text)
    text = re.sub(r'(\d{4}[년.]\d{2}[월.]\d{2}[일.]?)( \1)+', r'\1', text)
    return text.strip()

def clean_intro(text):
    bad_phrases = ["안녕하세요", "저는 내담자", "질문을 하셨는데요", "😊", "😌"]
    for phrase in bad_phrases:
        if phrase in text:
            text = text.split(phrase, 1)[-1]
    return text.strip()

# ✅ OpenAI API 응답 함수 (v1.x 구조)
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
    return clean_intro(clean_repetitions(response.choices[0].message.content.strip()))


class generator:
    def __init__(self, llm="openai", model=None, tokenizer=None):
        self.llm = llm
        self.history = []

        self.default_system_prompt = (
            "당신은 내담자의 고민을 공감하며 짧고 따뜻하게 조언하는 심리상담사입니다.\n"
            "절대 인사말이나 질문 반복 없이 바로 핵심 조언부터 시작하세요.\n"
            "답변은 2~3문장 이내로 끝내고, 80단어 이하로 자연스럽게 마무리하세요.\n"
            "예: '조금 쉬는 것도 괜찮아요. 나를 위한 시간을 가져보세요.'"
        )

    def get_llm_output(self, prompt):
        return get_openai_answer(
            user_prompt=prompt,
            system_prompt=self.default_system_prompt
        )

    def direct_answer(self, question):
        prompt = f"🙋 내담자 고민:\n{question}\n\n💬 상담사 응답:"
        return self.get_llm_output(prompt)

    def rag_answer(self, question, reference):
        prompt = (
            f"📚 심리상담 이론 참고:\n{reference}\n\n"
            f"🙋 내담자 고민:\n{question}\n\n💬 상담사 응답:"
        )
        return self.get_llm_output(prompt)

    def full_answer_with_history(self, question, reference, memory):
        prompt = (
            f"🧠 과거 상담 내용:\n{memory}\n\n"
            f"📚 심리상담 이론 참고:\n{reference}\n\n"
            f"🙋 내담자 고민:\n{question}\n\n💬 상담사 응답:"
        )
        return self.get_llm_output(prompt)
