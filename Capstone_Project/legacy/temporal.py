import json
import argparse
from tqdm import tqdm
from openai import OpenAI

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# ✅ JSON 데이터 로드
def load_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ✅ OpenAI API 기반 요약 생성
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
    return response.choices[0].message.content.strip()


# ✅ memory.json 생성 메인 함수
def memory_construct(args):
    data = load_dataset(args.data_path)
    outputs = []

    # 📌 프롬프트 구성
    system_prompt = (
        "당신은 심리상담 전문가이며, 상담 내용을 요약하는 역할입니다.\n"
        "내담자의 고민과 상담사의 응답을 기반으로, 핵심 내용만 간결하고 따뜻하게 요약해주세요."
    )

    template = (
        "### 역할: 당신은 내담자의 감정 흐름과 상담사의 반응을 요약하는 심리상담 전문가입니다.\n\n"
        "### 상담 기록:\n"
        "- 질문: {question}\n"
        "- 답변: {answer}\n\n"
        "### 요약해 주세요:"
    )

    for item in tqdm(data):
        question = item["question"].replace("사람 : ", "").strip()
        answer = item["answer"].replace("AI : ", "").strip()
        prompt = template.format(question=question, answer=answer)

        # 📡 OpenAI 응답
        memory_summary = get_openai_answer(prompt, system_prompt)

        outputs.append({
            "question": question,
            "memory": memory_summary
        })

    # 저장
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

    print(f"✅ 총 {len(outputs)}개 memory.json 항목 생성 완료")


# ✅ 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/dev.json", help="입력 데이터 경로")
    parser.add_argument("--output_path", type=str, default="data/memory.json", help="저장할 메모리 경로")
    args = parser.parse_args()

    memory_construct(args)
