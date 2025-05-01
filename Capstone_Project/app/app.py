import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

from app.chat_agent import ChatAgent
from app.wiki_searcher import WikiSearcher
from app.vector_manager import query_similar_chats, add_chat_to_vector_db

# ✅ 디버깅: app 경로 및 import 경로 확인
print("📂 [디버깅] 현재 실행 중인 app.py 경로:", os.path.abspath(__file__))
print("📂 [디버깅] sys.path 상의 import 검색 경로:")
for path in sys.path:
    print("    -", path)

# ✅ FastAPI 인스턴스 생성
app = FastAPI()

# ✅ 전역 wiki 검색기 (재사용)
wiki = WikiSearcher()

# ✅ Request / Response 데이터 모델 정의
class ChatRequest(BaseModel):
    user_input: str
    member_id: str
    persona: str = "위로형"

class ChatResponse(BaseModel):
    response: str
    emotion: Union[str, None]
    risk: Union[str, None]
    persona: str
    theory_refs: List[str]
    memory_summary: str

@app.post("/chat", response_model=ChatResponse)
def chat_with_ai(req: ChatRequest):
    """
    FastAPI 챗봇 응답 엔드포인트
    """
    if not req.user_input.strip():
        return ChatResponse(
            response="조금 더 구체적으로 말씀해주실 수 있을까요?",
            emotion=None,
            risk=None,
            persona=req.persona,
            theory_refs=[],
            memory_summary=""
        )

    # ✅ 벡터 기반 memory 요약
    similar_chats = query_similar_chats(req.member_id, req.user_input, top_k=3)
    memory_summary = "\n".join(similar_chats)

    # ✅ 상담 이론 검색
    theory_pairs = wiki.search(req.user_input, top_k=2)
    theory_refs = [f"[{name}] {desc}" for name, desc in theory_pairs]

    # ✅ 챗봇 응답 생성
    agent = ChatAgent(persona=req.persona)
    response = agent.respond(
        user_input=req.user_input,
        memory=memory_summary,
        theory=theory_pairs
    )

    # ✅ 대화 저장
    add_chat_to_vector_db(
        member_id=req.member_id,
        user_input=req.user_input,
        bot_response=response,
        persona=req.persona,
        emotion=agent.emotion,
        risk=agent.risk
    )

    return ChatResponse(
        response=response,
        emotion=agent.emotion,
        risk=agent.risk,
        persona=req.persona,
        theory_refs=theory_refs,
        memory_summary=memory_summary
    )


def chat_agent():
    return None


def vector_manager():
    return None