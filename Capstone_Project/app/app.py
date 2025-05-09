from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List

from app.chat_agent import ChatAgent
from app.wiki_searcher import WikiSearcher
from app.vector_manager import query_similar_chats, add_chat_to_vector_db
from app.db_manager import fetch_recent_dialogue
from app.memory_manager import summarize_memory, is_first_entry  # ✅ is_first_entry 추가

import os
import sys
import re

print("📂 [디버깅] 현재 실행 중인 app.py 경로:", os.path.abspath(__file__))
print("📂 [디버깅] sys.path 상의 import 검색 경로:")
for path in sys.path:
    print("    -", path)

app = FastAPI()
wiki = WikiSearcher()

# ✅ Request/Response 모델 정의
class ChatSendRequest(BaseModel):
    memberId: int
    message: str
    talkType: str

class ChatSendResponse(BaseModel):
    message: List[str]

class EnterRequest(BaseModel):
    memberId: int

class EnterResponse(BaseModel):
    summary: str

# ✅ 응답 문장을 문장 단위로 분리
def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

# ✅ 채팅 요청 처리
@app.post("/chat", response_model=ChatSendResponse)
def chat_with_ai(req: ChatSendRequest):
    if not req.message.strip():
        return ChatSendResponse(message=["조금 더 구체적으로 말씀해주시겠어요?"])

    similar_chats = query_similar_chats(str(req.memberId), req.message, top_k=3)
    theory_pairs = wiki.search(req.message, top_k=2)

    message_log = fetch_recent_dialogue(req.memberId, limit=100)

    agent = ChatAgent(persona=req.talkType)
    full_response = agent.respond(
        user_input=req.message,
        message_log=message_log,
        member_id=str(req.memberId),
        theory=theory_pairs
    )

    add_chat_to_vector_db(
        member_id=str(req.memberId),
        user_input=req.message,
        bot_response=full_response,
        persona=req.talkType,
        emotion=agent.emotion,
        risk=agent.risk
    )

    return ChatSendResponse(message=split_into_sentences(full_response))

# ✅ 과거 대화 요약 제공 (/summary)
@app.post("/summary", response_model=EnterResponse)
def enter_chat(req: EnterRequest):
    message_log = fetch_recent_dialogue(req.memberId, limit=100)
    print(f"📥 /summary: DB에서 {len(message_log)}개 대화 불러옴")

    # ✅ 첫 입장인 경우 summary 대신 NULL 반환
    if is_first_entry(str(req.memberId), message_log):
        print("🟡 첫 입장 확인됨 (NULL 출력)")
        return EnterResponse(summary="NULL")

    summary = summarize_memory(message_log)
    print(f"🧠 /summary 요약 결과: {summary}")
    return EnterResponse(summary=summary)
