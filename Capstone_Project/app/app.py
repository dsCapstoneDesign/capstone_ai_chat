# ✅ ▸ 수정된 app.py (한국 시간 + TalkType 적용)

import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta

from app.chat_agent import ChatAgent
from app.wiki_searcher import WikiSearcher
from app.vector_manager import query_similar_chats, add_chat_to_vector_db

print("\U0001F4C2 [디버깅] 현재 실행 중인 app.py 경로:", os.path.abspath(__file__))
print("\U0001F4C2 [디버깅] sys.path 상의 import 검색 경로:")
for path in sys.path:
    print("    -", path)

app = FastAPI()
wiki = WikiSearcher()

# ✅ 백어가 요청한 형식으로 변경
class ChatSendRequest(BaseModel):
    memberId: int
    message: str
    talkType: str  # 이전의 senderType 대신

class ChatSendResponse(BaseModel):
    memberId: int
    sendTime: str
    message: List[str]
    sender: str

# ✅ 문장 분리 함수
import re

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

@app.post("/chat", response_model=ChatSendResponse)
def chat_with_ai(req: ChatSendRequest):
    # 결과 가입이 없는 경우 fallback
    if not req.message.strip():
        korea_now = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d-%H:%M")
        return ChatSendResponse(
            memberId=req.memberId,
            sendTime=korea_now,
            message=["조금 더 구체적으로 말씀해주시겠어요?"],
            sender="bot"
        )

    # ✅ memory 요약 + wiki 이론 검색
    similar_chats = query_similar_chats(str(req.memberId), req.message, top_k=3)
    memory_summary = "\n".join(similar_chats)
    theory_pairs = wiki.search(req.message, top_k=2)

    agent = ChatAgent(persona=req.talkType)
    full_response = agent.respond(
        user_input=req.message,
        memory=memory_summary,
        theory=theory_pairs
    )

    # ✅ 대화 기록 저장
    add_chat_to_vector_db(
        member_id=str(req.memberId),
        user_input=req.message,
        bot_response=full_response,
        persona=req.talkType,
        emotion=agent.emotion,
        risk=agent.risk
    )

    korea_now = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d-%H:%M")
    return ChatSendResponse(
        memberId=req.memberId,
        sendTime=korea_now,
        message=split_into_sentences(full_response),
        sender="bot"
    )
