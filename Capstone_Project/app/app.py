# ✅ 수정된 app.py

import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime

from app.chat_agent import ChatAgent
from app.wiki_searcher import WikiSearcher
from app.vector_manager import query_similar_chats, add_chat_to_vector_db

print("\U0001F4C2 [디버깅] 현재 실행 중인 app.py 경로:", os.path.abspath(__file__))
print("\U0001F4C2 [디버깅] sys.path 상의 import 검색 경로:")
for path in sys.path:
    print("    -", path)

app = FastAPI()
wiki = WikiSearcher()

# ✅ 백엔드와 프론트에 맞는 request/response 모델
class ChatSendRequest(BaseModel):
    memberId: int
    message: str
    senderType: str

class ChatSendResponse(BaseModel):
    memberId: int
    sendTime: str
    message: List[str]
    sender: str

# ✅ 문장 단위로 분할하는 함수
import re
def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

@app.post("/chat", response_model=ChatSendResponse)
def chat_with_ai(req: ChatSendRequest):
    if not req.message.strip():
        return ChatSendResponse(
            memberId=req.memberId,
            sendTime=datetime.now().strftime("%Y-%m-%d-%H:%M"),
            message=["조금 더 구체적으로 말씀해주실 수 있을까요?"],
            sender="bot"
        )

    # ✅ memory 요약 + wiki 이론 검색
    similar_chats = query_similar_chats(str(req.memberId), req.message, top_k=3)
    memory_summary = "\n".join(similar_chats)
    theory_pairs = wiki.search(req.message, top_k=2)

    agent = ChatAgent(persona=req.senderType)
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
        persona=req.senderType,
        emotion=agent.emotion,
        risk=agent.risk
    )

    return ChatSendResponse(
        memberId=req.memberId,
        sendTime=datetime.now().strftime("%Y-%m-%d-%H:%M"),
        message=split_into_sentences(full_response),
        sender="bot"
    )
