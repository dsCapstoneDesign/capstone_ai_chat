# code/app.py

from fastapi import FastAPI
from pydantic import BaseModel

from chat_agent import ChatAgent
from memory_manager import summarize_memory
from wiki_searcher import WikiSearcher
from vector_manager import query_similar_chats, add_chat_to_vector_db

# FastAPI 서버 초기화
app = FastAPI()

# 전역 검색기
wiki = WikiSearcher()

class ChatRequest(BaseModel):
    user_input: str
    member_id: str
    persona: str = "위로형"

@app.post("/chat")
def chat_with_ai(req: ChatRequest):
    # ✅ 벡터 DB에서 요약용 대화 불러오기
    similar_chats = query_similar_chats(req.member_id, req.user_input, top_k=3)
    memory = summarize_memory([{"message": chat} for chat in similar_chats])

    # ✅ 상담 이론 검색 (RAG)
    theory_refs = wiki.search(req.user_input, top_k=2)
    theory_text = "\n".join(theory_refs)

    # ✅ 챗봇 응답 생성
    agent = ChatAgent(persona=req.persona)
    response = agent.respond(user_input=req.user_input, memory=memory, theory=theory_text)

    # ✅ 대화 내용을 벡터 DB에 저장
    add_chat_to_vector_db(req.member_id, req.user_input, response)

    return {"response": response}
