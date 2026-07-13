from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.mount("/static", StaticFiles(directory="static"), name="static")


# 채팅 메시지 하나의 구조 (OpenAI messages 포맷과 동일)
class Message(BaseModel):
    role: str
    content: str


# 요청 body 구조 정의: 지금까지의 전체 대화 히스토리
class ChatRequest(BaseModel):
    messages: list[Message]


@app.get("/")
def index():
    return FileResponse("static/index.html")


# 채팅 API 엔드포인트: 히스토리를 받아 응답을 추가해 반환 (stateless)
@app.post("/chat")
def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[m.model_dump() for m in request.messages],
    )

    reply = response.choices[0].message.content
    updated_messages = request.messages + [Message(role="assistant", content=reply)]

    return {"messages": updated_messages}