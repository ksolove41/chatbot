import base64
import os
import secrets
import time
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

load_dotenv()

APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORD = os.getenv("APP_PASSWORD")
if not APP_USERNAME or not APP_PASSWORD:
    raise ValueError("APP_USERNAME/APP_PASSWORD가 설정되지 않았습니다. .env 파일을 확인하세요.")

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 외부에서 URL만 알아도 함부로 못 쓰게 전체 요청에 Basic Auth를 건다
class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization", "")
        scheme, _, credentials = auth_header.partition(" ")

        if scheme.lower() == "basic":
            try:
                username, _, password = base64.b64decode(credentials).decode("utf-8").partition(":")
            except Exception:
                username, password = "", ""

            if secrets.compare_digest(username, APP_USERNAME) and secrets.compare_digest(password, APP_PASSWORD):
                return await call_next(request)

        return Response(status_code=401, headers={"WWW-Authenticate": "Basic"})


app.add_middleware(BasicAuthMiddleware)

app.mount("/static", StaticFiles(directory="static"), name="static")

# IP별 /chat 호출 횟수 제한 (API 비용 폭탄 방지용 최소 안전장치)
RATE_LIMIT = 10
RATE_WINDOW_SECONDS = 60
request_log: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(request: Request):
    ip = request.client.host
    now = time.time()
    timestamps = request_log[ip]

    while timestamps and timestamps[0] < now - RATE_WINDOW_SECONDS:
        timestamps.pop(0)

    if len(timestamps) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="요청이 너무 많습니다. 잠시 후 다시 시도하세요.")

    timestamps.append(now)


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
def chat(request: ChatRequest, http_request: Request):
    check_rate_limit(http_request)

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[m.model_dump() for m in request.messages],
    )

    reply = response.choices[0].message.content
    updated_messages = request.messages + [Message(role="assistant", content=reply)]

    return {"messages": updated_messages}