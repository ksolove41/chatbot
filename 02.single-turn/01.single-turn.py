"""
Single-Turn 챗봇

사용자의 질문 하나에 대해 한 번 응답하는 가장 기본적인 챗봇입니다.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# 시스템 프롬프트 = 페르소나 설정
system_prompt = "너는 사용자를 도와주는 상담사야."

# 사용자 입력 받기
user_input = input("사용자: ")

# API 호출
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ],
    temperature=0.7,
)

# 응답 출력
print("AI:", response.choices[0].message.content)
