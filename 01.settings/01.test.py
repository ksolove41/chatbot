"""
OpenAI API 연결 테스트

API 키가 정상적으로 설정되었는지 확인합니다.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일에서 환경변수 로드
load_dotenv()

# API 키 확인
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# 간단한 API 호출 테스트
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{"role": "user", "content": "한국의 수도는?"}],
)

# 결과 출력
print(response.choices[0].message.content)
print("\nAPI 연결 테스트 성공!")
