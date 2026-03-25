"""System Prompt - 역할 부여에 따른 답변 변화 비교"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

client = OpenAI(api_key=api_key)

question = "파이썬이 뭐야?"

# 역할 없이 질문
print("[역할 없음]")
response1 = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "user", "content": question},
    ],
    temperature=0.7,
)
print(response1.choices[0].message.content)

# 초등학교 선생님 역할
print("\n[역할: 초등학교 선생님]")
response2 = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": "너는 초등학교 3학년 담임선생님이야. 아이들이 이해할 수 있게 쉽고 재미있게 설명해줘."},
        {"role": "user", "content": question},
    ],
    temperature=0.7,
)
print(response2.choices[0].message.content)

# 시니어 개발자 역할
print("\n[역할: 시니어 개발자]")
response3 = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": "너는 10년차 백엔드 시니어 개발자야. 기술적으로 정확하고 실무 관점에서 답해줘."},
        {"role": "user", "content": question},
    ],
    temperature=0.7,
)
print(response3.choices[0].message.content)
