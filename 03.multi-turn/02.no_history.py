"""히스토리 없는 대화 - AI가 이전 대화를 기억하지 못하는 체험"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

client = OpenAI(api_key=api_key)

print("히스토리 없는 챗봇 (종료: exit)")

while True:
    try:
        user_input = input("\n사용자: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n종료합니다.")
        break

    if not user_input:
        continue
    if user_input.lower() == "exit":
        print("종료합니다.")
        break

    # 매번 새 메시지 1개만 보냄 (이전 대화 없음)
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
    )

    print(f"AI: {response.choices[0].message.content}")
