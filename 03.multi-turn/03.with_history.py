"""히스토리 있는 대화 - 이전 대화를 함께 전달하여 맥락 유지"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

client = OpenAI(api_key=api_key)

history = []  # 대화 히스토리 저장

print("히스토리 있는 챗봇 (종료: exit / 히스토리 확인: history)")

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

    # 히스토리 확인
    if user_input.lower() == "history":
        print(f"\n현재 히스토리 (총 {len(history)}개 메시지)")
        for i, msg in enumerate(history):
            role = "사용자" if msg["role"] == "user" else "AI"
            print(f"  [{i}] {role}: {msg['content'][:60]}")
        continue

    # 1. 사용자 메시지를 히스토리에 추가
    history.append({"role": "user", "content": user_input})

    # 2. 히스토리 전체를 API에 전달
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=history,
        temperature=0.7,
    )

    # 3. AI 응답도 히스토리에 추가
    assistant_message = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_message})

    print(f"AI: {assistant_message}")
