"""
Multi-Turn 챗봇

대화 히스토리를 유지하면서 연속 대화를 할 수 있는 챗봇입니다.
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
system_prompt = (
    "너는 사용자를 도와주는 상담사야. 공감적으로 답하고, "
    "불명확하면 짧게 되물어봐. 필요하면 단계별로 안내해줘."
)

# 대화 히스토리 초기화 (시스템 메시지 포함)
messages = [{"role": "system", "content": system_prompt}]

# 대화 루프
print("대화를 시작합니다. 'exit'으로 종료, 'reset'으로 히스토리 초기화.")
while True:
    try:
        user_input = input("사용자: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n종료합니다.")
        break

    if user_input.lower() == "exit":
        print("종료합니다.")
        break

    if user_input.lower() == "reset":
        messages = [{"role": "system", "content": system_prompt}]
        print("히스토리를 초기화했어요.")
        continue

    # 사용자 메시지를 히스토리에 추가
    messages.append({"role": "user", "content": user_input})

    # API 호출 (전체 히스토리 전달 → 멀티턴)
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        temperature=0.7,
    )

    # 응답 추출 및 히스토리에 추가
    assistant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})

    print("AI:", assistant_message)
