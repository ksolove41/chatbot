"""LangChain 스트리밍 출력 - 응답을 실시간으로 한 글자씩 출력"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 환경변수 로드
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

# 일반 호출: 전체 응답이 한번에 나옴
print("[일반 호출]")
response = llm.invoke("파이썬의 장점 3가지를 간단히 알려줘.")
print(response.content)

# 스트리밍 호출: 토큰 단위로 조각이 날아옴
print(f"\n[스트리밍 호출]")
for chunk in llm.stream("파이썬의 장점 3가지를 간단히 알려줘."):
    print(chunk.content, end="", flush=True)
print()

# 스트리밍 대화 루프
print(f"\n[스트리밍 대화] 종료: exit")

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

    print("AI: ", end="")
    for chunk in llm.stream(user_input):
        print(chunk.content, end="", flush=True)
    print()
