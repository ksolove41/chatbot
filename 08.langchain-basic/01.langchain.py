"""
LangChain 기본 사용법

LangChain + OpenAI를 사용하여 챗봇을 구성하는 기본 예제입니다.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 환경변수 로드
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# LangChain ChatOpenAI 모델 초기화
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

print("=" * 50)
print("LangChain 기본 예제")
print("=" * 50)

# 1. 단순 호출
print("\n[예제 1] 단순 질문")
response = llm.invoke("파이썬 프로그래밍의 장점 3가지를 간단히 알려줘.")
print(response.content)

# 2. 메시지 리스트로 호출 (system + user)
print("\n[예제 2] 시스템 프롬프트 + 사용자 질문")
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="너는 친절한 IT 전문 상담사야. 쉽게 설명해줘."),
    HumanMessage(content="API가 뭐야?"),
]
response = llm.invoke(messages)
print(response.content)

print("\n완료!")
