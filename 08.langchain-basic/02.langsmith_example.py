"""
LangSmith 추적 예제

LangSmith를 활용하여 LLM 호출을 추적하고 모니터링하는 방법을 보여줍니다.
.env에 LANGCHAIN_TRACING_V2=true 설정 시 자동으로 추적됩니다.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 환경변수 로드 (LangSmith 추적 자동 활성화)
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# LangSmith 설정 확인
print("LangSmith 설정 확인:")
print(f"  - Tracing: {os.environ.get('LANGCHAIN_TRACING_V2', 'false')}")
print(f"  - Project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
print()

# 모델 초기화
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

# 기본 질문 테스트 (LangSmith에 자동 추적)
print("=" * 50)
print("[예제] 기본 질문 (LangSmith에 자동 추적됨)")
print("=" * 50)

response = llm.invoke("파이썬의 주요 특징 3가지를 간단히 설명해주세요.")
print(response.content)

# LangSmith 대시보드 안내
print()
print("=" * 50)
print("LangSmith 대시보드에서 추적 결과 확인하기:")
print("  1. https://smith.langchain.com 접속")
print("  2. 로그인 후 프로젝트 선택")
print("  3. 최근 실행 내역에서 입출력/토큰/시간 확인")
print("=" * 50)
