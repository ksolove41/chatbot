"""LCEL 기본 - 파이프(|)로 프롬프트와 LLM 연결하기"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 환경변수 로드
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

# LLM 단독 호출
print("[1] LLM만 호출")
result = llm.invoke("파이썬을 한 줄로 설명해줘.")
print(f"결과: {result.content}")
print(f"타입: {type(result)}")

# 프롬프트 + LLM 파이프 연결
print(f"\n[2] prompt | llm (파이프로 연결)")

# 프롬프트 템플릿: {변수}를 넣을 수 있는 틀
prompt = ChatPromptTemplate.from_template("{topic}을(를) 한 줄로 설명해줘.")

# 파이프(|)로 연결 = 체인
chain = prompt | llm

# 변수만 넣으면 프롬프트 완성 → LLM 호출까지 자동
result = chain.invoke({"topic": "파이썬"})
print(f"결과: {result.content}")
print(f"타입: {type(result)}")

# 변수만 바꿔서 재사용
print(f"\n[3] 변수만 바꿔서 재사용")

topics = ["인공지능", "클라우드", "블록체인"]
for topic in topics:
    result = chain.invoke({"topic": topic})
    print(f"  [{topic}] {result.content}")
