"""복잡한 체인 - 체인끼리 연결하여 다단계 워크플로우 구성"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 환경변수 로드
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

# 1단계 체인: 주제 → 설명 생성
explain_chain = (
    ChatPromptTemplate.from_template(
        "{topic}을(를) 초등학생도 이해할 수 있게 3줄로 설명해줘."
    )
    | llm
    | StrOutputParser()
)

# 2단계 체인: 설명 → 퀴즈 생성
quiz_chain = (
    ChatPromptTemplate.from_template(
        "다음 설명을 읽고 O/X 퀴즈 2개를 만들어줘.\n"
        "형식: 'Q: (문제) → 정답: O 또는 X'\n\n"
        "설명:\n{explanation}"
    )
    | llm
    | StrOutputParser()
)

# 방법 1: 수동으로 체인 연결
print("[방법 1] 수동으로 체인 연결")

topic = "인공지능"
print(f"[주제] {topic}")

explanation = explain_chain.invoke({"topic": topic})
print(f"\n[1단계: 설명]\n{explanation}")

quiz = quiz_chain.invoke({"explanation": explanation})
print(f"\n[2단계: 퀴즈]\n{quiz}")

# 방법 2: RunnablePassthrough로 자동 연결
print(f"\n[방법 2] RunnablePassthrough로 자동 연결")

full_chain = (
    {"explanation": explain_chain}
    | quiz_chain
)

print(f"[주제] 블록체인")
result = full_chain.invoke({"topic": "블록체인"})
print(f"\n[결과 (설명→퀴즈 자동)]\n{result}")

# 방법 3: 3단계 체인 (설명 → 퀴즈 → 요약)
print(f"\n[방법 3] 3단계 체인 (설명 → 퀴즈 → 요약)")

summary_chain = (
    ChatPromptTemplate.from_template(
        "다음 퀴즈를 보고 학습 포인트를 2줄로 요약해줘:\n\n{quiz}"
    )
    | llm
    | StrOutputParser()
)

topic = "클라우드 컴퓨팅"
print(f"[주제] {topic}")

step1 = explain_chain.invoke({"topic": topic})
print(f"\n[1단계: 설명]\n{step1}")

step2 = quiz_chain.invoke({"explanation": step1})
print(f"\n[2단계: 퀴즈]\n{step2}")

step3 = summary_chain.invoke({"quiz": step2})
print(f"\n[3단계: 학습 요약]\n{step3}")
