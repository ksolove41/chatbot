"""프롬프트 템플릿 활용 - 사용자 입력으로 체인 실행"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 환경변수 로드
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

# 1. 주제 입력 → 설명 생성
print("[1] 주제를 입력하면 설명을 생성합니다")

explain_chain = (
    ChatPromptTemplate.from_template(
        "{topic}에 대해 초보자도 이해할 수 있게 3줄로 설명해줘."
    )
    | llm
    | StrOutputParser()
)

topic = input("설명할 주제를 입력하세요: ").strip()
if topic:
    result = explain_chain.invoke({"topic": topic})
    print(f"\n[{topic} 설명]\n{result}")

# 2. 역할 + 질문을 동시에 입력
print(f"\n[2] 역할과 질문을 직접 입력합니다")

role_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "너는 {role}이야. 그 역할에 맞게 답해줘."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

print("역할 예시: 요리사, 의사, 변호사, 여행 가이드, 개발자")
role = input("AI의 역할: ").strip()
question = input("질문: ").strip()

if role and question:
    result = role_chain.invoke({"role": role, "question": question})
    print(f"\n[{role}의 답변]\n{result}")

# 3. 역할을 고정하고 계속 질문
print(f"\n[3] 역할을 고정하고 계속 질문하기")
print("역할 예시: 요리사, 의사, 변호사, 여행 가이드, 개발자")
fixed_role = input("고정할 역할: ").strip()

if fixed_role:
    print(f"[{fixed_role}]에게 질문하세요! (종료: exit)")

    while True:
        try:
            q = input(f"\n질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not q:
            continue
        if q.lower() == "exit":
            print("종료합니다.")
            break

        result = role_chain.invoke({"role": fixed_role, "question": q})
        print(f"[{fixed_role}] {result}")
