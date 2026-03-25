"""프롬프트 템플릿 - 변수만 바꿔서 재사용하는 프롬프트"""

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

# 1. from_template() - 가장 기본적인 템플릿
print("[1] from_template() - 변수 넣기")

prompt1 = ChatPromptTemplate.from_template(
    "{topic}에 대해 {num}가지 핵심 포인트를 알려줘."
)

print(f"변수 목록: {prompt1.input_variables}")
print(f"완성된 프롬프트 미리보기:")
print(prompt1.format(topic="파이썬", num="3"))

chain = prompt1 | llm | StrOutputParser()
result = chain.invoke({"topic": "파이썬", "num": "3"})
print(f"\n[결과] {result}")

# 2. from_messages() - system/user 역할 분리
print(f"\n[2] from_messages() - 역할 분리")

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "너는 {role}이야. {style}"),
    ("human", "{question}"),
])

print(f"변수 목록: {prompt2.input_variables}")

chain2 = prompt2 | llm | StrOutputParser()

question = "클라우드 컴퓨팅이 뭐야?"
print(f"\n[질문] {question}")

result1 = chain2.invoke({
    "role": "초등학교 선생님",
    "style": "아이들이 이해할 수 있게 비유를 들어 설명해줘.",
    "question": question,
})
print(f"\n[초등학교 선생님] {result1}")

result2 = chain2.invoke({
    "role": "IT 전문 컨설턴트",
    "style": "기술 용어를 사용해서 정확하게 설명해줘.",
    "question": question,
})
print(f"\n[IT 컨설턴트] {result2}")

# 3. 같은 템플릿 재사용 - 변수만 바꾸면 됨
print(f"\n[3] 템플릿 재사용 - 변수만 바꾸기")

topics = ["머신러닝", "블록체인", "메타버스"]
for topic in topics:
    result = chain.invoke({"topic": topic, "num": "2"})
    print(f"[{topic}] {result}\n")
