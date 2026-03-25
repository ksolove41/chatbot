"""OpenAI 직접 호출 vs LangChain 비교"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 방법 1: OpenAI 직접 사용
print("[방법 1] OpenAI 직접 사용")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": "너는 요리 전문가야."},
        {"role": "user", "content": "김치찌개 레시피를 간단히 알려줘."},
    ],
    temperature=0.7,
)

result = response.choices[0].message.content
print(result)

# 방법 2: LangChain 사용
print(f"\n[방법 2] LangChain 사용")

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

response = llm.invoke([
    SystemMessage(content="너는 요리 전문가야."),
    HumanMessage(content="김치찌개 레시피를 간단히 알려줘."),
])

# .content로 바로 텍스트 접근
print(response.content)
