"""Wikipedia 검색 Agent - 외부 정보를 검색하여 답변"""

import os
import wikipedia
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 환경변수 로드
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 위키피디아 한국어 설정
wikipedia.set_lang("ko")

# 위키피디아 검색 도구
@tool
def search_wikipedia(query: str) -> str:
    """위키피디아에서 주제를 검색하여 요약 정보를 반환합니다."""
    try:
        results = wikipedia.search(query, results=3)
        if not results:
            return f"'{query}'에 대한 위키피디아 문서를 찾을 수 없습니다."
        summary = wikipedia.summary(results[0], sentences=3)
        return f"[{results[0]}]\n{summary}"
    except wikipedia.DisambiguationError as e:
        return f"여러 결과가 있습니다: {', '.join(e.options[:5])}"
    except wikipedia.PageError:
        return f"'{query}'에 대한 문서를 찾을 수 없습니다."

# 도구 직접 테스트
print("[도구 테스트] 위키피디아 검색")
result = search_wikipedia.invoke({"query": "파이썬 프로그래밍"})
print(result)

# Agent에 연결
tools = [search_wikipedia]
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 위키피디아를 검색하여 정확한 정보를 제공하는 도우미입니다. "
     "검색 결과를 바탕으로 사용자의 질문에 한국어로 친절하게 답하세요."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 테스트 질문
questions = [
    "세종대왕이 누구야?",
    "머신러닝이 뭐야?",
    "한국의 수도는 어디야?",
]

for q in questions:
    print(f"\n[질문] {q}")
    result = agent_executor.invoke({"input": q})
    print(f"[답변] {result['output']}")

# 직접 질문 (exit 입력으로 종료)
print("\n직접 질문해보세요. 종료: exit")

while True:
    try:
        user_input = input("\n질문: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n종료합니다.")
        break

    if not user_input:
        continue
    if user_input.lower() == "exit":
        print("종료합니다.")
        break

    result = agent_executor.invoke({"input": user_input})
    print(f"[답변] {result['output']}")
