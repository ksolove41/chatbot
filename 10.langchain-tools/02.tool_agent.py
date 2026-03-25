"""LangChain Agent - 도구 선택부터 실행까지 자동화"""

import os
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 환경변수 로드
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 도구 정의
@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """수학 표현식을 계산합니다. 예: '2 + 2', '10 * 5', 'sqrt(16)'"""
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sqrt": math.sqrt,
    }
    result = eval(expression, {"__builtins__": {}}, allowed_names)
    return str(float(result))

@tool
def get_word_length(word: str) -> int:
    """주어진 단어의 글자 수를 반환합니다."""
    return len(word)

tools = [get_current_time, calculate, get_word_length]

# Agent 구성 (agent_scratchpad는 Agent의 생각 과정 기록 공간)
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 도구를 활용하여 사용자의 질문에 정확히 답하는 도우미입니다."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 테스트
questions = [
    "지금 몇 시야?",
    "123 곱하기 456은 얼마야?",
    "'인공지능'이라는 단어는 몇 글자야?",
]

for q in questions:
    print(f"\n[질문] {q}")
    result = agent_executor.invoke({"input": q})
    print(f"[최종 답변] {result['output']}")
