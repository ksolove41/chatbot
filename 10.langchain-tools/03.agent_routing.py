"""Agent의 도구 선택 - 질문에 따라 적절한 도구를 자동으로 고른다"""

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

# 도구 4개 정의
@tool
def get_current_time() -> str:
    """현재 날짜와 시간을 반환합니다."""
    return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")

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

@tool
def translate_to_english(text: str) -> str:
    """한국어 텍스트를 영어로 번역합니다. 간단한 단어/문장만 지원합니다."""
    dictionary = {
        "안녕하세요": "Hello",
        "감사합니다": "Thank you",
        "파이썬": "Python",
        "인공지능": "Artificial Intelligence",
        "사랑": "Love",
        "컴퓨터": "Computer",
    }
    return dictionary.get(text, f"'{text}'의 번역을 찾을 수 없습니다.")

tools = [get_current_time, calculate, get_word_length, translate_to_english]

# Agent 구성
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 도구를 활용하여 사용자를 돕는 도우미입니다. "
     "도구가 필요하면 적절한 도구를 선택하고, "
     "도구가 필요 없는 질문에는 직접 답하세요."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 테스트: 다양한 질문으로 Agent의 도구 선택 관찰
questions = [
    "지금 몇 시야?",                          # 시간 도구
    "250 나누기 7은?",                         # 계산기 도구
    "'프로그래밍'은 몇 글자야?",               # 글자 수 도구
    "'감사합니다'를 영어로 뭐라고 해?",        # 번역 도구
    "파이썬이 뭐야?",                          # 도구 없이 직접 답변
    "'인공지능'은 몇 글자이고, 영어로는 뭐야?",  # 여러 도구 조합
]

for q in questions:
    print(f"\n[질문] {q}")
    result = agent_executor.invoke({"input": q})
    print(f"[최종 답변] {result['output']}")
