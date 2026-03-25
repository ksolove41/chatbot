"""LangChain Tool 정의 기본 - @tool 데코레이터와 bind_tools 사용"""

import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 환경변수 로드
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 도구 만들기 - @tool 데코레이터로 함수를 도구로 변환
@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_word_length(word: str) -> int:
    """주어진 단어의 글자 수를 반환합니다."""
    return len(word)

# 도구 확인
print(f"도구 이름: {get_current_time.name}")
print(f"도구 설명: {get_current_time.description}")
print(f"직접 호출: {get_current_time.invoke({})}")
print()
print(f"도구 이름: {get_word_length.name}")
print(f"도구 설명: {get_word_length.description}")
print(f"직접 호출: {get_word_length.invoke({'word': '안녕하세요'})}")

# LLM에 도구 연결 - bind_tools()로 사용 가능한 도구 목록 전달
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
tools = [get_current_time, get_word_length]
llm_with_tools = llm.bind_tools(tools)

# LLM이 도구를 선택하는지 확인
response = llm_with_tools.invoke("지금 몇 시야?")

print(f"\n[질문] 지금 몇 시야?")
print(f"[LLM 응답 텍스트] {response.content}")
print(f"[LLM이 선택한 도구] {response.tool_calls}")

# 선택된 도구 직접 실행 (LLM은 도구를 선택만 하고, 실행은 별도로 해야 함)
if response.tool_calls:
    tc = response.tool_calls[0]
    print(f"호출할 도구: {tc['name']}")
    print(f"전달할 인자: {tc.get('args', {})}")

    tool_map = {t.name: t for t in tools}
    result = tool_map[tc["name"]].invoke(tc.get("args", {}))
    print(f"실행 결과: {result}")
