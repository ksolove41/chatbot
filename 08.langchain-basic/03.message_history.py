"""
LangChain 메시지 히스토리 관리

세션별로 대화 기록을 유지하는 멀티턴 챗봇을 구성하는 예제입니다.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

# 환경변수 로드
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 모델 초기화
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

# 세션별 대화 기록 저장소
store = {}

def get_session_history(session_id: str):
    """세션 ID에 따른 대화 기록 반환 (없으면 새로 생성)"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 메시지 히스토리를 자동으로 관리하는 래퍼
with_message_history = RunnableWithMessageHistory(model, get_session_history)

print("=" * 50)
print("LangChain 메시지 히스토리 예제")
print("=" * 50)

# 세션 abc2에서 대화 시작
config = {"configurable": {"session_id": "abc2"}}

# 1) 자기소개
response = with_message_history.invoke(
    [HumanMessage(content="안녕? 난 김철수야.")],
    config=config,
)
print(f"[세션 abc2] AI: {response.content}")

# 2) 같은 세션에서 이름 확인 → 기억하고 있음
response = with_message_history.invoke(
    [HumanMessage(content="내 이름이 뭐지?")],
    config=config,
)
print(f"[세션 abc2] AI: {response.content}")

# 3) 새 세션 abc3에서 질문 → 기억 못함
config_new = {"configurable": {"session_id": "abc3"}}
response = with_message_history.invoke(
    [HumanMessage(content="내 이름이 뭐지?")],
    config=config_new,
)
print(f"[세션 abc3] AI: {response.content}")

# 4) 다시 abc2 세션 복귀 → 기억하고 있음
response = with_message_history.invoke(
    [HumanMessage(content="아까 우리가 무슨 얘기 했지?")],
    config=config,
)
print(f"[세션 abc2] AI: {response.content}")

# 세션 상태 출력
print()
print("활성 세션:")
for sid, history in store.items():
    print(f"  - 세션 {sid}: {len(history.messages)}개 메시지")
