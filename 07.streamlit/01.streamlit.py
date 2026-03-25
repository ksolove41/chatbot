"""
Streamlit 챗봇 UI

OpenAI API를 사용한 멀티턴 대화 웹 인터페이스입니다.
실행: uv run streamlit run 216.streamlit/01.streamlit.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# 환경설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# 시스템 프롬프트
SYSTEM_PROMPT = (
    "너는 사용자를 도와주는 상담사야. "
    "공감적으로 짧게 답하고, 필요한 경우 되물어봐."
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
    ]

# UI
st.title("ChatGPT 상담 챗봇")
st.caption("OpenAI API를 이용한 멀티턴 대화 예제")

# 기존 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 저장 및 출력
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # OpenAI API 호출 (스트리밍)
    with st.chat_message("assistant"):
        # 히스토리 구성 (시스템 프롬프트 + 대화 기록)
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        api_messages += st.session_state.messages

        stream = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=api_messages,
            temperature=0.7,
            stream=True,
        )
        response = st.write_stream(stream)

    # 어시스턴트 메시지 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
