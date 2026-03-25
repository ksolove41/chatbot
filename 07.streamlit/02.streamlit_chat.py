"""
Streamlit 스트리밍 챗봇 + 사이드바 설정

01의 기본 챗봇에 사이드바 설정(역할, temperature)을 추가하여
system prompt와 파라미터가 응답에 미치는 영향을 실시간으로 체험합니다.

실행: uv run streamlit run 07.streamlit/02.streamlit_chat.py
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

# ── 사이드바: 설정 패널 ──
st.sidebar.title("챗봇 설정")

# 역할 선택
role = st.sidebar.selectbox(
    "AI 역할",
    ["일반 상담사", "초등학교 선생님", "시니어 개발자", "영어 튜터"],
)

# 역할별 system prompt
role_prompts = {
    "일반 상담사": "너는 사용자를 도와주는 상담사야. 공감적으로 짧게 답해줘.",
    "초등학교 선생님": "너는 초등학교 선생님이야. 아이들이 이해할 수 있게 쉽고 재미있게 설명해줘. 이모지도 사용해줘.",
    "시니어 개발자": "너는 10년차 시니어 개발자야. 기술적으로 정확하고, 코드 예시를 포함해서 답해줘.",
    "영어 튜터": "너는 영어 튜터야. 한국어로 질문하면 영어로 답하고, 한국어 해석도 함께 제공해줘.",
}

# temperature 슬라이더
temperature = st.sidebar.slider(
    "Temperature (창의성)",
    min_value=0.0,
    max_value=1.5,
    value=0.7,
    step=0.1,
    help="낮을수록 일관된 답변, 높을수록 창의적인 답변",
)

# 대화 초기화 버튼
if st.sidebar.button("대화 초기화"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"현재 역할: {role}")
st.sidebar.caption(f"Temperature: {temperature}")

# ── 메인 영역 ──
st.title("AI 챗봇")
st.caption(f"역할: {role} | Temperature: {temperature}")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

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

    # API 호출 (스트리밍)
    with st.chat_message("assistant"):
        api_messages = [{"role": "system", "content": role_prompts[role]}]
        api_messages += st.session_state.messages

        stream = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=api_messages,
            temperature=temperature,
            stream=True,
        )
        response = st.write_stream(stream)

    # 어시스턴트 메시지 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
