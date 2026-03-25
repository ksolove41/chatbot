"""
실전 챗봇 2 - 언어 회화 튜터 챗봇

사용자가 선택한 언어로 회화 연습을 도와주는 튜터 챗봇입니다.
사용자가 한국어로 말하면 → 선택한 언어로 번역 + 발음 가이드
선택한 언어로 말하면 → 교정 + 자연스러운 표현 제안

실행: uv run streamlit run 12.chatbot-examples/02.language_tutor.py
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

# ── 언어별 시스템 프롬프트 ──
LANGUAGE_PROMPTS = {
    "영어": {
        "flag": "🇺🇸",
        "prompt": """너는 친절한 영어 회화 튜터야. 다음 규칙을 따라줘:

## 응답 규칙
1. 사용자가 한국어로 말하면:
   - 영어로 번역해줘
   - 발음을 한글로 표기해줘 (예: Hello → 헬로우)
   - 비슷한 표현 1개를 추가로 알려줘

2. 사용자가 영어로 말하면:
   - 문법이나 표현이 어색하면 교정해줘
   - 자연스러운 원어민 표현으로 바꿔줘
   - 잘했으면 칭찬하고 관련 표현을 알려줘

3. 대화를 이어가면서 자연스럽게 새로운 표현을 가르쳐줘
4. 난이도는 중학생 수준으로 맞춰줘
5. 매 답변 끝에 간단한 후속 질문을 해서 대화를 이어가줘

## 응답 형식
📝 번역/교정: (내용)
🗣️ 발음: (한글 발음)
💡 팁: (추가 표현이나 문법 설명)
❓ 다음 질문: (대화를 이어가는 질문)""",
    },
    "일본어": {
        "flag": "🇯🇵",
        "prompt": """너는 친절한 일본어 회화 튜터야. 다음 규칙을 따라줘:

## 응답 규칙
1. 사용자가 한국어로 말하면:
   - 일본어로 번역해줘 (히라가나/카타카나 + 한자)
   - 발음을 한글로 표기해줘 (예: ありがとう → 아리가토우)
   - 비슷한 표현 1개를 추가로 알려줘

2. 사용자가 일본어로 말하면:
   - 문법이나 표현이 어색하면 교정해줘
   - 자연스러운 표현으로 바꿔줘
   - 잘했으면 칭찬하고 관련 표현을 알려줘

3. 존댓말(です/ます)과 반말 두 가지를 함께 알려줘
4. 매 답변 끝에 간단한 후속 질문을 해서 대화를 이어가줘

## 응답 형식
📝 번역/교정: (내용)
🗣️ 발음: (한글 발음)
💡 팁: (추가 표현이나 문법 설명)
❓ 다음 질문: (대화를 이어가는 질문)""",
    },
    "중국어": {
        "flag": "🇨🇳",
        "prompt": """너는 친절한 중국어(만다린) 회화 튜터야. 다음 규칙을 따라줘:

## 응답 규칙
1. 사용자가 한국어로 말하면:
   - 중국어로 번역해줘 (간체자 + 병음)
   - 발음을 한글로 표기해줘 (예: 你好 nǐ hǎo → 니 하오)
   - 비슷한 표현 1개를 추가로 알려줘

2. 사용자가 중국어로 말하면:
   - 문법이나 표현이 어색하면 교정해줘
   - 자연스러운 표현으로 바꿔줘
   - 잘했으면 칭찬하고 관련 표현을 알려줘

3. 성조를 항상 표기해줘 (병음 포함)
4. 매 답변 끝에 간단한 후속 질문을 해서 대화를 이어가줘

## 응답 형식
📝 번역/교정: (내용)
🗣️ 발음: (한글 발음 + 병음)
💡 팁: (추가 표현이나 문법 설명)
❓ 다음 질문: (대화를 이어가는 질문)""",
    },
    "스페인어": {
        "flag": "🇪🇸",
        "prompt": """너는 친절한 스페인어 회화 튜터야. 다음 규칙을 따라줘:

## 응답 규칙
1. 사용자가 한국어로 말하면:
   - 스페인어로 번역해줘
   - 발음을 한글로 표기해줘 (예: Hola → 올라)
   - 비슷한 표현 1개를 추가로 알려줘

2. 사용자가 스페인어로 말하면:
   - 문법이나 표현이 어색하면 교정해줘
   - 자연스러운 표현으로 바꿔줘
   - 잘했으면 칭찬하고 관련 표현을 알려줘

3. 남미식과 스페인식 차이가 있으면 알려줘
4. 매 답변 끝에 간단한 후속 질문을 해서 대화를 이어가줘

## 응답 형식
📝 번역/교정: (내용)
🗣️ 발음: (한글 발음)
💡 팁: (추가 표현이나 문법 설명)
❓ 다음 질문: (대화를 이어가는 질문)""",
    },
}

# ── Streamlit UI ──
st.title("🌍 언어 회화 튜터")

# 사이드바 - 언어 선택
st.sidebar.title("설정")
selected_lang = st.sidebar.selectbox(
    "학습할 언어",
    list(LANGUAGE_PROMPTS.keys()),
)

lang_info = LANGUAGE_PROMPTS[selected_lang]

# 난이도 선택
difficulty = st.sidebar.selectbox(
    "난이도",
    ["입문 (인사, 자기소개)", "초급 (일상 대화)", "중급 (의견 표현)"],
)

# 상황 선택
situation = st.sidebar.selectbox(
    "회화 상황",
    ["자유 대화", "카페에서 주문", "길 묻기", "쇼핑하기", "자기소개"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**현재 언어**: {lang_info['flag']} {selected_lang}")
st.sidebar.markdown(f"**난이도**: {difficulty}")
st.sidebar.markdown(f"**상황**: {situation}")

if st.sidebar.button("대화 초기화"):
    st.session_state.tutor_messages = []
    st.rerun()

# 상황별 추가 지시
situation_prompt = ""
if situation != "자유 대화":
    situation_prompt = f"\n\n현재 회화 상황은 '{situation}'이야. 이 상황에 맞는 대화를 유도해줘."

difficulty_prompt = f"\n난이도는 '{difficulty}'에 맞춰줘."

# 전체 시스템 프롬프트 조합
full_prompt = lang_info["prompt"] + difficulty_prompt + situation_prompt

st.caption(f"{lang_info['flag']} {selected_lang} 회화 튜터 | {difficulty} | {situation}")

# 세션 상태 초기화
if "tutor_messages" not in st.session_state:
    st.session_state.tutor_messages = []

# 시작 메시지
if not st.session_state.tutor_messages:
    greetings = {
        "영어": "안녕하세요! 영어 회화 튜터입니다 😊\n한국어로 말해도 되고, 영어로 도전해봐도 좋아요!\n\n무엇부터 시작해볼까요?",
        "일본어": "안녕하세요! 일본어 회화 튜터입니다 😊\n한국어로 말해도 되고, 일본어로 도전해봐도 좋아요!\n\n무엇부터 시작해볼까요?",
        "중국어": "안녕하세요! 중국어 회화 튜터입니다 😊\n한국어로 말해도 되고, 중국어로 도전해봐도 좋아요!\n\n무엇부터 시작해볼까요?",
        "스페인어": "안녕하세요! 스페인어 회화 튜터입니다 😊\n한국어로 말해도 되고, 스페인어로 도전해봐도 좋아요!\n\n무엇부터 시작해볼까요?",
    }
    st.session_state.tutor_messages.append(
        {"role": "assistant", "content": greetings[selected_lang]}
    )

# 빠른 시작 버튼
st.markdown("**빠른 시작:**")
quick_starts = {
    "영어": ["안녕하세요를 영어로?", "오늘 날씨 좋다를 영어로", "Hi, how are you?"],
    "일본어": ["안녕하세요를 일본어로?", "감사합니다를 일본어로", "こんにちは"],
    "중국어": ["안녕하세요를 중국어로?", "감사합니다를 중국어로", "你好"],
    "스페인어": ["안녕하세요를 스페인어로?", "감사합니다를 스페인어로", "Hola"],
}

cols = st.columns(3)
for i, qs in enumerate(quick_starts[selected_lang]):
    if cols[i].button(qs, key=f"qs_{i}"):
        st.session_state.tutor_messages.append({"role": "user", "content": qs})
        st.rerun()

st.markdown("---")

# 기존 메시지 출력
for msg in st.session_state.tutor_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 입력 처리
prompt = st.chat_input(f"{selected_lang}로 말해보세요! (한국어도 OK)")

# 빠른 시작 버튼으로 추가된 메시지 처리
if not prompt and len(st.session_state.tutor_messages) >= 2:
    last_msg = st.session_state.tutor_messages[-1]
    if last_msg["role"] == "user" and "processed" not in last_msg:
        prompt = last_msg["content"]
        last_msg["processed"] = True
        with st.chat_message("user"):
            st.markdown(prompt)

if prompt:
    # 직접 입력한 경우
    if not any(m.get("processed") for m in st.session_state.tutor_messages if m.get("content") == prompt and m.get("role") == "user"):
        st.session_state.tutor_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # API 호출
    with st.chat_message("assistant"):
        api_messages = [{"role": "system", "content": full_prompt}]
        api_messages += [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.tutor_messages
        ]

        stream = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=api_messages,
            temperature=0.7,
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.tutor_messages.append({"role": "assistant", "content": response})
