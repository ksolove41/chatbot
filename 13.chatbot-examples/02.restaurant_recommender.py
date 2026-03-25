"""
실전 챗봇 2 - 맛집 추천 챗봇 (웹 검색 연동)

DuckDuckGo로 실제 웹 검색을 해서 맛집을 추천합니다.
사용자 조건 수집 → 검색 → AI가 정리해서 추천하는 구조입니다.

실행: uv run streamlit run 13.chatbot-examples/02.restaurant_recommender.py
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from duckduckgo_search import DDGS

# 환경설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

client = OpenAI(api_key=api_key)


# ── 검색 함수 ──
def search_restaurants(query: str) -> str:
    """DuckDuckGo로 맛집을 검색합니다."""
    try:
        results = DDGS().text(f"{query} 맛집 추천", max_results=5)
        if not results:
            return "검색 결과가 없습니다."

        search_text = ""
        for r in results:
            search_text += f"- {r['title']}: {r['body']}\n"
        return search_text
    except Exception as e:
        return f"검색 중 오류: {str(e)}"


# ── OpenAI Function Calling 도구 정의 ──
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "DuckDuckGo로 맛집 정보를 웹 검색합니다. 지역, 음식 종류 등을 포함한 검색어를 넣으세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색어. 예: '강남 일식 맛집', '홍대 데이트 레스토랑'"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ── 시스템 프롬프트 ──
SYSTEM_PROMPT = """너는 친절한 맛집 추천 전문가야.

## 역할
사용자에게 조건을 자연스럽게 물어보고, 충분한 조건이 모이면 웹 검색을 해서 실제 맛집을 추천해줘.

## 수집할 조건
1. 지역 (필수)
2. 음식 종류 또는 목적 (필수)
3. 인원수, 예산, 분위기 (선택)

## 추천 규칙
1. 조건을 한꺼번에 묻지 말고 자연스럽게 대화하며 물어봐
2. 최소 지역 + 음식종류/목적이 나오면 search_restaurants 도구로 검색해
3. 검색 결과를 바탕으로 3곳을 아래 형식으로 추천해:

### 🍽️ 추천 맛집

**1. [가게 이름]**
- 📍 위치: 위치 정보
- 💰 가격대: 예상 금액
- ⭐ 추천 이유: 검색 결과 기반 설명
- 💡 한줄평: 특징

4. 검색 결과가 부족하면 솔직하게 말하고, 다른 검색어로 재검색해
5. 추천 후 "다른 조건으로 다시 찾아드릴까요?" 물어봐

## 대화 스타일
- 친근하고 편안한 말투
- 음식 이모지를 적절히 사용
- 답변은 간결하게
"""


# ── 도구 실행 + 후속 응답 처리 ──
def process_with_tools(messages):
    """도구 호출이 포함된 응답을 처리합니다."""

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        tools=tools,
        temperature=0.7,
    )

    assistant_message = response.choices[0].message

    # 도구 호출이 없으면 바로 반환
    if not assistant_message.tool_calls:
        return assistant_message.content

    # 도구 호출 실행
    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        query = args.get("query", "")

        # 검색 실행 (UI에 검색 중 표시)
        st.info(f"🔍 '{query}' 검색 중...")
        search_result = search_restaurants(query)

        # 도구 결과를 메시지에 추가
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": search_result,
        })

    # 검색 결과를 바탕으로 최종 답변 생성 (스트리밍)
    final_response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        temperature=0.7,
        stream=True,
    )

    return final_response


# ── Streamlit UI ──
st.title("🍽️ 맛집 추천 챗봇")
st.caption("실제 웹 검색으로 맛집을 찾아드려요! (DuckDuckGo 검색 연동)")

# 사이드바
st.sidebar.title("빠른 선택")

quick_options = [
    "강남에서 점심 먹을 곳 추천해줘",
    "홍대 데이트 코스 추천",
    "혼밥하기 좋은 곳 알려줘",
    "종로 한식 맛집 추천해줘",
    "이태원 분위기 좋은 레스토랑",
]

for opt in quick_options:
    if st.sidebar.button(opt, key=f"quick_{opt}"):
        if "food_messages" not in st.session_state:
            st.session_state.food_messages = [
                {"role": "assistant", "content": "안녕하세요! 🍽️ 맛집 추천 챗봇이에요.\n\n어디서 뭘 먹고 싶으세요? 지역이나 음식 종류를 알려주세요!\n\n🔍 *실제 웹 검색으로 찾아드려요!*"}
            ]
        st.session_state.food_messages.append({"role": "user", "content": opt})
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("🔍 DuckDuckGo 웹 검색 연동")
st.sidebar.markdown("💰 검색 비용: **무료**")
st.sidebar.markdown("💳 OpenAI 모델 비용만 발생")

if st.sidebar.button("대화 초기화"):
    st.session_state.food_messages = [
        {"role": "assistant", "content": "안녕하세요! 🍽️ 맛집 추천 챗봇이에요.\n\n어디서 뭘 먹고 싶으세요? 지역이나 음식 종류를 알려주세요!\n\n🔍 *실제 웹 검색으로 찾아드려요!*"}
    ]
    st.rerun()

# 세션 상태 초기화
if "food_messages" not in st.session_state:
    st.session_state.food_messages = [
        {"role": "assistant", "content": "안녕하세요! 🍽️ 맛집 추천 챗봇이에요.\n\n어디서 뭘 먹고 싶으세요? 지역이나 음식 종류를 알려주세요!\n\n🔍 *실제 웹 검색으로 찾아드려요!*"}
    ]

# 기존 메시지 출력
for msg in st.session_state.food_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 입력 처리
prompt = st.chat_input("어떤 맛집을 찾으세요?")

# 빠른 선택 버튼 처리
if not prompt and len(st.session_state.food_messages) >= 2:
    last_msg = st.session_state.food_messages[-1]
    if last_msg["role"] == "user" and "processed" not in last_msg:
        prompt = last_msg["content"]
        last_msg["processed"] = True
        with st.chat_message("user"):
            st.markdown(prompt)

if prompt:
    # 직접 입력
    if not any(
        m.get("processed")
        for m in st.session_state.food_messages
        if m.get("content") == prompt and m.get("role") == "user"
    ):
        st.session_state.food_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # API 호출 (도구 포함)
    with st.chat_message("assistant"):
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        api_messages += [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.food_messages
        ]

        result = process_with_tools(api_messages)

        # 스트리밍 응답이면 write_stream, 문자열이면 바로 출력
        if isinstance(result, str):
            st.markdown(result)
            response = result
        else:
            response = st.write_stream(result)

    st.session_state.food_messages.append({"role": "assistant", "content": response})
