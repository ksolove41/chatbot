"""출력 파서 - AIMessage를 문자열이나 JSON으로 자동 변환"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# 환경변수 로드
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
prompt = ChatPromptTemplate.from_template("{food}의 칼로리를 한 줄로 알려줘.")

# 파서 없음 → AIMessage 객체
print("[파서 없음] prompt | llm")
chain_no_parser = prompt | llm
result = chain_no_parser.invoke({"food": "김치찌개"})
print(f"결과: {result}")
print(f"타입: {type(result)}")
print(f"텍스트 꺼내기: {result.content}")

# StrOutputParser → 바로 문자열
print(f"\n[StrOutputParser] prompt | llm | StrOutputParser()")
chain_str = prompt | llm | StrOutputParser()
result = chain_str.invoke({"food": "김치찌개"})
print(f"결과: {result}")
print(f"타입: {type(result)}")

# JsonOutputParser → 바로 dict
print(f"\n[JsonOutputParser] JSON으로 구조화")
chain_json = (
    ChatPromptTemplate.from_template(
        "다음 음식의 정보를 JSON으로 답해줘. "
        "키는 name, calories, category로 해줘.\n\n"
        "음식: {food}"
    )
    | llm
    | JsonOutputParser()
)

result = chain_json.invoke({"food": "김치찌개"})
print(f"결과: {result}")
print(f"타입: {type(result)}")
print(f"  이름: {result.get('name')}")
print(f"  칼로리: {result.get('calories')}")
print(f"  카테고리: {result.get('category')}")

# batch() - 여러 개 한번에 처리
print(f"\n[batch] 여러 개 한번에 처리")
foods = [{"food": "비빔밥"}, {"food": "떡볶이"}, {"food": "삼겹살"}]
results = chain_json.batch(foods)

for r in results:
    print(f"  {r.get('name', '?'):>6} | {str(r.get('calories', '?')):>10} | {r.get('category', '?')}")
