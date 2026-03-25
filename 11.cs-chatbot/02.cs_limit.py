"""고객 상담 챗봇의 한계 - FAQ가 많아지면 system prompt만으로는 부족"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

client = OpenAI(api_key=api_key)

# 상황 1: FAQ 5개 → system prompt에 넣기 (잘 됨)
small_faq = """
Q: 배송은 얼마나 걸려요? → 2~3 영업일
Q: 배송비는 얼마예요? → 3만원 이상 무료, 미만 시 3,000원
Q: 교환은 어떻게 해요? → 수령 후 7일 이내 가능
Q: 포인트는 어떻게 쓰나요? → 1,000포인트 이상부터 사용 가능
Q: 새벽 배송 되나요? → 수도권만 가능, 밤 11시 전 주문
"""

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": f"너는 쇼핑몰 상담사야. 아래 FAQ를 참고해서 답해줘.\n{small_faq}"},
        {"role": "user", "content": "배송비가 얼마예요?"},
    ],
    temperature=0.3,
)

print(f"[질문] 배송비가 얼마예요?")
print(f"[답변] {response.choices[0].message.content}")
print(f"[FAQ 크기] {len(small_faq)}자")

# 상황 2: FAQ 100개 → system prompt가 너무 길어짐
big_faq_lines = []
categories = ["배송", "교환", "환불", "포인트", "회원", "결제", "상품", "이벤트", "쿠폰", "앱"]
for i in range(100):
    cat = categories[i % len(categories)]
    big_faq_lines.append(f"Q: {cat} 관련 질문 {i+1}번 → {cat}에 대한 상세한 답변 내용이 여기에 들어갑니다. 이 답변은 꽤 길 수 있으며 여러 조건과 예외 사항을 포함합니다.")

big_faq = "\n".join(big_faq_lines)

# FAQ가 많아지면 생기는 문제
print(f"\n[FAQ 100개 크기] {len(big_faq):,}자 ({len(big_faq)//4:,} 토큰 추정)")
print(f"  - 매 질문마다 {len(big_faq):,}자를 보내야 함 (비용 증가)")
print(f"  - 토큰 한도 초과 위험")
print(f"  - AI가 정확한 답을 못 찾을 수 있음")
print(f"  - FAQ 추가/수정 시 코드 변경 필요")

# 해결 방법: RAG - 질문과 관련된 FAQ만 검색하여 전달
print(f"\n[해결] RAG: 1,000개 FAQ 중 관련된 3~5개만 골라서 전달")
