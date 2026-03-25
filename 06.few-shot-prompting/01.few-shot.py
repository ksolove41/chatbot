"""Few-Shot 프롬프팅 - 예시 여러 개로 패턴 학습"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

client = OpenAI(api_key=api_key)


def ask(prompt):
    """API 호출 헬퍼"""
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


# 긍정/부정/중립 각 1개씩 = 총 3개 예시
prompt_template = """다음 상품 리뷰의 감정을 분석해줘.

[예시 1 - 긍정]
리뷰: 제품이 사진과 똑같고 배송도 빨라서 만족합니다.
감정: 긍정
이유: 제품 품질과 배송 속도에 만족
점수: 5/5

[예시 2 - 부정]
리뷰: 포장이 찢어져서 왔고 제품에 스크래치가 있었어요. 실망입니다.
감정: 부정
이유: 포장 불량 및 제품 손상
점수: 1/5

[예시 3 - 중립]
리뷰: 쓸만한데 가격이 좀 비싼 느낌이에요. 할인하면 또 살 것 같아요.
감정: 중립
이유: 품질은 인정하나 가격 대비 아쉬움
점수: 3/5

[분석할 리뷰]
리뷰: {review}"""

reviews = [
    "배송도 빠르고 품질도 좋아요! 재구매 의사 있습니다.",
    "사이즈가 안 맞아서 교환했는데 교환도 느리고 불친절해요.",
    "가격 대비 그냥 그래요. 나쁘지는 않은데 특별하지도 않네요.",
]

for review in reviews:
    result = ask(prompt_template.format(review=review))
    print(f"[리뷰] {review}")
    print(f"[분석]\n{result}")
    print()
