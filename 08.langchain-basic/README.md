# LangChain 챗봇 개발 및 LangSmith 추적 가이드

LangChain을 사용한 챗봇 개발과 LangSmith를 통한 애플리케이션 추적 및 모니터링 방법을 설명합니다.

## 프로젝트 구조
```
08.langchain-basic/
├── 00.setting.md          # 환경 설정 가이드
├── 01.langchain.py        # 기본 LangChain 챗봇
├── 02.langsmith_example.py # LangSmith 추적 예제
├── 03.message_history.py  # 대화 히스토리 기능
└── README.md              # 이 파일
```

## 주요 파일 설명

- **01.langchain.py**: LangChain + OpenAI 기본 사용법
- **02.langsmith_example.py**: LangSmith 추적 활성화 예제
- **03.message_history.py**: 세션 기반 대화 히스토리 구현

## LangSmith 설정

### 1. 계정 생성
1. [LangSmith 사이트](https://smith.langchain.com) 접속
2. 계정 생성 및 로그인
3. API 키 발급

### 2. 환경변수 설정

`.env` 파일에 다음 내용 추가:

```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith 설정
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=my-chatbot-project
```

## 기본 사용 예제

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7,
)

response = llm.invoke("안녕하세요! 파이썬에 대해 설명해주세요.")
print(response.content)
```

## 실행 방법

```bash
# 기본 LangChain 챗봇
uv run python 08.langchain-basic/01.langchain.py

# LangSmith 추적 예제
uv run python 08.langchain-basic/02.langsmith_example.py

# 대화 히스토리
uv run python 08.langchain-basic/03.message_history.py
```

## 참고 자료

- [LangSmith 공식 문서](https://docs.smith.langchain.com/)
- [LangChain 공식 문서](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API 문서](https://platform.openai.com/docs)
