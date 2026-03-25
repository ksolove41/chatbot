# Kosta 챗봇 개발 워크샵

**1일 챗봇 개발 워크샵을 위한 교육 자료 저장소**

OpenAI API부터 LangChain, RAG까지 단계적으로 배우는 실습 중심 챗봇 개발 가이드

## 빠른 시작

### 1. 환경 설정
```bash
# Python 3.10+ 확인
python --version

# UV 패키지 매니저 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

### 2. API 키 설정
`.env` 파일 생성:
```bash
# OpenAI API 키 (필수)
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith 설정 (선택사항)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=kosta-chatbot-workshop
```

OpenAI API 키는 [OpenAI Platform](https://platform.openai.com/api-keys)에서 발급받을 수 있습니다.

### 3. 첫 실행
```bash
uv run python 01.settings/01.test.py
```

## 프로젝트 구조

```
01.settings/              # 환경 설정 및 API 테스트
02.single-turn/           # 단일 대화 (OpenAI API 직접 사용)
03.multi-turn/            # 멀티턴 대화 (대화 히스토리 관리)
04.zero-shot-prompting/   # Zero-shot 프롬프트 엔지니어링
05.one-shot-prompting/    # One-shot 프롬프트 엔지니어링
06.few-shot-prompting/    # Few-shot 프롬프트 엔지니어링
07.streamlit/             # Streamlit 웹 인터페이스
08.langchain-basic/       # LangChain 기본 + LangSmith
09.lcel-chain/            # LCEL 체인 구성
10.langchain-tools/       # LangChain Tool & Agent
11.stream-output/         # 스트리밍 출력
13.rag-faq/               # RAG FAQ 시스템
```

## 워크샵 진행 순서

### Phase 1: 기초 (2시간)
```bash
# 1. 환경 확인
uv run python 01.settings/01.test.py

# 2. 기본 챗봇
uv run python 02.single-turn/01.single-turn.py
uv run python 03.multi-turn/01.multi-turn.py

# 3. 프롬프트 엔지니어링
uv run python 04.zero-shot-prompting/01.zero-shot.py
uv run python 05.one-shot-prompting/01.one-shot.py
uv run python 06.few-shot-prompting/01.few-shot.py

# 4. 웹 인터페이스
uv run streamlit run 07.streamlit/01.streamlit.py
```

### Phase 2: LangChain (2시간)
```bash
# 5. LangChain 기본
uv run python 08.langchain-basic/01.langchain.py
uv run python 08.langchain-basic/02.langsmith_example.py
uv run python 08.langchain-basic/03.message_history.py

# 6. LCEL 체인
uv run python 09.lcel-chain/01.lcel_basic.py

# 7. 도구와 에이전트
uv run python 10.langchain-tools/01.tool_basic.py

# 8. 스트리밍
uv run python 11.stream-output/01.stream_basic.py
```

### Phase 3: RAG (2시간)
```bash
# 9. RAG FAQ 시스템
uv run streamlit run 13.rag-faq/start.py    # 시작 템플릿
uv run streamlit run 13.rag-faq/finish.py   # 완성 버전
```

## 학습 목표

- OpenAI API(gpt-4.1-nano)를 활용한 기본 챗봇 개발
- 프롬프트 엔지니어링 (Zero/One/Few-shot) 이해
- Streamlit으로 웹 인터페이스 구축
- LangChain을 통한 고급 챗봇 기능 구현
- RAG(Retrieval-Augmented Generation) 시스템 이해

## 주요 개념

### Temperature 설정
- **0.0**: 가장 예측 가능한 답변 (결정론적)
- **0.7**: 균형잡힌 창의성과 일관성
- **1.0**: 높은 창의성과 다양성

### LangChain 핵심 구성
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
response = llm.invoke("안녕하세요!")
```

## 문제 해결

### API 키 오류
- `.env` 파일에 `OPENAI_API_KEY`가 올바르게 설정되어 있는지 확인
- 키에 따옴표나 공백이 없는지 확인

### 패키지 설치 오류
```bash
# 가상환경 재생성
rm -rf .venv && uv venv && uv sync
```

### Streamlit 포트 충돌
```bash
uv run streamlit run app.py --server.port 8502
```

## 참고 자료

- [OpenAI API 문서](https://platform.openai.com/docs)
- [LangChain 문서](https://python.langchain.com/docs/get_started/introduction)
- [Streamlit 문서](https://docs.streamlit.io/)
- [LangSmith 문서](https://docs.smith.langchain.com/)
