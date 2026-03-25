# 실행 가이드

## 환경 설정

### 1. 의존성 설치
```bash
uv sync
```

### 2. 환경변수 설정
`.env` 파일:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## 전체 실행 순서

### Phase 1: 기초 (OpenAI API 직접 사용)

```bash
# 환경 확인
uv run python 01.settings/01.test.py

# 단일 턴 대화
uv run python 02.single-turn/01.single-turn.py

# 멀티 턴 대화
uv run python 03.multi-turn/01.multi-turn.py

# 프롬프트 엔지니어링
uv run python 04.zero-shot-prompting/01.zero-shot.py
uv run python 05.one-shot-prompting/01.one-shot.py
uv run python 06.few-shot-prompting/01.few-shot.py

# Streamlit 웹 인터페이스
uv run streamlit run 07.streamlit/01.streamlit.py
```

### Phase 2: LangChain

```bash
# LangChain 기본
uv run python 08.langchain-basic/01.langchain.py
uv run python 08.langchain-basic/02.langsmith_example.py
uv run python 08.langchain-basic/03.message_history.py

# LCEL 체인
uv run python 09.lcel-chain/01.lcel_basic.py

# 도구와 에이전트
uv run python 10.langchain-tools/01.tool_basic.py

# 스트리밍 출력
uv run python 11.stream-output/01.stream_basic.py
```

### Phase 3: RAG

```bash
# RAG FAQ 시스템
uv run streamlit run 13.rag-faq/start.py    # 시작 템플릿
uv run streamlit run 13.rag-faq/finish.py   # 완성 버전
```

## 문제 해결

### API 키 오류
```bash
# .env 파일 확인 (따옴표 없이 작성)
OPENAI_API_KEY=sk-...
```

### 패키지 설치 오류
```bash
rm -rf .venv && uv venv && uv sync
```

### Streamlit 포트 충돌
```bash
uv run streamlit run app.py --server.port 8502
```

## 학습 체크리스트

- [ ] 환경 설정 완료
- [ ] OpenAI API 키 발급 및 테스트
- [ ] 단일/멀티 턴 대화 이해
- [ ] 프롬프트 엔지니어링 3가지 방식 실습
- [ ] Streamlit 웹 인터페이스 구현
- [ ] LangChain 기본 개념 이해
- [ ] LCEL 체인 구성 방법 학습
- [ ] Tool과 Agent 사용법 습득
- [ ] 스트리밍 출력 구현
- [ ] RAG 시스템 구축 및 실행
