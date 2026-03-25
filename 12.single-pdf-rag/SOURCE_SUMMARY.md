# 03-single-pdf-rag 소스 요약

> PDF 한 개를 업로드 → 벡터DB에 저장 → 질문하면 답변하는 RAG 챗봇
> step1~step5까지 단계별로 개념을 하나씩 쌓아가는 구조

---

## 파일 구성

| 파일 | 역할 | 새로 배우는 개념 |
|------|------|-----------------|
| `step1_load_pdf.py` | PDF → Document 변환 | PyMuPDFLoader, Document 객체 |
| `step2_chunking.py` | Document → 청크 분할 | RecursiveCharacterTextSplitter, chunk_size/overlap |
| `step3_embedding.py` | 텍스트 → 벡터 변환 체험 | OpenAIEmbeddings, 코사인 유사도 |
| `step4_vector_store.py` | 청크 → ChromaDB 저장/검색 | Chroma, similarity_search |
| `step5_rag_chain.py` | 검색 + LLM 답변 생성 연결 | create_retrieval_chain, create_stuff_documents_chain |
| `app.py` | 완성 앱 | session_state, chat UI, expander 참조 문서 |
| `init_db.py` | 벡터DB 초기화 스크립트 | CLI에서 벡터DB 미리 생성 |

---

## 전체 데이터 흐름

```
PDF 업로드
  → [step1] PyMuPDFLoader.load() → Document 리스트 (페이지당 1개)
  → [step2] RecursiveCharacterTextSplitter → 작은 청크들
  → [step3] OpenAIEmbeddings → 숫자 벡터 (1536차원)
  → [step4] Chroma.from_documents() → 벡터DB에 저장

질문 입력
  → [step5] create_retrieval_chain.invoke({"input": 질문})
     ├─ retriever: ChromaDB에서 유사 청크 3개 검색
     ├─ create_stuff_documents_chain: 청크를 프롬프트에 삽입
     └─ ChatOpenAI (gpt-5-nano): 답변 생성
  → answer + context(출처) 반환
```

---

## step1_load_pdf.py — PDF → Document 변환

### 1~8번 줄: 독스트링 (설명문)

```python
"""
Step 1: 문서 로딩 — PDF를 Document 객체로 변환
...
"""
```

파이썬에서 `"""` 세 개로 감싸면 **여러 줄 문자열(주석)**입니다.
이 파일이 뭘 하는지 사람한테 알려주는 용도예요. 실행에는 아무 영향 없습니다.

### 10~15번 줄: import (라이브러리 불러오기)

```python
import streamlit as st
```
- **Streamlit** = 파이썬으로 웹 화면 만드는 라이브러리. `st`라는 별명으로 씁니다.
- 버튼, 파일 업로드, 텍스트 표시 등을 `st.button()`, `st.file_uploader()` 이런 식으로 호출하면 알아서 웹페이지를 만들어줍니다.

```python
from langchain_community.document_loaders import PyMuPDFLoader
```
- **LangChain** = AI 앱 만드는 프레임워크.
- **PyMuPDFLoader** = PDF 파일을 읽어서 텍스트로 바꿔주는 도구. 내부적으로 PyMuPDF 라이브러리를 씁니다.

```python
from streamlit.runtime.uploaded_file_manager import UploadedFile
```
- Streamlit에서 사용자가 업로드한 파일의 **타입 힌트**. "이 함수의 매개변수는 업로드된 파일이에요"라고 표시하는 용도입니다.

```python
import os
```
- 파이썬 기본 라이브러리. 파일 경로 조작, 폴더 만들기 등에 씁니다.

```python
from pathlib import Path
```
- 역시 파일 경로를 다루는 건데, `os`보다 현대적인 방식. `/`로 경로를 합칠 수 있어요.

```python
from dotenv import load_dotenv
```
- `.env` 파일에 적어둔 환경변수(예: API 키)를 읽어오는 라이브러리.

### 17~21번 줄: 환경 설정

```python
env_path = Path(__file__).parent.parent / ".env"
```
- `__file__` = 지금 실행 중인 이 파일의 경로
- `.parent` = 상위 폴더. `.parent.parent`는 두 단계 위 폴더
- `/ ".env"` = 거기서 `.env` 파일을 찾음
- 즉, `rag-system-example/.env` 파일을 가리킵니다.

```python
load_dotenv(dotenv_path=env_path)
```
- 그 `.env` 파일을 읽어서 환경변수로 등록. (예: `OPENAI_API_KEY=sk-...`)

```python
BASE_DIR = Path(__file__).resolve().parent
```
- 이 파일이 있는 폴더의 **절대 경로**를 저장. 나중에 임시 폴더 만들 때 기준점으로 씁니다.

### 24~32번 줄: save_uploadedfile 함수

```python
def save_uploadedfile(uploadedfile: UploadedFile) -> str:
```
- 함수 정의. 매개변수 `uploadedfile`은 사용자가 웹에서 올린 파일, 반환값은 `str`(저장된 파일 경로).

```python
    temp_dir = str(BASE_DIR / "PDF_임시폴더")
```
- 이 스크립트 옆에 "PDF_임시폴더"라는 폴더 경로를 만듭니다.

```python
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
```
- 폴더가 없으면 새로 만듭니다.

```python
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path
```
- 업로드된 파일의 내용을 `"wb"` (바이너리 쓰기) 모드로 디스크에 저장합니다.
- **왜 저장하나?** Streamlit의 업로드 파일은 메모리에만 있어서, PyMuPDFLoader가 읽으려면 실제 파일이 필요하기 때문입니다.

### 35~45번 줄: load_pdf 함수 (핵심!)

```python
def load_pdf(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
```
- **이 3줄이 이 파일의 핵심입니다.**
- `PyMuPDFLoader`에 PDF 경로를 넘기고, `.load()`를 호출하면 → **Document 객체 리스트**가 나옵니다.
- PDF가 5페이지면 `documents`에 5개의 Document가 들어갑니다.

각 Document는 이렇게 생겼습니다:
```python
Document(
    page_content="이 페이지의 텍스트 전체...",
    metadata={"page": 0, "source": "파일경로", ...}
)
```

```python
    for doc in documents:
        doc.metadata["file_path"] = pdf_path
        doc.metadata["filename"] = os.path.basename(pdf_path)
```
- 각 Document의 metadata에 파일 경로와 파일 이름을 추가합니다.
- `os.path.basename("/a/b/c.pdf")` → `"c.pdf"` (파일명만 추출)

### 48~108번 줄: Streamlit UI (웹 화면)

```python
def main():
    st.set_page_config("Step 1: 문서 로딩", layout="wide")
    st.header("Step 1: PDF → Document 변환")
```
- 웹페이지 제목 설정, 넓은 레이아웃, 큰 제목 표시.

```python
    st.markdown("""...""")
```
- 학습 설명을 마크다운 형식으로 화면에 표시.

```python
    pdf_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
```
- 파일 업로드 위젯. PDF만 받습니다. 사용자가 파일을 올리면 `pdf_file`에 값이 들어오고, 안 올리면 `None`.

```python
    if pdf_file:
```
- 파일이 업로드됐을 때만 아래 코드 실행.

```python
        with st.spinner("PDF 파일을 저장하는 중..."):
            pdf_path = save_uploadedfile(pdf_file)
        st.success(f"파일 저장 완료: {pdf_file.name}")
```
- `st.spinner` = "로딩 중..." 표시.
- 저장 끝나면 `st.success`로 초록색 성공 메시지.

```python
        with st.spinner("PDF를 Document로 변환하는 중..."):
            documents = load_pdf(pdf_path)
        st.success(f"총 **{len(documents)}**개 페이지를 Document로 변환했습니다.")
```
- 아까 만든 `load_pdf` 호출해서 Document 리스트를 받고, 몇 페이지인지 표시.

```python
        for i, doc in enumerate(documents):
            with st.expander(f"페이지 {i + 1} (길이: {len(doc.page_content)}자)"):
                st.json(doc.metadata)
                st.text_area(
                    "page_content (처음 500자)",
                    value=doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                    height=200,
                    key=f"doc_{i}"
                )
```
- 각 페이지를 접을 수 있는 패널(expander)로 보여줍니다.
- `st.json(doc.metadata)` → metadata를 JSON 형태로 표시
- `doc.page_content[:500]` → 텍스트 내용 중 처음 500자만 미리보기
- `key=f"doc_{i}"` → Streamlit 위젯마다 고유 키가 필요해서 페이지 번호를 키로 씀

```python
if __name__ == "__main__":
    main()
```
- 이 파일을 직접 실행할 때(`streamlit run step1_load_pdf.py`) `main()`을 호출합니다.

### step1 요약 흐름

```
사용자가 PDF 업로드
  → 디스크에 임시 저장 (save_uploadedfile)
  → PyMuPDFLoader로 읽기 (load_pdf)
  → Document 리스트 반환 (페이지당 1개)
  → 화면에 metadata + 텍스트 미리보기 표시
```

---

## step2_chunking.py — 텍스트 분할

> step1의 `save_uploadedfile`, `load_pdf` 함수를 import해서 사용합니다.
> ```python
> from step1_load_pdf import save_uploadedfile, load_pdf
> ```

### 12번 줄: 새로운 import

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
- **RecursiveCharacterTextSplitter** = 긴 텍스트를 작은 조각(청크)으로 잘라주는 도구.
- "Recursive"인 이유: 여러 구분자를 순서대로 시도합니다. 먼저 `\n\n`(빈 줄)로 끊어보고, 안 되면 `\n`(줄바꿈), 그래도 안 되면 ` `(공백), 최후에는 한 글자씩. 이렇게 하면 문장이 어색하게 잘리는 걸 최소화합니다.

### 23~30번 줄: chunk_documents 함수 (이 파일의 핵심!)

```python
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
```
- `documents` = step1에서 만든 Document 리스트
- `chunk_size=1000` = 한 청크의 최대 글자 수
- `chunk_overlap=200` = 청크끼리 겹치는 글자 수

```python
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
```
- 분할기를 만듭니다. "1000자 단위로 자르되, 200자는 겹쳐라."

```python
    chunks = text_splitter.split_documents(documents)
    return chunks
```
- `.split_documents()`에 Document 리스트를 넣으면 → 더 작은 Document 리스트가 나옵니다.
- 원래 metadata(페이지 번호 등)도 그대로 유지됩니다.

### 왜 이렇게 하나? 두 가지 이유:

1. **LLM 입력 제한**: GPT 같은 모델은 한 번에 처리할 수 있는 텍스트 길이가 제한되어 있음
2. **검색 정확도**: 100페이지 문서 전체보다, 관련 있는 작은 조각을 찾아서 넘기는 게 훨씬 정확함

### overlap이 왜 필요한가?

예를 들어:

원문: "서울의 인구는 약 950만 명이다. 이는 한국 전체 인구의 약 18%에 해당한다."

`chunk_size=30, overlap=0`으로 자르면:
- 청크1: "서울의 인구는 약 950만 명이다."
- 청크2: "이는 한국 전체 인구의 약 18%"

청크2만 보면 **"이는"이 뭘 가리키는지 모릅니다.** overlap이 있으면 앞 청크의 끝부분이 다음 청크 시작에 반복되어 문맥이 연결됩니다.

### UI 부분 (새로운 것만)

```python
col1, col2 = st.columns(2)
```
- 화면을 **2열**로 나눕니다. 왼쪽/오른쪽에 각각 위젯을 배치할 수 있음.

```python
with col1:
    chunk_size = st.slider("chunk_size (청크 크기)", 200, 2000, 1000, step=100)
with col2:
    chunk_overlap = st.slider("chunk_overlap (겹침)", 0, 500, 200, step=50)
```
- **슬라이더 위젯**. 사용자가 마우스로 값을 조절할 수 있습니다.
- `st.slider(라벨, 최솟값, 최댓값, 기본값, step=단위)`
- 즉, chunk_size는 200~2000 사이에서 100 단위로, chunk_overlap은 0~500 사이에서 50 단위로 조절 가능.
- 슬라이더를 움직이면 Streamlit이 자동으로 페이지를 다시 실행해서 결과가 바로 바뀝니다.

```python
st.metric("분할 전 (페이지)", len(documents))
```
- 숫자를 크게 보여주는 카드. "분할 전 (페이지): 5" 이런 식으로 표시.

```python
total_chars = sum(len(d.page_content) for d in documents)
```
- 모든 페이지의 글자 수를 합산. `sum(... for d in documents)`는 제너레이터 표현식으로, 각 document의 `page_content` 길이를 더합니다.

```python
avg_chars = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
```
- 청크들의 평균 글자 수. `// max(len(chunks), 1)`는 0으로 나누기 방지용입니다.

```python
page = chunk.metadata.get("page", 0) + 1
```
- metadata에서 페이지 번호를 가져옴. `.get("page", 0)`은 "page" 키가 없으면 0을 반환.
- `+1`은 0-based → 1-based 변환 (사람은 1페이지부터 세니까).

### step1 → step2에서 추가된 것

```
step1의 결과                     step2에서 추가된 것
─────────────                   ─────────────────
Document 리스트 (페이지 단위)   → RecursiveCharacterTextSplitter로 잘게 분할
                                → chunk_size, chunk_overlap 파라미터
                                → 분할 전/후 비교 UI
```

전체 흐름에서 보면:
```
PDF → [step2: Document] → [step3: 청크 분할] → step4: 임베딩 → step5: 벡터DB → ...
```

---

## step3_embedding.py — 임베딩 체험

### 12~13번 줄: 새로운 import

```python
from langchain_openai import OpenAIEmbeddings
```
- **OpenAIEmbeddings** = OpenAI의 임베딩 API를 호출해주는 LangChain 래퍼(wrapper).
- 텍스트를 넣으면 → 숫자 배열(벡터)이 나옵니다.

```python
import numpy as np
```
- **NumPy** = 파이썬의 수학/배열 계산 라이브러리. 벡터 연산(내적, 크기 등)에 씁니다.

### 22~26번 줄: get_embedding 함수

```python
def get_embedding(text: str) -> list:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector = embeddings.embed_query(text)
    return vector
```
- `text-embedding-3-small` = OpenAI의 임베딩 모델. 텍스트를 **1536개의 숫자**로 변환합니다.
- `.embed_query(text)` = 문자열 하나를 넣으면 → `[0.023, -0.041, 0.078, ...]` 이런 숫자 리스트가 나옴.

**임베딩이 뭔가?** 한마디로, **"의미를 숫자로 바꾸는 것"**입니다.
- "고양이" → `[0.2, 0.8, -0.1, ...]`
- "강아지" → `[0.19, 0.75, -0.08, ...]` ← 비슷한 의미니까 숫자도 비슷
- "주식" → `[-0.5, 0.1, 0.9, ...]` ← 다른 의미니까 숫자도 다름

이게 왜 중요하냐면, 나중에 "검색"을 할 때 텍스트 자체를 비교하는 게 아니라 **숫자끼리 비교해서 의미가 비슷한 문서를 찾을 수 있기 때문**입니다.

### 29~33번 줄: cosine_similarity 함수

```python
def cosine_similarity(vec1: list, vec2: list) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

**코사인 유사도** = 두 벡터가 얼마나 같은 방향을 가리키는지 측정. 결과는 -1 ~ 1 사이.

하나씩 풀면:
- `np.array(vec1)` → 파이썬 리스트를 NumPy 배열로 변환 (수학 연산 가능하게)
- `np.dot(a, b)` → 내적(dot product). 두 벡터의 각 원소를 곱한 뒤 전부 더한 값.
  - 예: `[1,2] · [3,4] = 1×3 + 2×4 = 11`
- `np.linalg.norm(a)` → 벡터의 크기(길이). `√(a₁² + a₂² + ... + aₙ²)`
- 나누는 이유: 벡터 길이에 상관없이 **방향만 비교**하려고. 긴 문서든 짧은 문장이든 공정하게 비교됨.

결과 해석:
- **1.0** = 완전히 같은 의미
- **0.8+** = 매우 유사
- **0.5 정도** = 약간 관련
- **0.2 이하** = 거의 관련 없음

### UI 부분 (새로운 것만)

```python
text1 = st.text_input("문장 1", value="파이썬은 배우기 쉬운 프로그래밍 언어입니다")
```
- `st.text_input` = 한 줄 텍스트 입력 상자. `value`는 기본값.

```python
if st.button("유사도 계산", type="primary"):
```
- `st.button` = 클릭 버튼. 누르면 `True` 반환. `type="primary"`는 파란색 강조 버튼.

```python
st.progress(min(similarity, 1.0))
```
- 프로그레스 바. 0.0~1.0 사이 값을 넣으면 바가 채워짐. 유사도를 시각적으로 보여줌.

```python
st.code(str([round(v, 4) for v in vec1[:10]]) + " ...")
```
- 1536개 숫자 중 처음 10개만 소수점 4자리로 반올림해서 코드 블록으로 표시.
- `[round(v, 4) for v in vec1[:10]]` = 리스트 컴프리헨션. vec1의 첫 10개 원소를 각각 반올림.

### 108~138번 줄: 여러 문장 비교 섹션

```python
compare_texts = st.text_area(
    "비교 문장들 (한 줄에 하나씩)",
    value="AI 기술이 혁신을 이끌고 있다\n머신러닝은...\n오늘 날씨가...",
    height=120
)
```
- `st.text_area` = 여러 줄 입력 상자. `st.text_input`과 달리 엔터로 줄바꿈 가능.

```python
sentences = [s.strip() for s in compare_texts.strip().split("\n") if s.strip()]
```
- 입력 텍스트를 `\n`(줄바꿈)으로 쪼개서 리스트로 만듦.
- `.strip()` = 앞뒤 공백/줄바꿈 제거.
- `if s.strip()` = 빈 줄은 제외.

```python
results.sort(key=lambda x: x[1], reverse=True)
```
- 결과를 유사도 높은 순으로 정렬.
- `lambda x: x[1]` → 튜플 `(문장, 유사도)`에서 유사도(두 번째 값)를 기준으로 정렬.
- `reverse=True` → 내림차순(높은 것부터).

```python
col_rank, col_sent, col_sim = st.columns([1, 6, 2])
```
- 3열 레이아웃인데, 비율이 `[1, 6, 2]`. 문장 열이 가장 넓음.

### step2 → step3에서 추가된 것

```
step2까지의 흐름                  step3에서 추가된 것
──────────────                   ────────────────
PDF → Document → 청크            → OpenAIEmbeddings로 텍스트를 벡터로 변환
                                 → 코사인 유사도로 의미 비교
                                 → "검색 = 벡터 간 유사도 비교" 원리 체험
```

전체 흐름:
```
PDF → [step2: Document] → [step3: 청크] → [step4: 임베딩 체험] → step5: 벡터DB 저장 → ...
```

---

## step4_vector_store.py — ChromaDB 저장과 검색

> step1의 `save_uploadedfile`, `load_pdf`와 step2의 `chunk_documents`를 import해서 사용합니다.
> ```python
> from step1_load_pdf import save_uploadedfile, load_pdf
> from step2_chunking import chunk_documents
> ```

### 15번 줄: 새로운 import

```python
from langchain_community.vectorstores import Chroma
```
- **Chroma** = 벡터 데이터베이스(ChromaDB)의 LangChain 래퍼.
- **벡터 데이터베이스가 뭔가?** 일반 DB는 텍스트/숫자를 저장하고 검색하지만, 벡터 DB는 임베딩(숫자 배열)을 저장하고 "비슷한 것"을 찾아줍니다.
  - 일반 DB 검색: `WHERE name = "홍길동"` (정확히 일치)
  - 벡터 DB 검색: "이 벡터와 가장 비슷한 벡터 3개 줘" (의미적 유사)

### 28번 줄

```python
CHROMA_DIR = str(BASE_DIR / "chroma_db")
```
- ChromaDB가 데이터를 저장할 폴더 경로. 이 폴더에 벡터들이 파일로 저장되어서, 프로그램을 껐다 켜도 남아있습니다.

### 31~53번 줄: save_to_vector_store 함수 (핵심 1)

```python
def save_to_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```
- step3에서 봤던 임베딩 모델을 준비합니다.

```python
    try:
        existing_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="documents"
        )
        existing_store._collection.delete(where={})
    except Exception:
        pass
```
- 이미 저장된 데이터가 있으면 싹 지웁니다 (덮어쓰기 용도).
- `Chroma(persist_directory=..., collection_name=...)` = 기존 DB에 연결
- `._collection.delete(where={})` = 조건 없이(`{}`) 전부 삭제
- `try/except` = DB가 아직 없으면 에러가 나는데, 그냥 무시하고 넘어감

```python
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="documents"
    )
    return vector_store
```
- **이게 이 파일의 핵심 한 줄입니다.**
- `Chroma.from_documents()`가 하는 일:
  1. 각 청크의 `page_content`를 → OpenAIEmbeddings로 벡터로 변환
  2. 그 벡터 + 원본 텍스트 + metadata를 → ChromaDB에 저장
  3. `persist_directory`에 파일로 영구 저장
- 즉, step1~3에서 했던 로드→분할→임베딩을 한 번에 처리해서 DB에 넣는 겁니다.

### 56~65번 줄: search_similar 함수 (핵심 2)

```python
def search_similar(query: str, k: int = 3):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="documents"
    )
```
- 저장해둔 ChromaDB에 다시 연결합니다.

```python
    results = vector_store.similarity_search(query, k=k)
    return results
```
- `similarity_search`가 하는 일:
  1. query 문자열을 → 임베딩 벡터로 변환
  2. DB에 저장된 모든 벡터와 → 코사인 유사도 비교 (step3에서 배운 것!)
  3. 가장 유사한 상위 k개의 → Document를 반환
- 예를 들어 `search_similar("핵심 내용 알려줘", k=3)` 하면:
  - "핵심 내용 알려줘"를 벡터로 바꾸고
  - DB에서 의미가 가장 비슷한 청크 3개를 찾아서
  - Document 리스트로 돌려줌

### UI 부분 (새로운 것만)

```python
tab1, tab2 = st.tabs(["📥 저장", "🔍 검색"])
```
- **탭 UI**. 화면 상단에 "저장" / "검색" 탭이 생기고, 클릭해서 전환합니다.

```python
with tab1:
    ...  # 저장 관련 UI
with tab2:
    ...  # 검색 관련 UI
```
- 각 탭 안에 별도의 위젯들을 배치.

```python
st.balloons()
```
- 저장 완료되면 풍선 애니메이션 표시. 순수하게 시각적 효과.

```python
query = st.text_input(
    "검색할 질문을 입력하세요",
    placeholder="예) 문서에서 핵심 내용을 알려줘"
)
```
- `placeholder` = 입력 전에 회색으로 보이는 힌트 텍스트. 실제 값은 아님.

```python
except Exception as e:
    st.error(f"벡터DB를 찾을 수 없습니다. ... 오류: {e}")
```
- 아직 저장을 안 했는데 검색하면 에러가 나니까, 사용자에게 안내 메시지를 보여줌.

### step3 → step4에서 추가된 것

```
step3까지의 흐름                       step4에서 추가된 것
──────────────                        ────────────────
PDF → Document → 청크 → 임베딩 체험    → ChromaDB에 실제 저장 (from_documents)
                                      → similarity_search로 유사 청크 검색
                                      → 저장/검색 탭 UI
```

전체 흐름:
```
PDF → [step2: Document] → [step3: 청크] → [step4: 임베딩 이해] → [step5: 벡터DB 저장+검색] → step6: RAG 체인
```

step4까지는 "검색"까지만 합니다. 찾은 문서를 LLM에 넘겨서 답변을 생성하는 건 step5입니다.

---

## step5_rag_chain.py — RAG 체인 (검색 + 생성)

### 14~18번 줄: 새로운 import들

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
```
- **ChatOpenAI** = OpenAI의 채팅 모델(GPT)을 호출하는 래퍼. step4의 OpenAIEmbeddings는 텍스트→숫자 변환용이었고, 이건 **답변 생성용**입니다.

```python
from langchain_core.prompts import ChatPromptTemplate
```
- **프롬프트 템플릿** = LLM에 보낼 메시지를 미리 틀로 만들어두는 것. `{context}`, `{input}` 같은 빈칸을 넣어두고, 나중에 값을 채워 넣습니다.

```python
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
```
- 이 두 개가 이 파일의 핵심입니다.
- **create_stuff_documents_chain** = 검색된 문서들을 프롬프트에 끼워넣는 체인
- **create_retrieval_chain** = 검색 + 문서 삽입 + 답변 생성을 하나로 묶는 체인

### 31~81번 줄: create_rag_chain 함수 (이 파일의 핵심!)

7단계로 나눠서 봅니다.

**1단계: 임베딩 모델**
```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```
- step4에서 저장할 때 쓴 것과 **같은 모델**을 써야 합니다. 다른 모델로 검색하면 숫자 체계가 달라서 결과가 엉망이 됩니다.

**2단계: 벡터 저장소 연결**
```python
vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="documents"
)
```
- step4에서 저장해둔 ChromaDB에 연결.

**3단계: LLM**
```python
llm = ChatOpenAI(model="gpt-5-nano", temperature=1)
```
- `gpt-5-nano` = 답변을 생성할 GPT 모델.
- `temperature=1` = 창의성 수준. 0이면 딱딱하고 정해진 답만, 높을수록 다양한 표현. 1은 기본값.

**4단계: 프롬프트 템플릿**
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 문서 기반 질문응답 AI 어시스턴트입니다.
...
- 컨텍스트에 없는 내용은 "문서에 해당 정보가 없습니다"라고 답변하세요
..."""),
    ("human", """다음 문서를 참고하여 질문에 답변해주세요:

[문서 내용]
{context}

---

질문: {input}""")
])
```
- `("system", ...)` = GPT에게 주는 역할 지시. "너는 문서 기반 QA 봇이야, 문서에 없으면 없다고 해."
- `("human", ...)` = 사용자 메시지 틀.
- `{context}` = 나중에 검색된 문서 내용이 여기에 들어감
- `{input}` = 나중에 사용자 질문이 여기에 들어감

**이게 RAG의 핵심 아이디어입니다.** GPT가 학습하지 않은 내용이라도, 관련 문서를 프롬프트에 넣어주면 그걸 읽고 답변할 수 있습니다.

**5단계: 문서 체인**
```python
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)
```
- "stuff" = "채워넣다"라는 뜻. 검색된 문서들을 `{context}` 자리에 전부 이어붙여서 넣습니다.
- 문서 3개가 검색됐으면, 3개 내용을 연결해서 하나의 `{context}`로 만들고, 프롬프트에 넣고, LLM에 보냄.

**6단계: 검색기(Retriever)**
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```
- step4의 `similarity_search`를 LangChain 체인에 연결 가능한 형태로 변환.
- `"k": 3` = 유사한 청크 3개를 검색.
- `as_retriever()` = "이 벡터 DB를 검색 도구로 쓸게"라는 의미.

**7단계: RAG 체인 조립**
```python
rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain
)
```
- retriever(검색기) + document_chain(문서→프롬프트→LLM)을 하나로 연결.
- 이 한 줄이 만드는 파이프라인:
```
질문 → retriever가 ChromaDB에서 유사 청크 3개 검색
     → document_chain이 청크들을 프롬프트 {context}에 삽입
     → LLM이 프롬프트를 읽고 답변 생성
     → {"answer": "...", "context": [...]} 반환
```

### UI 부분 (새로운 것만)

```python
if "rag_chain" not in st.session_state:
    with st.spinner("RAG 체인 초기화 중..."):
        st.session_state.rag_chain = create_rag_chain()
```
- `st.session_state` = Streamlit의 세션 저장소. 브라우저 탭이 열려있는 동안 값을 유지합니다.
- **왜 필요한가?** Streamlit은 위젯 값이 바뀔 때마다 스크립트 전체를 다시 실행합니다. session_state에 안 넣으면 질문할 때마다 RAG 체인을 새로 만들게 되어 느려집니다.
- `not in st.session_state` = 처음 한 번만 생성.

```python
response = st.session_state.rag_chain.invoke({"input": question})
```
- RAG 체인 실행. `{"input": question}` 하나만 넣으면 내부에서 검색→생성→응답까지 전부 알아서 합니다.

```python
st.write(response["answer"])
```
- `response["answer"]` = LLM이 생성한 답변 텍스트.

```python
context_docs = response.get("context", [])
```
- `response["context"]` = 검색에 사용된 Document 리스트. 어떤 문서를 참고했는지 출처로 보여줄 수 있음.

### step1~5 전체 흐름 완성!

```
[step1] PDF → Document (페이지 단위)
   ↓
[step2] Document → 청크 (작은 조각)
   ↓
[step3] 청크 → 임베딩 벡터 (의미를 숫자로)    ← 원리 이해
   ↓
[step4] 벡터 → ChromaDB 저장 + 유사도 검색    ← 저장 & 검색
   ↓
[step5] 검색 + LLM → RAG 체인                 ← 답변 생성
         │
         ├─ retriever: ChromaDB에서 관련 청크 검색
         ├─ prompt: 청크를 {context}에 삽입
         ├─ LLM: 문서를 읽고 답변 생성
         └─ 출력: answer + context(출처)
```

---

## app.py — 완성 앱

step1~5의 함수를 모두 활용하여 **사이드바 + 채팅 UI**로 통합한 앱.

- 채팅 탭: 질문 → RAG 답변 + 참조 문서(expander)
- 청크 미리보기 탭: 벡터DB에 저장된 청크 목록 확인

---

## init_db.py — 벡터DB 초기화

```python
from app import pdf_to_documents, chunk_documents, save_to_vector_store

pdf_documents = pdf_to_documents("문서.pdf")
smaller_documents = chunk_documents(pdf_documents)
save_to_vector_store(smaller_documents)
```
- Streamlit 없이 **커맨드라인에서** 벡터DB를 미리 생성하는 스크립트
- `app.py`의 함수들을 import해서 사용
- 웹 UI 없이도 PDF를 벡터DB에 넣을 수 있음

---

## 사용 모델

| 용도 | 모델 | 설명 |
|------|------|------|
| LLM (답변 생성) | `gpt-5-nano` | OpenAI 경량 모델, temperature=1 |
| 임베딩 (벡터 변환) | `text-embedding-3-small` | 1536차원, 빠르고 저렴 |

---

## 의존성 (requirements.txt)

```
streamlit              # 웹 UI
langchain              # AI 앱 프레임워크 (코어)
langchain-openai       # OpenAI 연동 (ChatOpenAI, OpenAIEmbeddings)
langchain-community    # 커뮤니티 도구 (PyMuPDFLoader, Chroma)
langchain-classic      # 클래식 체인 (create_retrieval_chain)
langchain-text-splitters  # 텍스트 분할기
chromadb               # 벡터 데이터베이스
pymupdf                # PDF 읽기 (fitz)
python-dotenv          # .env 파일 로드
```
