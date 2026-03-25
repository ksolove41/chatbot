# app.py 소스 요약 — 실전 완성 RAG 챗봇

> step2~6에서 배운 모든 것을 하나의 파일에 모은 **실전용 완성 앱**
> step7과 달리 PDF 페이지 이미지 뷰어, 청크 미리보기 탭, 문서 삭제, 답변 캐싱이 추가됨

---

## 전체 구조 (3단계로 나뉨)

```
app.py
│
├── 1단계: PDF → 벡터DB 저장 함수들 (34~89번 줄)
│   ├── save_uploadedfile()     ← step2에서 배운 것
│   ├── pdf_to_documents()      ← step2에서 배운 것
│   ├── chunk_documents()       ← step3에서 배운 것
│   └── save_to_vector_store()  ← step5에서 배운 것
│
├── 2단계: RAG 기능 함수들 (92~188번 줄)
│   ├── initialize_rag_system() ← 임베딩+LLM+벡터DB+체인을 한번에 초기화
│   ├── create_rag_chain()      ← step6에서 배운 것
│   └── process_question()      ← RAG 체인 실행 + 캐싱
│
├── 3단계: PDF 이미지 뷰어 함수들 (191~223번 줄)
│   ├── convert_pdf_to_images() ← PDF → PNG 이미지 변환
│   ├── display_pdf_page()      ← 이미지를 화면에 표시
│   └── natural_sort_key()      ← 파일명 자연 정렬
│
└── main() — Streamlit UI (226~462번 줄)
    ├── 사이드바: PDF 업로드 + 문서 정보 + 삭제
    └── 메인: 채팅 탭 (좌: 채팅, 우: PDF 뷰어) + 청크 미리보기 탭
```

---

## 1~9번 줄: 독스트링

```python
"""
RAG FAQ 챗봇 - Streamlit 버전 (OpenAI)

PDF 문서를 업로드하고 질문에 답변하는 RAG 챗봇
- PDF → 벡터DB 저장
- 질문 → 관련 문서 검색 → 답변 생성
- PDF 페이지 이미지로 표시
- OpenAI API 사용
"""
```
- 이 파일이 뭘 하는지 4줄로 요약. 실행에는 영향 없음.

---

## 11~29번 줄: import

```python
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from dotenv import load_dotenv
```
- step 파일들에서 이미 봤던 것들. Streamlit UI + 환경변수 로드.

```python
load_dotenv()
```
- `.env` 파일에서 `OPENAI_API_KEY` 등을 읽어옴. step 파일들은 `Path(__file__).parent.parent / ".env"`로 경로를 직접 지정했는데, 여기서는 그냥 `load_dotenv()`만 호출. 현재 폴더나 상위 폴더에서 자동으로 찾음.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader
```
- step2~6에서 하나씩 배웠던 것들을 **한 번에 전부 import**합니다.
- 새로 등장하는 건 `Document` (타입 힌트용) 정도.

```python
from typing import List, Dict, Any
import os
import fitz  # PyMuPDF
import re
from pathlib import Path
```
- `typing` = 함수 매개변수의 타입을 표시하는 용도. `List[Document]` 같은 힌트.
- `fitz` = **PyMuPDF 라이브러리**. `import fitz`로 불러옴 (이름이 다름에 주의). PDF 페이지를 이미지로 변환하는 데 사용.
- `re` = 정규표현식. 파일명을 자연 정렬할 때 사용.

```python
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
```
- step5에서 봤던 것과 동일. ChromaDB 저장 폴더 경로.

---

## 1단계: PDF → 벡터DB 저장 함수들 (34~89번 줄)

step2, 3, 5에서 배운 함수들을 그대로 모아놓은 것입니다.

### 37~44번 줄: save_uploadedfile (= step2)

```python
def save_uploadedfile(uploadedfile: UploadedFile) -> str:
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path
```
- Streamlit 업로드 파일(메모리)을 디스크에 저장. PyMuPDFLoader가 실제 파일을 필요로 하기 때문.

### 47~54번 줄: pdf_to_documents (= step2의 load_pdf)

```python
def pdf_to_documents(pdf_path: str) -> List[Document]:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata['file_path'] = pdf_path
        doc.metadata['filename'] = os.path.basename(pdf_path)
    return documents
```
- PDF → Document 리스트. 함수 이름만 `pdf_to_documents`로 바뀜, 동작은 같음.

### 57~62번 줄: chunk_documents (= step3)

```python
def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)
```
- Document → 작은 청크들. step3과 완전히 동일.

### 65~89번 줄: save_to_vector_store (= step5)

```python
def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 기존 컬렉션 삭제 (덮어씌우기)
    try:
        existing_store = Chroma(...)
        existing_store._collection.delete(where={})
    except:
        pass

    # 새로운 벡터 저장소 생성
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="documents"
    )
    return vector_store
```
- 청크 → 임베딩 → ChromaDB 저장. step5와 동일.

---

## 2단계: RAG 기능 함수들 (92~188번 줄)

### 95~126번 줄: initialize_rag_system (step6 확장)

```python
def initialize_rag_system(
    model_name: str = "gpt-5-nano",
    embedding_model: str = "text-embedding-3-small"
) -> tuple:
```
- step6의 `create_rag_chain`을 한 단계 감싼 **초기화 함수**.
- step6에서는 함수 안에서 임베딩/LLM/벡터DB를 직접 만들었는데, 여기서는 **외부에서 모델명을 바꿀 수 있게** 매개변수로 뺌.

```python
    embeddings = OpenAIEmbeddings(model=embedding_model)
    llm = ChatOpenAI(model=model_name, temperature=1)
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="documents"
    )
    rag_chain = create_rag_chain(llm, vector_store)
    return embeddings, llm, vector_store, rag_chain
```
- 4개를 한꺼번에 만들어서 튜플로 반환.
- **왜 4개 다 반환하나?** `vector_store`는 나중에 청크 미리보기에서 직접 접근할 때 쓰고, `rag_chain`은 질문 답변에 씀.

### 129~175번 줄: create_rag_chain (= step6)

```python
def create_rag_chain(llm, vector_store):
```
- step6과 거의 동일하지만, `llm`과 `vector_store`를 **매개변수로 받음** (step6에서는 함수 안에서 직접 생성).
- 프롬프트에 "사용자가 다양한 형태로 질문할 수 있습니다" 안내가 추가됨 → LLM이 "청약 1순위 알려줘", "청약 1순위가 뭐야?" 등 다양한 질문 형태를 잘 처리하도록.

### 178~188번 줄: process_question (app.py만의 기능)

```python
@st.cache_data
def process_question(user_question: str, _rag_chain):
    response = _rag_chain.invoke({"input": user_question})
    answer = response['answer']
    context_docs = response.get('context', [])
    return answer, context_docs
```

여기서 새로운 것:

**`@st.cache_data`** = Streamlit의 **캐싱 데코레이터**.
- 같은 질문이 들어오면 RAG 체인을 다시 실행하지 않고 **이전 결과를 재사용**합니다.
- 왜 필요한가? Streamlit은 위젯 값이 바뀔 때마다 전체 스크립트를 다시 실행하는데, 그때마다 API를 호출하면 느리고 비용이 듭니다.

**`_rag_chain`** (앞에 밑줄 `_`)
- Streamlit cache_data는 매개변수를 해시해서 캐시 키로 씁니다.
- 그런데 `rag_chain` 객체는 해시할 수 없어서 에러가 남.
- **앞에 `_`를 붙이면** Streamlit이 이 매개변수를 해시에서 제외합니다.

---

## 3단계: PDF 이미지 뷰어 함수들 (191~223번 줄)

step 파일들에는 없는, **app.py만의 기능**입니다.

### 193~214번 줄: convert_pdf_to_images

```python
@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)
    image_paths = []
```
- `fitz.open()` = PyMuPDF로 PDF 파일을 엶.
- `@st.cache_data(show_spinner=False)` = 캐싱하되 로딩 스피너는 안 보여줌 (외부에서 별도로 spinner를 표시하니까).

```python
    output_folder = "PDF_이미지"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
```
- 이미지 저장할 폴더를 만듦.

```python
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
```
- 각 페이지를 PNG 이미지로 변환하는 과정:
  1. `doc.load_page(page_num)` → 페이지 하나를 로드
  2. `zoom = dpi / 72` → PDF 기본 해상도는 72dpi. 250dpi로 만들려면 250/72 ≈ 3.47배 확대
  3. `fitz.Matrix(zoom, zoom)` → 가로/세로 확대 행렬
  4. `page.get_pixmap(matrix=mat)` → 확대된 해상도로 이미지 렌더링
  5. `pix.save(...)` → PNG 파일로 저장

### 217~219번 줄: display_pdf_page

```python
def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)
```
- PNG 파일을 읽어서 Streamlit 화면에 이미지로 표시. `width=600`으로 크기 고정.

### 222~223번 줄: natural_sort_key

```python
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]
```
- 파일명을 자연 정렬하는 함수. 왜 필요한가?
  - 일반 정렬: `page_1.png`, `page_10.png`, `page_2.png` (문자열 순서)
  - 자연 정렬: `page_1.png`, `page_2.png`, `page_10.png` (사람이 기대하는 순서)
- `re.split(r'(\d+)', "page_10.png")` → `['page_', '10', '.png']`
- 숫자 부분은 `int()`로 변환해서 숫자 크기로 비교.

---

## main() — Streamlit UI (226~462번 줄)

### 226~237번 줄: 초기 설정

```python
def main():
    st.set_page_config("FAQ", layout="wide")
```
- 브라우저 탭 제목 "FAQ", 넓은 레이아웃.

```python
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'doc_info' not in st.session_state:
        st.session_state.doc_info = None
```
- step7과 비교해서 `pdf_processed`가 추가됨. PDF가 처리됐는지 여부를 추적해서 채팅 입력을 활성화/비활성화합니다.

---

### 사이드바 (239~344번 줄)

```
┌─ 사이드바 ─────────────────┐
│ 설정                        │
│ ─────────────               │
│ 📄 PDF 문서                 │
│ [파일 업로드]               │
│ [PDF 문서 저장] 버튼        │
│ ─────────────               │
│ 📖 로드된 문서              │
│ - 파일명 / 크기             │
│ - 페이지 수 / 청크 수       │
│ [모든 문서 삭제] 버튼       │
│ ─────────────               │
│ 모델 정보                   │
│ - LLM: gpt-5-nano           │
│ - Embedding: text-embedding  │
│ ─────────────               │
│ 사용 방법                   │
│ 1. PDF 업로드               │
│ 2. 저장 클릭                │
│ 3. 질문 입력                │
└─────────────────────────────┘
```

#### PDF 업로드 + 저장 (249~272번 줄)

```python
if pdf_doc and upload_button:
    with st.spinner("PDF 문서를 저장하고 있습니다..."):
        pdf_path = save_uploadedfile(pdf_doc)           # step2
        pdf_documents = pdf_to_documents(pdf_path)       # step2
        smaller_documents = chunk_documents(pdf_documents) # step3
        save_to_vector_store(smaller_documents)           # step5
```
- step2→3→5를 순서대로 실행. 한 spinner 안에서 전부 처리.

```python
        st.session_state.doc_info = {
            'filename': pdf_doc.name,
            'size': f"{pdf_doc.size / 1024 / 1024:.2f} MB",
            'pages': len(pdf_documents),
            'chunks': len(smaller_documents)
        }
```
- 문서 정보를 딕셔너리로 저장. `pdf_doc.size / 1024 / 1024` = 바이트 → MB 변환. `:.2f` = 소수점 2자리.

```python
        _, _, st.session_state.vector_store, st.session_state.rag_chain = initialize_rag_system()
```
- RAG 시스템 초기화. `_`는 "이 값은 안 쓸 거야"라는 파이썬 관례. embeddings와 llm은 버리고 vector_store와 rag_chain만 저장.

```python
    with st.spinner("PDF 페이지를 이미지로 변환하는 중입니다..."):
        images = convert_pdf_to_images(pdf_path)
        st.session_state.images = images
```
- PDF를 이미지로 변환 (app.py만의 기능). 오른쪽 PDF 뷰어에서 사용.

#### 기존 벡터DB 자동 로드 (274~301번 줄)

```python
elif st.session_state.rag_chain is None and os.path.exists(CHROMA_DIR):
    embeddings, llm, vector_store, st.session_state.rag_chain = initialize_rag_system()
```
- PDF를 새로 업로드하지 않았더라도, 이전에 저장한 벡터DB가 디스크에 있으면 자동으로 불러옴.
- **왜 필요?** 브라우저를 껐다 다시 켰을 때, 이전에 저장한 문서를 다시 업로드 안 해도 되게.

```python
    try:
        collection = vector_store._collection
        results = collection.get()
        if results and results['metadatas']:
            first_meta = results['metadatas'][0]
            filename = first_meta.get('filename', 'Unknown')
            max_page = max([m.get('page', 0) for m in results['metadatas']]) + 1
            ...
    except Exception as e:
        ...
```
- 벡터DB에서 metadata를 꺼내서 파일명, 페이지 수를 복원. 파일 크기는 알 수 없어서 `'N/A'`.
- `_collection.get()` = ChromaDB의 내부 API. 저장된 모든 문서의 metadata를 가져옴.

#### 문서 삭제 (312~334번 줄)

```python
if st.button("모든 문서 삭제", use_container_width=True):
    import shutil, gc
    if st.session_state.get('vector_store'):
        try:
            st.session_state.vector_store.delete_collection()
        except Exception:
            pass
    st.session_state.rag_chain = None
    st.session_state.vector_store = None
    gc.collect()
    try:
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
    except Exception:
        pass
```
- 삭제 과정:
  1. `delete_collection()` → ChromaDB 컬렉션을 메모리에서 삭제
  2. `rag_chain = None`, `vector_store = None` → 참조 해제
  3. `gc.collect()` → 파이썬 가비지 컬렉터 강제 실행 (메모리 해제)
  4. `shutil.rmtree(CHROMA_DIR)` → 디스크에서 폴더째 삭제
- `use_container_width=True` = 버튼이 사이드바 전체 너비를 차지.

```python
    for key in ['images', 'last_context_docs', 'page_number', 'last_question']:
        st.session_state.pop(key, None)
    st.rerun()
```
- 관련 세션 데이터를 전부 정리하고 페이지를 새로고침.
- `.pop(key, None)` = 키가 있으면 삭제, 없으면 무시 (에러 안 남).

---

### 메인 화면 — 채팅 탭 (346~409번 줄)

```
┌─ 채팅 탭 ──────────────────────────────────────────────┐
│                                                         │
│  ┌─ 왼쪽: 채팅 ────────┐  ┌─ 오른쪽: PDF 뷰어 ───────┐ │
│  │ 👤 질문: ...         │  │                           │ │
│  │ 🤖 답변: ...         │  │  [PDF 페이지 이미지]      │ │
│  │                      │  │                           │ │
│  │ 📚 관련 문서         │  │  [◀ 이전]    [다음 ▶]    │ │
│  │ 청크1: ...           │  │                           │ │
│  │ [🔍 PDF 3페이지 보기]│  │                           │ │
│  │                      │  │                           │ │
│  │ [질문 입력...]       │  │                           │ │
│  └──────────────────────┘  └───────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

#### 채팅 기록 표시 (357~359번 줄)

```python
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
```
- 저장된 대화 기록을 처음부터 다시 그림. Streamlit은 매번 다시 실행되니까 이렇게 해야 기록이 유지됨.

#### 질문 입력 + 답변 생성 (362~385번 줄)

```python
prompt = st.chat_input(
    "질문을 입력해주세요 ...",
    disabled=not st.session_state.pdf_processed
)
```
- `disabled=not st.session_state.pdf_processed` → PDF를 아직 안 올렸으면 입력창 비활성화.

```python
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_question = prompt
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            answer, context_docs = process_question(prompt, st.session_state.rag_chain)
        st.markdown(answer)

    st.session_state.last_context_docs = context_docs
    st.session_state.messages.append({"role": "assistant", "content": answer})
```
- step7과 비슷하지만, `process_question()`을 통해 **캐싱된 답변**을 받음.
- `last_context_docs`를 session_state에 저장 → 관련 문서 표시 + PDF 뷰어 연동에 사용.

#### 관련 문서 + PDF 페이지 연동 (387~409번 줄)

```python
for idx, document in enumerate(st.session_state.last_context_docs):
    page = document.metadata.get('page', 0) + 1
    st.text_area(f"청크 {idx + 1}", value=content, height=150,
                 key=f"chunk_{idx}_{st.session_state.get('last_question', '')}")

    if st.button(f"🔍 PDF {page}페이지 보기", key=f"btn_{page}_{idx}"):
        st.session_state.page_number = str(page)
        st.rerun()
```
- 각 관련 청크 아래에 **"PDF N페이지 보기" 버튼**이 붙음.
- 클릭하면 `page_number`를 session_state에 저장 → `st.rerun()`으로 페이지 새로고침 → 오른쪽에 해당 PDF 이미지가 표시됨.
- `key=f"btn_{page}_{idx}"` → 같은 페이지 번호 버튼이 여러 개일 수 있어서 `idx`를 추가해 고유하게 만듦.

---

### 메인 화면 — PDF 뷰어 (411~437번 줄)

```python
with right_column:
    if st.session_state.get("page_number"):
        page_number = int(st.session_state.page_number)
        image_folder = "PDF_이미지"

        if os.path.exists(image_folder):
            images = sorted(os.listdir(image_folder), key=natural_sort_key)
            image_paths = [os.path.join(image_folder, img) for img in images]

            if 0 < page_number <= len(image_paths):
                display_pdf_page(image_paths[page_number - 1], page_number)
```
- "PDF_이미지" 폴더에서 해당 페이지 이미지를 찾아서 표시.
- `natural_sort_key`로 정렬해서 `page_1`, `page_2`, ..., `page_10` 순서가 올바르게.

```python
                nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
                with nav_col1:
                    if page_number > 1 and st.button("◀ 이전"):
                        st.session_state.page_number = str(page_number - 1)
                        st.rerun()
                with nav_col3:
                    if page_number < len(image_paths) and st.button("다음 ▶"):
                        st.session_state.page_number = str(page_number + 1)
                        st.rerun()
```
- 이전/다음 페이지 네비게이션 버튼.
- 첫 페이지면 "이전" 안 보이고, 마지막 페이지면 "다음" 안 보임.

---

### 메인 화면 — 청크 미리보기 탭 (439~458번 줄)

```python
with tab_chunks:
    st.subheader("저장된 청크 미리보기")
    if st.session_state.get("vector_store"):
        collection = st.session_state.vector_store._collection
        result = collection.get(include=["documents", "metadatas"])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])

        for i, (doc, meta) in enumerate(zip(docs, metas)):
            page = meta.get("page", "?")
            page_display = int(page) + 1 if isinstance(page, (int, float)) else page
            with st.expander(f"청크 #{i+1} — 페이지 {page_display}"):
                st.text(doc)
```
- ChromaDB에 저장된 **모든 청크를 직접 확인**할 수 있는 탭.
- `collection.get(include=["documents", "metadatas"])` → 텍스트 + metadata를 가져옴.
- `zip(docs, metas)` → 텍스트와 metadata를 쌍으로 묶어서 반복.
- `isinstance(page, (int, float))` → 페이지 번호가 숫자인지 확인 (혹시 문자열이면 그대로 표시).

---

## 전체 실행 흐름 요약

```
[사용자가 PDF 업로드 + "저장" 클릭]
  → save_uploadedfile()     → 디스크에 임시 저장
  → pdf_to_documents()      → Document 리스트 (페이지당 1개)
  → chunk_documents()       → 작은 청크들 (1000자 단위)
  → save_to_vector_store()  → ChromaDB에 벡터 저장
  → initialize_rag_system() → RAG 체인 생성
  → convert_pdf_to_images() → 페이지별 PNG 이미지 생성

[사용자가 질문 입력]
  → process_question()
    → rag_chain.invoke({"input": 질문})
      ├─ retriever: ChromaDB에서 유사 청크 3개 검색
      ├─ stuff chain: 청크를 {context}에 삽입
      └─ ChatOpenAI: 문서를 읽고 답변 생성
    → answer + context_docs 반환
  → 왼쪽: 답변 표시 + 관련 청크 + "PDF 페이지 보기" 버튼
  → 오른쪽: 해당 페이지 이미지 표시
```
