"""
Step 2: 텍스트 분할 — RecursiveCharacterTextSplitter

학습 목표:
- 왜 문서를 분할(chunking)해야 하는지 이해
- chunk_size와 chunk_overlap의 의미
- RecursiveCharacterTextSplitter 사용법
- 분할 전/후 비교
"""

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from dotenv import load_dotenv

from step1_load_pdf import save_uploadedfile, load_pdf

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Document를 작은 청크로 분할"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


############################### Streamlit UI ##########################

def main():
    st.set_page_config("03-single-pdf-rag | Step 2: 텍스트 분할", layout="wide")
    st.header("Step 2: 텍스트 분할 (Chunking)")

    st.markdown("""
    ### 이 단계에서 배우는 것:
    1. **왜 분할?**: LLM에는 입력 길이 제한이 있고, 작은 단위가 검색 정확도가 높음
    2. **chunk_size**: 한 청크의 최대 글자 수
    3. **chunk_overlap**: 청크 간 겹치는 글자 수 (문맥 유지)

    **비유**: 교과서 전체 → 핵심 노트 카드 여러 장으로 나누기
    ```
    [    청크 1    ]
              [    청크 2    ]     ← overlap 부분이 겹침
                        [    청크 3    ]
    ```
    """)

    st.divider()

    pdf_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    if pdf_file:
        # PDF 로드 (step1 복습)
        pdf_path = save_uploadedfile(pdf_file)
        documents = load_pdf(pdf_path)

        st.success(f"PDF 로드 완료: {len(documents)}개 페이지")

        # 청킹 파라미터 조절
        st.subheader("청킹 파라미터 조절")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("chunk_size (청크 크기)", 200, 2000, 1000, step=100)
        with col2:
            chunk_overlap = st.slider("chunk_overlap (겹침)", 0, 500, 200, step=50)

        # 청킹 실행
        chunks = chunk_documents(documents, chunk_size, chunk_overlap)

        # 결과 비교
        st.subheader("분할 결과")

        col_before, col_after = st.columns(2)
        with col_before:
            st.metric("분할 전 (페이지)", len(documents))
            total_chars = sum(len(d.page_content) for d in documents)
            st.metric("총 글자 수", f"{total_chars:,}")

        with col_after:
            st.metric("분할 후 (청크)", len(chunks))
            avg_chars = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
            st.metric("평균 청크 크기", f"{avg_chars:,}자")

        # 청크 내용 확인
        st.subheader("청크 내용 확인")
        for i, chunk in enumerate(chunks):
            page = chunk.metadata.get("page", 0) + 1
            with st.expander(f"청크 #{i + 1} (페이지 {page}, {len(chunk.page_content)}자)"):
                st.text_area(
                    "내용",
                    value=chunk.page_content,
                    height=150,
                    key=f"chunk_{i}"
                )

        st.divider()
        st.info("""
        ### 다음 단계 (step3)에서는:
        - 텍스트를 **숫자 벡터(임베딩)**로 변환하는 원리를 체험합니다
        - "의미를 숫자로 표현한다"는 것이 무엇인지 이해합니다
        """)


if __name__ == "__main__":
    main()
