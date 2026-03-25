"""
Step 1: 문서 로딩 — PDF를 Document 객체로 변환

학습 목표:
- PDF 파일을 LangChain Document 객체로 변환하는 방법
- Document 객체의 구조: page_content + metadata
- PyMuPDFLoader의 사용법
"""

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_DIR = Path(__file__).resolve().parent


def save_uploadedfile(uploadedfile: UploadedFile) -> str:
    """업로드된 PDF 파일을 임시 폴더에 저장"""
    temp_dir = str(BASE_DIR / "PDF_임시폴더")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path


def load_pdf(pdf_path: str):
    """PDF 파일을 LangChain Document 리스트로 변환"""
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # metadata에 파일 정보 추가
    for doc in documents:
        doc.metadata["file_path"] = pdf_path
        doc.metadata["filename"] = os.path.basename(pdf_path)

    return documents


############################### Streamlit UI ##########################

def main():
    st.set_page_config("03-single-pdf-rag | Step 1: 문서 로딩", layout="wide")
    st.header("Step 1: PDF → Document 변환")

    st.markdown("""
    ### 이 단계에서 배우는 것:
    1. **PDF 로딩**: `PyMuPDFLoader`로 PDF를 읽기
    2. **Document 객체**: LangChain의 기본 문서 단위
       - `page_content`: 텍스트 내용
       - `metadata`: 페이지 번호, 파일명 등 부가 정보

    **핵심 코드**:
    ```python
    loader = PyMuPDFLoader("문서.pdf")
    documents = loader.load()  # → List[Document]
    ```
    """)

    st.divider()

    pdf_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    if pdf_file:
        # 1. 파일 저장
        with st.spinner("PDF 파일을 저장하는 중..."):
            pdf_path = save_uploadedfile(pdf_file)
        st.success(f"파일 저장 완료: {pdf_file.name}")

        # 2. Document로 변환
        with st.spinner("PDF를 Document로 변환하는 중..."):
            documents = load_pdf(pdf_path)

        st.success(f"총 **{len(documents)}**개 페이지를 Document로 변환했습니다.")

        # 3. Document 내용 확인
        st.subheader("Document 내용 확인")

        for i, doc in enumerate(documents):
            with st.expander(f"페이지 {i + 1} (길이: {len(doc.page_content)}자)"):
                # metadata 표시
                st.json(doc.metadata)
                # 내용 미리보기 (처음 500자)
                st.text_area(
                    "page_content (처음 500자)",
                    value=doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                    height=200,
                    key=f"doc_{i}"
                )

        st.divider()
        st.info("""
        ### 다음 단계 (step2)에서는:
        - 긴 Document를 **작은 청크로 분할**하는 방법을 배웁니다
        - 왜 분할이 필요한지, chunk_size와 overlap의 의미를 이해합니다
        """)


if __name__ == "__main__":
    main()
