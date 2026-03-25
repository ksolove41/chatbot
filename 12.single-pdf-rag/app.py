"""
RAG FAQ 챗봇 - Streamlit 버전 (OpenAI)

PDF 문서를 업로드하고 질문에 답변하는 RAG 챗봇
- PDF → 벡터DB 저장
- 질문 → 관련 문서 검색 → 답변 생성
- OpenAI API 사용
"""

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List, Dict, Any
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")

############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

## 1: 임시폴더에 파일 저장
def save_uploadedfile(uploadedfile: UploadedFile) -> str:
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

## 2: 저장된 PDF 파일을 Document로 변환
def pdf_to_documents(pdf_path: str) -> List[Document]:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    # metadata에 file_path 추가 (나중에 참조용)
    for doc in documents:
        doc.metadata['file_path'] = pdf_path
        doc.metadata['filename'] = os.path.basename(pdf_path)
    return documents

## 3: Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

## 4: Document를 벡터DB로 저장
def save_to_vector_store(documents: List[Document]) -> None:
    # OpenAI 임베딩 모델 사용
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 기존 컬렉션 삭제 (덮어씌우기)
    try:
        existing_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="documents"
        )
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


############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################

## RAG 시스템 초기화
def initialize_rag_system(
    model_name: str = "gpt-5-nano",
    embedding_model: str = "text-embedding-3-small"
) -> tuple:
    """
    RAG 시스템 초기화

    Returns:
        (embeddings, llm, vector_store, rag_chain)
    """
    # OpenAI 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(
        model=embedding_model
    )

    # OpenAI LLM 초기화
    llm = ChatOpenAI(
        model=model_name,
        temperature=1
    )

    # 벡터 저장소 로드
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="documents"
    )

    # RAG 체인 생성
    rag_chain = create_rag_chain(llm, vector_store)

    return embeddings, llm, vector_store, rag_chain


def create_rag_chain(llm, vector_store):
    """RAG 체인 생성 (최신 LangChain 문법)"""
    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 문서 기반 질문응답 AI 어시스턴트입니다.

중요 규칙:
- 제공된 컨텍스트에서 답을 찾아주세요
- 컨텍스트에 없는 내용은 "문서에 해당 정보가 없습니다"라고 답변하세요
- 한국어로 답변하세요
- 간결하고 명확하게 답변하세요

사용자가 다양한 형태로 질문할 수 있습니다:
- "핵심 내용 알려줘"
- "이 문서가 뭐에 대한 건지 알려줘"
- "주요 내용이 뭐야?"
- "요약해줘"

모든 질문 형태에서 핵심 질문을 파악하고 컨텍스트에서 답을 찾아주세요."""),
        ("human", """다음 문서를 참고하여 질문에 답변해주세요:

[문서 내용]
{context}

---

질문: {input}""")
    ])

    # 문서 체인
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # 검색 체인
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # RAG 체인
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    return rag_chain


## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question: str, _rag_chain):
    """사용자 질문 처리"""
    # RAG 체인 실행
    response = _rag_chain.invoke({"input": user_question})

    answer = response['answer']
    context_docs = response.get('context', [])

    return answer, context_docs


def main():
    st.set_page_config("03-single-pdf-rag", layout="wide")

    # 세션 상태 초기화
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'doc_info' not in st.session_state:
        st.session_state.doc_info = None

    # 사이드바
    with st.sidebar:
        st.header("설정")
        st.markdown("---")

        # PDF 업로드
        st.subheader("📄 PDF 문서")
        pdf_doc = st.file_uploader("PDF 파일을 업로드 해주세요", type=["pdf"])
        upload_button = st.button("PDF 문서 저장")

        if pdf_doc and upload_button:
            with st.spinner("PDF 문서를 저장하고 있습니다..."):
                # PDF 저장 및 벡터DB 생성
                pdf_path = save_uploadedfile(pdf_doc)
                pdf_documents = pdf_to_documents(pdf_path)
                smaller_documents = chunk_documents(pdf_documents)
                save_to_vector_store(smaller_documents)

                # 문서 정보 저장
                st.session_state.doc_info = {
                    'filename': pdf_doc.name,
                    'size': f"{pdf_doc.size / 1024 / 1024:.2f} MB",
                    'pages': len(pdf_documents),
                    'chunks': len(smaller_documents)
                }

                # RAG 시스템 초기화
                _, _, st.session_state.vector_store, st.session_state.rag_chain = initialize_rag_system()
                st.session_state.pdf_processed = True
                st.success("PDF 문서가 저장되었습니다!")

        # RAG 체인이 초기화되지 않은 경우 기존 벡터 저장소 로드 시도
        elif st.session_state.rag_chain is None and os.path.exists(CHROMA_DIR):
            embeddings, llm, vector_store, st.session_state.rag_chain = initialize_rag_system()
            st.session_state.pdf_processed = True
            st.session_state.vector_store = vector_store

            # 벡터 DB에서 문서 정보 추출
            try:
                collection = vector_store._collection
                results = collection.get()
                if results and results['metadatas']:
                    # 첫 번째 메타데이터에서 파일명 추출
                    first_meta = results['metadatas'][0]
                    filename = first_meta.get('filename', 'Unknown')

                    # 최대 페이지 번호 찾기 (실제 페이지 수)
                    max_page = max([m.get('page', 0) for m in results['metadatas']]) + 1

                    st.session_state.doc_info = {
                        'filename': filename,
                        'size': 'N/A',
                        'pages': max_page,
                        'chunks': len(results['ids'])
                    }
            except Exception as e:
                st.write(f"디버그: {str(e)}")
                st.session_state.doc_info = None
            st.info("기존 문서를 로드했습니다.")

        # 로드된 문서 정보 표시
        if st.session_state.doc_info:
            st.markdown("---")
            st.markdown("### 📖 로드된 문서")
            st.markdown(f"**파일명:** {st.session_state.doc_info['filename']}")
            st.markdown(f"**크기:** {st.session_state.doc_info['size']}")
            st.markdown(f"**페이지:** {st.session_state.doc_info['pages']}페이지")
            st.markdown(f"**청크:** {st.session_state.doc_info['chunks']}개")

            if st.button("모든 문서 삭제", use_container_width=True):
                import shutil, gc
                # ChromaDB 컬렉션 삭제 및 참조 해제
                if st.session_state.get('vector_store'):
                    try:
                        st.session_state.vector_store.delete_collection()
                    except Exception:
                        pass
                st.session_state.rag_chain = None
                st.session_state.vector_store = None
                gc.collect()
                # 디스크에서 삭제
                try:
                    if os.path.exists(CHROMA_DIR):
                        shutil.rmtree(CHROMA_DIR)
                except Exception:
                    pass
                st.session_state.pdf_processed = False
                st.session_state.doc_info = None
                st.session_state.messages = []
                for key in ['last_context_docs', 'page_number', 'last_question']:
                    st.session_state.pop(key, None)
                st.rerun()

        st.markdown("---")
        st.markdown("### 모델 정보")
        st.markdown("- **LLM:** gpt-5-nano")
        st.markdown("- **Embedding:** text-embedding-3-small")
        st.markdown("---")
        st.markdown("### 사용 방법")
        st.markdown("1. PDF 파일 업로드")
        st.markdown("2. 'PDF 문서 저장' 클릭")
        st.markdown("3. 질문 입력")

    # 메인 화면
    st.header("Single PDF RAG")

    tab_chat, tab_chunks = st.tabs(["💬 채팅", "🔍 청크 미리보기"])

    with tab_chat:
        # 채팅 기록 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 질문 입력
        prompt = st.chat_input(
            "질문을 입력해주세요 (예: 문서에서 핵심 내용을 알려줘)",
            disabled=not st.session_state.pdf_processed
        )

        # 새 질문이 입력된 경우
        if prompt:
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.last_question = prompt
            with st.chat_message("user"):
                st.markdown(prompt)

            # 어시스턴트 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    answer, context_docs = process_question(prompt, st.session_state.rag_chain)
                st.markdown(answer)

            # 관련 문서를 세션 상태에 저장
            st.session_state.last_context_docs = context_docs

            # 어시스턴트 메시지 추가
            st.session_state.messages.append({"role": "assistant", "content": answer})

        # 관련 문서 표시
        if st.session_state.get("last_context_docs"):
            if not prompt:
                st.markdown("---")
            st.markdown("### 참조 문서")
            for idx, document in enumerate(st.session_state.last_context_docs):
                page = document.metadata.get('page', 0) + 1
                with st.expander(f"청크 {idx + 1} — 페이지 {page}"):
                    st.text(document.page_content)

    with tab_chunks:
        st.subheader("저장된 청크 미리보기")
        if st.session_state.get("vector_store"):
            try:
                collection = st.session_state.vector_store._collection
                result = collection.get(include=["documents", "metadatas"])
                docs = result.get("documents", [])
                metas = result.get("metadatas", [])

                st.info(f"총 **{len(docs)}**개 청크가 저장되어 있습니다.")

                for i, (doc, meta) in enumerate(zip(docs, metas)):
                    page = meta.get("page", "?")
                    page_display = int(page) + 1 if isinstance(page, (int, float)) else page
                    with st.expander(f"청크 #{i+1} — 페이지 {page_display}"):
                        st.text(doc)
            except Exception as e:
                st.error(f"청크 조회 실패: {e}")
        else:
            st.warning("벡터 저장소가 비어 있습니다. PDF를 먼저 업로드해주세요.")


if __name__ == "__main__":
    main()
