"""
벡터 DB 초기화 스크립트
"""
import sys
sys.path.append('.')

from app import pdf_to_documents, chunk_documents, save_to_vector_store, convert_pdf_to_images

# 샘플 PDF로 벡터 DB 생성
pdf_path = "★ 2024 주택청약 FAQ.pdf"
print(f"PDF 로드 중: {pdf_path}")

pdf_documents = pdf_to_documents(pdf_path)
print(f"페이지 수: {len(pdf_documents)}")

smaller_documents = chunk_documents(pdf_documents)
print(f"청크 수: {len(smaller_documents)}")

save_to_vector_store(smaller_documents)
print("벡터 DB 생성 완료!")

# PDF 이미지 변환
print("PDF 페이지 이미지 변환 중...")
images = convert_pdf_to_images(pdf_path)
print(f"이미지 변환 완료: {len(images)}페이지")
