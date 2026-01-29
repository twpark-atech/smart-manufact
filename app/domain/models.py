# ==============================================================================
# 목적 : 프로젝트에서 사용하는 모델 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-23
# AI 활용 여부 :
# ==============================================================================

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass(frozen=True)
class ParseArtifactsResult:
    """PDF 파싱/임베딩/인덱싱 파이프라인의 최종 산출물 DTO 클래스.
    ingest_pdf() 실행 후 생성되는 주요 식별자, 산출물 경로, 사용된 인덱스명, 처리량/성공/실패 카운트를 묶어 반환합니다.
    
    Attributes:
        doc_id: 문서 식별자.
        doc_sha256: 원본 PDF의 sha256 해시.
        doc_title: 문서 제목.
        source_uri: 입력 소스 경로.
        output_dir: 결과 루트 디렉토리.
        assets_root: 문서별 assets 루트.
        pages_dir: 페이지 PNG 저장 디렉토리.
        md_path: 페이지 OCR 결과를 합친 Markdown 캐시 파일 경로.
        pdf_uri: 업로드된 PDF의 s3://.. URI.
        text_index: 텍스트 청크 인덱스명.
        image_index: 이미지 인덱스명.
        pages_staging_index: OCR 페이지 staging 인덱스명.
        images_staging_index: 이미지 staging 인덱스명.
        table_index: 테이블 인덱스명.
        tables_staging_index: 테이블 staging 인덱스명.
        page_count: 전체 페이지 수.
        extracted_image_count: 이미지 crop 추출 개수.
        generated_desc_count: 이미지 description 생성 개수.
        staged_page_count: pages_staging에 status=done으로 적재된 페이지 수.
        failed_page_count: pages_staging에 status=failed으로 적재된 페이지 수.
        staged_image_count: images_staging에 적재된 이미지 수.
        indexed_image_count: image_index에 최종 적재된 이미지 수.
        staged_table_count: tables_staging에 적재된 테이블 수.
        indexed_table_docs_count: table_index에 적재된 table(doc_type=table) 문서 수.
        indexed_table_rows_count: table_index에 적재된 row(doc_type=row) 문서 수.
        chunk_count: 생성된 텍스트 청크 수.
        indexed_chunk_count: text_index에 적재된 텍스트 청크 수.
        mode: 실행 모드("fresh" | "from_pages_staging").
    """
    doc_id: str
    doc_sha256: str
    doc_title: str
    source_uri: str

    output_dir: str
    assets_root: str
    pages_dir: str
    md_path: str

    pdf_uri: str
    text_index: str
    image_index: str
    pages_staging_index: str
    images_staging_index: str

    table_index: str
    tables_staging_index: str

    page_count: int
    extracted_image_count: int
    generated_desc_count: int

    staged_page_count: int
    failed_page_count: int
    staged_image_count: int
    indexed_image_count: int

    staged_table_count: int
    indexed_table_docs_count: int
    indexed_table_rows_count: int

    chunk_count: int
    indexed_chunk_count: int

    mode: str

    def to_dict(self) -> Dict[str, Any]:
        """dataclass를 dict로 변환합니다.
        
        Returns:
            dataclasses.asdict(self) 결과 dict.
        """
        return asdict(self)

@dataclass(frozen=True)
class ParseDocxResult:
    """Docx 파싱/임베딩/인덱싱 파이프라인의 최종 산출물 DTO 클래스.
    ingest_docx() 실행 후 생성되는 주요 식별자, 산출물 경로, 사용된 인덱스명, 처리량/성공/실패 카운트를 묶어 반환합니다.

    Attributes:
        doc_id: 문서 식별자.
        doc_sha256: 원본 PDF의 sha256 해시.
        doc_title: 문서 제목.
        source_uri: 입력 소스 경로.
        output_dir: 결과 루트 디렉토리.
        md_path: 페이지 OCR 결과를 합친 Markdown 캐시 파일 경로.
        chunk_count: 생성된 텍스트 청크 수.
        indexed_chunk_count: text_index에 적재된 텍스트 청크 수.
    """
    doc_id: str
    doc_sha256: str
    doc_title: str
    source_uri: str
    output_dir: str
    md_path: str
    chunk_count: int
    indexed_chunk_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
