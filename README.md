# Parser + RAG

## Document Parsing

### 필수 작업

1. LibreOffice 설치
    - Docx 안의 DrawingML을 렌더링하기 위한 필수 작업입니다.
```bash
sudo apt-get update
sudo apt-get install -y libreoffice

export DOCLING_LIBREOFFICE_CMD=/usr/bin/libreoffice
```