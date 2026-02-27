# [Step3 - 서비스 개발]

## 1) 구현 범위

- LangGraph 기반 Multi-Agent 워크플로우 구현
- RAG 파이프라인(문서 로딩, 청킹, FAISS + BM25 하이브리드 검색) 구현
- Structured Output(Pydantic) 기반 최종 응답 생성
- Streamlit UI + FastAPI API + CLI 실행 경로 제공
- `.env` 기반 Azure OpenAI 설정 로딩 및 모듈 분리
- API 예외 처리(400/500) 기반 안정적 오류 응답 구성

## 2) 핵심 파일

- `src/workflow/engine.py`: LangGraph 오케스트레이션, 서비스 진입점
- `src/retrieval/documents.py`: 데이터 로딩/청킹
- `src/retrieval/hybrid.py`: FAISS + BM25 검색
- `src/agents/tools.py`: Tool Calling 도구 함수
- `src/agents/schemas.py`: Structured Output 스키마
- `src/api/app.py`: FastAPI 엔드포인트
- `src/ui/streamlit_app.py`: Streamlit UI
- `main.py`: CLI 실행 엔트리

## 3) 실행 방법

### 의존성 설치

```powershell
pip install -r requirements-final.txt
```

### 환경변수 설정

`final-project/.env` 파일에 아래 값 설정:

- `AOAI_ENDPOINT`
- `AOAI_API_KEY`
- `AOAI_DEPLOY_GPT4O`
- `AOAI_DEPLOY_EMBED_ADA` (또는 `AOAI_EMBEDDING_DEPLOYMENT`)
- `AOAI_API_VERSION`

### CLI 실행

```powershell
python main.py --query "백엔드 이직 준비를 위한 2주 계획을 작성해줘" --target-role "백엔드 개발자"
```

### FastAPI 실행

```powershell
python scripts/run_api.py
```

### Streamlit 실행

```powershell
python scripts/run_streamlit.py
```

## 4) 기본 검증 결과

- `pytest` 실행 결과: `3 passed`
- 검증 대상:
  - 세션 메모리 동작
  - Resume Tool/Interview Tool 기본 응답

## 5) 필수 기술 요소 매핑 표

| 필수 항목 | 반영 내용 | 구현 위치 |
|---|---|---|
| 1) Prompt Engineering | 역할 기반 프롬프트(Supervisor/Resume/Interview), Few-shot 예시 반영, 구조화 출력 지시 | `src/workflow/engine.py`, `src/agents/schemas.py` |
| 2) LangChain/LangGraph Multi-Agent | Multi-Agent 그래프 구성, Tool Calling, 세션 메모리 활용 | `src/workflow/engine.py`, `src/agents/tools.py`, `src/utils/memory.py` |
| 3) RAG | 문서 로딩/전처리/청킹, 임베딩, FAISS + BM25 하이브리드 검색, 근거 출처 반환 (`.txt/.md/.csv/.pdf/.docx/.xlsx`) | `src/retrieval/documents.py`, `src/retrieval/hybrid.py`, `data/knowledge/*` |
| 4) 서비스 개발/패키징 | Streamlit UI, FastAPI 백엔드, CLI 엔트리, 실행 스크립트, `.env` 설정 관리, API 예외 처리 | `src/ui/streamlit_app.py`, `src/api/app.py`, `main.py`, `scripts/*`, `src/config/*` |

## 6) 최근 현행화 반영 사항

- RAG 문서 로더 지원 포맷 확대: `.pdf`, `.docx`, `.xlsx` 추가
- FastAPI `/chat` 예외 처리 강화: `ValueError -> 400`, 기타 예외 -> `500`
- 실행/운영 문서 업데이트: README Troubleshooting 섹션 추가
