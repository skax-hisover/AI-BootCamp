# [Step3 - 서비스 개발]

## 1) 구현 범위

- LangGraph 기반 Multi-Agent 워크플로우 구현
- RAG 파이프라인(문서 로딩, 청킹, FAISS + BM25 하이브리드 검색) 구현
- Structured Output(Pydantic) 기반 최종 응답 생성
- Streamlit UI + FastAPI API + CLI 실행 경로 제공
- `.env` 기반 Azure OpenAI 설정 로딩 및 모듈 분리
- API 예외 처리(400/500) 기반 안정적 오류 응답 구성
- Streamlit 실행 입력 기록 관리(조회/삭제/다시 불러오기) 및 세션 자동 관리

## 2) 핵심 파일

- `src/workflow/engine.py`: LangGraph 오케스트레이션, 서비스 진입점
- `src/retrieval/documents.py`: 데이터 로딩/청킹
- `src/retrieval/hybrid.py`: FAISS + BM25 검색
- `src/agents/tools.py`: Tool Calling 도구 함수
- `src/agents/schemas.py`: Structured Output 스키마
- `src/api/app.py`: FastAPI 엔드포인트
- `src/ui/streamlit_app.py`: Streamlit UI
- `main.py`: CLI 실행 엔트리

## 2-1) 구조도/플로우 산출물

- 아키텍처 다이어그램: `docs/images/system_architecture.png`
- 서비스 플로우 다이어그램: `docs/images/service_flow_sequence.png`
- 포함 내용:
  - LangGraph 주요 노드/엣지(`supervisor -> rag -> resume/interview -> synthesis`)
  - RAG 컨텍스트/참고 출처(`rag_context`, `rag_refs`)가 Supervisor/Synthesis에 주입되는 데이터 흐름

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
| 4) 서비스 개발/패키징 | Streamlit UI(이력서 파일 업로드 + 실행 입력 기록 조회/삭제 + 다시 불러오기), FastAPI 백엔드, CLI 엔트리, 실행 스크립트, `.env` 설정 관리, API 예외 처리 | `src/ui/streamlit_app.py`, `src/api/app.py`, `main.py`, `scripts/*`, `src/config/*` |

## 6) 최근 현행화 반영 사항

- RAG 문서 로더 지원 포맷 확대: `.pdf`, `.docx`, `.xlsx` 추가
- FastAPI `/chat` 예외 처리 강화: `ValueError -> 400`, 기타 예외 -> `500`
- 실행/운영 문서 업데이트: README Troubleshooting 섹션 추가
- Streamlit UI 업데이트: 세션 ID 자동 관리, 새 대화 시작, 실행 입력 기록의 다시 불러오기(질문/직무/이력서/결과 복원)
- Supervisor 라우팅 고도화: `resume_only/interview_only/full/plan_only` 분류 + LangGraph 조건부 엣지 분기 적용
- Tool loop 고도화: 단일 호출이 아닌 최대 N회(기본 4회) 도구 재호출 루프 및 종료 토큰(`DONE`) 처리
- 세션 메모리 영속화: 메모리 JSON 파일 저장/로드로 서버 재시작 후 대화 이력 유지
- RAG Agent 강화: 쿼리 리라이트, 직무 힌트 기반 문서 우선순위(스코어 부스팅), route 메타 노트 반영