# JobPilot AI - Final Project

JobPilot AI는 취업/이직 준비를 위한 End-to-End 멀티 에이전트 서비스입니다.

## 구현 완료 항목

- **프롬프트 엔지니어링**: Supervisor/Resume/Interview 에이전트 역할 기반 프롬프트
- **멀티 에이전트(LangGraph)**: Supervisor -> RAG -> (Resume/Interview/Plan) -> Synthesis 라우트 기반 분기 실행
- **요구사항-인터페이스 정합성**: `ChatRequest`에서 `jd_text`를 명시적으로 받아 "JD vs 이력서 갭" 분석 지원
- **에이전트 자율 정책**: 이력서 텍스트가 없을 때 Resume/Interview 에이전트의 로컬 fallback 정책 적용
- **RAG**: 문서 로딩(`.txt/.md/.csv/.pdf/.docx/.xlsx`), 청킹, FAISS 벡터 검색 + BM25 키워드 검색
- **RAG 품질**: 인덱스 영속화(`data/index/faiss`), 메타데이터 기반 컨텍스트(page/paragraph/sheet/row), route-aware 카테고리 필터 + no-hit 재검색 fallback, 전용 리랭크 레이어
- **RAG 점수 안정성**: 하이브리드 결합 전 FAISS distance min-max 정규화 적용으로 가중치 튜닝 예측성 향상
- **RAG 추적성**: references에 rank/source/chunk/location/snippet 정보를 포함해 citation-근거 연결 강화
- **RAG 재현성**: 형태소 분석 백엔드(`kiwi/okt/fallback`)와 검색 가중치를 `retriever_meta.json`에 기록
- **구조화 출력**: Pydantic 스키마 기반 최종 응답 생성
- **Route-aware 출력 정책**: synthesis 단계에서 라우트별 최소 섹션 규칙 적용, 불필요 섹션은 빈 배열로 처리
- **구조화 출력 내구성**: Resume/Interview/Synthesis 파싱 실패 시 노드 단위 degrade fallback
- **근거 연결성 강화**: Resume/Interview notes에 `evidence_map` 포함, 최종 불릿에 citation(`[1][2]`) 스타일 적용
- **그래프 체크포인팅**: LangGraph checkpointer와 `thread_id=session_id` 연동(실행 상태 복원 기반)
- **그래프 상태 재사용 캐시**: 파일 기반 invoke 캐시(`graph_state_cache.json` + lock)로 동일 요청 재실행 시 재시작 후에도 결과 재사용
- **JD-이력서 갭 도구**: `jd_resume_gap_score`(필수/우대 키워드 매칭률 + 누락 Top-N) 추가 및 Resume Agent tool loop 연동
- **선택형 신뢰도 메타데이터**: `ChatResponse`에 `route/routing_reason/rag_low_confidence/cached_state_hit` 포함, Streamlit 디버그 토글로 표시 가능
- **동시성 보호**: SessionMemory read/write 구간에 파일 락(`session_memory.json.lock`) 적용
- **서비스 패키징**: FastAPI 백엔드 + Streamlit UI
- **안정성**: API 레벨 예외 처리를 통한 사용자 친화적 오류 응답
- **오류 계약 표준화**: API/CLI/UI 공통 `error_code/detail` 페이로드 적용
- **모듈형 코드 구조**: `src/` 하위 config/retrieval/workflow/ui/api로 재사용 가능한 구조화

## 폴더 구조

- `data/knowledge`: RAG 원천 문서 (`.txt/.md/.csv/.pdf/.docx/.xlsx`)
  - 권장 카테고리: `job_postings/`, `jd/`, `interview_guides/`, `portfolio_examples/`
- `docs`: 제출용 기획/설계/개발 문서 (`step2_planning_design.md`, `step3_service_development.md`)
- `scripts`: 실행 스크립트
- `src/config`: `.env` 로딩 및 모델 클라이언트
- `src/retrieval`: 데이터 로딩, 청킹, 하이브리드 검색
- `src/agents`: 도구 및 구조화 응답 스키마
- `src/workflow`: LangGraph 워크플로우 및 서비스 레이어
- `src/api`: FastAPI 앱
- `src/ui`: Streamlit 앱

## 제출 문서

- [Step2 - 기획 및 설계](./docs/step2_planning_design.md)
- [Step3 - 서비스 개발](./docs/step3_service_development.md)
- [System Architecture Image](./docs/images/system_architecture.png)
- [Service Flow Image](./docs/images/service_flow_sequence.png)
- [Evidence - Agent Execution Log](./docs/evidence/agent_execution_log.md)
- [Evidence - Final Answer JSON](./docs/evidence/agent_final_answer.json)
- [Evidence - E2E Test Checklist](./docs/evidence/e2e_test_checklist.md)
- [Evidence Generator Script](./scripts/generate_submission_evidence.py)
- [Differentiation Metrics Evaluator](./scripts/evaluate_differentiation_metrics.py)

## 환경 설정

1. 의존성 설치:

```powershell
pip install -r requirements-final.txt
```

2. 이 폴더에 `.env` 생성:

```powershell
Copy-Item .env.example .env
```

3. `.env` 필수 값 입력:

- `AOAI_ENDPOINT`
- `AOAI_API_KEY`
- `AOAI_DEPLOY_GPT4O`
- `AOAI_DEPLOY_EMBED_ADA` (또는 `AOAI_EMBEDDING_DEPLOYMENT`)
- `AOAI_API_VERSION`
- `MEMORY_MAX_SESSIONS` (선택, 기본 `200`)
- `MEMORY_TTL_SECONDS` (선택, 기본 `86400`)
- `SESSION_MEMORY_PERSIST_ENABLED` (선택, 기본 `true`; 세션 메모리 디스크 저장 on/off)
- `SESSION_MEMORY_PII_MASK` (선택, 기본 `false`; 세션 메모리 저장 전 이메일/전화번호 마스킹)
- `UI_HISTORY_PERSIST_ENABLED` (선택, 기본 `true`; UI 실행 기록 디스크 로드/저장 on/off)
- `UI_HISTORY_PII_MASK` (선택, 기본 `false`; UI 실행 기록 저장 전 이메일/전화번호 마스킹)
- `INDEX_FORCE_REBUILD` (선택, 기본 `false`; `true` 설정 시 FAISS/청크 캐시 강제 재생성)
- `VECTOR_WEIGHT` (선택, 기본 `0.6`)
- `BM25_WEIGHT` (선택, 기본 `0.4`)
- `RAG_EVIDENCE_SCORE_THRESHOLD` (선택, 기본 `0.45`; low-confidence 모드 임계치)
- `RERANK_ENABLED` (선택, 기본 `true`; 전용 리랭크 레이어 사용 on/off)
- `RERANK_PROVIDER` (선택, 기본 `heuristic`; `heuristic|cross_encoder|llm`, 현재 `cross_encoder/llm`은 heuristic fallback)

## 빠른 시작 (배포 관점)

1) `.env` 준비  
- `Copy-Item .env.example .env` 실행 후 Azure OpenAI 값을 채웁니다.

2) `data/knowledge` 예시 문서 준비  
- 최소 1개 이상 문서를 아래 카테고리 중 하나에 넣습니다.
  - `data/knowledge/job_postings/`
  - `data/knowledge/jd/`
  - `data/knowledge/interview_guides/`
  - `data/knowledge/portfolio_examples/`
- 지원 포맷: `.txt/.md/.csv/.pdf/.docx/.xlsx`

### 지식 문서 최소 체크리스트

- [ ] 카테고리별 최소 1개 문서 확보: `job_postings/`, `jd/`, `interview_guides/`, `portfolio_examples/`
- [ ] 문서 최신성 기준: 수집/수정 기준 3개월 이내 문서 우선, 오래된 문서는 라벨링 또는 교체
- [ ] 금지 콘텐츠 제외: 개인정보 원문(주민번호/연락처), 저작권 위반 원문 전문, 근거 불명확 루머/홍보성 자료

3) 실행  
- API: `python scripts/run_api.py`  
- UI: `python scripts/run_streamlit.py`  
- 접속: `http://localhost:8501`

## 실행 옵션

### 1) CLI (빠른 테스트)

```powershell
python main.py --query "백엔드 이직을 위해 이력서 개선 포인트와 2주 계획을 작성해줘" --target-role "백엔드 개발자" --jd-text "채용 공고/JD 텍스트"
```

### 2) FastAPI

```powershell
python scripts/run_api.py
```

- Health: `http://127.0.0.1:8000/health`
- Chat endpoint: `POST http://127.0.0.1:8000/chat`

### 3) Streamlit

```powershell
python scripts/run_streamlit.py
```

- UI는 이력서 파일 업로드를 지원합니다: `.txt`, `.md`, `.pdf`, `.docx`, `.xlsx`
- UI는 JD/공고 파일 업로드 및 텍스트 입력(선택)을 지원하며, 갭 분석에 반영됩니다.
- 세션 ID는 내부에서 자동 관리되며, 새 대화는 `새 대화 시작` 버튼으로 생성합니다.
- 사이드바에서 실행 입력 기록 조회/삭제를 제공합니다.
- `다시 불러오기`로 질문/직무/이력서/JD/이전 결과를 복원할 수 있습니다.
- 사이드바에 "인덱스 사전 빌드/로드" 및 개인정보 옵션(기록 저장 토글, PII 마스킹 토글)이 있습니다.
- RAG 인덱스는 디스크 캐시를 사용하므로 첫 실행은 느릴 수 있고 이후 실행은 빨라집니다.
- 대용량 입력 방어 적용: 입력창 실시간 카운터(`max_chars`) + 질문/이력서 하드 제한 + 사이드바 이력서 자동 압축 옵션

## 문제 해결 (Troubleshooting)

- `ModuleNotFoundError: No module named 'src'`
  - 반드시 `final-project` 루트에서 실행하세요.
  - 직접 uvicorn 명령 대신 `python scripts/run_api.py`를 사용하세요.
- `Missing environment variables: AOAI_ENDPOINT`
  - `final-project/.env`와 필수 키를 확인하세요.
- 지식 문서 업데이트 후 검색/인덱스가 오래된 것 같을 때
  - `.env`에 `INDEX_FORCE_REBUILD=true`를 설정하고 1회 재실행하세요.
  - 이후 정상 캐시 기동을 위해 `false`로 되돌리세요.
- `/chat`에서 API `400` 또는 `500`이 반환될 때
  - `400`: 입력/설정 이슈(예: 환경변수 누락)
  - `500`: 런타임/모델/검색 이슈(터미널 오류 로그 확인)
  - 에러 계약 예시(최상위): `{"error_code":"CONFIG_MISSING_ENV","detail":"Missing environment variables: AOAI_ENDPOINT"}`
- 구조화 출력 파싱이 간헐적으로 불안정할 때
  - 워크플로우에 Resume/Interview/Synthesis 노드별 fallback degrade 응답이 적용되어 있습니다.
  - 모델/배포 상태를 확인하고 한국어 전용 프롬프트 모드를 유지하세요.
- 세션 메모리/인덱스 파일 동시 쓰기 우려가 있을 때
  - `SessionMemory`는 write/read-update-write 구간에 파일 락을 사용합니다.
  - 다중 사용자 고부하 운영에서는 외부 저장소(SQLite/Redis) + 워커 분리를 권장합니다.

## 차별성 지표 자동화

```powershell
python scripts/evaluate_differentiation_metrics.py --cases data/eval/sample_queries.json
```

- 지표: 라우팅 정확도(`expected_route`가 있을 때), 근거 포함률, 플랜 품질률
- 기준 조정: `--min-routing-accuracy`, `--min-reference-rate`, `--min-plan-quality-rate`
