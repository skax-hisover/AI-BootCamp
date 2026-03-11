# [Step3 - 서비스 개발]

## 1) 구현 범위

- LangGraph 기반 Multi-Agent 워크플로우 구현
- RAG 파이프라인(문서 로딩, 청킹, FAISS + BM25 하이브리드 검색) 구현
- Structured Output(Pydantic) 기반 최종 응답 생성
- Streamlit UI + FastAPI API + CLI 실행 경로 제공
- `.env` 기반 Azure OpenAI 설정 로딩 및 모듈 분리
- API 예외 처리(400/500) 기반 안정적 오류 응답 구성
- Streamlit 실행 입력 기록 관리(조회/삭제/다시 불러오기) 및 세션 자동 관리
- `ChatRequest.jd_text` 기반 JD/공고 텍스트 입력 경로 제공(공고-이력서 갭 분석 명시화)

## 2) 핵심 파일

- `src/workflow/engine.py`: LangGraph 오케스트레이션, 서비스 진입점
- `src/workflow/contracts.py`: `ChatRequest`/`ChatResponse` 스키마 (`jd_text` 포함)
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
- E2E 테스트 체크리스트: `docs/evidence/e2e_test_checklist.md`
- 실행 증빙 로그: `docs/evidence/agent_execution_log.md`
- 최종 응답 JSON: `docs/evidence/agent_final_answer.json`
- 포함 내용:
  - LangGraph 주요 노드/엣지(`supervisor -> rag -> (resume/interview/plan) -> synthesis`)
  - RAG 컨텍스트/참고 출처(`rag_context`, `rag_refs`)가 Supervisor/Synthesis에 주입되는 데이터 흐름

## 3) 실행 방법

### 의존성 설치

```powershell
pip install -r requirements-final.txt
# (선택) Okt 형태소 분석 활성화 시: pip install konlpy==0.6.0
```

### 환경변수 설정

`final-project/.env` 파일에 아래 값 설정:

- `AOAI_ENDPOINT`
- `AOAI_API_KEY`
- `AOAI_DEPLOY_GPT4O`
- `AOAI_DEPLOY_EMBED_ADA` (또는 `AOAI_EMBEDDING_DEPLOYMENT`)
- `AOAI_API_VERSION`
- `MEMORY_MAX_SESSIONS` (선택, 기본 200)
- `MEMORY_TTL_SECONDS` (선택, 기본 86400초)
- `SESSION_MEMORY_PERSIST_ENABLED` (선택, 기본 true / 세션 메모리 디스크 저장 on/off)
- `SESSION_MEMORY_PII_MASK` (선택, 기본 false / 세션 메모리 저장 전 이메일/전화 마스킹)
- `UI_HISTORY_PERSIST_ENABLED` (선택, 기본 true / UI 실행기록 디스크 로드·저장 on/off)
- `UI_HISTORY_PII_MASK` (선택, 기본 false / UI 실행기록 저장 전 이메일/전화 마스킹)
- `UI_HISTORY_STORAGE_MODE` (선택, 기본 summary / `summary|full`, 기본은 길이/해시/미리보기 + 요약 응답 저장)
- `UI_PAGE_ICON_MODE` (선택, 기본 emoji / `emoji|default`)
- `UI_PAGE_ICON_EMOJI` (선택, 기본 💼 / emoji 모드 아이콘)
- `INDEX_FORCE_REBUILD` (선택, 기본 false / true 시 인덱스 강제 재생성)
- `VECTOR_WEIGHT` (선택, 기본 0.6 / 하이브리드 벡터 점수 가중치)
- `BM25_WEIGHT` (선택, 기본 0.4 / 하이브리드 BM25 점수 가중치)
- `RAG_EVIDENCE_SCORE_THRESHOLD` (선택, 기본 0.45 / 상위 점수 임계치 미만 시 근거 부족 모드 전환)
- `EPHEMERAL_JD_BASE_SCORE` (선택, 기본 0.42 / 업로드 JD 임시 근거 기본 점수)
- `EPHEMERAL_RESUME_BASE_SCORE` (선택, 기본 0.36 / 업로드 이력서 임시 근거 기본 점수)
- `EPHEMERAL_OVERLAP_WEIGHT` (선택, 기본 0.35 / 임시 근거 점수에 lexical overlap 반영 비율)
- `FEW_SHOT_MAX_EXAMPLES` (선택, 기본 1 / 직무별 Few-shot 주입 개수 상한)
- `RERANK_ENABLED` (선택, 기본 true / 리랭크 레이어 사용 on/off)
- `RERANK_PROVIDER` (선택, 기본 heuristic / `heuristic|cross_encoder|llm`, 현재 `cross_encoder/llm`은 heuristic fallback)
- `RERANK_MAX_PER_SOURCE` (선택, 기본 2 / top-k 내 동일 source 문서 최대 청크 수)
- `RETRIEVAL_MAX_CHUNKS_PER_FILE` (선택, 기본 0 / retrieval 단계 source당 청크 상한, 0이면 비활성)
- `ALLOW_UNCATEGORIZED_IN_FILTER` (선택, 기본 true / route 필터에서 `uncategorized` 허용 여부, 운영 단계에서 false로 점진적 tighten 가능)
- `UNCATEGORIZED_RATIO_WARN_THRESHOLD` (선택, 기본 0.5 / `uncategorized_ratio >= threshold`일 때 카테고리 품질 경고)
- `FAISS_ALLOW_DANGEROUS_DESERIALIZATION` (선택, 기본 false / 로컬 개발에서만 필요 시 true opt-in)
- `FINAL_ANSWER_CACHE_ENABLED` (선택, 기본 true / 동일 요청 결과 캐시 사용 on/off, 레거시 `GRAPH_STATE_CACHE_ENABLED` 호환)
- `FINAL_ANSWER_CACHE_BYPASS_CONTEXTUAL` (선택, 기본 true / "이전 대화/다시/이어서" 질의 시 캐시 자동 우회, 레거시 `GRAPH_STATE_CACHE_BYPASS_CONTEXTUAL` 호환)
- `FINAL_ANSWER_CACHE_MAX_PER_SESSION` (선택, 기본 5 / 세션별 캐시 보관 개수, `session_id + request_signature` 기반 LRU, 레거시 `GRAPH_STATE_CACHE_MAX_PER_SESSION` 호환)
- `STATE_STORE_BACKEND` (선택, 기본 file / `file|sqlite|redis` 스위치, 현재는 file 구현 우선 + 비file 값은 fallback)
- `STATE_STORE_DSN` (선택, 기본 빈값 / SQLite·Redis 확장용 DSN 예약 필드)

### CLI 실행

```powershell
python main.py --query "백엔드 이직 준비를 위한 2주 계획을 작성해줘" --target-role "백엔드 개발자" --jd-text "채용 공고/JD 텍스트"
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

## 4-1) 차별성 검증 지표(운영 체크)

- **라우팅 정확도**: 의도 라벨 셋(`resume_only`, `interview_only`, `plan_only`, `full`) 기준 수동 평가 정확도
- **근거 포함률**: 응답에서 `references` 1개 이상 포함 비율 및 근거-응답 정합성 점검
- **실행 플랜 품질**: `two_week_plan` 항목의 구체성(행동/우선순위/기간) 5점 척도 평가
- **자동화 실행 경로**:
  ```powershell
  python scripts/evaluate_differentiation_metrics.py --cases data/eval/sample_queries.json
  ```

- 리랭크 다양성 비교는 `--compare-rerank-on-off`로 실행하며, 동일 케이스를 `RERANK_ENABLED=true/false`로 각각 돌려 duplicate-source 비율 차이를 확인합니다.
- 임계치 검증이 필요하면 `--min-rerank-diversity-gain 0.01`을 함께 사용해 `off-on` 개선량이 기준 미달일 때 즉시 실패 처리합니다.
  - 현재 샘플 구성은 총 25건(`resume_only` 5, `interview_only` 5, `plan_only` 5, `full` 5, 모호 질의 5)으로 라우트 균형 + 경계조건을 함께 검증
  - 샘플 질의셋의 `expected_route`가 있을 경우 Top-1 라우팅 정확도를 계산
  - `references >= 1` 비율, `two_week_plan >= 4` 비율을 함께 계산
  - 분포 보정 검증: `--min-per-labeled-route`, `--min-ambiguous-cases`로 라벨/모호 질의 최소 개수를 함께 검증
  - 임계치 미달 시 non-zero 종료코드로 CI/배치 점검 가능

## 4-2) RAG 범위/안전 정책

- 지식 범위: `data/knowledge` 내 채용공고, JD, 면접가이드, 포트폴리오 예시 카테고리 문서 중심
- 라이선스 원칙: 공개 활용 가능 문서/직접 작성 요약본 우선, 저작권 제약 원문은 전문 저장 지양
- 안전장치: 근거 부족 시 일반 조언으로 전환하고 단정형 표현 제한, 필요 시 문서 업로드 안내
- 메타데이터 적용 수준 구분:
  - **권장(현재 로더)**: `src/retrieval/documents.py`는 본문/카테고리 로딩 시 문서별 `*.meta.json` sidecar를 병합하고, 최소 필드(`collected_at/source_url/curator/license`)를 metadata에 반영(누락/파싱 오류 시 경고)
  - **강제/검증(전처리 스크립트)**: `scripts/validate_knowledge_metadata.py`로 `*.meta.json` 필수 필드(`collected_at/source_url/curator/license`)를 배치 검증(`--strict` 시 실패 코드 반환)

### 메타데이터 검증 실행(선택)

```powershell
python scripts/validate_knowledge_metadata.py --strict
```

## 5) 필수 기술 요소 매핑 표

| 필수 항목 | 반영 내용 | 구현 위치 |
|---|---|---|
| 1) Prompt Engineering | 역할 기반 프롬프트(Supervisor/Resume/Interview), Few-shot 예시 반영(외부 파일 로드+fallback), 근거 기반 요약/생각 과정 비노출 지시, 합격 확률/결과 보장 표현 금지, 구조화 출력 지시 | `src/workflow/engine.py`, `src/workflow/prompts.py`, `src/agents/schemas.py`, `data/prompts/few_shots/*` |
| 2) LangChain/LangGraph Multi-Agent | Multi-Agent 그래프 구성(Supervisor/RAG/Resume/Interview/Plan), Tool Calling, 세션 메모리 활용, LangGraph Checkpointer(`thread_id=session_id`) 기반 실행 상태 복원 | `src/workflow/engine.py`, `src/agents/tools.py`, `src/utils/memory.py` |
| 3) RAG | 문서 로딩/전처리/청킹, 임베딩, FAISS + BM25 하이브리드 검색, 근거 출처 반환 (`.txt/.md/.csv/.pdf/.docx/.xlsx`) | `src/retrieval/documents.py`, `src/retrieval/hybrid.py`, `data/knowledge/*` |
| 4) 서비스 개발/패키징 | Streamlit UI(이력서 파일 업로드 + 실행 입력 기록 조회/삭제 + 다시 불러오기), FastAPI 백엔드, CLI 엔트리, 실행 스크립트, `.env` 설정 관리, API 예외 처리 | `src/ui/streamlit_app.py`, `src/api/app.py`, `main.py`, `scripts/*`, `src/config/*` |

## 6) 최근 현행화 반영 사항

- RAG 문서 로더 지원 포맷 확대: `.pdf`, `.docx`, `.xlsx` 추가
- UI 업로드 허용 형식 일치: Streamlit 이력서/JD 업로드 확장자에 `.csv`를 포함해 로더(`.txt/.md/.csv/.pdf/.docx/.xlsx`)와 동일 정책으로 정렬
- FastAPI `/chat` 예외 처리 강화: `ValueError -> 400`, 기타 예외 -> `500`
- 실행/운영 문서 업데이트: README Troubleshooting 섹션 추가
- Streamlit UI 업데이트: 세션 ID 자동 관리, 새 대화 시작, 실행 입력 기록의 다시 불러오기(질문/직무/이력서/결과 복원)
- 실행 출처 UX 명확화: 결과 payload에 `run_id/result_source/executed_at`를 포함하고, 화면 상단에 "새 실행 vs 히스토리 복원" 상태를 명시해 갱신 출처 혼동을 완화
- 에러 처리 표준화: `JobPilotError` 기반 `error_code/detail` 계약으로 API/CLI/UI 분기 로직 일원화
- 인덱스 빌드 UX 보강: Streamlit에 "인덱스 사전 빌드/로드" 관리 버튼 및 첫 실행 지연 안내 추가
- 개인정보 옵션 보강: 실행 입력 기록 파일 저장 on/off, 저장 전 이메일/전화번호 마스킹 옵션 추가
- JD 입력 경로 추가: CLI/UI/워크플로우 전 구간에 `jd_text` 전달 및 프롬프트 반영(공고-이력서 갭 비교 명시)
- Supervisor 라우팅 고도화: `resume_only/interview_only/full/plan_only` 분류 + LangGraph 조건부 엣지 분기 적용
- Supervisor 2단 라우팅: "면접 제외/계획 제외/이력서 제외" 등 명시 표현은 1차 휴리스틱으로 우선 반영하고, 그 외는 LLM 라우팅으로 처리
- 라우팅 정의 상수화: 휴리스틱 키워드/제외어/LLM 라우팅 규칙 블록을 공통 상수(`src/workflow/prompts.py`)로 분리해 휴리스틱/LLM 라우터가 동일 정의 집합을 공유
- 라우팅 부정문 내성 보강: `~제외는 아니고`, `~제외하지 말고` 패턴을 휴리스틱 예외 규칙으로 추가해 `resume_only` 질의의 `plan_only` 오분류를 완화
- 회귀 테스트 고정: 부정/예외 문맥 케이스를 `tests/test_workflow_routes.py`에 추가해 라우팅 안정성을 테스트로 재현 가능하게 관리
- Tool loop 가드레일 강화: 최대 N회(기본 4회) 루프 + 동일 tool+args 반복 호출 감지 시 중단 + 도구 라운드 상한 후 강제 요약 전환
- Tool 출력 구조화: `resume_keyword_match_score`/`interview_question_bank` 결과를 JSON으로 표준화하고, Agent 프롬프트에서 1회 이상 반영 규칙을 명시
- Plan Agent 분리: `plan_node` 추가, `full/plan_only`에서 실행해 우선순위/일정/검증 방법을 독립적으로 의사결정
- plan_only 경로 정합성 보강: `plan_only`도 `rag_node(top_k=2 경량)`를 거쳐 근거를 확보한 뒤 Plan Agent/Synthesis로 전달
- 세션 메모리 영속화: 메모리 JSON 파일 저장/로드로 서버 재시작 후 대화 이력 유지
- LangGraph Checkpointer 연동: `thread_id=session_id` 기준으로 그래프 실행 상태 복원 기반 마련
- 체크포인터 역할 분리 명시: `MemorySaver`는 프로세스 내 런타임 복원용, 서버 재시작 이후 복원/재사용은 `session_memory.json` + `final_answer_cache.json`이 담당
- RAG Agent 강화: 쿼리 리라이트, 직무 힌트 기반 문서 우선순위(스코어 부스팅), route 메타 노트 반영
- RAG Agent 자율 보강: `rag_low_confidence`일 때 1회 재검색(쿼리 확장 + 무필터 탐색) 후 재랭크하는 로컬 복구 정책 적용
- Resume/Interview Agent 자율 보강: `rag_low_confidence`일 때 각 노드가 로컬 쿼리로 추가 검색을 수행해 근거를 보강하고, 보강된 컨텍스트/refs를 이후 노드에 전달
- RAG 리랭크 레이어 분리: 검색(`HybridRetriever.search`) 이후 `rerank` 확장 포인트 추가(향후 cross-encoder/LLM 리랭커 교체 용이)
- 중간 산출물 구조화: `ResumeNotes`, `InterviewNotes` 스키마 추가 및 Resume/Interview 노드 structured output 적용
- 근거 연결성 강화: `ResumeNotes/InterviewNotes`에 `evidence_map` 추가(항목 -> 근거 chunk 번호 매핑)
- 에이전트 로컬 정책 강화: 이력서 텍스트 미제공 시 Resume/Interview 노드가 갭 분석/공통 질문 중심으로 독립 전환
- 노드 독립성 강화: Resume/Interview/Plan/Supervisor/RAG 노드별 모델 온도와 시스템 역할 지시를 분리해 의사결정 편향을 완화
- 오류 내성 강화: Resume/Interview/Synthesis structured output 실패 시 fallback 결과로 degrade 처리
- Specialist fallback 타입 보장 강화: `_fallback_resume_notes/_fallback_interview_notes/_fallback_plan_notes`에서 리스트 필드·근거 스니펫·reason 포맷(`ErrorCodes`)을 정규화해 계약 일관성 보강
- route-aware 출력 규칙: `synthesis` 단계에서 라우트별 최소 섹션 규칙 적용(예: `plan_only`는 계획 중심, 불필요 섹션은 빈 배열)
- 라우트 최소 개수 정렬: `resume_only/interview_only`는 `two_week_plan` 최소 개수를 0으로 조정해 기획 시나리오(플랜 제외)와 일치
- Synthesis 안정화: 규칙을 필수/권장으로 분리하고, 최소 개수/빈 배열/citation 보정은 코드 후처리에서 강제
- Synthesis citation 규칙 단일화: references는 임의 생성하지 않되 불릿에는 기본 `[1]` citation을 우선 표기하도록 지시하고, references/evidence_map 정합성은 후처리로 보정
- 요약 정책 통일: `plan_only` 포함 모든 라우트에서 `summary`는 항상 제공(계획 전용 라우트는 1~2문장 요약)
- 카테고리 필터 보완: route-aware 필터에 `uncategorized`를 허용해 루트 문서 데이터도 검색 누락 없이 반영
- 카테고리 필터 운영 스위치: `ALLOW_UNCATEGORIZED_IN_FILTER`로 `uncategorized` 허용 여부를 환경별로 제어해 데이터 성숙도에 따라 잡음을 점진 축소
- citation 강제: 최종 액션 불릿에 `[1][2]` 형태의 근거 번호 표기를 프롬프트 수준에서 요구
- RAG 인덱스 영속화: `FAISS.save_local/load_local` + 청크/시그니처 메타 저장으로 재시작 시 인덱스 재사용
- 인덱스 무효화 안정성 강화: corpus signature에 파일 fingerprint 반영 + `INDEX_FORCE_REBUILD` 옵션 지원
- 검색 품질 보강: 길이 페널티 리랭킹, 동일 파일 청크 수 제한(다양성), 카테고리 필터 파라미터 지원
- 검색 다양성 강화: `rerank_hits` 단계에서 source당 최대 청크 수(`RERANK_MAX_PER_SOURCE`)를 적용해 references 중복을 완화
- 필터 내구성 보강: route-aware category filter 결과가 비는 경우 무필터 재검색 fallback으로 근거 회수율 개선
- FAISS 캐시 로드 안전성 보강: `retriever_meta.json.cache_hashes`와 `index.faiss/index.pkl` SHA-256 검증이 통과한 경우에만 캐시 로드를 허용
- 메타데이터 강화: PDF 페이지 번호, DOCX 문단 번호, XLSX 시트/행 정보를 컨텍스트 및 refs에 노출
- references 추적성 강화: rank/source/location/chunk_id/snippet 정보를 포함한 문자열 포맷으로 citation 연결성 향상
- 점수 해석성 강화: references에 `score_breakdown`(vector/bm25/fused/length penalty/rerank boosts) 메타를 포함해 디버그 튜닝 근거를 제공
- 점수 스케일 정합성: rerank 이후 점수도 0~1 범위로 클리핑해 `RAG_EVIDENCE_SCORE_THRESHOLD`의 절대값 해석을 일관되게 유지
- references 타입 정렬: `FinalAnswer.references`와 `ChatResponse.references`를 구조화 객체 목록으로 통일해 synthesis 단계 타입 불일치 리스크를 완화
- 한국어 BM25 개선: kiwi/konlpy 형태소 분석 옵션(설치 시 자동 활용), 미설치 시 조사 제거 기반 fallback 토크나이저 사용
- 재현성 강화: `retriever_meta.json`에 tokenizer backend(`kiwi/okt/fallback`)와 하이브리드 가중치(`vector_weight`, `bm25_weight`) 기록
- RAG 안전정책 코드 반영: 최고 점수가 `RAG_EVIDENCE_SCORE_THRESHOLD` 미만이면 근거 부족 모드로 전환해 보수적 표현을 우선
- 벡터 점수 정규화 안정화: FAISS distance를 쿼리-독립 변환(`1/(1+d)`)으로 0~1 스케일링해 low-confidence 임계치 해석의 일관성 확보
- 중복 제어 역할 분리: rerank 활성 시 retrieval 단계 source 상한(`RETRIEVAL_MAX_CHUNKS_PER_FILE`)은 비활성화하고, 최종 다양성 제어는 `RERANK_MAX_PER_SOURCE`에서 단일 강제로 운영
- 카테고리 품질 진단 추가: `HybridRetriever.build()`에서 카테고리 분포와 `uncategorized` 비율을 점검하고, 비중 과다 시 경고를 출력하며 `retriever_meta.json`에 진단 결과 기록
- UI 품질 경고 연결: Streamlit 사이드바에서 `retriever_meta.json`의 `category_quality_warning`/`uncategorized_ratio`를 즉시 노출해 데이터 거버넌스 이슈를 실행 전에 인지 가능
- 리랭크 확장성 분리: 휴리스틱 리랭커를 `src/retrieval/rerank.py`로 분리하고 `RERANK_ENABLED`, `RERANK_PROVIDER` 설정 기반 on/off·전략 전환 포인트 제공
- 업로드 입력 근거 강화: `rag_node`에서 `jd_text/resume_text`를 임시 청크로 생성해 검색 후보에 혼합(ephemeral evidence)하여 공고-이력서 갭 분석의 직접 근거성을 보강
- 업로드 이력서 카테고리 분리: `resume_text` 기반 임시 청크는 `resume_upload`로 분류해 지식 문서 카테고리(`portfolio_examples`)와 의미를 분리하고, rerank에서 별도 가중으로 반영
- 체크포인터 실효성 보강: `MemorySaver`와 별개로 `final_answer_cache.json`(파일+락) 기반 invoke 전/후 캐시를 추가해 동일 입력 재실행 시 결과 재사용(재시작 이후에도 캐시 복원) 지원
  - 용어 정리: 위 캐시는 "그래프 중간 상태"가 아니라 정규화된 최종 `ChatResponse` payload 재사용 캐시
- 캐시 안전장치 추가: `FINAL_ANSWER_CACHE_ENABLED`, `FINAL_ANSWER_CACHE_BYPASS_CONTEXTUAL` 옵션과 맥락형 질의("이전 대화/다시/이어서") 자동 우회 규칙으로 캐시 오적용 위험 완화
- 도구 도메인 특화 강화: `jd_resume_gap_score` 도구를 추가해 JD 필수/우대 키워드 매칭률과 누락 역량 top-N을 Resume Agent가 정량 근거로 반영
- 도구 매칭 신뢰도 보강: `tools.py`에 kiwi 토크나이저(설치 시) + 동의어 정규화(`RDBMS↔DB`, `Fast-API↔fastapi` 등)를 적용해 표기 변형에 대한 강건성 향상
- Tool Calling 진입점 명확화: `_run_tool_loop_structured_with_trace()`에서 `bind_tools()`로 도구 목록을 모델에 주입하고 `tool_calls`를 모델이 자율 선택하는 흐름을 코드 주석으로 명시
- 디버그 신뢰도 노출 확장: `ChatResponse`에 `route`, `routing_reason`, `rag_low_confidence`, `cached_state_hit`, `node_status` 옵션 필드를 추가하고 Streamlit에서 선택적으로 표시
- 출처 메타 UX 토글: Streamlit 사이드바에서 `참고 출처 메타데이터 표시`를 켜면 references의 `collected_at/source_url/curator/license`를 선택적으로 노출
- 노드 부분 실패 격리: Resume/Interview/Plan/Synthesis 중 일부가 fallback(degraded)되어도 전체 응답은 유지하고 `node_status`로 실패 노드와 `error_code`를 전달
- API 에러 계약 단순화: FastAPI `exception_handler`로 `JobPilotError`를 최상위 `{error_code, detail}` 형태로 직렬화해 클라이언트 파싱 복잡도 완화
- API 계약 명시 강화: `/chat` 엔드포인트에 `response_model=ChatResponse`를 지정해 OpenAPI 문서/클라이언트 계약 안정성을 강화
- UI route-aware 안내 보강: 비활성 섹션을 숨기는 대신 라우트별 생략 안내 문구(예: resume_only에서 면접/플랜 생략)를 조건부 표시
- 동시성 범위 확장: `ui_input_history.json` 로드/저장에도 파일 락(`ui_input_history.json.lock`)을 적용해 멀티세션 경합 내구성 강화
- 원자적 파일 저장 적용: `session_memory.json`/`final_answer_cache.json`/`chunks.json`/`retriever_meta.json`/`ui_input_history.json`은 temp 파일 후 `replace` 방식으로 저장해 부분쓰기 리스크 완화
- JD 입력 방어 대칭화: Streamlit에 JD 자동 압축(앞/뒤 유지) 옵션 및 목표 글자 수 설정을 추가해 긴 공고 텍스트 처리 비용을 완화
- 업로드 UX 보강: 파일 업로드 반영 방식을 `덮어쓰기/추가하기`로 선택 가능하게 하고 업로드 시그니처로 중복 반영(누적 혼선)을 방지
- 파일 파서 공통화: Streamlit 업로드 파서와 RAG 문서 로더가 `src/utils/file_extract.py`를 공통 사용해 포맷별 파싱/예외 처리를 단일화
- Streamlit 오류 가이드 보강: 지식문서 미존재/환경변수 누락 시 사용자 안내 메시지 및 해결 가이드 표시
- 지식 로드 실패 가시성 보강: 인덱싱 시 문서 로드 실패 목록을 `retriever_meta.json`에 기록하고, Streamlit 사이드바 버튼으로 요약 조회 지원
- 멀티유저 방어 로직: SessionMemory에 TTL/최대 세션 수 제한 추가(메모리 누적 방지)
- 대용량 입력 방어: Streamlit에서 질문/이력서 입력란 내부 실시간 카운터(`max_chars`) + 하드 제한 및 긴 이력서 자동 압축(앞/뒤 중심) 옵션 제공
- 파일 경합 완화: `SessionMemory` 저장/조회 경로에 파일 락 적용(`session_memory.json.lock`)
- 히스토리 스키마 마이그레이션: `record_version` 기반 `migrate_history_record()`를 통해 구버전 기록(v0)도 로드시 최신 스키마(v1)로 정규화
- 실행 재현성 강화: `requirements-final.txt` 핵심 의존성 버전 고정 및 `.env.example` 제공
- 도구 반영 검증 강화: `resume_node/interview_node`에서 ToolMessage(JSON) 반영 여부를 키워드/질문 기준으로 점검하고, 미반영 시 검증 피드백을 포함해 1회 재시도하도록 보완
- `plan_only` 요약 가독성 강화: `normalize_final_answer_by_route()` 후처리에서 summary를 1~2문장(과도 길이 시 절단)으로 강제해 UI 카드 길이 편차를 안정화
- 요약 분리 보정 가드레일: 문장 분리가 불안정한 한국어 종결형 케이스를 대비해 line(최대 2줄)/char(최대 길이) 기반 보정을 함께 적용
- 더미 콘텐츠 완화: 최소 개수 미달 시 `"추가 권장 액션 N"` 대신 근거/입력 부족을 명시하고 경력연차·지원회사·핵심 프로젝트 등 추가 질문을 유도하는 fallback 문구로 대체
- citation 정합성 자기검증: `synthesis_node`에서 불릿별 citation([1]~[N]) 유효성을 점검하고 미달 시 자기검증 프롬프트로 1회 재작성해 근거-문장 연결성을 강화
- 데이터 구조 정합성 보강: `data/knowledge/job_postings|jd|interview_guides|portfolio_examples` 예시 폴더/샘플 문서를 추가해 route-aware 카테고리 필터가 실제 데이터에서도 의미 있게 동작하도록 정리
- 루트 문서 카테고리 보정: `data/knowledge` 루트 파일은 파일명 규칙으로 카테고리를 자동 추론해 `uncategorized` 과다를 완화
- references 계약 구조화: `ChatResponse.references`를 `{rank, source, chunk_id, location, score, category, snippet}` 객체 리스트로 통일해 UI/후처리 활용성을 개선
- 지표 증빙 인코딩 안정화: `evaluate_differentiation_metrics.py --output` 옵션으로 UTF-8 저장을 명시 지원해 `metrics_run_output.txt` 깨짐 문제를 예방

## 7) 체크포인터/캐시 책임 경계 및 마이그레이션

```mermaid
flowchart LR
    A[MemorySaver<br/>LangGraph runtime checkpointer] -->|프로세스 내 실행 중 복원| G[Graph Execution]
    B[session_memory.json] -->|재시작 이후 대화 이력 복원| G
    C[final_answer_cache.json] -->|재시작 이후 동일 요청 최종 응답 재사용| G
    D[graph_state_cache.json<br/>(legacy)] -->|일회성 마이그레이션| C
```

- 책임 경계: `MemorySaver`는 **프로세스 내(runtime) 그래프 상태 복원** 전용, `session_memory.json/final_answer_cache.json`은 **재시작 이후 복원/재사용** 전용입니다.
- 레거시 제거 플랜: `scripts/migrate_cache.py`로 `graph_state_cache.json -> final_answer_cache.json`를 일괄 이관하고, 이관 완료 후 `--strict` 기준으로 레거시 파일 잔존을 운영 점검 항목으로 관리합니다.