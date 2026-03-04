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
- `INDEX_FORCE_REBUILD` (선택, 기본 false / true 시 인덱스 강제 재생성)
- `VECTOR_WEIGHT` (선택, 기본 0.6 / 하이브리드 벡터 점수 가중치)
- `BM25_WEIGHT` (선택, 기본 0.4 / 하이브리드 BM25 점수 가중치)
- `RAG_EVIDENCE_SCORE_THRESHOLD` (선택, 기본 0.45 / 상위 점수 임계치 미만 시 근거 부족 모드 전환)

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
  - 샘플 질의셋의 `expected_route`가 있을 경우 Top-1 라우팅 정확도를 계산
  - `references >= 1` 비율, `two_week_plan >= 4` 비율을 함께 계산
  - 임계치 미달 시 non-zero 종료코드로 CI/배치 점검 가능

## 4-2) RAG 범위/안전 정책

- 지식 범위: `data/knowledge` 내 채용공고, JD, 면접가이드, 포트폴리오 예시 카테고리 문서 중심
- 라이선스 원칙: 공개 활용 가능 문서/직접 작성 요약본 우선, 저작권 제약 원문은 전문 저장 지양
- 안전장치: 근거 부족 시 일반 조언으로 전환하고 단정형 표현 제한, 필요 시 문서 업로드 안내

## 5) 필수 기술 요소 매핑 표

| 필수 항목 | 반영 내용 | 구현 위치 |
|---|---|---|
| 1) Prompt Engineering | 역할 기반 프롬프트(Supervisor/Resume/Interview), Few-shot 예시 반영, 근거 기반 요약/생각 과정 비노출 지시, 구조화 출력 지시 | `src/workflow/engine.py`, `src/agents/schemas.py` |
| 2) LangChain/LangGraph Multi-Agent | Multi-Agent 그래프 구성(Supervisor/RAG/Resume/Interview/Plan), Tool Calling, 세션 메모리 활용, LangGraph Checkpointer(`thread_id=session_id`) 기반 실행 상태 복원 | `src/workflow/engine.py`, `src/agents/tools.py`, `src/utils/memory.py` |
| 3) RAG | 문서 로딩/전처리/청킹, 임베딩, FAISS + BM25 하이브리드 검색, 근거 출처 반환 (`.txt/.md/.csv/.pdf/.docx/.xlsx`) | `src/retrieval/documents.py`, `src/retrieval/hybrid.py`, `data/knowledge/*` |
| 4) 서비스 개발/패키징 | Streamlit UI(이력서 파일 업로드 + 실행 입력 기록 조회/삭제 + 다시 불러오기), FastAPI 백엔드, CLI 엔트리, 실행 스크립트, `.env` 설정 관리, API 예외 처리 | `src/ui/streamlit_app.py`, `src/api/app.py`, `main.py`, `scripts/*`, `src/config/*` |

## 6) 최근 현행화 반영 사항

- RAG 문서 로더 지원 포맷 확대: `.pdf`, `.docx`, `.xlsx` 추가
- FastAPI `/chat` 예외 처리 강화: `ValueError -> 400`, 기타 예외 -> `500`
- 실행/운영 문서 업데이트: README Troubleshooting 섹션 추가
- Streamlit UI 업데이트: 세션 ID 자동 관리, 새 대화 시작, 실행 입력 기록의 다시 불러오기(질문/직무/이력서/결과 복원)
- 에러 처리 표준화: `JobPilotError` 기반 `error_code/detail` 계약으로 API/CLI/UI 분기 로직 일원화
- 인덱스 빌드 UX 보강: Streamlit에 "인덱스 사전 빌드/로드" 관리 버튼 및 첫 실행 지연 안내 추가
- 개인정보 옵션 보강: 실행 입력 기록 파일 저장 on/off, 저장 전 이메일/전화번호 마스킹 옵션 추가
- JD 입력 경로 추가: CLI/UI/워크플로우 전 구간에 `jd_text` 전달 및 프롬프트 반영(공고-이력서 갭 비교 명시)
- Supervisor 라우팅 고도화: `resume_only/interview_only/full/plan_only` 분류 + LangGraph 조건부 엣지 분기 적용
- Tool loop 가드레일 강화: 최대 N회(기본 4회) 루프 + 동일 tool+args 반복 호출 감지 시 중단 + 도구 라운드 상한 후 강제 요약 전환
- Tool 출력 구조화: `resume_keyword_match_score`/`interview_question_bank` 결과를 JSON으로 표준화하고, Agent 프롬프트에서 1회 이상 반영 규칙을 명시
- Plan Agent 분리: `plan_node` 추가, `full/plan_only`에서 실행해 우선순위/일정/검증 방법을 독립적으로 의사결정
- plan_only 경로 정합성 보강: `plan_only`도 `rag_node(top_k=2 경량)`를 거쳐 근거를 확보한 뒤 Plan Agent/Synthesis로 전달
- 세션 메모리 영속화: 메모리 JSON 파일 저장/로드로 서버 재시작 후 대화 이력 유지
- LangGraph Checkpointer 연동: `thread_id=session_id` 기준으로 그래프 실행 상태 복원 기반 마련
- RAG Agent 강화: 쿼리 리라이트, 직무 힌트 기반 문서 우선순위(스코어 부스팅), route 메타 노트 반영
- RAG 리랭크 레이어 분리: 검색(`HybridRetriever.search`) 이후 `rerank` 확장 포인트 추가(향후 cross-encoder/LLM 리랭커 교체 용이)
- 중간 산출물 구조화: `ResumeNotes`, `InterviewNotes` 스키마 추가 및 Resume/Interview 노드 structured output 적용
- 근거 연결성 강화: `ResumeNotes/InterviewNotes`에 `evidence_map` 추가(항목 -> 근거 chunk 번호 매핑)
- 에이전트 로컬 정책 강화: 이력서 텍스트 미제공 시 Resume/Interview 노드가 갭 분석/공통 질문 중심으로 독립 전환
- 오류 내성 강화: Resume/Interview/Synthesis structured output 실패 시 fallback 결과로 degrade 처리
- route-aware 출력 규칙: `synthesis` 단계에서 라우트별 최소 섹션 규칙 적용(예: `plan_only`는 계획 중심, 불필요 섹션은 빈 배열)
- 라우트 최소 개수 정렬: `resume_only/interview_only`는 `two_week_plan` 최소 개수를 0으로 조정해 기획 시나리오(플랜 제외)와 일치
- Synthesis 안정화: 규칙을 필수/권장으로 분리하고, 최소 개수/빈 배열/citation 보정은 코드 후처리에서 강제
- 요약 정책 통일: `plan_only` 포함 모든 라우트에서 `summary`는 항상 제공(계획 전용 라우트는 1~2문장 요약)
- 카테고리 필터 보완: route-aware 필터에 `uncategorized`를 허용해 루트 문서 데이터도 검색 누락 없이 반영
- citation 강제: 최종 액션 불릿에 `[1][2]` 형태의 근거 번호 표기를 프롬프트 수준에서 요구
- RAG 인덱스 영속화: `FAISS.save_local/load_local` + 청크/시그니처 메타 저장으로 재시작 시 인덱스 재사용
- 인덱스 무효화 안정성 강화: corpus signature에 파일 fingerprint 반영 + `INDEX_FORCE_REBUILD` 옵션 지원
- 검색 품질 보강: 길이 페널티 리랭킹, 동일 파일 청크 수 제한(다양성), 카테고리 필터 파라미터 지원
- 필터 내구성 보강: route-aware category filter 결과가 비는 경우 무필터 재검색 fallback으로 근거 회수율 개선
- 메타데이터 강화: PDF 페이지 번호, DOCX 문단 번호, XLSX 시트/행 정보를 컨텍스트 및 refs에 노출
- references 추적성 강화: rank/source/location/chunk_id/snippet 정보를 포함한 문자열 포맷으로 citation 연결성 향상
- 한국어 BM25 개선: kiwi/konlpy 형태소 분석 옵션(설치 시 자동 활용), 미설치 시 조사 제거 기반 fallback 토크나이저 사용
- 재현성 강화: `retriever_meta.json`에 tokenizer backend(`kiwi/okt/fallback`)와 하이브리드 가중치(`vector_weight`, `bm25_weight`) 기록
- RAG 안전정책 코드 반영: 최고 점수가 `RAG_EVIDENCE_SCORE_THRESHOLD` 미만이면 근거 부족 모드로 전환해 보수적 표현을 우선
- 벡터 점수 정규화 안정화: FAISS distance를 retrieval set 기준 min-max 정규화 후 BM25와 결합해 가중치 튜닝 예측 가능성 향상
- Streamlit 오류 가이드 보강: 지식문서 미존재/환경변수 누락 시 사용자 안내 메시지 및 해결 가이드 표시
- 멀티유저 방어 로직: SessionMemory에 TTL/최대 세션 수 제한 추가(메모리 누적 방지)
- 대용량 입력 방어: Streamlit에서 질문/이력서 입력란 내부 실시간 카운터(`max_chars`) + 하드 제한 및 긴 이력서 자동 압축(앞/뒤 중심) 옵션 제공
- 파일 경합 완화: `SessionMemory` 저장/조회 경로에 파일 락 적용(`session_memory.json.lock`)
- 실행 재현성 강화: `requirements-final.txt` 핵심 의존성 버전 고정 및 `.env.example` 제공