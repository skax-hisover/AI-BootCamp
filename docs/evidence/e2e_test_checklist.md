# E2E 테스트 체크리스트 (샘플 `jd_text` 포함)

CLI/FastAPI/Streamlit 통합 검증을 위한 제출용 체크리스트입니다.

## 0) 사전 준비

- [ ] 프로젝트 루트로 이동: `D:\AI-BootCamp\final-project`
- [ ] 의존성 설치: `pip install -r requirements-final.txt`
- [ ] `.env`에 필수 AOAI 설정값 입력
- [ ] (선택) 저장/개인정보 관련 변수 확인: `UI_HISTORY_PERSIST_ENABLED`, `UI_HISTORY_PII_MASK`, `SESSION_MEMORY_PERSIST_ENABLED`, `SESSION_MEMORY_PII_MASK`
- [ ] `data/knowledge`에 최소 1개 이상의 문서가 있는지 확인

샘플 `jd_text`:

```text
백엔드 개발자 채용 공고
- Python/FastAPI 기반 API 설계 및 운영 경험
- RDBMS(MySQL/PostgreSQL) 성능 최적화 경험
- 장애 대응 및 로그/모니터링 기반 트러블슈팅 경험
- Docker/AWS 배포 경험
- 협업 커뮤니케이션 및 문서화 역량
```

샘플 `resume_text`:

```text
Python과 FastAPI로 사내 API를 개발/운영했습니다.
MySQL 사용 경험이 있으며 AWS EC2 배포 경험이 있습니다.
장애 대응은 기본적인 로그 확인 수준으로 수행했습니다.
```

## 1) CLI E2E

### 1-1. JD + 이력서 동시 실행

- [ ] 실행:

```powershell
python main.py --session-id "cli-e2e-1" --target-role "백엔드 개발자" --query "공고와 이력서의 갭을 분석하고, 이력서 개선 5개와 2주 계획을 작성해줘." --jd-text "백엔드 개발자 채용 공고 - Python/FastAPI, DB 튜닝, 장애 대응, Docker/AWS, 협업 문서화" --resume-text "FastAPI API 개발 2년, MySQL 사용, AWS 배포 경험"
```

- [ ] 검증:
  - [ ] JSON 응답이 반환된다.
  - [ ] `summary`, `resume_improvements`, `interview_preparation`, `two_week_plan`, `references` 키가 존재한다.
  - [ ] `references`가 1개 이상 포함된다.
  - [ ] `two_week_plan`이 4개 이상 포함된다.

### 1-2. 라우팅 성향 확인

- [ ] 이력서 중심 질의(면접 제외 명시)로 실행한다.
- [ ] 응답이 이력서 개선 중심으로 출력되는지 확인한다.

## 2) FastAPI E2E

### 2-1. 서버 실행

- [ ] 실행: `python scripts/run_api.py`
- [ ] 헬스체크 확인: `http://127.0.0.1:8000/health`

### 2-2. `jd_text` 포함 `/chat` 호출

- [ ] 실행:

```powershell
$body = @{
  session_id = "api-e2e-1"
  user_query = "JD 대비 이력서 갭과 면접 준비 포인트, 2주 계획을 작성해줘."
  target_role = "백엔드 개발자"
  jd_text = "Python/FastAPI, DB 성능 최적화, 장애 대응, Docker/AWS, 협업 문서화"
  resume_text = "FastAPI API 개발 2년, MySQL 사용, AWS 배포 경험"
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
```

- [ ] 검증:
  - [ ] HTTP 200 응답
  - [ ] 필수 응답 필드 존재
  - [ ] 한국어 응답
  - [ ] `references` 포함

### 2-3. 에러 처리 확인(선택)

- [ ] 설정/입력 오류 시 400/500 안내 메시지가 명확한지 확인한다.

## 3) Streamlit E2E

### 3-1. UI 실행

- [ ] 실행: `python scripts/run_streamlit.py`
- [ ] 접속: `http://localhost:8501`

### 3-2. JD 포함 입력/실행 시나리오

- [ ] 질문 입력 및 목표 직무 선택
- [ ] `JD/공고 텍스트(선택)` 입력(또는 JD 파일 업로드)
- [ ] `이력서 텍스트(선택)` 입력(또는 이력서 파일 업로드)
- [ ] `에이전트 실행` 클릭

- [ ] 검증:
  - [ ] 결과 카드(요약/이력서/면접/계획/출처) 렌더링
  - [ ] 실행 입력 기록 저장
  - [ ] `다시 불러오기` 시 질문/직무/JD/이력서/결과 복원
  - [ ] `새 대화 시작` 시 새 세션으로 초기화
  - [ ] 입력창 `max_chars` 카운터 정상 동작
  - [ ] 업로드 반영 방식(`덮어쓰기/추가하기`) 전환 시 이력서/JD 텍스트가 의도한 방식으로 반영되고 중복 누적이 발생하지 않는지 확인
  - [ ] 응답 요약(summary)에 책임 한계 고지 문구(법/세무/노무 비전문 영역 제외 + 최신 공고/회사 정책 원문 확인)가 포함되는지 확인
  - [ ] 맥락형 질의(예: "이전 대화 기준으로 다시") 실행 시 `cached_state_hit=false`로 캐시 우회가 동작하는지(디버그 메타 표시 ON 기준) 확인
  - [ ] 디버그 메타에서 `node_status`가 노드별(`ok/degraded/skipped`)로 표시되고, 부분 실패 시 `error_code/detail`이 노출되는지 확인
  - [ ] (선택) `인덱스 사전 빌드/로드` 버튼 정상 동작
  - [ ] (선택) `지식 문서 로드 실패 요약` 버튼 클릭 시 인덱싱 실패 파일 목록/오류가 표시되는지 확인
  - [ ] (선택) 기록 저장 OFF/PII 마스킹 옵션 동작 확인

### 3-3. 재시작 후 기록 유지 확인

- [ ] Streamlit 종료(`Ctrl+C`) 후 재실행
- [ ] 사이드바 기록이 유지되는지 확인(기록 저장 옵션 ON 기준)
- [ ] `실행 입력 기록 파일 저장 사용` OFF 시 디스크 로드/저장이 중단되고 세션 메모리만 사용하는 정책이 적용되는지 확인

## 4) 차별성 지표 자동 검증

- [ ] 실행:

```powershell
python scripts/evaluate_differentiation_metrics.py --cases data/eval/sample_queries.json
```

- [ ] 검증:
  - [ ] 라우팅 정확도 출력
  - [ ] 근거 포함률 출력
  - [ ] 플랜 품질률 출력
  - [ ] 임계치 충족 시 `[PASS] All thresholds satisfied.` 출력

## 5) 제출 산출물 점검

- [ ] `docs/evidence/agent_execution_log.md` 최신화
- [ ] `docs/evidence/agent_final_answer.json` 최신화
- [ ] Streamlit 캡처 준비: `docs/evidence/streamlit_main_capture.png`
- [ ] `.env` 제출 안전 점검: `python scripts/check_env_submission_safety.py --strict` 실행 후 경고가 없거나, `.env`는 제출물에서 제외했는지 확인
- [ ] (선택) 지표 자동검증 실행 결과 캡처 첨부

