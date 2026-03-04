# E2E Test Checklist (with sample jd_text)

Submit-ready checklist for CLI/FastAPI/Streamlit end-to-end verification.

## 0) Prerequisites

- [ ] Move to project root: `D:\AI-BootCamp\final-project`
- [ ] Install dependencies: `pip install -r requirements-final.txt`
- [ ] Configure `.env` with required AOAI keys
- [ ] Ensure at least one document exists in `data/knowledge`

Sample `jd_text`:

```text
백엔드 개발자 채용 공고
- Python/FastAPI 기반 API 설계 및 운영 경험
- RDBMS(MySQL/PostgreSQL) 성능 최적화 경험
- 장애 대응 및 로그/모니터링 기반 트러블슈팅 경험
- Docker/AWS 배포 경험
- 협업 커뮤니케이션 및 문서화 역량
```

Sample `resume_text`:

```text
Python과 FastAPI로 사내 API를 개발/운영했습니다.
MySQL 사용 경험이 있으며 AWS EC2 배포 경험이 있습니다.
장애 대응은 기본적인 로그 확인 수준으로 수행했습니다.
```

## 1) CLI E2E

### 1-1. Run with JD + resume

- [ ] Run:

```powershell
python main.py --session-id "cli-e2e-1" --target-role "백엔드 개발자" --query "공고와 이력서의 갭을 분석하고, 이력서 개선 5개와 2주 계획을 작성해줘." --jd-text "백엔드 개발자 채용 공고 - Python/FastAPI, DB 튜닝, 장애 대응, Docker/AWS, 협업 문서화" --resume-text "FastAPI API 개발 2년, MySQL 사용, AWS 배포 경험"
```

- [ ] Validate:
  - [ ] JSON response returned
  - [ ] Keys exist: `summary`, `resume_improvements`, `interview_preparation`, `two_week_plan`, `references`
  - [ ] `references` includes at least 1 item
  - [ ] `two_week_plan` includes at least 4 items

### 1-2. Route tendency check

- [ ] Run a resume-focused query (exclude interview explicitly)
- [ ] Confirm response is resume-improvement focused

## 2) FastAPI E2E

### 2-1. Start server

- [ ] Run: `python scripts/run_api.py`
- [ ] Confirm health endpoint is reachable: `http://127.0.0.1:8000/health`

### 2-2. Call `/chat` with `jd_text`

- [ ] Run:

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

- [ ] Validate:
  - [ ] HTTP 200
  - [ ] Required fields present
  - [ ] Korean response
  - [ ] `references` present

### 2-3. Error handling (optional)

- [ ] Confirm 400/500 guidance messages are clear for invalid config/input

## 3) Streamlit E2E

### 3-1. Start UI

- [ ] Run: `python scripts/run_streamlit.py`
- [ ] Open: `http://localhost:8501`

### 3-2. Input/run scenario with JD

- [ ] Enter query and choose target role
- [ ] Fill `JD/공고 텍스트(선택)` (or upload JD file)
- [ ] Fill `이력서 텍스트(선택)` (or upload resume file)
- [ ] Click `에이전트 실행`

- [ ] Validate:
  - [ ] Result cards rendered (summary/resume/interview/plan/references)
  - [ ] Input history stores run
  - [ ] `다시 불러오기` restores query/role/JD/resume/result
  - [ ] `새 대화 시작` resets state with new session
  - [ ] `max_chars` counters work in input boxes

### 3-3. Persistence across restart

- [ ] Stop Streamlit (`Ctrl+C`) and run again
- [ ] Confirm sidebar history is still available

## 4) Differentiation Metrics Automation

- [ ] Run:

```powershell
python scripts/evaluate_differentiation_metrics.py --cases data/eval/sample_queries.json
```

- [ ] Validate:
  - [ ] Routing accuracy printed
  - [ ] Reference inclusion rate printed
  - [ ] Plan quality rate printed
  - [ ] `[PASS] All thresholds satisfied.` printed when thresholds are met

## 5) Submission Artifacts

- [ ] `docs/evidence/agent_execution_log.md` updated
- [ ] `docs/evidence/agent_final_answer.json` updated
- [ ] Streamlit screenshot prepared: `docs/evidence/streamlit_main_capture.png`
- [ ] (Optional) Attach metrics run output capture

