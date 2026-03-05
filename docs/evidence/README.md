# 제출 산출물(Evidence)

최종 제출 시 아래 3개 파일을 포함하면 됩니다.

1. `streamlit_main_capture.png`  
   - Streamlit 메인 화면(입력 + 출력 결과가 함께 보이도록) 캡처 파일
2. `agent_execution_log.md`  
   - Supervisor 라우팅 + RAG 근거(refs) 포함 실행 로그
3. `agent_final_answer.json`  
   - 구조화 최종 결과(JSON)

## 생성 방법

### 1) 에이전트 실행 로그/결과 JSON 자동 생성

프로젝트 루트(`final-project`)에서:

```powershell
python scripts/generate_submission_evidence.py
```

필요 시 입력 커스터마이즈:

```powershell
python scripts/generate_submission_evidence.py --query "이력서 개선 포인트 5개만 제시해줘" --target-role "백엔드 개발자"
```

### 2) Streamlit 화면 캡처 저장

1. `python scripts/run_streamlit.py` 실행
2. `http://localhost:8501/` 접속
3. 질문 실행 후 결과 카드가 보이는 화면 캡처
4. 캡처 이미지를 `docs/evidence/streamlit_main_capture.png`로 저장

### 3) 차별성 지표 자동검증 로그(최신 기준)

- 최신 로그 파일: `docs/evidence/metrics_run_output.txt` (route 케이스 `resume_only/interview_only/plan_only/full` 4개 기준, `[PASS] All thresholds satisfied.` 확인)
