# JobPilot AI - Final Project

JobPilot AI is an end-to-end multi-agent service for job preparation and career transition support.

## What Is Implemented

- **Prompt Engineering**: role-based prompts for Supervisor/Resume/Interview agents
- **Multi-Agent (LangGraph)**: Supervisor -> RAG -> Resume Agent -> Interview Agent -> Synthesis
- **Agent Autonomy Policy**: Resume/Interview agents apply local fallback policy when resume text is missing
- **RAG**: document loading (`.txt/.md/.csv/.pdf/.docx/.xlsx`), chunking, FAISS vector retrieval + BM25 keyword retrieval
- **RAG Quality**: index persistence (`data/index/faiss`), metadata-aware context (page/paragraph/sheet/row), route-aware category filtering, dedicated rerank layer
- **Structured Output**: final response generated with Pydantic schema
- **Resilient Structured Output**: node-level degrade fallback for Resume/Interview/Synthesis parsing failures
- **Graph Checkpointing**: LangGraph checkpointer wired with `thread_id=session_id` for execution-state restoration baseline
- **Concurrency Guard**: SessionMemory read/write sections protected with file lock (`session_memory.json.lock`)
- **Service Packaging**: FastAPI backend and Streamlit UI
- **Resilience**: API-level exception handling for user-friendly error responses
- **Modular Code**: reusable modules under `src/` for config/retrieval/workflow/ui/api

## Folder Layout

- `data/knowledge`: RAG source documents (`.txt/.md/.csv/.pdf/.docx/.xlsx`)
  - recommended categories: `job_postings/`, `jd/`, `interview_guides/`, `portfolio_examples/`
- `docs`: planning/design docs for submission (`step2_planning_design.md`, `step3_service_development.md`)
- `scripts`: runnable launch scripts
- `src/config`: `.env` loading and model clients
- `src/retrieval`: data loading, chunking, hybrid retrieval
- `src/agents`: tools and structured response schema
- `src/workflow`: LangGraph workflow and service layer
- `src/api`: FastAPI app
- `src/ui`: Streamlit app

## Submission Docs

- [Step2 - 기획 및 설계](./docs/step2_planning_design.md)
- [Step3 - 서비스 개발](./docs/step3_service_development.md)
- [System Architecture Image](./docs/images/system_architecture.png)
- [Service Flow Image](./docs/images/service_flow_sequence.png)
- [Evidence - Agent Execution Log](./docs/evidence/agent_execution_log.md)
- [Evidence - Final Answer JSON](./docs/evidence/agent_final_answer.json)
- [Evidence Generator Script](./scripts/generate_submission_evidence.py)

## Environment Setup

1. Install dependencies:

```powershell
pip install -r requirements-final.txt
```

2. Create `.env` in this folder:

```powershell
Copy-Item .env.example .env
```

3. Fill required values in `.env`:

- `AOAI_ENDPOINT`
- `AOAI_API_KEY`
- `AOAI_DEPLOY_GPT4O`
- `AOAI_DEPLOY_EMBED_ADA` (or `AOAI_EMBEDDING_DEPLOYMENT`)
- `AOAI_API_VERSION`
- `MEMORY_MAX_SESSIONS` (optional, default `200`)
- `MEMORY_TTL_SECONDS` (optional, default `86400`)
- `INDEX_FORCE_REBUILD` (optional, default `false`; set `true` to force rebuild FAISS/chunk cache)

## Quick Start (Deploy View)

1) `.env` 준비  
- `Copy-Item .env.example .env` 실행 후 Azure OpenAI 값을 채웁니다.

2) `data/knowledge` 예시 문서 준비  
- 최소 1개 이상 문서를 아래 카테고리 중 하나에 넣습니다.
  - `data/knowledge/job_postings/`
  - `data/knowledge/jd/`
  - `data/knowledge/interview_guides/`
  - `data/knowledge/portfolio_examples/`
- 지원 포맷: `.txt/.md/.csv/.pdf/.docx/.xlsx`

3) 실행  
- API: `python scripts/run_api.py`  
- UI: `python scripts/run_streamlit.py`  
- 접속: `http://localhost:8501`

## Run Options

### 1) CLI (quick test)

```powershell
python main.py --query "백엔드 이직을 위해 이력서 개선 포인트와 2주 계획을 작성해줘" --target-role "백엔드 개발자"
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

- UI supports resume file upload: `.txt`, `.md`, `.pdf`, `.docx`, `.xlsx`
- Session ID is managed internally; use `새 대화 시작` button for a fresh session
- Sidebar provides input history view/delete for previous agent runs
- `다시 불러오기` restores query, target role, resume text, and previous agent result
- RAG index is cached on disk; first run can be slower, later runs are faster due to index reuse
- Large text guardrails are enabled: input-box real-time counters (`max_chars`) + hard limits for query/resume, and resume auto-compression options in sidebar

## Troubleshooting

- `ModuleNotFoundError: No module named 'src'`
  - Always run from `final-project` root.
  - Use `python scripts/run_api.py` instead of direct uvicorn commands.
- `Missing environment variables: AOAI_ENDPOINT`
  - Check `final-project/.env` and required keys.
- Retrieval/index seems stale after knowledge file update
  - Set `INDEX_FORCE_REBUILD=true` in `.env` and rerun once.
  - Then reset to `false` for normal cached startup.
- API returns `400` or `500` on `/chat`
  - `400`: input/config issues (e.g., missing env variables)
  - `500`: runtime/model/retrieval issues; check terminal error details
- Structured output parse instability (intermittent)
  - Workflow has fallback degrade responses in Resume/Interview/Synthesis nodes.
  - Verify model/deployment health and keep prompts in Korean-only mode for consistency.
- Concurrent write concerns for session memory/index files
  - `SessionMemory` uses file lock for write/read-update-write sections.
  - For heavier multi-user production, prefer external store (e.g., SQLite/Redis) with worker separation.