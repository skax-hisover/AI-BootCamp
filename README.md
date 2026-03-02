# JobPilot AI - Final Project

JobPilot AI is an end-to-end multi-agent service for job preparation and career transition support.

## What Is Implemented

- **Prompt Engineering**: role-based prompts for Supervisor/Resume/Interview agents
- **Multi-Agent (LangGraph)**: Supervisor -> RAG -> Resume Agent -> Interview Agent -> Synthesis
- **RAG**: document loading (`.txt/.md/.csv/.pdf/.docx/.xlsx`), chunking, FAISS vector retrieval + BM25 keyword retrieval
- **Structured Output**: final response generated with Pydantic schema
- **Service Packaging**: FastAPI backend and Streamlit UI
- **Resilience**: API-level exception handling for user-friendly error responses
- **Modular Code**: reusable modules under `src/` for config/retrieval/workflow/ui/api

## Folder Layout

- `data/knowledge`: RAG source documents (`.txt/.md/.csv/.pdf/.docx/.xlsx`)
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

## Troubleshooting

- `ModuleNotFoundError: No module named 'src'`
  - Always run from `final-project` root.
  - Use `python scripts/run_api.py` instead of direct uvicorn commands.
- `Missing environment variables: AOAI_ENDPOINT`
  - Check `final-project/.env` and required keys.
- API returns `400` or `500` on `/chat`
  - `400`: input/config issues (e.g., missing env variables)
  - `500`: runtime/model/retrieval issues; check terminal error details
