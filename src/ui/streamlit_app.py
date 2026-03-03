"""Streamlit UI for JobPilot AI."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.config import load_settings
from src.workflow import ChatRequest, JobPilotService


@st.cache_resource
def get_service() -> JobPilotService:
    return JobPilotService()


def _history_file_path() -> Path:
    settings = load_settings()
    return settings.index_dir / "ui_input_history.json"


def _load_persisted_history() -> list[dict]:
    path = _history_file_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return raw if isinstance(raw, list) else []


def _save_persisted_history(history: list[dict]) -> None:
    path = _history_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_uploaded_text(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.getvalue()

    if name.endswith(".txt") or name.endswith(".md"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("cp949", errors="ignore")

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ModuleNotFoundError:
            st.warning("PDF 파싱을 위해 pypdf 설치가 필요합니다.")
            return ""
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    if name.endswith(".docx"):
        try:
            from docx import Document as DocxDocument
        except ModuleNotFoundError:
            st.warning("DOCX 파싱을 위해 python-docx 설치가 필요합니다.")
            return ""
        doc = DocxDocument(BytesIO(data))
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(lines).strip()

    if name.endswith(".xlsx"):
        try:
            xls = pd.ExcelFile(BytesIO(data))
        except Exception as exc:
            st.warning(f"XLSX 파일 파싱에 실패했습니다: {exc}")
            return ""
        blocks: list[str] = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet).fillna("")
            rows = [" | ".join(map(str, row)) for row in df.values.tolist()]
            blocks.append(f"[sheet: {sheet}]\n" + "\n".join(rows))
        return "\n\n".join(blocks).strip()

    st.warning("지원하지 않는 파일 형식입니다. txt/md/pdf/docx/xlsx 파일을 업로드해 주세요.")
    return ""


def run() -> None:
    st.set_page_config(page_title="JobPilot AI", page_icon=":briefcase:", layout="wide")
    st.title("JobPilot AI - 취업/이직 멀티 에이전트 코파일럿")
    st.caption("Resume Agent + Interview Agent + RAG Agent")

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session-{uuid4().hex[:8]}"
    if "input_history" not in st.session_state:
        st.session_state.input_history = _load_persisted_history()
    if "resume_text_input" not in st.session_state:
        st.session_state.resume_text_input = ""
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    if "target_role_input" not in st.session_state:
        st.session_state.target_role_input = "백엔드 개발자"
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    with st.sidebar:
        st.subheader("입력 설정")
        st.caption("세션은 내부적으로 자동 관리됩니다.")
        if st.button("새 대화 시작", use_container_width=True):
            st.session_state.session_id = f"session-{uuid4().hex[:8]}"
            st.session_state.resume_text_input = ""
            st.session_state.query_input = ""
            st.session_state.target_role_input = "백엔드 개발자"
            st.session_state.last_response = None
            st.success("새 세션으로 전환되었습니다.")
            st.rerun()

        with st.expander("실행 입력 기록(조회/삭제)", expanded=False):
            history = st.session_state.input_history
            if not history:
                st.caption("아직 실행 기록이 없습니다.")
            else:
                for idx, item in enumerate(reversed(history), start=1):
                    real_idx = len(history) - idx
                    st.markdown(f"**{idx}. {item['query']}**")
                    st.caption(
                        f"직무: {item['target_role']} | 세션: {item['session_id']} | 이력서 길이: {item['resume_len']}자"
                    )
                    col_a, col_b = st.columns(2)
                    if col_a.button("다시 불러오기", key=f"load_{real_idx}", use_container_width=True):
                        st.session_state.session_id = item.get("session_id", st.session_state.session_id)
                        st.session_state.query_input = item.get("query", "")
                        st.session_state.target_role_input = item.get("target_role", "백엔드 개발자")
                        st.session_state.resume_text_input = item.get("resume_text", "")
                        st.session_state.last_response = item.get("response")
                        st.info("질문/직무/이력서/결과를 복원했습니다.")
                        st.rerun()
                    if col_b.button("삭제", key=f"delete_{real_idx}", use_container_width=True):
                        del st.session_state.input_history[real_idx]
                        _save_persisted_history(st.session_state.input_history)
                        st.rerun()
                if st.button("기록 전체 삭제", type="secondary", use_container_width=True):
                    st.session_state.input_history = []
                    _save_persisted_history(st.session_state.input_history)
                    st.rerun()

        st.selectbox(
            "목표 직무",
            ["백엔드 개발자", "데이터 분석가", "PM"],
            key="target_role_input",
        )

    st.text_area(
        "질문/요청",
        placeholder="예) 백엔드 이직을 위해 이력서 개선 포인트와 2주 계획을 만들어줘",
        key="query_input",
    )
    uploaded_resume = st.file_uploader(
        "이력서 파일 업로드(선택)",
        type=["txt", "md", "pdf", "docx", "xlsx"],
        help="업로드하면 파일 내용이 아래 이력서 텍스트에 자동 반영됩니다.",
    )

    extracted_resume = ""
    if uploaded_resume is not None:
        extracted_resume = _extract_uploaded_text(uploaded_resume)
        if extracted_resume:
            st.success(f"파일 분석 완료: {uploaded_resume.name}")
            st.session_state.resume_text_input = extracted_resume

    resume_text = st.text_area(
        "이력서 텍스트(선택)",
        height=220,
        placeholder="여기에 직접 붙여넣거나, 위에서 파일 업로드를 사용하세요.",
        key="resume_text_input",
    )
    resume_text = st.session_state.resume_text_input

    if st.button("에이전트 실행", type="primary", use_container_width=True):
        query = st.session_state.query_input
        target_role = st.session_state.target_role_input
        if not query.strip():
            st.warning("질문/요청을 입력해 주세요.")
            return

        try:
            with st.spinner("멀티 에이전트가 분석 중입니다..."):
                service = get_service()
                response = service.run(
                    ChatRequest(
                        session_id=st.session_state.session_id,
                        user_query=query,
                        target_role=target_role,
                        resume_text=resume_text,
                    )
                )
        except ValueError as exc:
            message = str(exc)
            lower = message.lower()
            if "no knowledge documents found" in lower:
                settings = load_settings()
                st.error("지식 문서가 없어 RAG를 실행할 수 없습니다.")
                st.info(
                    "다음 경로에 예시 파일을 넣어주세요: "
                    f"`{settings.knowledge_dir}` (.txt/.md/.csv/.pdf/.docx/.xlsx)"
                )
                with st.expander("해결 가이드", expanded=False):
                    st.markdown(
                        "- 1) `data/knowledge` 아래에 최소 1개 문서를 추가하세요.\n"
                        "- 2) 예: `job_postings/sample_jd.md`, `interview_guides/backend_qna.txt`\n"
                        "- 3) 다시 `에이전트 실행`을 누르세요."
                    )
                return

            if "missing environment variables" in lower:
                st.error("필수 환경변수가 누락되어 실행할 수 없습니다.")
                st.info("`final-project/.env` 파일의 AOAI 관련 항목을 확인하세요.")
                with st.expander("필수 환경변수 목록", expanded=False):
                    st.code(
                        "AOAI_ENDPOINT\nAOAI_API_KEY\nAOAI_DEPLOY_GPT4O\nAOAI_DEPLOY_EMBED_ADA\nAOAI_API_VERSION"
                    )
                return

            st.error(f"실행 오류: {message}")
            return
        except Exception as exc:
            st.error(f"서비스 실행 중 예기치 못한 오류가 발생했습니다: {exc}")
            return

        st.session_state.input_history.append(
            {
                "session_id": st.session_state.session_id,
                "query": query.strip(),
                "target_role": target_role,
                "resume_text": resume_text,
                "resume_len": len(resume_text),
                "response": response.model_dump(),
            }
        )
        _save_persisted_history(st.session_state.input_history)
        st.session_state.last_response = response.model_dump()
        # Rerun to refresh sidebar history immediately after append.
        st.rerun()

    if st.session_state.last_response:
        latest = st.session_state.last_response
        st.success("분석 완료")
        st.subheader("요약")
        st.write(latest["summary"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("이력서 개선")
            for item in latest["resume_improvements"]:
                st.markdown(f"- {item}")
        with col2:
            st.subheader("면접 준비")
            for item in latest["interview_preparation"]:
                st.markdown(f"- {item}")

        st.subheader("2주 실행 계획")
        for item in latest["two_week_plan"]:
            st.markdown(f"- {item}")

        st.subheader("참고 출처")
        for item in latest["references"]:
            st.markdown(f"- {item}")


if __name__ == "__main__":
    run()
