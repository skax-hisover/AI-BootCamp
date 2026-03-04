"""Streamlit UI for JobPilot AI."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.common import JobPilotError
from src.config import load_settings
from src.utils.pii import mask_pii_payload
from src.workflow import ChatRequest, JobPilotService

MAX_QUERY_CHARS = 1500
DEFAULT_MAX_RESUME_CHARS = 30000
DEFAULT_MAX_JD_CHARS = 12000


@st.cache_resource(show_spinner="지식 인덱스 생성/로딩 중입니다. 첫 실행은 다소 시간이 걸릴 수 있습니다...")
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


def _show_error_by_code(error_code: str, detail: str) -> None:
    lower = (error_code or "").upper()
    if lower == "KNOWLEDGE_EMPTY":
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
    if lower == "CONFIG_MISSING_ENV":
        st.error("필수 환경변수가 누락되어 실행할 수 없습니다.")
        st.info("`final-project/.env` 파일의 AOAI 관련 항목을 확인하세요.")
        with st.expander("필수 환경변수 목록", expanded=False):
            st.code(
                "AOAI_ENDPOINT\nAOAI_API_KEY\nAOAI_DEPLOY_GPT4O\nAOAI_DEPLOY_EMBED_ADA\nAOAI_API_VERSION"
            )
        return
    st.error(f"실행 오류({error_code}): {detail}")


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


def _compress_long_text(text: str, target_chars: int) -> tuple[str, bool]:
    if len(text) <= target_chars:
        return text, False
    if target_chars < 200:
        return text[:target_chars], True
    head = int(target_chars * 0.7)
    tail = max(target_chars - head, 0)
    compressed = f"{text[:head]}\n\n... [중간 내용 생략] ...\n\n{text[-tail:]}"
    return compressed, True


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
    if "jd_text_input" not in st.session_state:
        st.session_state.jd_text_input = ""
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "max_resume_chars" not in st.session_state:
        st.session_state.max_resume_chars = DEFAULT_MAX_RESUME_CHARS
    if "auto_compress_resume" not in st.session_state:
        st.session_state.auto_compress_resume = False
    if "resume_target_chars" not in st.session_state:
        st.session_state.resume_target_chars = 18000
    if "max_jd_chars" not in st.session_state:
        st.session_state.max_jd_chars = DEFAULT_MAX_JD_CHARS
    if "persist_history_enabled" not in st.session_state:
        st.session_state.persist_history_enabled = True
    if "mask_pii_enabled" not in st.session_state:
        st.session_state.mask_pii_enabled = False

    with st.sidebar:
        st.subheader("입력 설정")
        st.caption("세션은 내부적으로 자동 관리됩니다.")
        if st.button("새 대화 시작", use_container_width=True):
            st.session_state.session_id = f"session-{uuid4().hex[:8]}"
            st.session_state.resume_text_input = ""
            st.session_state.jd_text_input = ""
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
                        f"직무: {item['target_role']} | 세션: {item['session_id']} | "
                        f"이력서 길이: {item['resume_len']}자 | JD 길이: {item.get('jd_len', 0)}자"
                    )
                    col_a, col_b = st.columns(2)
                    if col_a.button("다시 불러오기", key=f"load_{real_idx}", use_container_width=True):
                        st.session_state.session_id = item.get("session_id", st.session_state.session_id)
                        st.session_state.query_input = item.get("query", "")
                        st.session_state.target_role_input = item.get("target_role", "백엔드 개발자")
                        st.session_state.resume_text_input = item.get("resume_text", "")
                        st.session_state.jd_text_input = item.get("jd_text", "")
                        st.session_state.last_response = item.get("response")
                        st.info("질문/직무/JD/이력서/결과를 복원했습니다.")
                        st.rerun()
                    if col_b.button("삭제", key=f"delete_{real_idx}", use_container_width=True):
                        del st.session_state.input_history[real_idx]
                        if st.session_state.persist_history_enabled:
                            _save_persisted_history(st.session_state.input_history)
                        st.rerun()
                if st.button("기록 전체 삭제", type="secondary", use_container_width=True):
                    st.session_state.input_history = []
                    if st.session_state.persist_history_enabled:
                        _save_persisted_history(st.session_state.input_history)
                    st.rerun()

        st.selectbox(
            "목표 직무",
            ["백엔드 개발자", "데이터 분석가", "PM"],
            key="target_role_input",
        )
        with st.expander("대용량 입력 방어 설정", expanded=False):
            st.number_input(
                "이력서 최대 글자 수(하드 제한)",
                min_value=5000,
                max_value=50000,
                step=1000,
                key="max_resume_chars",
                help="이 값을 초과하면 실행 전에 차단됩니다.",
            )
            st.checkbox(
                "긴 이력서 자동 압축(앞/뒤 핵심만 유지)",
                key="auto_compress_resume",
                help="업로드/입력된 텍스트가 길면 앞/뒤 중심으로 자동 압축합니다.",
            )
            st.number_input(
                "자동 압축 목표 글자 수",
                min_value=2000,
                max_value=30000,
                step=500,
                key="resume_target_chars",
                disabled=not st.session_state.auto_compress_resume,
            )
            st.number_input(
                "JD/공고 최대 글자 수(하드 제한)",
                min_value=2000,
                max_value=30000,
                step=500,
                key="max_jd_chars",
                help="JD/공고 텍스트 입력 제한입니다.",
            )
        with st.expander("서비스/개인정보 설정", expanded=False):
            st.caption("운영 환경에서 저장 정책과 마스킹 정책을 조정할 수 있습니다.")
            st.checkbox(
                "실행 입력 기록 파일 저장 사용",
                key="persist_history_enabled",
                help="해제 시 실행 기록은 현재 세션 메모리에만 유지되고 파일에는 저장되지 않습니다.",
            )
            st.checkbox(
                "저장 전 이메일/전화번호 마스킹",
                key="mask_pii_enabled",
                help="실행 입력 기록 저장 전 민감정보를 [EMAIL_MASKED]/[PHONE_MASKED]로 치환합니다.",
            )
            if st.button("인덱스 사전 빌드/로드", use_container_width=True):
                with st.spinner("지식 인덱스 사전 빌드/로드 중입니다..."):
                    get_service.clear()
                    get_service()
                st.success("인덱스 사전 준비가 완료되었습니다.")
            st.caption("첫 실행에서는 인덱스 생성으로 30~90초 이상 소요될 수 있습니다.")

    if len(st.session_state.query_input or "") > MAX_QUERY_CHARS:
        st.session_state.query_input = (st.session_state.query_input or "")[:MAX_QUERY_CHARS]
        st.warning(f"질문/요청은 최대 {MAX_QUERY_CHARS:,}자까지 입력할 수 있어 자동으로 잘렸습니다.")

    st.text_area(
        "질문/요청",
        placeholder="예) 백엔드 이직을 위해 이력서 개선 포인트와 2주 계획을 만들어줘",
        key="query_input",
        max_chars=MAX_QUERY_CHARS,
        help=f"최대 {MAX_QUERY_CHARS:,}자까지 입력할 수 있습니다. 입력 중 글자 수가 실시간 표시됩니다.",
    )
    st.caption(
        f"질문/요청은 최대 {MAX_QUERY_CHARS:,}자까지 입력 가능합니다. "
        "입력창 내부 카운터가 실시간 기준입니다."
    )
    uploaded_resume = st.file_uploader(
        "이력서 파일 업로드(선택)",
        type=["txt", "md", "pdf", "docx", "xlsx"],
        help="업로드하면 파일 내용이 아래 이력서 텍스트에 자동 반영됩니다.",
    )
    uploaded_jd = st.file_uploader(
        "JD/공고 파일 업로드(선택)",
        type=["txt", "md", "pdf", "docx", "xlsx"],
        help="업로드하면 파일 내용이 아래 JD/공고 텍스트에 자동 반영됩니다.",
    )

    extracted_resume = ""
    if uploaded_resume is not None:
        extracted_resume = _extract_uploaded_text(uploaded_resume)
        if extracted_resume:
            if st.session_state.auto_compress_resume:
                extracted_resume, compressed = _compress_long_text(
                    extracted_resume, int(st.session_state.resume_target_chars)
                )
                if compressed:
                    st.info("업로드 텍스트가 길어 자동 압축(앞/뒤 중심)되었습니다.")
            st.success(f"파일 분석 완료: {uploaded_resume.name}")
            st.session_state.resume_text_input = extracted_resume

    if uploaded_jd is not None:
        extracted_jd = _extract_uploaded_text(uploaded_jd)
        if extracted_jd:
            st.success(f"JD 파일 분석 완료: {uploaded_jd.name}")
            st.session_state.jd_text_input = extracted_jd

    max_resume_chars = int(st.session_state.max_resume_chars)
    if len(st.session_state.resume_text_input or "") > max_resume_chars:
        st.session_state.resume_text_input = (st.session_state.resume_text_input or "")[
            :max_resume_chars
        ]
        st.warning(
            f"이력서 텍스트는 최대 {max_resume_chars:,}자까지 입력할 수 있어 자동으로 잘렸습니다."
        )

    resume_text = st.text_area(
        "이력서 텍스트(선택)",
        height=220,
        placeholder="여기에 직접 붙여넣거나, 위에서 파일 업로드를 사용하세요.",
        key="resume_text_input",
        max_chars=max_resume_chars,
        help=f"최대 {max_resume_chars:,}자까지 입력할 수 있습니다. 입력 중 글자 수가 실시간 표시됩니다.",
    )
    resume_text = st.session_state.resume_text_input
    st.caption(
        f"이력서 텍스트는 최대 {max_resume_chars:,}자까지 입력 가능합니다. "
        "입력창 내부 카운터가 실시간 기준입니다."
    )
    max_jd_chars = int(st.session_state.max_jd_chars)
    if len(st.session_state.jd_text_input or "") > max_jd_chars:
        st.session_state.jd_text_input = (st.session_state.jd_text_input or "")[:max_jd_chars]
        st.warning(
            f"JD/공고 텍스트는 최대 {max_jd_chars:,}자까지 입력할 수 있어 자동으로 잘렸습니다."
        )
    st.text_area(
        "JD/공고 텍스트(선택)",
        height=180,
        placeholder="채용공고/JD 텍스트를 붙여넣거나 위에서 파일 업로드를 사용하세요.",
        key="jd_text_input",
        max_chars=max_jd_chars,
        help=f"최대 {max_jd_chars:,}자까지 입력할 수 있습니다. 입력 중 글자 수가 실시간 표시됩니다.",
    )
    jd_text = st.session_state.jd_text_input
    st.caption(
        f"JD/공고 텍스트는 최대 {max_jd_chars:,}자까지 입력 가능합니다. "
        "입력창 내부 카운터가 실시간 기준입니다."
    )

    if st.button(
        "에이전트 실행",
        type="primary",
        use_container_width=True,
    ):
        query = st.session_state.query_input
        target_role = st.session_state.target_role_input
        resume_text_for_run = resume_text
        jd_text_for_run = jd_text
        if not query.strip():
            st.warning("질문/요청을 입력해 주세요.")
            return
        if len(query) > MAX_QUERY_CHARS:
            st.warning(f"질문/요청은 최대 {MAX_QUERY_CHARS:,}자까지 입력할 수 있습니다.")
            return
        if len(resume_text_for_run) > max_resume_chars:
            if st.session_state.auto_compress_resume:
                resume_text_for_run, compressed = _compress_long_text(
                    resume_text_for_run, int(st.session_state.resume_target_chars)
                )
                if compressed:
                    st.session_state.resume_text_input = resume_text_for_run
                    st.info("실행 전 긴 이력서 텍스트를 자동 압축했습니다.")
            if len(resume_text_for_run) > max_resume_chars:
                st.error(
                    f"이력서 텍스트 길이({len(resume_text_for_run):,}자)가 제한"
                    f"({max_resume_chars:,}자)을 초과했습니다."
                )
                st.info("텍스트를 줄이거나 '긴 이력서 자동 압축' 옵션을 켜고 다시 시도하세요.")
                return

        try:
            with st.spinner("멀티 에이전트가 분석 중입니다..."):
                service = get_service()
                response = service.run(
                    ChatRequest(
                        session_id=st.session_state.session_id,
                        user_query=query,
                        target_role=target_role,
                        resume_text=resume_text_for_run,
                        jd_text=jd_text_for_run,
                    )
                )
        except JobPilotError as exc:
            _show_error_by_code(exc.error_code, exc.detail)
            return
        except Exception as exc:
            st.error(f"서비스 실행 중 예기치 못한 오류가 발생했습니다: {exc}")
            return

        record = {
            "session_id": st.session_state.session_id,
            "query": query.strip(),
            "target_role": target_role,
            "resume_text": resume_text_for_run,
            "jd_text": jd_text_for_run,
            "resume_len": len(resume_text_for_run),
            "jd_len": len(jd_text_for_run),
            "response": response.model_dump(),
        }
        if st.session_state.mask_pii_enabled:
            record = mask_pii_payload(record)
        st.session_state.input_history.append(record)
        if st.session_state.persist_history_enabled:
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
            resume_items = latest.get("resume_improvements", []) or []
            if resume_items:
                st.subheader("이력서 개선")
                for item in resume_items:
                    st.markdown(f"- {item}")
        with col2:
            interview_items = latest.get("interview_preparation", []) or []
            if interview_items:
                st.subheader("면접 준비")
                for item in interview_items:
                    st.markdown(f"- {item}")

        plan_items = latest.get("two_week_plan", []) or []
        if plan_items:
            st.subheader("2주 실행 계획")
            for item in plan_items:
                st.markdown(f"- {item}")

        references = latest.get("references", []) or []
        if references:
            st.subheader("참고 출처")
            for item in references:
                st.markdown(f"- {item}")


if __name__ == "__main__":
    run()
