"""Streamlit UI for JobPilot AI."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime
from uuid import uuid4

import streamlit as st
from filelock import FileLock

from src.common import JobPilotError
from src.config import load_settings
from src.ui.history_record import HISTORY_RECORD_VERSION, build_history_record, migrate_history_record
from src.ui.input_merge import merge_uploaded_text
from src.utils.file_extract import extract_text_from_upload
from src.utils.io import atomic_write_text
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


def _history_lock_path() -> Path:
    settings = load_settings()
    return settings.index_dir / "ui_input_history.json.lock"


def _retriever_meta_path() -> Path:
    settings = load_settings()
    return settings.index_dir / "retriever_meta.json"


def _load_retriever_meta() -> dict:
    path = _retriever_meta_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _load_persisted_history() -> list[dict]:
    path = _history_file_path()
    lock = FileLock(str(_history_lock_path()))
    if not path.exists():
        return []
    try:
        with lock:
            raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    normalized: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            normalized.append(_normalize_history_item(item))
    return normalized


def _normalize_history_item(item: dict) -> dict:
    return migrate_history_record(item)


def _save_persisted_history(history: list[dict]) -> None:
    path = _history_file_path()
    lock = FileLock(str(_history_lock_path()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        atomic_write_text(
            path,
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


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
    try:
        return extract_text_from_upload(name, data)
    except RuntimeError as exc:
        st.warning(str(exc))
        return ""
    except Exception as exc:
        st.warning(f"파일 파싱에 실패했습니다: {exc}")
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


def _upload_signature(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    digest = hashlib.sha1(data).hexdigest()
    return f"{uploaded_file.name}:{len(data)}:{digest}"


def _resolve_page_icon(settings) -> str | None:
    if settings.ui_page_icon_mode == "default":
        return None
    return settings.ui_page_icon_emoji or "💼"


def run() -> None:
    settings = load_settings()
    page_icon = _resolve_page_icon(settings)
    if page_icon is None:
        st.set_page_config(page_title="JobPilot AI", layout="wide")
    else:
        st.set_page_config(page_title="JobPilot AI", page_icon=page_icon, layout="wide")
    st.title("JobPilot AI - 취업/이직 멀티 에이전트 코파일럿")
    st.caption("Resume Agent + Interview Agent + RAG Agent")

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session-{uuid4().hex[:8]}"
    if "persist_history_enabled" not in st.session_state:
        st.session_state.persist_history_enabled = settings.ui_history_persist_enabled
    if "mask_pii_enabled" not in st.session_state:
        st.session_state.mask_pii_enabled = settings.ui_history_pii_mask_enabled
    if "input_history" not in st.session_state:
        st.session_state.input_history = (
            _load_persisted_history() if st.session_state.persist_history_enabled else []
        )
    if "_persist_history_prev" not in st.session_state:
        st.session_state._persist_history_prev = st.session_state.persist_history_enabled
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
    if "last_response_origin" not in st.session_state:
        st.session_state.last_response_origin = ""
    if "max_resume_chars" not in st.session_state:
        st.session_state.max_resume_chars = DEFAULT_MAX_RESUME_CHARS
    if "auto_compress_resume" not in st.session_state:
        st.session_state.auto_compress_resume = False
    if "resume_target_chars" not in st.session_state:
        st.session_state.resume_target_chars = 18000
    if "max_jd_chars" not in st.session_state:
        st.session_state.max_jd_chars = DEFAULT_MAX_JD_CHARS
    if "auto_compress_jd" not in st.session_state:
        st.session_state.auto_compress_jd = False
    if "jd_target_chars" not in st.session_state:
        st.session_state.jd_target_chars = 8000
    if "upload_apply_mode" not in st.session_state:
        st.session_state.upload_apply_mode = "덮어쓰기"
    if "last_resume_upload_sig" not in st.session_state:
        st.session_state.last_resume_upload_sig = ""
    if "last_jd_upload_sig" not in st.session_state:
        st.session_state.last_jd_upload_sig = ""
    if "show_debug_meta" not in st.session_state:
        st.session_state.show_debug_meta = False
    if "show_reference_metadata" not in st.session_state:
        st.session_state.show_reference_metadata = False
    if "history_storage_mode" not in st.session_state:
        st.session_state.history_storage_mode = settings.ui_history_storage_mode

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
                        f"이력서 길이: {item['resume_len']}자 | JD 길이: {item.get('jd_len', 0)}자 | "
                        f"run_id: {item.get('run_id', 'n/a')} | "
                        f"저장모드: {item.get('storage_mode', 'full')} | "
                        f"스키마 v{item.get('record_version', 1)}"
                    )
                    col_a, col_b = st.columns(2)
                    if col_a.button("다시 불러오기", key=f"load_{real_idx}", use_container_width=True):
                        st.session_state.session_id = item.get("session_id", st.session_state.session_id)
                        st.session_state.query_input = item.get("query", "")
                        st.session_state.target_role_input = item.get("target_role", "백엔드 개발자")
                        st.session_state.resume_text_input = item.get("resume_text", "")
                        st.session_state.jd_text_input = item.get("jd_text", "")
                        loaded_response = item.get("response") if isinstance(item.get("response"), dict) else {}
                        loaded_response = dict(loaded_response or {})
                        loaded_response["run_id"] = str(
                            loaded_response.get("run_id")
                            or item.get("run_id")
                            or f"history-{uuid4().hex[:8]}"
                        )
                        loaded_response["result_source"] = "history_load"
                        loaded_response["executed_at"] = str(
                            loaded_response.get("executed_at")
                            or datetime.now().isoformat(timespec="seconds")
                        )
                        st.session_state.last_response = loaded_response
                        st.session_state.last_response_origin = "history_load"
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
            st.checkbox(
                "긴 JD/공고 자동 압축(앞/뒤 핵심만 유지)",
                key="auto_compress_jd",
                help="JD/공고 텍스트가 길면 앞/뒤 중심으로 자동 압축합니다.",
            )
            st.number_input(
                "JD 자동 압축 목표 글자 수",
                min_value=1500,
                max_value=20000,
                step=500,
                key="jd_target_chars",
                disabled=not st.session_state.auto_compress_jd,
            )
            st.radio(
                "업로드 텍스트 반영 방식",
                ["덮어쓰기", "추가하기"],
                key="upload_apply_mode",
                help="파일 업로드 시 기존 입력값을 대체하거나 뒤에 이어붙일 수 있습니다.",
            )
        with st.expander("서비스/개인정보 설정", expanded=False):
            st.caption("운영 환경에서 저장 정책과 마스킹 정책을 조정할 수 있습니다.")
            retriever_meta = _load_retriever_meta()
            quality_warning = str(retriever_meta.get("category_quality_warning", "") or "").strip()
            if quality_warning:
                uncategorized_ratio = float(retriever_meta.get("uncategorized_ratio", 0.0) or 0.0)
                category_distribution = retriever_meta.get("category_distribution", {})
                if quality_warning == "uncategorized_ratio_high":
                    st.warning(
                        "지식 문서 카테고리 품질 경고: `uncategorized` 비율이 높습니다 "
                        f"({uncategorized_ratio:.1%}). "
                        "카테고리 폴더 정리 또는 메타데이터 검증(`validate_knowledge_metadata.py --strict`)을 권장합니다."
                    )
                elif quality_warning == "missing_required_categories":
                    st.warning(
                        "지식 문서 카테고리 품질 경고: 필수 카테고리 일부가 누락되었습니다. "
                        "데이터 거버넌스 검증(`validate_knowledge_metadata.py --strict`)을 권장합니다."
                    )
                if isinstance(category_distribution, dict) and category_distribution:
                    st.caption(f"카테고리 분포: {category_distribution}")
            st.checkbox(
                "실행 입력 기록 파일 저장 사용",
                key="persist_history_enabled",
                help="해제 시 디스크 기록은 로드/저장을 모두 수행하지 않으며, 현재 세션 메모리만 사용합니다.",
            )
            st.checkbox(
                "저장 전 이메일/전화번호 마스킹",
                key="mask_pii_enabled",
                help="실행 입력 기록 저장 전 민감정보를 [EMAIL_MASKED]/[PHONE_MASKED]로 치환합니다.",
            )
            st.radio(
                "기록 저장 모드",
                ["summary", "full"],
                key="history_storage_mode",
                help=(
                    "summary(기본): 텍스트 원문 대신 길이/해시/미리보기 + 요약 응답만 저장. "
                    "full: 입력/응답 원문 전체 저장."
                ),
            )
            if st.button("인덱스 사전 빌드/로드", use_container_width=True):
                with st.spinner("지식 인덱스 사전 빌드/로드 중입니다..."):
                    get_service.clear()
                    get_service()
                st.success("인덱스 사전 준비가 완료되었습니다.")
            if st.button("지식 문서 로드 실패 요약", use_container_width=True):
                meta = _load_retriever_meta()
                if not meta:
                    st.info("retriever_meta.json이 없어 아직 요약을 표시할 수 없습니다. 먼저 인덱스를 빌드/로드하세요.")
                else:
                    failures = meta.get("document_load_failures", [])
                    failure_count = int(meta.get("document_load_failure_count", 0) or 0)
                    if not isinstance(failures, list):
                        failures = []
                    if failure_count <= 0:
                        st.success("최근 인덱싱 기준 문서 로드 실패가 없습니다.")
                    else:
                        st.warning(f"최근 인덱싱에서 문서 로드 실패 {failure_count}건이 감지되었습니다.")
                        for idx, item in enumerate(failures[:20], start=1):
                            if not isinstance(item, dict):
                                continue
                            file_name = str(item.get("file", "unknown"))
                            error_text = str(item.get("error", "unknown error"))
                            st.markdown(f"- [{idx}] `{file_name}`")
                            st.caption(error_text)
                        if failure_count > len(failures):
                            st.caption(
                                f"메타에는 최대 {len(failures)}건만 저장되며, 전체 실패 건수는 {failure_count}건입니다."
                            )
            st.caption("첫 실행에서는 인덱스 생성으로 30~90초 이상 소요될 수 있습니다.")
        with st.expander("디버그/신뢰도 표시", expanded=False):
            st.checkbox(
                "라우팅 근거/근거 신뢰도 메타 표시",
                key="show_debug_meta",
                help="출력 하단에 route, routing_reason, rag_low_confidence 정보를 표시합니다.",
            )
            st.checkbox(
                "참고 출처 메타데이터 표시",
                key="show_reference_metadata",
                help="references에 포함된 수집일/출처 URL/큐레이터/라이선스 정보를 표시합니다.",
            )
            if st.session_state.persist_history_enabled != st.session_state._persist_history_prev:
                if st.session_state.persist_history_enabled:
                    st.session_state.input_history = _load_persisted_history()
                    st.success("기록 저장을 켜서 디스크 기록을 다시 로드했습니다.")
                else:
                    st.session_state.input_history = []
                    st.info("기록 저장 OFF: 디스크 기록 로드/저장을 중단하고 세션 메모리만 사용합니다.")
                st.session_state._persist_history_prev = st.session_state.persist_history_enabled
                st.rerun()

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
        type=["txt", "md", "csv", "pdf", "docx", "xlsx"],
        help="업로드하면 파일 내용이 아래 이력서 텍스트에 자동 반영됩니다.",
    )
    uploaded_jd = st.file_uploader(
        "JD/공고 파일 업로드(선택)",
        type=["txt", "md", "csv", "pdf", "docx", "xlsx"],
        help="업로드하면 파일 내용이 아래 JD/공고 텍스트에 자동 반영됩니다.",
    )

    extracted_resume = ""
    if uploaded_resume is not None:
        current_sig = _upload_signature(uploaded_resume)
        if current_sig != st.session_state.last_resume_upload_sig:
            extracted_resume = _extract_uploaded_text(uploaded_resume)
            if extracted_resume:
                if st.session_state.auto_compress_resume:
                    extracted_resume, compressed = _compress_long_text(
                        extracted_resume, int(st.session_state.resume_target_chars)
                    )
                    if compressed:
                        st.info("업로드 텍스트가 길어 자동 압축(앞/뒤 중심)되었습니다.")
                st.success(f"파일 분석 완료: {uploaded_resume.name}")
                st.session_state.resume_text_input = merge_uploaded_text(
                    st.session_state.resume_text_input,
                    extracted_resume,
                    st.session_state.upload_apply_mode,
                )
                st.session_state.last_resume_upload_sig = current_sig
    else:
        st.session_state.last_resume_upload_sig = ""

    if uploaded_jd is not None:
        current_sig = _upload_signature(uploaded_jd)
        if current_sig != st.session_state.last_jd_upload_sig:
            extracted_jd = _extract_uploaded_text(uploaded_jd)
            if extracted_jd:
                if st.session_state.auto_compress_jd:
                    extracted_jd, compressed = _compress_long_text(
                        extracted_jd,
                        int(st.session_state.jd_target_chars),
                    )
                    if compressed:
                        st.info("JD/공고 텍스트가 길어 자동 압축(앞/뒤 중심)되었습니다.")
                st.success(f"JD 파일 분석 완료: {uploaded_jd.name}")
                st.session_state.jd_text_input = merge_uploaded_text(
                    st.session_state.jd_text_input,
                    extracted_jd,
                    st.session_state.upload_apply_mode,
                )
                st.session_state.last_jd_upload_sig = current_sig
    else:
        st.session_state.last_jd_upload_sig = ""

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

    if not (st.session_state.resume_text_input or "").strip():
        st.info(
            "이력서 텍스트가 비어 있습니다. 이력서 전용 요청은 실행 시 계획 중심(예: plan_only)으로 "
            "자동 조정될 수 있습니다."
        )
    if not (st.session_state.jd_text_input or "").strip():
        st.caption(
            "JD/공고 텍스트가 없으면 공고-이력서 갭 분석의 구체성이 낮아질 수 있습니다."
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

        run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
        response_payload = response.model_dump()
        response_payload["run_id"] = run_id
        response_payload["result_source"] = "new_run"
        response_payload["executed_at"] = datetime.now().isoformat(timespec="seconds")
        record = build_history_record(
            session_id=st.session_state.session_id,
            query=query,
            target_role=target_role,
            resume_text=resume_text_for_run,
            jd_text=jd_text_for_run,
            response_payload=response_payload,
            run_id=run_id,
            storage_mode=st.session_state.history_storage_mode,
        )
        if int(record.get("record_version", 0) or 0) < HISTORY_RECORD_VERSION:
            record["record_version"] = HISTORY_RECORD_VERSION
        if st.session_state.mask_pii_enabled:
            record = mask_pii_payload(record)
        st.session_state.input_history.append(record)
        if st.session_state.persist_history_enabled:
            _save_persisted_history(st.session_state.input_history)
        st.session_state.last_response = response_payload
        st.session_state.last_response_origin = "new_run"
        # Rerun to refresh sidebar history immediately after append.
        st.rerun()

    if st.session_state.last_response:
        latest = st.session_state.last_response
        route = str(latest.get("route", "")).lower()
        run_id = str(latest.get("run_id", "") or "n/a")
        result_source = str(latest.get("result_source", "") or st.session_state.last_response_origin or "n/a")
        executed_at = str(latest.get("executed_at", "") or "").strip()
        source_label = "새 실행 결과" if result_source == "new_run" else "히스토리 불러오기"
        time_label = f" | 시각: {executed_at}" if executed_at else ""
        st.success(f"분석 완료 - {source_label} | run_id: {run_id}{time_label}")
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
            elif route in {"resume_only", "plan_only"}:
                st.caption("요청 라우트에 따라 면접 준비 섹션은 생략되었습니다.")

        plan_items = latest.get("two_week_plan", []) or []
        if plan_items:
            st.subheader("2주 실행 계획")
            for item in plan_items:
                st.markdown(f"- {item}")
            gap_notice = str(latest.get("input_gap_notice", "") or "").strip()
            if gap_notice.lower() in {"none", "null", "nan"}:
                gap_notice = ""
            if gap_notice:
                st.info(gap_notice)
        elif route in {"resume_only", "interview_only"}:
            st.caption("요청 라우트에 따라 2주 실행 계획 섹션은 생략되었습니다.")

        if (not (latest.get("resume_improvements", []) or [])) and route in {"interview_only", "plan_only"}:
            st.caption("요청 라우트에 따라 이력서 개선 섹션은 생략되었습니다.")

        references = latest.get("references", []) or []
        if references:
            st.subheader("참고 출처")
            for item in references:
                if isinstance(item, dict):
                    st.markdown(
                        "- "
                        f"[{item.get('rank', '?')}] {item.get('source', 'unknown')} "
                        f"(chunk={item.get('chunk_id', 'na')}, location={item.get('location', 'n/a')}, "
                        f"score={item.get('score', 0.0)})"
                    )
                    snippet = str(item.get("snippet", "")).strip()
                    if snippet:
                        st.caption(f"snippet: {snippet}")
                    if st.session_state.show_reference_metadata:
                        collected_at = str(item.get("collected_at", "") or "").strip()
                        source_url = str(item.get("source_url", "") or "").strip()
                        curator = str(item.get("curator", "") or "").strip()
                        license_text = str(item.get("license", "") or "").strip()
                        if collected_at:
                            st.caption(f"collected_at: {collected_at}")
                        if source_url:
                            st.markdown(f"[source_url]({source_url})")
                        if curator:
                            st.caption(f"curator: {curator}")
                        if license_text:
                            st.caption(f"license: {license_text}")
                    if st.session_state.show_debug_meta:
                        breakdown = item.get("score_breakdown")
                        if isinstance(breakdown, dict) and breakdown:
                            st.caption(f"score_breakdown: {breakdown}")
                else:
                    st.markdown(f"- {item}")
        if st.session_state.show_debug_meta:
            with st.expander("라우팅/신뢰도 메타", expanded=False):
                st.write(f"- route: `{latest.get('route', 'n/a')}`")
                st.write(f"- routing_reason: {latest.get('routing_reason', 'n/a')}")
                st.write(f"- rag_low_confidence: `{latest.get('rag_low_confidence', 'n/a')}`")
                st.write(f"- cached_state_hit: `{latest.get('cached_state_hit', False)}`")
                node_status = latest.get("node_status")
                if isinstance(node_status, dict):
                    st.markdown("- node_status:")
                    for node_name in ("supervisor", "rag", "resume", "interview", "plan", "synthesis"):
                        item = node_status.get(node_name)
                        if not isinstance(item, dict):
                            continue
                        status = str(item.get("status", "ok")).strip() or "ok"
                        code = str(item.get("error_code", "") or "").strip()
                        detail = str(item.get("detail", "") or "").strip()
                        line = f"  - `{node_name}`: `{status}`"
                        if code:
                            line += f", code=`{code}`"
                        if detail:
                            line += f", detail={detail}"
                        st.write(line)


if __name__ == "__main__":
    run()
