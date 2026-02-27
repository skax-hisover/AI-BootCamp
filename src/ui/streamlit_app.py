"""Streamlit UI for JobPilot AI."""

from __future__ import annotations

import streamlit as st

from src.workflow import ChatRequest, JobPilotService


@st.cache_resource
def get_service() -> JobPilotService:
    return JobPilotService()


def run() -> None:
    st.set_page_config(page_title="JobPilot AI", page_icon=":briefcase:", layout="wide")
    st.title("JobPilot AI - 취업/이직 멀티 에이전트 코파일럿")
    st.caption("Resume Agent + Interview Agent + RAG Agent")

    if "session_id" not in st.session_state:
        st.session_state.session_id = "streamlit-default"

    with st.sidebar:
        st.subheader("입력 설정")
        target_role = st.selectbox(
            "목표 직무",
            ["백엔드 개발자", "데이터 분석가", "PM"],
            index=0,
        )
        session_id = st.text_input("세션 ID", value=st.session_state.session_id)
        st.session_state.session_id = session_id.strip() or "streamlit-default"

    query = st.text_area("질문/요청", placeholder="예) 백엔드 이직을 위해 이력서 개선 포인트와 2주 계획을 만들어줘")
    resume_text = st.text_area("이력서 텍스트(선택)", height=200)

    if st.button("에이전트 실행", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("질문/요청을 입력해 주세요.")
            return

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

        st.success("분석 완료")
        st.subheader("요약")
        st.write(response.summary)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("이력서 개선")
            for item in response.resume_improvements:
                st.markdown(f"- {item}")
        with col2:
            st.subheader("면접 준비")
            for item in response.interview_preparation:
                st.markdown(f"- {item}")

        st.subheader("2주 실행 계획")
        for item in response.two_week_plan:
            st.markdown(f"- {item}")

        st.subheader("참고 출처")
        for item in response.references:
            st.markdown(f"- {item}")


if __name__ == "__main__":
    run()
