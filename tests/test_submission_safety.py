from scripts.check_env_submission_safety import _dangerous_faiss_setting_warning, _is_true_like


def test_is_true_like_variants() -> None:
    assert _is_true_like("true")
    assert _is_true_like("YES")
    assert _is_true_like("1")
    assert not _is_true_like("false")


def test_dangerous_faiss_setting_warning() -> None:
    warning = _dangerous_faiss_setting_warning({"FAISS_ALLOW_DANGEROUS_DESERIALIZATION": "true"})
    assert warning is not None
    assert "FAISS_ALLOW_DANGEROUS_DESERIALIZATION=true" in warning
    assert _dangerous_faiss_setting_warning({"FAISS_ALLOW_DANGEROUS_DESERIALIZATION": "false"}) is None
