import pytest

from frontend.streaming import typewriter_chunks


def test_typewriter_chunks_preserve_exact_answer_without_waiting():
    waits: list[float] = []
    chunks = list(
        typewriter_chunks(
            "员工应先提交申请。[1]",
            chunk_size=2,
            delay_seconds=0.01,
            sleeper=waits.append,
        )
    )

    assert "".join(chunks) == "员工应先提交申请。[1]"
    assert all(len(chunk) <= 2 for chunk in chunks)
    assert waits == [0.01] * len(chunks)


@pytest.mark.parametrize(
    ("chunk_size", "delay_seconds"),
    [(0, 0.01), (2, -0.01)],
)
def test_typewriter_chunks_reject_invalid_configuration(chunk_size, delay_seconds):
    with pytest.raises(ValueError):
        list(
            typewriter_chunks(
                "回答",
                chunk_size=chunk_size,
                delay_seconds=delay_seconds,
                sleeper=lambda _: None,
            )
        )
