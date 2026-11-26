from scripts.qwen35_cap_compare import completed_answer


def test_completion_requires_final_answer_and_stop():
    assert completed_answer('reason</think>\n\n', 'stop') is None
    assert completed_answer('reason</think>\nanswer', 'length') is None
    assert completed_answer('reason</think>\nanswer', 'stop')['content'] == 'answer'
