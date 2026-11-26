from workloads.readout.real_readout_gsm8k import answer


def test_exact_final_numeric_line():
    assert answer("work\n#### +1,000.000") == "1000"
    assert answer("#### -0.00") == "0"
    assert answer("#### 1.00000000000000000000000000001") != "1"
    assert answer("#### 123\nmore reasoning") is None
    assert answer("#### 1,00") is None
