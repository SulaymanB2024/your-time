from screen_storage_guard import STOP_BELOW_BYTES, should_stop


def test_stop_before_storage_pressure():
    assert should_stop(STOP_BELOW_BYTES - 1, True, False)
    assert should_stop(STOP_BELOW_BYTES - 1, False, True)
    assert not should_stop(STOP_BELOW_BYTES, True, True)
    assert not should_stop(STOP_BELOW_BYTES - 1, False, False)
