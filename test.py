from parkrun import hms_to_minutes


def test_hms_to_minutes():
    assert hms_to_minutes("1:00") == 1.0
    assert hms_to_minutes("1:30") == 1.5
    assert hms_to_minutes("2:45") == 2.75
    assert hms_to_minutes("3:00") == 3.0
    assert hms_to_minutes("0:30") == 0.5
    assert hms_to_minutes("0:00") == 0.0

if __name__ == "__main__":
    test_hms_to_minutes()