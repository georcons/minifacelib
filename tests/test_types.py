from minifacelib.models.base.types import parse_gender, parse_emotion


def test_parse_gender_variants() -> None:
    """parse_gender should normalize common gender representations."""
    assert parse_gender("male") == "male"
    assert parse_gender("M") == "male"
    assert parse_gender("female") == "female"
    assert parse_gender("F") == "female"
    assert parse_gender(" something ") == "unknown"


def test_parse_emotion_variants() -> None:
    """parse_emotion should normalize common emotion representations."""
    assert parse_emotion("happy") == "happy"
    assert parse_emotion("HAPPY") == "happy"
    assert parse_emotion("neutral") == "neutral"
    assert parse_emotion("unknown") == "unknown"
    assert parse_emotion("???") == "unknown"
