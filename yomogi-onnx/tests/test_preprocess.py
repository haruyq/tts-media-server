from yomogi_onnx.preprocess import UserReadings, preprocess_discord


def test_longest_user_dictionary_match_preserves_neighbors() -> None:
    readings = UserReadings(
        {
            "VR": "ぶいあーる",
            "VRChat": "ぶいあーるちゃっと",
        }
    )
    assert readings.replace("新VRChat民とVR機器") == "新ぶいあーるちゃっと民とぶいあーる機器"


def test_discord_cleanup() -> None:
    result = preprocess_discord(
        "@太郎 #一般 https://example.com <a:dance:123> 🙂 ```print('x')```\n\n\nVRChat!!!",
        max_length=500,
    )
    assert "太郎" in result.text
    assert "一般" in result.text
    assert "ゆーあーるえる" in result.text
    assert "dance" in result.text
    assert "コード省略" in result.text
    assert "ぶいあーるちゃっと" in result.text
    assert "!!!" not in result.text
    assert result.removed_emoji_count > 0


def test_reports_truncation() -> None:
    result = preprocess_discord("あ" * 600, max_length=500)
    assert len(result.text) == 500
    assert result.truncated
