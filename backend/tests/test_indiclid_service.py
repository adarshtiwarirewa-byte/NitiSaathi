
import pytest

from backend.app.services.indiclid_service import (
    IndicLIDService,
    FTN_PATH,
    FTR_PATH,
    BERT_PATH,
    TOKENIZER_PATH,
)


@pytest.fixture(scope="module")
def indiclid_service():
    """Load the official models once for this test module."""

    required_paths = [
        FTN_PATH,
        FTR_PATH,
        BERT_PATH,
        TOKENIZER_PATH,
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        pytest.skip(
            f"IndicLID model files unavailable: {missing}"
        )

    return IndicLIDService()


TEST_CASES = [
    (
        "यह योजना छोटे व्यवसायों को वित्तीय सहायता प्रदान करती है।",
        "hin_Deva",
        "IndicLID-FTN",
    ),
    (
        "தமிழ்நாடு அரசு புதிய நலத்திட்டத்தை அறிவித்துள்ளது.",
        "tam_Tamil",
        "IndicLID-FTN",
    ),
    (
        "Mujhe is sarkari yojana ke liye aavedan kaise karna hai?",
        "hin_Latn",
        "IndicLID-FTR",
    ),
    (
        "Indha thittathukku yaar thagudhiyaanavanga?",
        "tam_Latn",
        "IndicLID-FTR",
    ),
    (
        "Small businesses can apply for financial assistance under this government scheme.",
        "eng_Latn",
        "IndicLID-FTR",
    ),
]


@pytest.mark.parametrize(
    "text, expected_language, expected_model",
    TEST_CASES,
)
def test_language_detection(
    indiclid_service,
    text,
    expected_language,
    expected_model,
):
    result = indiclid_service.detect(text)

    assert result["language_label"] == expected_language
    assert result["detection_model"] == expected_model
    assert isinstance(result["detection_raw_score"], float)


def test_empty_text_rejected(indiclid_service):
    with pytest.raises(ValueError):
        indiclid_service.detect("   ")
