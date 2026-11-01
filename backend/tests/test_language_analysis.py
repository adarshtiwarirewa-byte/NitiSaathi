import unicodedata

from backend.app.services.language_analysis import analyze_page_languages


class FakeIndicLIDService:
    def detect(self, text: str) -> dict:
        if "தமிழ்நாடு" in text or any("TAMIL" in unicodedata.name(ch, "") for ch in text):
            language = ("tam", "tam_Tamil", "Tamil")
        elif "संस्कृत" in text:
            language = ("san", "san_Deva", "Deva")
        elif any("DEVANAGARI" in unicodedata.name(ch, "") for ch in text):
            language = ("hin", "hin_Deva", "Deva")
        else:
            language = ("eng", "eng_Latn", "Latn")

        return {
            "language_code": language[0],
            "language_label": language[1],
            "script": language[2],
            "detection_model": "fake-indiclid",
            "detection_raw_score": 0.99,
        }


def test_single_page_can_contain_multiple_languages_and_scripts():
    text = (
        "यह सरकारी योजना छोटे व्यवसायों को वित्तीय सहायता प्रदान करती है।\n\n"
        "Government assistance is available to eligible small businesses.\n\n"
        "தமிழ்நாடு அரசு சிறு தொழில்களுக்கு நிதி உதவி வழங்குகிறது.\n\n"
        "संस्कृत भाषा प्राचीन भारतीय साहित्य की महत्वपूर्ण भाषा है।"
    )

    result = analyze_page_languages(
        text,
        FakeIndicLIDService(),
        page_number=7,
    )

    assert set(result["page_languages"]) == {
        "hin", "eng", "tam", "san"
    }
    assert {"Devanagari", "Latin", "Tamil"}.issubset(
        set(result["detected_scripts"])
    )
    assert len(result["language_segments"]) >= 4

    for segment in result["language_segments"]:
        assert text[segment["start_char"]:segment["end_char"]] == segment["text"]
        assert segment["detection_status"] == "detected"


def test_mixed_scripts_inside_one_sentence_are_separate_segments():
    text = (
        "हिन्दी में सरकारी सहायता छोटे व्यवसायों को वित्तीय लाभ मिलता है "
        "Government assistance is available for eligible citizens and businesses "
        "தமிழ்நாடு அரசு சிறு தொழில்களுக்கு நிதி உதவி வழங்குகிறது"
    )

    result = analyze_page_languages(
        text,
        FakeIndicLIDService(),
        page_number=2,
    )

    assert {"hin", "eng", "tam"}.issubset(
        set(result["page_languages"])
    )


def test_short_fragments_are_not_force_classified():
    text = "GST / 2026 / OK"

    result = analyze_page_languages(
        text,
        FakeIndicLIDService(),
        page_number=1,
    )

    assert result["page_languages"] == []
    assert all(
        segment["detection_status"] == "too_short"
        for segment in result["language_segments"]
    )
