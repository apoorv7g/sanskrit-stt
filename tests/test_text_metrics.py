from sanskrit_stt.metrics import cer, edit_distance
from sanskrit_stt.text import normalize, normalize_for_scoring


def test_normalize_strips_vedic_accents_and_spaces():
    assert normalize("  अ॒र्जु॑न   उवाच ") == "अर्जुन उवाच"


def test_scoring_drops_dandas_and_digits():
    assert normalize_for_scoring("केशव ॥ १ ॥") == "केशव"


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a"]) == 1


def test_cer_ignores_sandhi_spacing():
    assert cer("न च संन्यसनादेव", "न च संन्यसनादेव") == 0
    assert cer("चेत्कर्मणस्ते", "चेत् कर्मणस्ते") == 0
    assert cer("abcd", "abxd") == 0.25
