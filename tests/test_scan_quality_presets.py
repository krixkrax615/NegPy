from negpy.desktop.view.sidebar.scan import _QUALITY_PRESETS, _QUALITY_SNR, ScanCaptureMode, quality_minutes


def test_presets_are_distinct_and_described():
    states = [state for *_head, state in _QUALITY_PRESETS]
    assert len(set(states)) == len(states)
    for key, label, desc, time_text, state in _QUALITY_PRESETS:
        assert key and label and desc and time_text > 0
        mode, passes, samples, ae = state
        assert isinstance(mode, ScanCaptureMode) and passes >= 1 and samples >= 1


def test_multi_exposure_preset_does_not_claim_auto_exposure():
    dense = next(state for key, *_r, state in _QUALITY_PRESETS if key == "dense")
    assert dense[0] == ScanCaptureMode.ADAPTIVE_ME and dense[3] is False


def test_time_estimates_follow_the_resolution():
    assert quality_minutes(13.0, 3600) == "about 13 min"
    assert quality_minutes(13.0, 1800) == "about 6 min"
    assert quality_minutes(13.0, 7200) == "about 26 min"
    assert quality_minutes(1.0, 1800) == "under 1 min"


def test_dense_film_max_stacks_the_multi_exposure_preset():
    mode, passes, samples, ae = next(state for key, *_r, state in _QUALITY_PRESETS if key == "dense_max")
    assert mode == ScanCaptureMode.ADAPTIVE_MULTI_PASS and passes == 2 and samples == 2 and ae is False


def test_every_preset_states_its_snr_gain():
    assert set(_QUALITY_SNR) == {key for key, *_rest in _QUALITY_PRESETS}
    assert _QUALITY_SNR["fast"] == "×1"
