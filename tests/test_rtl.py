import asyncio

import pytest

import deep_eye
from deep_eye import render, trace
from deep_eye.tui import DeepEyeApp

AR = "مرحبا، كيف حالك اليوم؟"          # logical order: مرحبا، كيف حالك اليوم؟
AR_WORDS = AR.split(" ")
HE = "שלום עולם טוב"


@pytest.fixture(autouse=True)
def _reset_mode():
    yield
    render.rtl_mode = "off"


def _words(text: str) -> str:
    render.rtl_mode = "words"
    return render.clean(text)


def test_off_changes_nothing():
    assert render.clean(AR) == AR


def test_words_reverses_word_order_and_keeps_letters():
    assert _words(AR) == " ".join(reversed(AR_WORDS))


def test_words_works_for_hebrew():
    assert _words(HE) == " ".join(reversed(HE.split(" ")))


def test_words_leaves_left_to_right_text_alone():
    assert _words("hello how are you") == "hello how are you"
    assert _words("a b\nc d") == "a b\nc d"


def test_words_keeps_quotes_and_punctuation_at_the_edges():
    assert _words(f'  "q": "{AR}"') == f'  "q": "{" ".join(reversed(AR_WORDS))}"'
    assert _words('x "مرحبا بك".') == 'x "بك مرحبا".'


def test_words_reverses_each_run_and_each_line_separately():
    out = _words("one مرحبا بك two هنا ذلك\nمرحبا بك")
    assert out == "one بك مرحبا two ذلك هنا\nبك مرحبا"


def test_words_reverses_words_across_extra_spaces():
    assert _words("a  مرحبا  بك").split() == ["a", "بك", "مرحبا"]


def test_control_characters_are_still_escaped():
    esc, backslash = chr(27), chr(92)
    assert _words(esc + "[31m مرحبا").startswith(backslash + "x1b")


@pytest.mark.skipif(not render.full_rtl_available(), reason="deep-eye[rtl] not installed")
def test_full_joins_letters():
    render.rtl_mode = "full"
    out = render.clean(AR)
    assert out != AR and any("ﹰ" <= ch <= "﻿" for ch in out)  # presentation forms


def test_parse_rtl_mode():
    assert render.parse_rtl_mode("words") == "words"
    assert render.parse_rtl_mode("1") == "words"
    assert render.parse_rtl_mode("full") == "full"
    assert render.parse_rtl_mode("bogus") == "off"
    assert render.parse_rtl_mode(None) == "off"


def test_b_key_cycles_modes_and_the_detail_pane_uses_it(tmp_path, monkeypatch):
    monkeypatch.setattr("deep_eye.tui.save_setting", lambda *a: None)
    monkeypatch.setattr("deep_eye.tui.load_settings", lambda: {})
    monkeypatch.delenv("DEEP_EYE_RTL", raising=False)
    deep_eye.configure(trace_dir=tmp_path)

    @trace
    def agent(q):
        return AR

    agent("x")

    async def scenario():
        app = DeepEyeApp()
        async with app.run_test() as pilot:
            assert render.rtl_mode == "off"
            await pilot.press("b")
            assert render.rtl_mode == "words"
            await pilot.press("b")
            assert render.rtl_mode == ("full" if render.full_rtl_available() else "off")
    asyncio.run(scenario())
