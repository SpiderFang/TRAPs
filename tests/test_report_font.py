"""``traps.report_font``：CJK 字型解析順序、字元涵蓋 gate、fail-closed 與 rcParams 還原。

解析器的單元測試以暫存假字型檔搭配 monkeypatch 的 ``findfont``／charmap 檢查進行，
不依賴執行環境是否安裝 CJK 字型；只有「必要字元真的被系統 Noto Sans CJK 涵蓋」這項
檢查在沒有字型檔時 skip。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from traps import report_font
from traps.report_font import (
    CJK_FONT_CANDIDATES,
    FIGURE_TEXT,
    REQUIRED_FIGURE_GLYPHS,
    STUDY_SITE_NAMES_ZH,
    CJKFontSelection,
    cjk_font_context,
    resolve_cjk_font,
)
from traps.sites import DEFAULT_STUDY_DESIGN_PATH, load_study_design

SYSTEM_NOTO_SANS = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def _fake_font_file(tmp_path: Path, name: str = "FakeCJK.otf", payload: bytes = b"fake-font-bytes") -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_candidate_order_is_fixed() -> None:
    assert CJK_FONT_CANDIDATES == (
        "Noto Sans CJK TC", "Noto Sans TC", "Source Han Sans TC", "PingFang TC", "Heiti TC", "Arial Unicode MS",
    )


def test_required_glyphs_cover_every_figure_string_and_site_name() -> None:
    """必要字元集合由實際圖面文字、五站名稱與 ASCII／符號組成，且站名與研究設計一致。"""

    for text in FIGURE_TEXT.values():
        assert set(text) <= REQUIRED_FIGURE_GLYPHS
    for name in STUDY_SITE_NAMES_ZH:
        assert set(name) <= REQUIRED_FIGURE_GLYPHS
    design = load_study_design(DEFAULT_STUDY_DESIGN_PATH)
    assert {site["study_site_name_zh"] for site in design["study_sites"]} == set(STUDY_SITE_NAMES_ZH)
    for char in "貢寮龜山島西側新竹外海南灣連江":
        assert char in REQUIRED_FIGURE_GLYPHS
    assert "MODEL_ONLY 表層短期吸引潛勢" == FIGURE_TEXT["science_model_only"]
    assert "©" in REQUIRED_FIGURE_GLYPHS and "−" in REQUIRED_FIGURE_GLYPHS


@pytest.mark.skipif(not SYSTEM_NOTO_SANS.is_file(), reason="系統沒有安裝 Noto Sans CJK")
def test_system_noto_sans_cjk_covers_required_glyphs() -> None:
    """有安裝 Noto Sans CJK 時，其 FreeType charmap 必須涵蓋全部必要字元。"""

    assert report_font._font_covers_required_glyphs(SYSTEM_NOTO_SANS)


def test_resolver_tries_candidates_in_order_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """所有候選都找不到：依固定順序、fallback_to_default=False 逐一嘗試，最後拋固定 ValueError。"""

    calls: list[tuple[str, bool]] = []

    def _findfont(properties, *args, fallback_to_default=True, **kwargs):  # noqa: ANN001, ANN002, ANN003
        calls.append((properties.get_family()[0], fallback_to_default))
        raise ValueError("Failed to find font /secret/local/path")

    monkeypatch.setattr(report_font.font_manager, "findfont", _findfont)
    with pytest.raises(ValueError, match="繁體中文字型") as info:
        resolve_cjk_font()
    assert [family for family, _ in calls] == list(CJK_FONT_CANDIDATES)
    assert all(fallback is False for _, fallback in calls)
    assert "/secret" not in str(info.value)  # 不洩漏底層路徑
    assert "--no-figures" in str(info.value)


def test_resolver_selects_first_covering_candidate_and_hashes_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """前面候選缺字時改試下一個；回傳 basename、檔案 SHA-256 與必要字元數，不含絕對路徑。"""

    payload = b"pretend-this-is-a-font"
    good = _fake_font_file(tmp_path, "GoodTC.otf", payload)
    weak = _fake_font_file(tmp_path, "WeakTC.otf", b"weak")
    paths = {"Noto Sans CJK TC": weak, "Noto Sans TC": good, "Source Han Sans TC": _fake_font_file(tmp_path, "Later.otf")}
    tried: list[str] = []

    def _findfont(properties, *args, fallback_to_default=True, **kwargs):  # noqa: ANN001, ANN002, ANN003
        family = properties.get_family()[0]
        tried.append(family)
        if family not in paths:
            raise ValueError("not found")
        return str(paths[family])

    monkeypatch.setattr(report_font.font_manager, "findfont", _findfont)
    monkeypatch.setattr(report_font, "_font_covers_required_glyphs", lambda path: Path(path).name == "GoodTC.otf")
    selection = resolve_cjk_font()
    assert tried == ["Noto Sans CJK TC", "Noto Sans TC"]  # 第二個通過後不再往後找
    assert selection == CJKFontSelection(
        family="Noto Sans TC", filename="GoodTC.otf", file_sha256=hashlib.sha256(payload).hexdigest(),
        required_glyph_count=len(REQUIRED_FIGURE_GLYPHS),
    )
    assert "/" not in selection.filename
    assert selection.as_manifest_dict()["family"] == "Noto Sans TC"


def test_resolver_first_candidate_wins_when_it_covers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    font = _fake_font_file(tmp_path)
    tried: list[str] = []

    def _findfont(properties, *args, fallback_to_default=True, **kwargs):  # noqa: ANN001, ANN002, ANN003
        tried.append(properties.get_family()[0])
        return str(font)

    monkeypatch.setattr(report_font.font_manager, "findfont", _findfont)
    monkeypatch.setattr(report_font, "_font_covers_required_glyphs", lambda path: True)
    assert resolve_cjk_font().family == "Noto Sans CJK TC"
    assert tried == ["Noto Sans CJK TC"]


def test_resolver_rejects_symlink_and_missing_glyph_coverage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """符號連結與缺字的字型都視為不合格，全部不合格時 fail closed。"""

    target = _fake_font_file(tmp_path, "Real.otf")
    link = tmp_path / "Link.otf"
    link.symlink_to(target)
    monkeypatch.setattr(report_font.font_manager, "findfont", lambda *a, **k: str(link))
    monkeypatch.setattr(report_font, "_font_covers_required_glyphs", lambda path: True)
    with pytest.raises(ValueError, match="繁體中文字型"):
        resolve_cjk_font()

    monkeypatch.setattr(report_font.font_manager, "findfont", lambda *a, **k: str(target))
    monkeypatch.setattr(report_font, "_font_covers_required_glyphs", lambda path: False)
    with pytest.raises(ValueError, match="繁體中文字型"):
        resolve_cjk_font()


def test_selection_validates_record_shape() -> None:
    count = len(REQUIRED_FIGURE_GLYPHS)
    with pytest.raises(ValueError):
        CJKFontSelection("Comic Sans", "a.ttf", "0" * 64, count)
    with pytest.raises(ValueError):
        CJKFontSelection("Noto Sans CJK TC", "/abs/a.ttf", "0" * 64, count)
    with pytest.raises(ValueError):
        CJKFontSelection("Noto Sans CJK TC", "a.ttf", "A" * 64, count)
    with pytest.raises(ValueError):
        CJKFontSelection("Noto Sans CJK TC", "a.ttf", "0" * 64, count + 1)


def test_cjk_font_context_sets_and_restores_rcparams() -> None:
    import matplotlib

    selection = CJKFontSelection("Noto Sans CJK TC", "a.ttc", "0" * 64, len(REQUIRED_FIGURE_GLYPHS))
    before_family = list(matplotlib.rcParams["font.family"])
    before_minus = matplotlib.rcParams["axes.unicode_minus"]
    with cjk_font_context(selection):
        assert list(matplotlib.rcParams["font.family"]) == ["Noto Sans CJK TC", "DejaVu Sans"]
        assert matplotlib.rcParams["axes.unicode_minus"] is False
    assert list(matplotlib.rcParams["font.family"]) == before_family
    assert matplotlib.rcParams["axes.unicode_minus"] == before_minus
    with pytest.raises(RuntimeError), cjk_font_context(selection):
        raise RuntimeError("boom")
    assert list(matplotlib.rcParams["font.family"]) == before_family  # 例外時也還原
