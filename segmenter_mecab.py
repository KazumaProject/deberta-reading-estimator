from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Morph:
    surf: str
    reading: str  # hiragana or ""


_KATAKANA_RANGE = (0x30A1, 0x30F6)  # small a .. small ke


def _katakana_to_hiragana(s: str) -> str:
    out: List[str] = []
    for ch in s:
        code = ord(ch)
        if _KATAKANA_RANGE[0] <= code <= _KATAKANA_RANGE[1]:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _normalize_reading(yomi: str) -> str:
    y = (yomi or "").strip()
    if not y or y in ("*", "＊"):
        return ""
    return _katakana_to_hiragana(y)


def _contains_mecabrc_arg(args: str) -> bool:
    a = (args or "").strip()
    if not a:
        return False
    # rough but practical
    return " -r " in f" {a} " or a.startswith("-r") or " -r\t" in f" {a} "


def _find_existing_mecabrc() -> Optional[str]:
    candidates = [
        os.environ.get("MECABRC", "").strip(),
        "/etc/mecabrc",
        "/usr/local/etc/mecabrc",
        "/opt/homebrew/etc/mecabrc",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists() and p.is_file():
            return str(p)
    return None


class MeCabSegmenter:
    """
    MeCab (fugashi) segmenter plugin.

    Config keys (via --segmenter-config JSON):
      - mecab_args: str (passed to fugashi Tagger/GenericTagger)
      - mecabrc: str (explicit mecabrc path; inject "-r <path>")
      - dict_dir: str (inject "-d <path>")
      - use_feature_reading: bool (default True)
      - prefer_pron: bool (default True)
      - force_generic: bool (default False)  # always use GenericTagger
    """

    def __init__(
        self,
        mecab_args: str = "",
        use_feature_reading: bool = True,
        prefer_pron: bool = True,
        mecabrc: Optional[str] = None,
        dict_dir: Optional[str] = None,
        force_generic: bool = False,
    ):
        try:
            from fugashi import Tagger, GenericTagger  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "MeCabSegmenter requires fugashi.\n"
                "Install: pip install -U fugashi\n"
                "And install MeCab + dictionary (e.g. apt install mecab mecab-ipadic-utf8)."
            ) from e

        self._use_feature_reading = bool(use_feature_reading)
        self._prefer_pron = bool(prefer_pron)

        args = (mecab_args or "").strip()

        # inject mecabrc
        if mecabrc:
            mecabrc = str(mecabrc).strip()
            if mecabrc and not _contains_mecabrc_arg(args):
                args = (args + f" -r {mecabrc}").strip()

        # inject dict_dir
        if dict_dir:
            dict_dir = str(dict_dir).strip()
            if dict_dir and " -d " not in f" {args} ":
                args = (args + f" -d {dict_dir}").strip()

        # auto-discover mecabrc if not set
        if not _contains_mecabrc_arg(args) and not os.environ.get("MECABRC"):
            found = _find_existing_mecabrc()
            if found:
                args = (args + f" -r {found}").strip()

        # build tagger
        self._tagger = None
        init_err: Optional[BaseException] = None

        if bool(force_generic):
            try:
                self._tagger = GenericTagger(args)
            except Exception as e:
                init_err = e
        else:
            # try Tagger first
            try:
                self._tagger = Tagger(args)
            except RuntimeError as e:
                # IMPORTANT: dictionary format unknown -> use GenericTagger
                msg = str(e)
                if "Unknown dictionary format" in msg:
                    try:
                        self._tagger = GenericTagger(args)
                    except Exception as e2:
                        init_err = e2
                else:
                    init_err = e
            except Exception as e:
                init_err = e

        if self._tagger is None:
            found = _find_existing_mecabrc()
            raise RuntimeError(
                "Failed initializing MeCab (fugashi).\n"
                f"  args='{args}'\n"
                f"  MECABRC env='{os.environ.get('MECABRC', '')}'\n"
                f"  auto_found_mecabrc='{found or ''}'\n"
                f"  error='{init_err}'\n"
                "\n"
                "Try one of these:\n"
                "  - Use GenericTagger explicitly:\n"
                "      --segmenter-config '{\"force_generic\":true}'\n"
                "  - Provide dictionary dir explicitly (example path varies):\n"
                "      --segmenter-config '{\"dict_dir\":\"/usr/lib/x86_64-linux-gnu/mecab/dic/ipadic\"}'\n"
            ) from init_err

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(
            mecab_args=str(config.get("mecab_args", "") or ""),
            mecabrc=config.get("mecabrc", None),
            dict_dir=config.get("dict_dir", None),
            use_feature_reading=bool(config.get("use_feature_reading", True)),
            prefer_pron=bool(config.get("prefer_pron", True)),
            force_generic=bool(config.get("force_generic", False)),
        )

    def _extract_yomi(self, token) -> str:
        if not self._use_feature_reading:
            return ""

        feat = getattr(token, "feature", None)

        # 1) object-style access (if available)
        for attr_path in (("pron",), ("pronunciation",), ("kana",), ("reading",)):
            try:
                v = feat
                for a in attr_path:
                    v = getattr(v, a)
                if isinstance(v, str) and v:
                    y = _normalize_reading(v)
                    if y:
                        return y
            except Exception:
                pass

        # 2) CSV feature string fallback (GenericTagger will usually land here)
        try:
            fs = str(feat)
            parts = fs.split(",")

            cand_pron: Optional[str] = None
            cand_read: Optional[str] = None

            # IPADIC-ish: ..., base, reading, pron
            if len(parts) >= 8:
                cand_read = parts[-2]
                cand_pron = parts[-1]

            if self._prefer_pron and cand_pron and cand_pron not in ("*", "＊"):
                y = _normalize_reading(cand_pron)
                if y:
                    return y

            if cand_read and cand_read not in ("*", "＊"):
                y = _normalize_reading(cand_read)
                if y:
                    return y
        except Exception:
            pass

        return ""

    def tokenize(self, text: str) -> List[Morph]:
        s = (text or "").strip()
        if not s:
            return []

        out: List[Morph] = []
        for tok in self._tagger(s):
            surf = tok.surface
            yomi = self._extract_yomi(tok)
            out.append(Morph(surf=surf, reading=yomi))
        return out
