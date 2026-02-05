from __future__ import annotations

import argparse
import re
import sys
from typing import Iterable, List, Optional

from datasets import load_dataset


WIKI40B_MARKERS_RE = re.compile(
    r"(_START_ARTICLE_|_START_SECTION_|_START_PARAGRAPH_|_NEWLINE_)", flags=0
)

# ざっくり文分割（日本語）
SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")


def clean_wiki40b_text(s: str) -> str:
    if not s:
        return ""
    # marker -> space/newline
    s = s.replace("_NEWLINE_", "\n")
    s = WIKI40B_MARKERS_RE.sub(" ", s)

    # 余分な空白整理
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", "\n", s)
    return s.strip()


def split_sentences(s: str) -> List[str]:
    if not s:
        return []
    # 行ごとに分割してから文分割（見出し/段落の混在を軽減）
    out: List[str] = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in SENT_SPLIT_RE.split(line) if p and p.strip()]
        out.extend(parts)
    return out


def iter_text_fields(example: dict) -> Iterable[str]:
    """
    wiki40b 互換の想定:
    - 'text' カラムがあることが多い
    - もし違っても、文字列カラムを全部拾って最大の本文っぽいものを使う
    """
    if "text" in example and isinstance(example["text"], str):
        yield example["text"]
        return

    # fallback: 文字列フィールドを全部候補に
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            yield v


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build source text lines from range3/wiki40b-ja (1 line = 1 sentence)."
    )
    ap.add_argument("--out", required=True, help="output txt path (utf-8)")
    ap.add_argument(
        "--target_word",
        required=True,
        help="keep sentences containing this word exactly once",
    )
    ap.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="dataset split",
    )
    ap.add_argument("--dataset", default="range3/wiki40b-ja", help="HF dataset name")
    ap.add_argument(
        "--streaming", action="store_true", help="use streaming=True (no full download)"
    )
    ap.add_argument(
        "--max_len", type=int, default=120, help="max sentence length (0 disables)"
    )
    ap.add_argument("--min_len", type=int, default=1, help="min sentence length")
    ap.add_argument(
        "--max_lines", type=int, default=200000, help="stop after N lines written"
    )
    ap.add_argument(
        "--text_filter",
        default=None,
        help="optional substring filter (must be contained)",
    )
    ap.add_argument(
        "--dedup", action="store_true", help="deduplicate lines (memory heavy if huge)"
    )
    args = ap.parse_args()

    target = args.target_word
    max_len = args.max_len
    min_len = args.min_len
    text_filter: Optional[str] = args.text_filter

    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    seen = set() if args.dedup else None
    written = 0

    # 出力は逐次書き込み（巨大になるので）
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in ds:
            # 本文候補を拾って処理
            for raw in iter_text_fields(ex):
                cleaned = clean_wiki40b_text(raw)
                if not cleaned:
                    continue

                for sent in split_sentences(cleaned):
                    if not sent:
                        continue
                    if len(sent) < min_len:
                        continue
                    if max_len > 0 and len(sent) > max_len:
                        continue
                    if sent.count(target) != 1:
                        continue
                    if text_filter is not None and text_filter not in sent:
                        continue

                    if seen is not None:
                        if sent in seen:
                            continue
                        seen.add(sent)

                    f.write(sent + "\n")
                    written += 1

                    if written % 2000 == 0:
                        print(f"[progress] written={written}", file=sys.stderr)

                    if written >= args.max_lines:
                        print(
                            f"[done] reached max_lines={args.max_lines}",
                            file=sys.stderr,
                        )
                        return

    print(f"[done] written={written} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
