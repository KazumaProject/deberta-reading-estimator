from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, cast

import numpy as np
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

# -------------------------
# Optional progress (tqdm)
# -------------------------
def _get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


Representation = Literal["hidden", "logits"]

# -------------------------
# Reading normalization
# -------------------------
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
    if not y or y == "*":
        return ""
    return _katakana_to_hiragana(y)


# -------------------------
# Segmenter interface
# -------------------------
@dataclass
class Morph:
    surf: str
    reading: str  # hiragana if available; otherwise ""


class Segmenter(Protocol):
    """
    Segmenter minimal interface:
      - tokenize(text) -> List[Morph]
    """

    def tokenize(self, text: str) -> List[Morph]:
        ...


class ConfigurableSegmenter(Protocol):
    """
    Optional plugin interface:
      - from_config(config: dict) -> Segmenter
    """

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Segmenter:
        ...


class WhitespaceSegmenter:
    def tokenize(self, text: str) -> List[Morph]:
        s = (text or "").strip()
        if not s:
            return []
        toks = [t for t in re.split(r"\s+", s) if t]
        return [Morph(surf=t, reading="") for t in toks]


class SudachiSegmenter:
    """
    SudachiPy segmenter (optional dependency)
    config:
      - sudachi_mode: A/B/C
    """

    def __init__(self, mode: str = "C"):
        try:
            from sudachipy import dictionary as sudachi_dictionary  # type: ignore
            from sudachipy import tokenizer as sudachi_tokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "SudachiSegmenter requires sudachipy. Install: pip install sudachipy sudachidict-core"
            ) from e

        mode = (mode or "C").upper()
        if mode not in ("A", "B", "C"):
            raise ValueError("Sudachi split mode must be A/B/C")

        self._tokenizer = sudachi_dictionary.Dictionary().create()
        if mode == "A":
            self._mode = sudachi_tokenizer.Tokenizer.SplitMode.A
        elif mode == "B":
            self._mode = sudachi_tokenizer.Tokenizer.SplitMode.B
        else:
            self._mode = sudachi_tokenizer.Tokenizer.SplitMode.C

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Segmenter:
        return cls(mode=str(config.get("sudachi_mode", "C")))

    def tokenize(self, text: str) -> List[Morph]:
        ms = self._tokenizer.tokenize(text, self._mode)
        out: List[Morph] = []
        for m in ms:
            surf = m.surface()
            reading = _normalize_reading(m.reading_form())
            out.append(Morph(surf=surf, reading=reading))
        return out


class JumanppSegmenter:
    """
    Juman++ segmenter (optional dependency)

    Key point:
      - rhoknp.Jumanpp has a startup "sanity check" which can time out in some environments.
      - We default to skip_sanity_check=True to avoid failing at startup.
      - rcfile is passed via jumanpp CLI options: ["-r", rcfile].

    config:
      - juman_timeout: seconds for apply_to_sentence()
      - juman_executable: path/name for jumanpp (default: "jumanpp")
      - juman_options: list of extra CLI options (JSON array in CLI)
      - juman_rcfile: optional path (converted into ["-r", path])
      - juman_skip_sanity_check: bool (default: True)
    """

    def __init__(
        self,
        timeout: float = 10.0,
        *,
        executable: str = "jumanpp",
        options: Optional[List[str]] = None,
        rcfile: Optional[str] = None,
        skip_sanity_check: bool = True,
    ):
        try:
            from rhoknp import Jumanpp  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "JumanppSegmenter requires rhoknp + Juman++ runtime. Install: pip install rhoknp "
                "and ensure jumanpp is available in PATH."
            ) from e

        opt: List[str] = []
        if options:
            opt.extend([str(x) for x in options])
        if rcfile:
            # jumanpp supports "-r <rcfile>"
            opt.extend(["-r", str(rcfile)])

        # NOTE: rhoknp.Jumanpp accepts executable/options/skip_sanity_check
        self._jumanpp = Jumanpp(
            executable=str(executable),
            options=opt if opt else None,
            skip_sanity_check=bool(skip_sanity_check),
        )
        self._timeout = float(timeout)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Segmenter:
        timeout = float(config.get("juman_timeout", 10.0))
        executable = str(config.get("juman_executable", "jumanpp"))
        rcfile = config.get("juman_rcfile", None)
        options = config.get("juman_options", None)
        skip = config.get("juman_skip_sanity_check", True)

        opt_list: Optional[List[str]] = None
        if isinstance(options, list):
            opt_list = [str(x) for x in options]
        elif options is None:
            opt_list = None
        else:
            raise ValueError("juman_options must be a list (JSON array)")

        return cls(
            timeout=timeout,
            executable=executable,
            options=opt_list,
            rcfile=str(rcfile) if rcfile else None,
            skip_sanity_check=bool(skip),
        )

    def tokenize(self, text: str) -> List[Morph]:
        # apply_to_sentence timeout is separate from startup sanity check
        result = self._jumanpp.apply_to_sentence(text, timeout=self._timeout)

        out: List[Morph] = []
        for mrph in result.morphemes:
            surf = getattr(mrph, "surf", "")
            reading = _normalize_reading(getattr(mrph, "reading", ""))
            out.append(Morph(surf=surf, reading=reading))
        return out


# -------------------------
# Plugin loader
# -------------------------
def _load_segmenter_from_plugin(spec: str, config: Dict[str, Any]) -> Segmenter:
    """
    spec format: "some_module:ClassName"
    - Class must implement tokenize(text)->List[Morph]
    - Optionally implements classmethod from_config(config)->Segmenter
    """
    if ":" not in spec:
        raise ValueError("Plugin segmenter spec must be 'module:ClassName'")

    mod_name, cls_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise ValueError(f"Plugin class not found: {spec}")

    if hasattr(cls, "from_config") and callable(getattr(cls, "from_config")):
        seg = cast(ConfigurableSegmenter, cls).from_config(config)  # type: ignore[arg-type]
        return cast(Segmenter, seg)

    try:
        seg = cls()
    except TypeError as e:
        raise ValueError(
            f"Plugin class {spec} could not be instantiated with no-arg constructor. "
            "Implement from_config(config) or provide a no-arg __init__."
        ) from e

    if not hasattr(seg, "tokenize") or not callable(getattr(seg, "tokenize")):
        raise ValueError(f"Plugin segmenter {spec} does not implement tokenize(text)")
    return cast(Segmenter, seg)


def _build_segmenter(name: str, *, config: Dict[str, Any]) -> Tuple[Segmenter, str, Dict[str, Any]]:
    """
    Returns (segmenter_instance, segmenter_name_for_metadata, segmenter_config_for_metadata)
    name:
      - auto: sudachi -> juman -> whitespace
      - sudachi / juman / whitespace
      - plugin: module:ClassName
    """
    nm = (name or "auto").strip()

    if ":" in nm:
        seg = _load_segmenter_from_plugin(nm, config)
        return seg, nm, config

    low = nm.lower()
    if low == "whitespace":
        return WhitespaceSegmenter(), "whitespace", {}

    if low == "sudachi":
        seg = SudachiSegmenter.from_config(config)
        return seg, "sudachi", {"sudachi_mode": config.get("sudachi_mode", "C")}

    if low == "juman":
        seg = JumanppSegmenter.from_config(config)
        meta: Dict[str, Any] = {
            "juman_timeout": config.get("juman_timeout", 10.0),
            "juman_executable": config.get("juman_executable", "jumanpp"),
            "juman_skip_sanity_check": config.get("juman_skip_sanity_check", True),
        }
        if config.get("juman_rcfile"):
            meta["juman_rcfile"] = config["juman_rcfile"]
        if config.get("juman_options"):
            meta["juman_options"] = config["juman_options"]
        return seg, "juman", meta

    if low != "auto":
        raise ValueError("--segmenter must be auto/sudachi/juman/whitespace or 'module:ClassName' plugin")

    # auto
    try:
        seg = SudachiSegmenter.from_config(config)
        return seg, "sudachi", {"sudachi_mode": config.get("sudachi_mode", "C")}
    except Exception:
        pass
    try:
        seg = JumanppSegmenter.from_config(config)
        meta2: Dict[str, Any] = {
            "juman_timeout": config.get("juman_timeout", 10.0),
            "juman_executable": config.get("juman_executable", "jumanpp"),
            "juman_skip_sanity_check": config.get("juman_skip_sanity_check", True),
        }
        if config.get("juman_rcfile"):
            meta2["juman_rcfile"] = config["juman_rcfile"]
        if config.get("juman_options"):
            meta2["juman_options"] = config["juman_options"]
        return seg, "juman", meta2
    except Exception:
        pass
    return WhitespaceSegmenter(), "whitespace", {}


# -------------------------
# ReadingEstimator (segmenter-agnostic)
# -------------------------
class ReadingEstimator:
    _MASK_VARIANT_RE = re.compile(
        r"(?i)("
        r"\[\s*mask\s*\]"
        r"|<\s*mask\s*>"
        r"|＜\s*mask\s*＞"
        r"|\(\s*mask\s*\)"
        r"|\{\s*mask\s*\}"
        r")"
    )

    def __init__(
        self,
        model_name: str,
        references: Dict[str, Dict[str, List[str]]],
        *,
        evaluation_type: str = "most_similar",
        device: str = "cpu",
        representation: Representation = "hidden",
        batch_size: int = 16,
        show_progress: bool = True,
        mask_aliases: Optional[List[str]] = None,
        segmenter_name: str = "auto",
        segmenter_config: Optional[Dict[str, Any]] = None,
        segmenter: Optional[Segmenter] = None,
        _skip_compile: bool = False,
    ):
        if evaluation_type not in ("most_similar", "average"):
            raise ValueError("evaluation_type must be 'most_similar' or 'average'")
        if representation not in ("hidden", "logits"):
            raise ValueError("representation must be 'hidden' or 'logits'")
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if self.tokenizer.mask_token is None or self.tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer has no mask_token/mask_token_id. Requires masked-LM tokenizer.")

        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.representation: Representation = representation
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.evaluation_type = evaluation_type

        self.segmenter_config = segmenter_config or {}
        if segmenter is not None:
            self.segmenter = segmenter
            self.segmenter_name = segmenter_name
        else:
            seg, seg_name_meta, seg_conf_meta = _build_segmenter(segmenter_name, config=self.segmenter_config)
            self.segmenter = seg
            self.segmenter_name = seg_name_meta
            self.segmenter_config = seg_conf_meta

        self.mask_aliases = mask_aliases or ["[MASK]", "[ MASK ]", "[mask]", "[ mask ]"]

        # model
        if self.representation == "hidden":
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        self.model.eval()
        self.model.to(self.device)

        self.references = deepcopy(references)

        self.reference_vectors: Dict[str, Dict[str, List[Tuple[np.ndarray, str]]]] = {}
        if not _skip_compile:
            self._prepare_references_inplace()
            self.reference_vectors = self._calculate_reference_vectors()

    def _normalize_mask_in_text(self, text: str) -> str:
        s = text
        for a in self.mask_aliases:
            s = s.replace(a, self.tokenizer.mask_token)
        s = self._MASK_VARIANT_RE.sub(self.tokenizer.mask_token, s)
        mt_pat = re.escape(self.tokenizer.mask_token)
        s = re.sub(mt_pat, f" {self.tokenizer.mask_token} ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _split_reference(self, text: str) -> str:
        s = self._normalize_mask_in_text(text)
        mrphs = self.segmenter.tokenize(s)
        spaced = " ".join([m.surf for m in mrphs]).strip()
        spaced = self._normalize_mask_in_text(spaced)
        return spaced

    def _prepare_references_inplace(self) -> None:
        for key, values in self.references.items():
            for reading, texts in values.items():
                self.references[key][reading] = [self._split_reference(t) for t in texts]

    def update_references(self, references: Dict[str, Dict[str, List[str]]]) -> None:
        self.references = deepcopy(references)
        self._prepare_references_inplace()
        self.reference_vectors = self._calculate_reference_vectors()

    def _calculate_reference_vectors(self) -> Dict[str, Dict[str, List[Tuple[np.ndarray, str]]]]:
        tqdm = _get_tqdm()
        it_desc = (
            f"Compiling references ({self.representation}, bs={self.batch_size}, device={self.device}, "
            f"segmenter={self.segmenter_name})"
        )
        use_tqdm = (tqdm is not None) and self.show_progress

        reference_vectors: Dict[str, Dict[str, List[Tuple[np.ndarray, str]]]] = {}

        flat_items: List[Tuple[str, str, str]] = []
        for key, readings in self.references.items():
            for reading, examples in readings.items():
                for text in examples:
                    flat_items.append((key, reading, text))

        if use_tqdm:
            pbar = tqdm(total=len(flat_items), desc=it_desc)
        else:
            pbar = None
            if self.show_progress:
                print(it_desc, file=sys.stderr)
                print(f"Total examples: {len(flat_items)}", file=sys.stderr)

        for key in self.references.keys():
            reference_vectors[key] = {}
            for reading in self.references[key].keys():
                reference_vectors[key][reading] = []

        if not flat_items:
            if pbar is not None:
                pbar.close()
            return reference_vectors

        for i in range(0, len(flat_items), self.batch_size):
            batch = flat_items[i : i + self.batch_size]
            texts = [t for _, _, t in batch]

            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)
                if self.representation == "hidden":
                    hs = outputs.last_hidden_state
                    lg = None
                else:
                    lg = getattr(outputs, "logits", None)
                    if lg is None:
                        raise ValueError("Model outputs have no logits; use --repr hidden")
                    hs = None

            input_ids = inputs["input_ids"]
            mask_id = self.tokenizer.mask_token_id
            mask_positions = (input_ids == mask_id).nonzero(as_tuple=False)

            first_mask_pos: List[Optional[int]] = [None] * len(batch)
            for b, t in mask_positions.tolist():
                if first_mask_pos[b] is None:
                    first_mask_pos[b] = t

            for bi, (key, reading, text) in enumerate(batch):
                t = first_mask_pos[bi]
                if t is None:
                    raise ValueError(
                        "Reference text has no mask token after normalization.\n"
                        f"  key={key}\n  reading={reading}\n  text='{text}'\n"
                        f"  mask_token='{self.tokenizer.mask_token}'"
                    )

                if self.representation == "hidden":
                    assert hs is not None
                    vec_t = hs[bi, t].detach().to("cpu").numpy()
                else:
                    assert lg is not None
                    vec_t = lg[bi, t].detach().to("cpu").numpy()

                reference_vectors[key][reading].append((vec_t, text))

            if pbar is not None:
                pbar.update(len(batch))

        if pbar is not None:
            pbar.close()
        return reference_vectors

    def _get_most_similar_reading(self, key: str, vec: np.ndarray) -> Optional[str]:
        if key not in self.reference_vectors:
            return None
        max_similarity = -1e9
        predicted: Optional[str] = None
        for reading, values in self.reference_vectors[key].items():
            for ref_vec, _ in values:
                sim = _cosine_similarity(vec, ref_vec)
                if sim > max_similarity:
                    max_similarity = sim
                    predicted = reading
        return predicted

    def _get_average_similar_reading(self, key: str, vec: np.ndarray) -> Optional[str]:
        if key not in self.reference_vectors:
            return None
        max_similarity = -1e9
        predicted: Optional[str] = None
        for reading, values in self.reference_vectors[key].items():
            if not values:
                continue
            sims = [_cosine_similarity(vec, ref_vec) for ref_vec, _ in values]
            sim = float(sum(sims) / len(sims))
            if sim > max_similarity:
                max_similarity = sim
                predicted = reading
        return predicted

    def _infer_mask_vectors(self, masked_texts: List[str]) -> List[np.ndarray]:
        if not masked_texts:
            return []

        inputs = self.tokenizer(masked_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            if self.representation == "hidden":
                hs = outputs.last_hidden_state
                lg = None
            else:
                lg = getattr(outputs, "logits", None)
                if lg is None:
                    raise ValueError("Model outputs have no logits; use --repr hidden")
                hs = None

        input_ids = inputs["input_ids"]
        mask_id = self.tokenizer.mask_token_id
        mask_positions = (input_ids == mask_id).nonzero(as_tuple=False)

        first_mask_pos: List[Optional[int]] = [None] * input_ids.size(0)
        for b, t in mask_positions.tolist():
            if first_mask_pos[b] is None:
                first_mask_pos[b] = t

        vecs: List[np.ndarray] = []
        for bi in range(input_ids.size(0)):
            t = first_mask_pos[bi]
            if t is None:
                raise ValueError(
                    "No mask token found in a masked_text (after tokenization).\n"
                    f"  masked_text='{masked_texts[bi]}'\n"
                    f"  mask_token='{self.tokenizer.mask_token}'"
                )

            if self.representation == "hidden":
                assert hs is not None
                vec = hs[bi, t].detach().to("cpu").numpy()
            else:
                assert lg is not None
                vec = lg[bi, t].detach().to("cpu").numpy()
            vecs.append(vec)

        return vecs

    def _infer_mask_vector(self, masked_text: str) -> np.ndarray:
        return self._infer_mask_vectors([masked_text])[0]

    def get_reading_prediction(self, text: str) -> List[Tuple[str, str]]:
        return self.get_reading_predictions([text])[0]

    def get_reading_predictions(self, texts: List[str]) -> List[List[Tuple[str, str]]]:
        if not texts:
            return []

        outputs: List[List[Tuple[str, str]]] = []
        all_masked_texts: List[str] = []
        masked_map: List[Tuple[int, int, str, str]] = []

        for ti, text in enumerate(texts):
            mrphs = self.segmenter.tokenize(text)

            pairs: List[Tuple[str, str]] = []
            for m in mrphs:
                y = m.reading if m.reading else m.surf
                pairs.append((m.surf, y))
            outputs.append(pairs)

            counts: Dict[str, int] = {}
            for m in mrphs:
                counts[m.surf] = counts.get(m.surf, 0) + 1

            for idx, m in enumerate(mrphs):
                surf = m.surf
                if surf in self.references and counts.get(surf, 0) == 1:
                    masked_spaced = " ".join(
                        [self.tokenizer.mask_token if j == idx else mrphs[j].surf for j in range(len(mrphs))]
                    ).strip()
                    masked_spaced = self._normalize_mask_in_text(masked_spaced)
                    all_masked_texts.append(masked_spaced)

                    fallback = m.reading if m.reading else m.surf
                    masked_map.append((ti, idx, surf, fallback))

        if not all_masked_texts:
            return outputs

        chooser = self._get_most_similar_reading if self.evaluation_type == "most_similar" else self._get_average_similar_reading

        inferred_vecs: List[np.ndarray] = []
        for i in range(0, len(all_masked_texts), self.batch_size):
            inferred_vecs.extend(self._infer_mask_vectors(all_masked_texts[i : i + self.batch_size]))

        for mi, (ti, token_idx, surf, fallback_yomi) in enumerate(masked_map):
            vec = inferred_vecs[mi]
            pred = chooser(surf, vec)
            word, _ = outputs[ti][token_idx]
            outputs[ti][token_idx] = (word, pred if pred is not None else fallback_yomi)

        return outputs

    def get_optimized_reading(self, word: str, left_context: str, right_context: str, current_reading: str) -> str:
        if word not in self.references:
            return current_reading

        left_m = self.segmenter.tokenize(left_context)
        right_m = self.segmenter.tokenize(right_context)
        left_spaced = " ".join([m.surf for m in left_m]).strip()
        right_spaced = " ".join([m.surf for m in right_m]).strip()

        masked_text = f"{left_spaced} {self.tokenizer.mask_token} {right_spaced}".strip()
        masked_text = self._normalize_mask_in_text(masked_text)

        vec = self._infer_mask_vector(masked_text)
        predicted = self._get_most_similar_reading(word, vec)
        return predicted if predicted is not None else current_reading

    def save_compiled(self, path: str = "compiled_data.pkl") -> None:
        data = {
            "model_name": self.model_name,
            "evaluation_type": self.evaluation_type,
            "representation": self.representation,
            "batch_size": self.batch_size,
            "mask_aliases": self.mask_aliases,
            "references": self.references,
            "reference_vectors": self.reference_vectors,
            "segmenter_name": self.segmenter_name,
            "segmenter_config": self.segmenter_config,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Compiled data saved to {path}")

    @staticmethod
    def load_compiled(
        path: str = "compiled_data.pkl",
        *,
        device: str = "cpu",
        batch_size: int = 16,
        show_progress: bool = False,
        segmenter_name: Optional[str] = None,
        segmenter_config: Optional[Dict[str, Any]] = None,
    ) -> "ReadingEstimator":
        if not os.path.exists(path):
            raise FileNotFoundError(f"No compiled data file found at {path}")

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        seg_name = segmenter_name if segmenter_name is not None else loaded.get("segmenter_name", "auto")
        seg_conf = segmenter_config if segmenter_config is not None else loaded.get("segmenter_config", {}) or {}

        obj = ReadingEstimator(
            model_name=loaded["model_name"],
            references={},
            evaluation_type=loaded.get("evaluation_type", "most_similar"),
            device=device,
            representation=loaded.get("representation", "hidden"),
            batch_size=batch_size,
            show_progress=show_progress,
            mask_aliases=loaded.get("mask_aliases", None),
            segmenter_name=seg_name,
            segmenter_config=seg_conf,
            _skip_compile=True,
        )
        obj.references = loaded["references"]
        obj.reference_vectors = loaded["reference_vectors"]
        return obj


# -------------------------
# CLI helpers
# -------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ReadingEstimator CLI (segmenter-agnostic + ruri)")

    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu/cuda")
    p.add_argument("--batch-size", type=int, default=16, help="batch size for compiling & inference")
    p.add_argument("--no-progress", action="store_true", help="disable progress output")

    # segmenter
    p.add_argument(
        "--segmenter",
        default="auto",
        help="segmenter backend: auto/sudachi/juman/whitespace OR plugin 'module:ClassName'",
    )
    p.add_argument("--sudachi-mode", default="C", choices=["A", "B", "C"], help="Sudachi split mode (A/B/C)")

    # Juman config
    p.add_argument(
        "--juman-timeout",
        type=float,
        default=10.0,
        help="Juman++ timeout seconds for apply_to_sentence (default: 10.0)",
    )
    p.add_argument("--juman-rcfile", default=None, help="Optional path to jumanpp rcfile (passed as -r <rcfile>)")

    # NEW: startup options
    p.add_argument(
        "--juman-skip-sanity-check",
        action="store_true",
        help="Skip rhoknp's Juman++ startup sanity check (recommended if startup times out).",
    )
    p.add_argument(
        "--juman-executable",
        default="jumanpp",
        help="Juman++ executable (default: jumanpp). You can set full path like /usr/local/bin/jumanpp.",
    )
    p.add_argument(
        "--juman-options",
        default=None,
        help='Extra options for jumanpp as JSON array. Example: \'["-s","1"]\'',
    )

    # Plugin config (generic)
    p.add_argument(
        "--segmenter-config",
        default=None,
        help="JSON string for plugin/segmenter config. Example: '{\"foo\":1,\"bar\":\"x\"}'",
    )

    # model
    p.add_argument("--model", default="cl-nagoya/ruri-v3-30m", help="HF model name")
    p.add_argument("--eval", default="most_similar", choices=["most_similar", "average"], help="evaluation type")
    p.add_argument("--repr", default="hidden", choices=["hidden", "logits"], help="representation: hidden / logits")

    # compiled or references
    p.add_argument("--compiled", default=None, help="compiled pkl path to load")
    p.add_argument("--references", default=None, help="references.json path (used when not using --compiled)")
    p.add_argument("--build-compiled", default=None, help="output path to save compiled pkl, then exit")

    # input (multi)
    p.add_argument("-t", "--text", action="append", default=None, help="input text (can be specified multiple times)")
    p.add_argument(
        "-f",
        "--file",
        action="append",
        default=None,
        help="read input texts from file (utf-8). one line = one input. can be specified multiple times",
    )
    p.add_argument("--stdin", action="store_true", help="read additional input texts from stdin (one line = one input)")
    p.add_argument("--interactive", action="store_true", help="interactive REPL mode (single-line loop)")

    # output
    p.add_argument(
        "--format",
        default="reading",
        choices=["reading", "pairs", "json", "jsonl"],
        help="output format: reading/pairs/json/jsonl",
    )
    p.add_argument("--out", default=None, help="output file path (default: stdout)")

    # timing
    p.add_argument("--time", action="store_true", help="print inference time (seconds) to stderr")
    p.add_argument("--load-time", action="store_true", help="print model/loading time (seconds) to stderr")

    # optimized reading
    p.add_argument("--opt-word", default=None, help="word for get_optimized_reading (e.g. 水)")
    p.add_argument("--opt-left", default="", help="left context")
    p.add_argument("--opt-right", default="", help="right context")
    p.add_argument("--opt-current", default="", help="current reading")

    return p


def _load_references(path: str) -> Dict[str, Dict[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_lines_from_file(path: str) -> List[str]:
    txt = Path(path).read_text(encoding="utf-8")
    lines = [line.strip() for line in txt.splitlines()]
    return [line for line in lines if line]


def _collect_inputs(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []
    if args.text:
        for t in args.text:
            if t is None:
                continue
            s = str(t).strip()
            if s:
                texts.append(s)
    if args.file:
        for fp in args.file:
            if fp is None:
                continue
            texts.extend(_read_lines_from_file(fp))
    if args.stdin:
        piped = sys.stdin.read()
        lines = [line.strip() for line in piped.splitlines()]
        texts.extend([line for line in lines if line])
    return texts


def _format_one(predicted_pairs: List[Tuple[str, str]], fmt: str) -> str:
    if fmt == "reading":
        return "".join([yomi for _, yomi in predicted_pairs])
    if fmt == "pairs":
        return " ".join([f"{w}:{y}" for w, y in predicted_pairs])
    return ""


def _write_output(out_path: Optional[str], content: str) -> None:
    if out_path:
        Path(out_path).write_text(content, encoding="utf-8")
    else:
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")


def _merge_segmenter_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Base config = CLI dedicated flags + optional JSON config string
    JSON config overrides.
    """
    conf: Dict[str, Any] = {
        "sudachi_mode": args.sudachi_mode,
        "juman_timeout": args.juman_timeout,
        "juman_executable": args.juman_executable,
        # IMPORTANT: default True (avoid startup hang); allow override
        "juman_skip_sanity_check": True if args.juman_skip_sanity_check else True,
    }

    if args.juman_rcfile:
        conf["juman_rcfile"] = args.juman_rcfile

    if args.juman_options:
        try:
            opt = json.loads(args.juman_options)
            if not isinstance(opt, list):
                raise ValueError("juman-options must be a JSON array")
            conf["juman_options"] = opt
        except Exception as e:
            raise SystemExit(f"ERROR: --juman-options is invalid JSON array: {e}")

    if args.segmenter_config:
        try:
            extra = json.loads(args.segmenter_config)
            if not isinstance(extra, dict):
                raise ValueError("segmenter-config must be a JSON object")
            conf.update(extra)
        except Exception as e:
            raise SystemExit(f"ERROR: --segmenter-config is invalid JSON: {e}")

    return conf


def main() -> None:
    args = _build_argparser().parse_args()
    show_progress = not args.no_progress

    seg_conf = _merge_segmenter_config(args)

    # build-compiled mode
    if args.build_compiled is not None:
        if not args.references:
            raise SystemExit("ERROR: --build-compiled requires --references references.json")

        refs = _load_references(args.references)

        t0_load = time.perf_counter()
        predictor = ReadingEstimator(
            model_name=args.model,
            references=refs,
            evaluation_type=args.eval,
            device=args.device,
            representation=args.repr,
            batch_size=args.batch_size,
            show_progress=show_progress,
            segmenter_name=args.segmenter,
            segmenter_config=seg_conf,
        )
        load_elapsed = time.perf_counter() - t0_load

        predictor.save_compiled(args.build_compiled)

        if args.load_time:
            print(f"[time] load_sec={load_elapsed:.6f}", file=sys.stderr)
        return

    # load predictor
    t0_load = time.perf_counter()
    if args.compiled:
        predictor = ReadingEstimator.load_compiled(
            args.compiled,
            device=args.device,
            batch_size=args.batch_size,
            show_progress=False,
            segmenter_name=args.segmenter,
            segmenter_config=seg_conf,
        )
    else:
        if not args.references:
            raise SystemExit("ERROR: Provide --compiled predictor.pkl OR --references references.json")
        refs = _load_references(args.references)
        predictor = ReadingEstimator(
            model_name=args.model,
            references=refs,
            evaluation_type=args.eval,
            device=args.device,
            representation=args.repr,
            batch_size=args.batch_size,
            show_progress=show_progress,
            segmenter_name=args.segmenter,
            segmenter_config=seg_conf,
        )
    load_elapsed = time.perf_counter() - t0_load

    if args.load_time:
        print(f"[time] load_sec={load_elapsed:.6f}", file=sys.stderr)

    # optimized reading
    if args.opt_word is not None:
        t0 = time.perf_counter()
        res = predictor.get_optimized_reading(
            word=args.opt_word,
            left_context=args.opt_left,
            right_context=args.opt_right,
            current_reading=args.opt_current,
        )
        elapsed = time.perf_counter() - t0

        _write_output(args.out, res + "\n")
        if args.time:
            print(f"[time] optimized_infer_sec={elapsed:.6f}", file=sys.stderr)
        return

    # interactive
    if args.interactive:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue

            t0 = time.perf_counter()
            predicted = predictor.get_reading_prediction(line)
            elapsed = time.perf_counter() - t0

            if args.format in ("reading", "pairs"):
                out = _format_one(predicted, args.format)
                _write_output(args.out, out + "\n")
            elif args.format == "json":
                out = json.dumps([{"word": w, "reading": y} for w, y in predicted], ensure_ascii=False)
                _write_output(args.out, out + "\n")
            else:  # jsonl
                out = json.dumps(
                    {"input": line, "tokens": [{"word": w, "reading": y} for w, y in predicted]},
                    ensure_ascii=False,
                )
                _write_output(args.out, out + "\n")

            if args.time:
                print(f"[time] infer_sec={elapsed:.6f}", file=sys.stderr)
        return

    # batch inputs
    inputs = _collect_inputs(args)
    if not inputs:
        raise SystemExit("ERROR: No input. Provide --text/--file/--stdin or use --interactive.")

    t0 = time.perf_counter()
    batch_predicted = predictor.get_reading_predictions(inputs)
    elapsed = time.perf_counter() - t0

    if args.format in ("reading", "pairs"):
        lines = [_format_one(pairs, args.format) for pairs in batch_predicted]
        _write_output(args.out, "\n".join(lines) + "\n")
    elif args.format == "json":
        obj: List[Any] = []
        for inp, pairs in zip(inputs, batch_predicted):
            obj.append({"input": inp, "tokens": [{"word": w, "reading": y} for w, y in pairs]})
        _write_output(args.out, json.dumps(obj, ensure_ascii=False) + "\n")
    else:  # jsonl
        out_lines: List[str] = []
        for inp, pairs in zip(inputs, batch_predicted):
            out_lines.append(
                json.dumps({"input": inp, "tokens": [{"word": w, "reading": y} for w, y in pairs]}, ensure_ascii=False)
            )
        _write_output(args.out, "\n".join(out_lines) + "\n")

    if args.time:
        print(f"[time] infer_total_sec={elapsed:.6f} (n_inputs={len(inputs)})", file=sys.stderr)


if __name__ == "__main__":
    main()
