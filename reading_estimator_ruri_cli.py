from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

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


class ReadingEstimator:
    """
    references.json の例（構造）:
    {
      "水": {
        "みず": ["油は[MASK]を弾く", "生理食塩[MASK]"],
        "すい": ["[MASK]溶液", "王[MASK]は金も溶かす"]
      }
    }

    - Juman++ は使わない（重いので排除）。
    - text は「生テキスト」のまま tokenizer に渡す。
    - "[MASK]" / "[ MASK ]" 等は tokenizer.mask_token に置換して統一する。
    """

    def __init__(
        self,
        model_name: str,
        references: Dict[str, Dict[str, List[str]]],
        evaluation_type: str = "most_similar",
        device: str = "cpu",
        representation: Representation = "hidden",
        batch_size: int = 16,
        show_progress: bool = True,
        mask_aliases: Optional[List[str]] = None,
    ):
        if evaluation_type not in ("most_similar", "average"):
            raise ValueError("evaluation_type must be 'most_similar' or 'average'")
        if representation not in ("hidden", "logits"):
            raise ValueError("representation must be 'hidden' or 'logits'")
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # ruri-v3 は mask_token が "<mask>" のはず（モデルに依存）
        if self.tokenizer.mask_token is None or self.tokenizer.mask_token_id is None:
            raise ValueError(
                "Tokenizer has no mask_token/mask_token_id. "
                "This script requires a masked-LM style tokenizer."
            )

        self.device = device
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.representation: Representation = representation
        self.batch_size = batch_size
        self.show_progress = show_progress

        # model load (representation によって変える)
        if self.representation == "hidden":
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        self.model.eval()
        self.model.to(self.device)

        self.references = deepcopy(references)
        self.evaluation_type = evaluation_type

        # "[MASK]" 表記ゆれ
        self.mask_aliases = mask_aliases or ["[MASK]", "[ MASK ]", "[mask]", "[ mask ]"]

        # references を mask置換（Juman++分割はしない）
        for key, values in self.references.items():
            for reading, texts in values.items():
                self.references[key][reading] = [self._normalize_mask(t) for t in texts]

        # 参照ベクトルを前計算
        self.reference_vectors = self._calculate_reference_vectors()

    def update_references(self, references: Dict[str, Dict[str, List[str]]]) -> None:
        self.references = deepcopy(references)
        for key, values in self.references.items():
            for reading, texts in values.items():
                self.references[key][reading] = [self._normalize_mask(t) for t in texts]
        self.reference_vectors = self._calculate_reference_vectors()

    def _normalize_mask(self, text: str) -> str:
        # 生テキスト上で mask 表記を tokenizer.mask_token に統一
        s = text
        for a in self.mask_aliases:
            s = s.replace(a, self.tokenizer.mask_token)
        return s

    def _calculate_reference_vectors(
        self,
    ) -> Dict[str, Dict[str, List[Tuple[np.ndarray, str]]]]:
        """
        reference_vectors[kanji][reading] = [(vec, text), ...]
        vec は
          representation="hidden" -> (hidden_size,)
          representation="logits" -> (vocab_size,)
        """
        tqdm = _get_tqdm()

        it_desc = (
            f"Compiling references ({self.representation}, bs={self.batch_size}, device={self.device})"
        )
        use_tqdm = (tqdm is not None) and self.show_progress

        reference_vectors: Dict[str, Dict[str, List[Tuple[np.ndarray, str]]]] = {}

        flat_items: List[Tuple[str, str, str]] = []  # (kanji, reading, text)
        for kanji, readings in self.references.items():
            for reading, examples in readings.items():
                for text in examples:
                    flat_items.append((kanji, reading, text))

        if use_tqdm:
            pbar = tqdm(total=len(flat_items), desc=it_desc)
        else:
            pbar = None
            if self.show_progress:
                print(it_desc)
                print(f"Total examples: {len(flat_items)}")

        for kanji in self.references.keys():
            reference_vectors[kanji] = {}
            for reading in self.references[kanji].keys():
                reference_vectors[kanji][reading] = []

        if len(flat_items) == 0:
            if pbar is not None:
                pbar.close()
            return reference_vectors

        # バッチ推論
        for i in range(0, len(flat_items), self.batch_size):
            batch = flat_items[i : i + self.batch_size]
            texts = [t for _, _, t in batch]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)
                if self.representation == "hidden":
                    hs = outputs.last_hidden_state  # (B, T, H)
                    lg = None
                else:
                    # AutoModelForMaskedLM の場合 logits がある
                    lg = getattr(outputs, "logits", None)
                    if lg is None:
                        raise ValueError("Model outputs have no logits; use --repr hidden")
                    hs = None

            input_ids = inputs["input_ids"]  # (B, T)
            mask_id = self.tokenizer.mask_token_id
            mask_positions = (input_ids == mask_id).nonzero(as_tuple=False)

            first_mask_pos: List[Optional[int]] = [None] * len(batch)
            for b, t in mask_positions.tolist():
                if first_mask_pos[b] is None:
                    first_mask_pos[b] = t

            for bi, (kanji, reading, text) in enumerate(batch):
                t = first_mask_pos[bi]
                if t is None:
                    raise ValueError(f"Reference text has no mask token: {text}")

                if self.representation == "hidden":
                    assert hs is not None
                    vec_t = hs[bi, t].detach().to("cpu").numpy()
                else:
                    assert lg is not None
                    vec_t = lg[bi, t].detach().to("cpu").numpy()

                reference_vectors[kanji][reading].append((vec_t, text))

            if pbar is not None:
                pbar.update(len(batch))
            else:
                if self.show_progress and (i == 0 or (i // self.batch_size) % 20 == 0):
                    done = min(i + self.batch_size, len(flat_items))
                    print(f"  {done}/{len(flat_items)} examples processed...")

        if pbar is not None:
            pbar.close()

        return reference_vectors

    def _get_most_similar_reading(self, kanji: str, vec: np.ndarray) -> Optional[str]:
        max_similarity = -1e9
        predicted_reading: Optional[str] = None

        if kanji not in self.reference_vectors:
            return None

        for reading, values in self.reference_vectors[kanji].items():
            for ref_vec, _text in values:
                sim = _cosine_similarity(vec, ref_vec)
                if sim > max_similarity:
                    max_similarity = sim
                    predicted_reading = reading

        return predicted_reading

    def _get_average_similar_reading(
        self, kanji: str, vec: np.ndarray
    ) -> Optional[str]:
        max_similarity = -1e9
        predicted_reading: Optional[str] = None

        if kanji not in self.reference_vectors:
            return None

        for reading, values in self.reference_vectors[kanji].items():
            if not values:
                continue
            sims = [_cosine_similarity(vec, ref_vec) for ref_vec, _ in values]
            sim = float(sum(sims) / len(sims))
            if sim > max_similarity:
                max_similarity = sim
                predicted_reading = reading

        return predicted_reading

    def _infer_mask_vectors(self, masked_texts: List[str]) -> List[np.ndarray]:
        if len(masked_texts) == 0:
            return []

        inputs = self.tokenizer(
            masked_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
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
                raise ValueError("No mask token found in a masked_text (after tokenization).")

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
        """
        Juman++を使わないので「形態素ごとの (surf, reading)」は作れない。
        代わりに、この関数は:
        - 入力文中の「references に含まれるキー文字（例: 水）」を探す
        - その各位置を 1 文字だけ mask した文を作る（最初の出現だけ）
        - 参照ベクトルと比較して読みを返す
        出力は [(token, reading)] ではなく「文字単位の推定結果リスト」を返す形に変更する。
        ただし互換のため、ここでは:
          - 基本は text 全体に対して、推定できたものだけ dict で返す形式を採用する。
        """
        if len(texts) == 0:
            return []

        # 互換性のため、戻り値は「pairs風」にする:
        # [(original_text, json_string_of_predictions)]
        # CLI側で format=reading/pairs/json/jsonl を整形する。
        results: List[List[Tuple[str, str]]] = []

        all_masked_texts: List[str] = []
        masked_map: List[Tuple[int, str, str]] = []  # (ti, target_char, fallback)

        # 参照対象（単漢字キー）を先に集める
        keys = list(self.references.keys())

        for ti, text in enumerate(texts):
            # 見つかった推定をここに入れる
            # 例: {"水": "みず"}
            # fallback は未推定時に空文字にする
            found: Dict[str, str] = {}

            # 文中の各キーについて最初の出現位置だけ推定（重いので）
            for k in keys:
                pos = text.find(k)
                if pos < 0:
                    continue

                # 1文字mask（kが複数文字なら全体をmaskしたいが、あなたの例は単漢字キー前提）
                masked_text = text[:pos] + self.tokenizer.mask_token + text[pos + len(k) :]
                masked_text = self._normalize_mask(masked_text)

                all_masked_texts.append(masked_text)
                masked_map.append((ti, k, ""))  # fallback空
                found[k] = ""  # placeholder

            # いったん placeholder を results に入れておく（後で埋める）
            results.append([(text, json.dumps(found, ensure_ascii=False))])

        if len(all_masked_texts) == 0:
            return results

        chooser = (
            self._get_most_similar_reading
            if self.evaluation_type == "most_similar"
            else self._get_average_similar_reading
        )

        inferred_vecs: List[np.ndarray] = []
        for i in range(0, len(all_masked_texts), self.batch_size):
            chunk = all_masked_texts[i : i + self.batch_size]
            inferred_vecs.extend(self._infer_mask_vectors(chunk))

        # results を復元して該当キーを埋める
        # results[ti] は [(text, json_str)]
        # json_str を dict に戻して更新
        for mi, (ti, key_char, _fallback) in enumerate(masked_map):
            vec = inferred_vecs[mi]
            pred = chooser(key_char, vec)

            text, json_str = results[ti][0]
            d = json.loads(json_str)
            d[key_char] = pred if pred is not None else ""
            results[ti][0] = (text, json.dumps(d, ensure_ascii=False))

        return results

    def save_compiled(self, path: str = "compiled_data.pkl") -> None:
        data = {
            "model_name": self.model_name,
            "evaluation_type": self.evaluation_type,
            "representation": self.representation,
            "reference_vectors": self.reference_vectors,
            "references": self.references,
            "mask_aliases": self.mask_aliases,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Compiled data saved to {path}")

    @staticmethod
    def load_compiled(
        path: str = "compiled_data.pkl",
        device: str = "cpu",
        batch_size: int = 16,
        show_progress: bool = False,
    ) -> "ReadingEstimator":
        if not os.path.exists(path):
            raise FileNotFoundError(f"No compiled data file found at {path}")

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        obj = ReadingEstimator(
            model_name=loaded["model_name"],
            references=loaded.get("references", {}),
            evaluation_type=loaded.get("evaluation_type", "most_similar"),
            device=device,
            representation=loaded.get("representation", "hidden"),
            batch_size=batch_size,
            show_progress=show_progress,
            mask_aliases=loaded.get("mask_aliases", None),
        )
        # ここで再計算しないよう上書き
        obj.reference_vectors = loaded["reference_vectors"]
        obj.references = loaded["references"]
        return obj


# -------------------------
# CLI helpers
# -------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ReadingEstimator (NO Juman++) CLI for ruri-v3-310m"
    )

    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu/cuda")
    p.add_argument("--batch-size", type=int, default=16, help="batch size")
    p.add_argument("--no-progress", action="store_true", help="disable progress output")

    p.add_argument(
        "--model",
        default="cl-nagoya/ruri-v3-310m",
        help="HF model name (default: cl-nagoya/ruri-v3-310m)",
    )
    p.add_argument(
        "--eval",
        default="most_similar",
        choices=["most_similar", "average"],
        help="evaluation type",
    )
    p.add_argument(
        "--repr",
        default="hidden",
        choices=["hidden", "logits"],
        help="representation: hidden / logits",
    )

    p.add_argument("--compiled", default=None, help="compiled pkl path to load")
    p.add_argument("--references", default=None, help="references.json path")

    p.add_argument(
        "--build-compiled",
        default=None,
        help="output path to save compiled pkl, then exit",
    )

    p.add_argument(
        "-t",
        "--text",
        action="append",
        default=None,
        help="input text (can be specified multiple times)",
    )
    p.add_argument(
        "-f",
        "--file",
        action="append",
        default=None,
        help="read input texts from file (utf-8). one line = one input. can be specified multiple times",
    )
    p.add_argument("--stdin", action="store_true", help="read additional input texts from stdin")

    p.add_argument(
        "--format",
        default="jsonl",
        choices=["json", "jsonl"],
        help="output format (this no-juman version outputs per-input dict): json/jsonl",
    )
    p.add_argument("--out", default=None, help="output file path (default: stdout)")

    p.add_argument("--time", action="store_true", help="print inference time (seconds) to stderr")
    p.add_argument("--load-time", action="store_true", help="print model/loading time (seconds) to stderr")

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


def _write_output(out_path: Optional[str], content: str) -> None:
    if out_path:
        Path(out_path).write_text(content, encoding="utf-8")
    else:
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")


def main() -> None:
    args = _build_argparser().parse_args()
    show_progress = not args.no_progress

    # 1) build-compiled
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
        )
        load_elapsed = time.perf_counter() - t0_load

        predictor.save_compiled(args.build_compiled)

        if args.load_time:
            print(f"[time] load_sec={load_elapsed:.6f}", file=sys.stderr)
        return

    # 2) load predictor
    t0_load = time.perf_counter()
    if args.compiled:
        predictor = ReadingEstimator.load_compiled(
            args.compiled,
            device=args.device,
            batch_size=args.batch_size,
            show_progress=False,
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
        )
    load_elapsed = time.perf_counter() - t0_load

    if args.load_time:
        print(f"[time] load_sec={load_elapsed:.6f}", file=sys.stderr)

    inputs = _collect_inputs(args)
    if len(inputs) == 0:
        raise SystemExit("ERROR: No input. Provide --text/--file/--stdin.")

    t0 = time.perf_counter()
    batch_predicted = predictor.get_reading_predictions(inputs)
    elapsed = time.perf_counter() - t0

    # batch_predicted: List[ [(text, json_dict_str)] ]
    if args.format == "json":
        obj: List[Any] = []
        for item in batch_predicted:
            text, pred_json = item[0]
            obj.append({"input": text, "predictions": json.loads(pred_json)})
        _write_output(args.out, json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        lines: List[str] = []
        for item in batch_predicted:
            text, pred_json = item[0]
            lines.append(
                json.dumps(
                    {"input": text, "predictions": json.loads(pred_json)},
                    ensure_ascii=False,
                )
            )
        _write_output(args.out, "\n".join(lines) + "\n")

    if args.time:
        print(f"[time] infer_total_sec={elapsed:.6f} (n_inputs={len(inputs)})", file=sys.stderr)


if __name__ == "__main__":
    main()
