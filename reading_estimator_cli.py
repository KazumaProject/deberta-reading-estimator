from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal, Any

import numpy as np
import torch
from pyknp import Juman
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel


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
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
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

    - text は _split_reference() で Juman により形態素分割され、スペース区切りになります。
    - "[MASK]" または "[ MASK ]" は tokenizer.mask_token に置換されます。
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
    ):
        """
        Args:
            model_name: HF model name
            references: 参照データ
            evaluation_type: most_similar / average
            device: cpu / cuda
            representation:
                - "hidden": [MASK]位置の隠れ状態ベクトルで比較（速い＆pkl軽い 推奨）
                - "logits": 語彙logitsベクトルで比較（pkl重い＆遅いが元の挙動に近い）
            batch_size: 参照計算時・推論時のバッチサイズ
            show_progress: 進捗表示するか
        """
        if evaluation_type not in ("most_similar", "average"):
            raise ValueError(
                "evaluation_type must be 'most_similar' or 'average'")
        if representation not in ("hidden", "logits"):
            raise ValueError("representation must be 'hidden' or 'logits'")
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        self.jumanpp = Juman()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            # LM head を使わないので軽い
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        self.model.eval()
        self.model.to(self.device)

        self.references = deepcopy(references)
        self.evaluation_type = evaluation_type

        # references を形態素分割＆mask置換
        for key, values in self.references.items():
            for reading, texts in values.items():
                self.references[key][reading] = [
                    self._split_reference(text) for text in texts]

        # 参照ベクトルを前計算
        self.reference_vectors = self._calculate_reference_vectors()

    def update_references(self, references: Dict[str, Dict[str, List[str]]]) -> None:
        self.references = deepcopy(references)
        for key, values in self.references.items():
            for reading, texts in values.items():
                self.references[key][reading] = [
                    self._split_reference(text) for text in texts]
        self.reference_vectors = self._calculate_reference_vectors()

    def _split_reference(self, text: str) -> str:
        result = self.jumanpp.analysis(text)
        spaced = " ".join([mrph.midasi for mrph in result.mrph_list()])

        # 表記ゆれをまとめて mask に
        spaced = spaced.replace("[ MASK ]", self.tokenizer.mask_token)
        spaced = spaced.replace("[MASK]", self.tokenizer.mask_token)
        spaced = spaced.replace("[ mask ]", self.tokenizer.mask_token)
        spaced = spaced.replace("[mask]", self.tokenizer.mask_token)
        return spaced

    def _calculate_reference_vectors(self) -> Dict[str, Dict[str, List[Tuple[np.ndarray, str]]]]:
        """
        reference_vectors[kanji][reading] = [(vec, text), ...]

        vec は
          representation="hidden" -> (hidden_size,) 例: 768
          representation="logits" -> (vocab_size,)  例: 32000
        """
        tqdm = _get_tqdm()

        it_desc = f"Compiling references ({self.representation}, bs={self.batch_size}, device={self.device})"
        use_tqdm = (tqdm is not None) and self.show_progress

        reference_vectors: Dict[str,
                                Dict[str, List[Tuple[np.ndarray, str]]]] = {}

        # すべての例文をフラットにして、バッチ推論してから元の構造に戻す
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

        # 初期化
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
            batch = flat_items[i: i + self.batch_size]
            texts = [t for _, _, t in batch]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                if self.representation == "hidden":
                    outputs = self.model(**inputs)
                    hs = outputs.last_hidden_state  # (B, T, H)
                    lg = None
                else:
                    outputs = self.model(**inputs)
                    lg = outputs.logits  # (B, T, V)
                    hs = None

            # 各サンプルの mask 位置を取得して vec を抜き出す
            input_ids = inputs["input_ids"]  # (B, T)
            mask_id = self.tokenizer.mask_token_id
            mask_positions = (input_ids == mask_id).nonzero(as_tuple=False)

            # サンプルごとに「最初のmask位置」を取る
            first_mask_pos: List[Optional[int]] = [None] * len(batch)
            for b, t in mask_positions.tolist():
                if first_mask_pos[b] is None:
                    first_mask_pos[b] = t

            for bi, (kanji, reading, text) in enumerate(batch):
                t = first_mask_pos[bi]
                if t is None:
                    raise ValueError(
                        f"Reference text has no [MASK] after conversion: {text}")

                if self.representation == "hidden":
                    assert hs is not None
                    vec_t = hs[bi, t].detach().to("cpu").numpy()  # (H,)
                else:
                    assert lg is not None
                    vec_t = lg[bi, t].detach().to("cpu").numpy()  # (V,)

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

    def _get_average_similar_reading(self, kanji: str, vec: np.ndarray) -> Optional[str]:
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
        """
        masked_texts: 「スペース区切り形態素」かつ mask_token を含むテキスト群
        まとめて tokenizer → model を回して [MASK] 位置ベクトルだけ取り出す（高速）。
        """
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
            if self.representation == "hidden":
                outputs = self.model(**inputs)
                hs = outputs.last_hidden_state  # (B, T, H)
                lg = None
            else:
                outputs = self.model(**inputs)
                lg = outputs.logits  # (B, T, V)
                hs = None

        input_ids = inputs["input_ids"]  # (B, T)
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
                    "No [MASK] token found in a masked_text (after tokenization).")
            if self.representation == "hidden":
                assert hs is not None
                vec = hs[bi, t].detach().to("cpu").numpy()
            else:
                assert lg is not None
                vec = lg[bi, t].detach().to("cpu").numpy()
            vecs.append(vec)

        return vecs

    def _infer_mask_vector(self, masked_text: str) -> np.ndarray:
        # 互換のため残す（内部ではバッチ版推奨）
        return self._infer_mask_vectors([masked_text])[0]

    def get_reading_prediction(self, text: str) -> List[Tuple[str, str]]:
        return self.get_reading_predictions([text])[0]

    def get_reading_predictions(self, texts: List[str]) -> List[List[Tuple[str, str]]]:
        """
        複数テキストをまとめて推論（推論はバッチ化して高速化）。
        - 形態素解析(Juman)はテキストごとに実施
        - 推論(Transformer)は [MASK] 文を集約してバッチ推論
        """
        if len(texts) == 0:
            return []

        # どのトークン(midasi)が文中に何回出ているかは「mrph列」で数える（text.countより安定）
        # ここで "推定対象" の masked_text を全部集める
        all_masked_texts: List[str] = []
        # マッピング: masked_text index -> (text_index, token_index, target_midasi, fallback_yomi)
        masked_map: List[Tuple[int, int, str, str]] = []

        # 出力をまずはJumanの yomi で埋める（必要箇所だけ推定で置換）
        outputs: List[List[Tuple[str, str]]] = []

        for ti, text in enumerate(texts):
            result = self.jumanpp.analysis(text)
            mrphs = result.mrph_list()

            # 初期出力（まずは元のyomi）
            pairs: List[Tuple[str, str]] = [(m.midasi, m.yomi) for m in mrphs]
            outputs.append(pairs)

            # midasi の出現回数（トークン列で数える）
            counts: Dict[str, int] = {}
            for m in mrphs:
                counts[m.midasi] = counts.get(m.midasi, 0) + 1

            # 推定する対象だけ masked_text を作成
            for idx, target in enumerate(mrphs):
                midasi = target.midasi
                if midasi in self.references and counts.get(midasi, 0) == 1:
                    masked_text = " ".join(
                        [self.tokenizer.mask_token if j ==
                            idx else mrphs[j].midasi for j in range(len(mrphs))]
                    ).strip()
                    all_masked_texts.append(masked_text)
                    masked_map.append((ti, idx, midasi, target.yomi))

        if len(all_masked_texts) == 0:
            return outputs

        chooser = (
            self._get_most_similar_reading
            if self.evaluation_type == "most_similar"
            else self._get_average_similar_reading
        )

        # masked_texts を self.batch_size で刻んで推論（GPU/CPUで高速）
        inferred_vecs: List[np.ndarray] = []
        for i in range(0, len(all_masked_texts), self.batch_size):
            chunk = all_masked_texts[i: i + self.batch_size]
            inferred_vecs.extend(self._infer_mask_vectors(chunk))

        # 推定結果を outputs に反映
        for mi, (ti, token_idx, midasi, fallback_yomi) in enumerate(masked_map):
            vec = inferred_vecs[mi]
            pred = chooser(midasi, vec)
            # token_idx の reading を置換
            word, _old = outputs[ti][token_idx]
            outputs[ti][token_idx] = (
                word, pred if pred is not None else fallback_yomi)

        return outputs

    def get_optimized_reading(
        self, word: str, left_context: str, right_context: str, current_reading: str
    ) -> str:
        if word not in self.references:
            return current_reading

        left_result = self.jumanpp.analysis(left_context)
        right_result = self.jumanpp.analysis(right_context)
        left_spaced = " ".join(
            [item.midasi for item in left_result.mrph_list()])
        right_spaced = " ".join(
            [item.midasi for item in right_result.mrph_list()])

        masked_text = f"{left_spaced} {self.tokenizer.mask_token} {right_spaced}".strip(
        )
        vec = self._infer_mask_vector(masked_text)

        predicted = self._get_most_similar_reading(word, vec)
        return predicted if predicted is not None else current_reading

    def save_compiled(self, path: str = "compiled_data.pkl") -> None:
        """
        reference_vectors を pickle 保存
        """
        data = {
            "model_name": self.model_name,
            "evaluation_type": self.evaluation_type,
            "representation": self.representation,
            "reference_vectors": self.reference_vectors,
            "references": self.references,
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
            references={},  # 後から復元
            evaluation_type=loaded.get("evaluation_type", "most_similar"),
            device=device,
            representation=loaded.get("representation", "hidden"),
            batch_size=batch_size,
            show_progress=show_progress,
        )
        obj.reference_vectors = loaded["reference_vectors"]
        obj.references = loaded["references"]
        return obj


# -------------------------
# CLI helpers
# -------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ReadingEstimator CLI (multi-input + batched inference)")

    # 共通
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "cuda"], help="cpu/cuda")
    p.add_argument("--batch-size", type=int, default=16,
                   help="batch size for compiling & inference")
    p.add_argument("--no-progress", action="store_true",
                   help="disable progress output")

    # モデル・方式
    p.add_argument(
        "--model", default="ku-nlp/deberta-v2-base-japanese", help="HF model name")
    p.add_argument("--eval", default="most_similar",
                   choices=["most_similar", "average"], help="evaluation type")
    p.add_argument("--repr", default="hidden",
                   choices=["hidden", "logits"], help="representation: hidden / logits")

    # 予測（compiled を使う or references から作る）
    p.add_argument("--compiled", default=None,
                   help="compiled pkl path to load")
    p.add_argument("--references", default=None,
                   help="references.json path (used when not using --compiled)")

    # コンパイル作成モード（これがあると “pkl生成だけ” で終わる）
    p.add_argument("--build-compiled", default=None,
                   help="output path to save compiled pkl, then exit")

    # 入力（複数）
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
    p.add_argument("--stdin", action="store_true",
                   help="read additional input texts from stdin (one line = one input)")
    p.add_argument("--interactive", action="store_true",
                   help="interactive REPL mode (single-line loop)")

    # 出力
    p.add_argument(
        "--format",
        default="reading",
        choices=["reading", "pairs", "json", "jsonl"],
        help="output format: reading/pairs/json/jsonl",
    )
    p.add_argument("--out", default=None,
                   help="output file path (default: stdout)")

    # timing
    p.add_argument("--time", action="store_true",
                   help="print inference time (seconds) to stderr")
    p.add_argument("--load-time", action="store_true",
                   help="print model/loading time (seconds) to stderr")

    # optimized reading
    p.add_argument("--opt-word", default=None,
                   help="word for get_optimized_reading (e.g. 水)")
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

    # --text (複数回)
    if args.text:
        for t in args.text:
            if t is None:
                continue
            s = str(t).strip()
            if s:
                texts.append(s)

    # --file (複数回) : 1行=1入力
    if args.file:
        for fp in args.file:
            if fp is None:
                continue
            lines = _read_lines_from_file(fp)
            texts.extend(lines)

    # --stdin : 1行=1入力
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
    # json / jsonl は呼び出し元で組み立てる
    return ""


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

    # 1) build-compiled モード（pkl生成のみ）
    if args.build_compiled is not None:
        if not args.references:
            raise SystemExit(
                "ERROR: --build-compiled requires --references references.json")

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

    # 2) predictor をロード（compiled優先）
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
            raise SystemExit(
                "ERROR: Provide --compiled predictor.pkl OR --references references.json")
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

    # optimized reading 単発
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

    # interactive（1行ずつだが内部は単発）
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
                _write_output(args.out, out + "\n" if args.out else out + "\n")
            elif args.format == "json":
                out = json.dumps(
                    [{"word": w, "reading": y} for w, y in predicted],
                    ensure_ascii=False,
                )
                _write_output(args.out, out + "\n")
            else:  # jsonl
                out = json.dumps(
                    {"input": line, "tokens": [
                        {"word": w, "reading": y} for w, y in predicted]},
                    ensure_ascii=False,
                )
                _write_output(args.out, out + "\n")

            if args.time:
                print(f"[time] infer_sec={elapsed:.6f}", file=sys.stderr)
        return

    # ---- 複数入力モード（--text/--file/--stdin をまとめて処理） ----
    inputs = _collect_inputs(args)

    if len(inputs) == 0:
        raise SystemExit(
            "ERROR: No input. Provide --text/--file/--stdin or use --interactive.")

    t0 = time.perf_counter()
    batch_predicted = predictor.get_reading_predictions(inputs)
    elapsed = time.perf_counter() - t0

    # 出力組み立て
    if args.format in ("reading", "pairs"):
        lines = [_format_one(pairs, args.format) for pairs in batch_predicted]
        _write_output(args.out, "\n".join(lines) + "\n")
    elif args.format == "json":
        # 全入力分を1つのJSON配列で返す
        obj: List[Any] = []
        for inp, pairs in zip(inputs, batch_predicted):
            obj.append(
                {
                    "input": inp,
                    "tokens": [{"word": w, "reading": y} for w, y in pairs],
                }
            )
        _write_output(args.out, json.dumps(obj, ensure_ascii=False) + "\n")
    else:  # jsonl
        # 1行=1JSON（後処理向き）
        out_lines: List[str] = []
        for inp, pairs in zip(inputs, batch_predicted):
            out_lines.append(
                json.dumps(
                    {"input": inp, "tokens": [
                        {"word": w, "reading": y} for w, y in pairs]},
                    ensure_ascii=False,
                )
            )
        _write_output(args.out, "\n".join(out_lines) + "\n")

    if args.time:
        print(
            f"[time] infer_total_sec={elapsed:.6f} (n_inputs={len(inputs)})", file=sys.stderr)


if __name__ == "__main__":
    main()
