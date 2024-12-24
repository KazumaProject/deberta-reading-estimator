from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from pyknp import Juman  # JUMAN tokenizer を使用
import json
import numpy as np
from copy import deepcopy
import os
import pickle


class ReadingEstimator:
    def __init__(self, model_name, references, evaluation_type="most_similar"):
        """
        Args:
            model_name (str): 使用するモデルの名前
            references (dict): 参照データ
            evaluation_type (str): 評価方法
             - most_similar: コサイン類似度が最も高い読みを予測
             - average: すべての参照データのコサイン類似度の平均が最も高い読みを予測
        """
        self.jumanpp = Juman()  # Jumanを初期化
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.references = deepcopy(references)
        self.evaluation_type = evaluation_type
        # replace [MASK] with tokenizer.mask_token
        for key, values in self.references.items():
            for reading, texts in values.items():
                self.references[key][reading] = [
                    self._split_reference(text) for text in texts
                ]
        self.reference_logits = self._calculate_reference_logits()

    def update_references(self, references):
        self.references = deepcopy(references)
        for key, values in self.references.items():
            for reading, texts in values.items():
                self.references[key][reading] = [
                    self._split_reference(text) for text in texts
                ]
        self.reference_logits = self._calculate_reference_logits()

    def _calculate_reference_logits(self):
        # reference_logitsの初期化
        reference_logits = {}
        for kanji, readings in self.references.items():
            reference_logits[kanji] = {}
            for reading, examples in readings.items():
                reference_logits[kanji][reading] = []
                for text in examples:
                    inputs = self.tokenizer(text, return_tensors="pt")
                    outputs = self.model(**inputs)
                    mask_token_index = torch.where(
                        inputs["input_ids"][0] == self.tokenizer.mask_token_id
                    )[0]
                    reference_logits[kanji][reading].append(
                        (outputs.logits[0, mask_token_index].detach().numpy(), text)
                    )
        return reference_logits

    def _get_most_similar_reading(self, kanji, logit):
        # assert logit == (1, 32000)
        max_similarity = 0
        predicted_reading = None
        # 与えられた漢字に対する全ての読みを確認
        for reading, values in self.reference_logits[kanji].items():
            for value, text in values:
                # use: cosine similarity
                similarity = np.dot(logit[0], value[0]) / (
                    np.linalg.norm(logit) * np.linalg.norm(value)
                )
                # print(f"{reading}, {similarity:04f}, {text}")
                if similarity > max_similarity:
                    max_similarity = similarity
                    predicted_reading = reading
        return predicted_reading

    def _get_average_similar_reading(self, kanji, logit):
        max_similarity = 0
        predicted_reading = None
        # 与えられた漢字に対する全ての読みを確認
        for reading, values in self.reference_logits[kanji].items():
            similarity_sum = 0
            for value, _ in values:
                similarity_sum += torch.nn.functional.cosine_similarity(
                    logit, value, dim=1
                ).item()
            similarity = similarity_sum / len(values)
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_reading = reading
        return predicted_reading

    def _split_reference(self, text):
        # referenceのテキストを形態素解析し、半角スペースで分割する
        result = self.jumanpp.analysis(text)
        text = " ".join([mrph.midasi for mrph in result.mrph_list()])
        text = text.replace("[ MASK ]", self.tokenizer.mask_token)
        return text

    def get_reading_prediction(self, text):
        # Jumanでテキストを形態素解析し、分割する
        result = self.jumanpp.analysis(text)
        predicted_readings = []

        for mrph in result.mrph_list():
            # FIXME: 一文に複数回出現する場合に対応
            if mrph.midasi in self.references and text.count(mrph.midasi) == 1:  # 原形が対象の読み分け単語に含まれる場合
                masked_text = " ".join([
                    self.tokenizer.mask_token if mrph.midasi == item.midasi else item.midasi for item in result.mrph_list()
                ])
                inputs = self.tokenizer(masked_text, return_tensors="pt")
                outputs = self.model(**inputs)
                mask_token_index = torch.where(
                    inputs["input_ids"][0] == self.tokenizer.mask_token_id
                )[0]
                get_reading = self._get_most_similar_reading if self.evaluation_type == "most_similar" else self._get_average_similar_reading
                predicted_reading = get_reading(
                    mrph.midasi, outputs.logits[0, mask_token_index].detach().numpy()
                )
                predicted_readings.append((mrph.midasi, predicted_reading))
            else:
                predicted_readings.append((mrph.midasi, mrph.yomi))
        return predicted_readings

    def get_optimized_reading(self, word, left_context, right_context, current_reading):
        """
        単語が参照データに存在する場合、左文脈と右文脈を考慮して最適な読みを返す。
        存在しない場合、現在の読み候補をそのまま返す。

        Args:
            word (str): 対象の単語
            left_context (str): 単語の左文脈
            right_context (str): 単語の右文脈
            current_reading (str): 現在の読み候補

        Returns:
            str: 最適な読み
        """
        # 単語が参照データに存在しない場合は現在の読みを返す
        if word not in self.references:
            return current_reading

        # 単語が参照データに存在する場合、コンテキストを用いて最適な読みを決定
        # 左文脈と右文脈を結合して[MASK]トークンを挿入
        left_result = self.jumanpp.analysis(left_context)
        right_result = self.jumanpp.analysis(right_context)
        left_result = " ".join([item.midasi for item in left_result.mrph_list()])
        right_result = " ".join([item.midasi for item in right_result.mrph_list()])

        masked_text = f"{left_result} {self.tokenizer.mask_token} {right_result}"
        inputs = self.tokenizer(masked_text, return_tensors="pt")
        outputs = self.model(**inputs)

        # MASKトークンの位置を取得
        mask_token_index = torch.where(
            inputs["input_ids"][0] == self.tokenizer.mask_token_id
        )[0]

        # コサイン類似度を用いて最適な読みを取得
        logit = outputs.logits[0, mask_token_index].detach().numpy()
        predicted_reading = self._get_most_similar_reading(word, logit)

        return predicted_reading

    def save_compiled(self, path="compiled_data.pkl"):
        """
        reference_logitsをpickle形式で保存する。
        Args:
            path (str): 保存先のパス
        """
        data_to_save = {
            "reference_logits": self.reference_logits,
            "references": self.references,
            "evaluation_type": self.evaluation_type,
            "model_name": self.model_name
        }
        with open(path, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"Compiled data saved to {path}")

    def load_compiled(path="compiled_data.pkl"):
        """
        pickle形式のデータを読み込み、reference_logitsを復元する。
        Args:
            path (str): 読み込むファイルのパス
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No compiled data file found at {path}")
        with open(path, "rb") as f:
            loaded_data = pickle.load(f)
        self = ReadingEstimator(loaded_data["model_name"], dict(), loaded_data["evaluation_type"])
        self.reference_logits = loaded_data["reference_logits"]
        self.references = loaded_data["references"]
        print(f"Compiled data loaded from {path}")
        return self

if __name__ == "__main__":
    # 使用例
    references = json.load(open("references.json", "r"))
    # 「水」以外のkeyを削除
    # references = {key: references[key] for key in references if key == "水"}
    # predictor = ReadingEstimator("ku-nlp/deberta-v2-base-japanese", references, evaluation_type="most_similar")
    predictor = ReadingEstimator.load_compiled("./predictor.pkl")

    texts = [
        "結局世の中は金が全てです",
        "王水は金も溶かす強力な溶液です",
        "金正日が来日した",
        "ピアノを弾くのが好きです",
        "ギターの弦を弾くと音が出ます",
        "油は水を弾く",
        "生理食塩水",
        "学校に行った",
        "開会式を行った",
        "君と僕の間で何か隠し事があるのは良くない",
        "紅葉が綺麗に色づく季節になりました",
        "紅葉した山の景色",
        "北の方に向かって進む",
        "例のあの方がいらっしゃいました",
        "その件については私に任せてください",
        "件の人物を探し出す",
        "集合の元aに対して、-aが常に存在する",
        "彼の元には多くの人が集まった",
    ]

    for text in texts:
        predicted_readings = predictor.get_reading_prediction(text)
        print(f"Original text: {text}")
        joined_yomi = "".join([yomi for _, yomi in predicted_readings])
        print(f"Predicted readings: {joined_yomi}")

    # `get_optimized_reading`の使用例
    word = "水"
    left_context = "油は"
    right_context = "を弾く"
    current_reading = "すい"
    optimized_reading = predictor.get_optimized_reading(
        word, left_context, right_context, current_reading
    )
    print(f"Optimized Reading: {optimized_reading}")

    # predictor.save_compiled("./predictor.pkl")
