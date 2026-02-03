from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = "ku-nlp/deberta-v2-base-japanese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

# # Inference
text = f"私 は {tokenizer.mask_token} を 稼ぎ たい です"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits.shape)  # torch.Size([1, 12, 2])
print(outputs.logits)
# # [MASK]トークンの位置を特定
mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
print(mask_token_index)

# # 最も確率の高いトークンを取得
predicted_token_id = outputs.logits[0, mask_token_index].argmax().item()
predicted_token = tokenizer.decode(predicted_token_id)

# # 結果を表示
print(f"Original text: {text}")
print(f"Predicted text: {text.replace(tokenizer.mask_token, predicted_token)}")

# fill_mask = pipeline(
#     "fill-mask", model="ku-nlp/deberta-v2-base-japanese"
# )
# for r in fill_mask("[MASK] を 稼ぐ 。"):
#     print(r)
# for r in fill_mask("[MASK] と 銀 の 採掘 。"):
#     print(r)


references = {
    "かね": [
        f"私 は {tokenizer.mask_token} を 稼ぎ たい です",
        f"たくさんの {tokenizer.mask_token} が あれば なんでも 買え ます",
    ],
    "きん": [
        f"{tokenizer.mask_token} と 銀 の 採掘 を 行う 仕事",
        f"私 は {tokenizer.mask_token} メダル を 取り たい です",
        f"先週の {tokenizer.mask_token} の 価格 は どう でした か",
        f"毎週 {tokenizer.mask_token} 曜日 に 授業 が あります",
    ],
    "きむ": [
        f"{tokenizer.mask_token} 先生 は とても 厳しい です",
        f"韓国の {tokenizer.mask_token} は とても 有名 です",
        f"{tokenizer.mask_token} 正恩 は 北朝鮮 の 最高 指導者 です",
    ],
}
refrence_logits = {}
for key, values in references.items():
    refrence_logits[key] = []
    for text in values:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        mask_token_index = torch.where(
            inputs["input_ids"][0] == tokenizer.mask_token_id
        )[0]
        refrence_logits[key].append(outputs.logits[0, mask_token_index])


def get_most_similar_token(logit, refrence_logits):
    max_similarity = 0
    predicted_token = None
    for key, values in refrence_logits.items():
        for value in values:
            similarity = torch.nn.functional.cosine_similarity(
                logit, value, dim=1
            ).item()
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_token = key
    return predicted_token


def get_reading_prediction(text, refrence_logits):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
    predicted_reading = get_most_similar_token(
        outputs.logits[0, mask_token_index], refrence_logits
    )
    return predicted_reading


# 以下の「金」は何と読むか？
text = "結局 世の中 は 金 が 全て です".replace("金", tokenizer.mask_token)
predicted_reading = get_reading_prediction(text, refrence_logits)
print(f"Original text: {text}")
print(f"Predicted reading: {predicted_reading}")

# 以下の「金」は何と読むか？
text = "王水は 金 も 溶かす 強力 な 溶液 です".replace("金", tokenizer.mask_token)
predicted_reading = get_reading_prediction(text, refrence_logits)
print(f"Original text: {text}")
print(f"Predicted reading: {predicted_reading}")

# 以下の「金」は何と読むか？
text = "金 正日 が 来日 した".replace("金", tokenizer.mask_token)
predicted_reading = get_reading_prediction(text, refrence_logits)
print(f"Original text: {text}")
print(f"Predicted reading: {predicted_reading}")
