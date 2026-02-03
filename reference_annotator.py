"""
リファレンスデータを構成するためのアノテーションツール
"""

from reading_estimator import ReadingEstimator
import argparse
import json
import random


def save_references(references, output_reference_file):
    with open(output_reference_file, "w") as f:
        json.dump(references, f, ensure_ascii=False, indent=4)


def main(args):
    assert len(args.source_file) > 0, "source_file must be specified"
    references = json.load(open(args.reference_file, "r"))
    if args.target_word not in references:
        references[args.target_word] = {}

    target_references = references[args.target_word]
    estimator = ReadingEstimator(
        "ku-nlp/deberta-v2-base-japanese",
        {args.target_word: references[args.target_word]},
    )

    source_texts = []
    for source_file in args.source_file:
        with open(source_file, "r") as f:
            for line in f:
                # count must be one
                if len(line) < 100 and line.count(args.target_word) == 1:
                    if args.text_filter is not None and args.text_filter not in line:
                        continue
                    source_texts.append(line.strip())
    print(f"Found {len(source_texts)} possible lines containing {args.target_word}")

    try:
        i = 0
        # randomize
        random.shuffle(source_texts)
        for text in source_texts:
            try:
                predicted_readings = estimator.get_reading_prediction(text)
            except Exception as e:
                print("Invalid input", text)
                print("Error:", e)
                continue
            if args.target_word in [midasi for midasi, _ in predicted_readings]:
                readings = [yomi for _, yomi in predicted_readings]
                if None in readings:
                    print("Invalid input", text)
                    continue
                description = "".join(
                    [
                        midasi if midasi != args.target_word else f"{midasi}({yomi})"
                        for midasi, yomi in predicted_readings
                    ]
                )
                masked_text = text.replace(args.target_word, "[MASK]")
                selected_reading = [
                    yomi
                    for midasi, yomi in predicted_readings
                    if midasi == args.target_word
                ][0]
                if masked_text in target_references[selected_reading]:
                    # すでにリファレンスに含まれている場合はスキップ
                    continue

                print(f"Original text: {text}")
                print(f"Description  : {description}")
                print("Is it correct? (y/n/skip)")
                answer = input()
                if answer == "y":
                    target_references[selected_reading].append(masked_text)
                    print("Added to references")
                elif answer == "n":
                    print("Which reading is correct?")
                    # referencesの中から正しい読みを選択
                    candidates = list(enumerate(target_references))
                    for j, reading in candidates:
                        print(j, reading)
                    print(j + 1, "Other")
                    selected_index = None
                    while selected_index is None:
                        try:
                            selected_index = int(input())
                            if selected_index not in range(len(candidates) + 1):
                                selected_index = None
                                print("Please input a valid index")
                        except ValueError:
                            print("Please input a valid index")
                            continue
                    if selected_index == j + 1:
                        print("Please input the correct reading")
                        correct_reading = input()
                        target_references[correct_reading] = [masked_text]
                        print("Added to references")
                    else:
                        selected_reading = candidates[selected_index][1]
                        if masked_text not in target_references[selected_reading]:
                            target_references[selected_reading].append(masked_text)
                elif answer == "skip":
                    print("Skipped")
                    continue
                else:
                    print("Invalid input")
                    continue
                # 区切り
                print(f"annotated {i} lines")
                print()
                i += 1
                if i % 10 == 0:
                    # save
                    save_references(references, args.output_reference_file)
                    print("Updating references...")
                    estimator.update_references({args.target_word: target_references})
    except Exception as e:
        print("Invalid input", text)
        print("Saving references...")
        save_references(references, args.output_reference_file)
        print("Saved")
        raise e
    except KeyboardInterrupt:
        print("Saving references...")
        save_references(references, args.output_reference_file)
        print("Saved")

    print("All lines are annotated")
    save_references(references, args.output_reference_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # source txt files
    parser.add_argument(
        "--source_file",
        nargs="*",
        type=str,
        default=[],
        help="アノテーション対象のファイルパス",
    )
    # target word
    parser.add_argument(
        "--target_word",
        type=str,
        help="アノテーション対象の単語（midasi）",
    )
    # current reference json file (optional)
    parser.add_argument(
        "--reference_file",
        type=str,
        default="./references.json",
        help="既存のリファレンスデータのファイルパス",
    )
    # output reference json file (optional)
    parser.add_argument(
        "--output_reference_file",
        type=str,
        default="./updated_references.json",
        help="新しいリファレンスデータのファイルパス",
    )
    # text filter (optional)
    parser.add_argument(
        "--text_filter",
        type=str,
        default=None,
        help="アノテーション対象のテキストのフィルター",
    )
    args = parser.parse_args()

    main(args)

    # mv updated_references.json references.json
    print("Move updated_references.json to references.json? (y/n)")
    answer = input()
    if answer == "y":
        import shutil

        shutil.move("updated_references.json", "references.json")
        print("Moved")
    else:
        print("Exit without moving.")
