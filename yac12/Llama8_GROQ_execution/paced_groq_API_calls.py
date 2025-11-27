import json
from pathlib import Path
import pandas as pd
import csv
import os
import ast
import time
from groq import Groq

def reply_to_values(response):
    values_list = response.split(",")
    for idx, value in enumerate(values_list):
        values_list[idx] = "".join([c for c in value if c.isdigit()])
    return values_list

def write_out(out_file_name, results_dict):
    out_annotation_file = Path(out_file_name)
    if not out_annotation_file.exists():
        with out_annotation_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_dict.keys())
            writer.writeheader()
            writer.writerow(results_dict)
    else:
        with out_annotation_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_dict.keys())
            writer.writerow(results_dict)

def groq_API_calling(dataset, model, raters, test = False):

    DATASET = str(dataset)
    DATASET_ID = DATASET[-6:-4]
    MODEL = (model).replace(":", "-")
    TASK_INSTRUCTIONS = open(Path("instructions", DATASET_ID + "_task_instructions.txt"), "r", encoding="utf-8").read()
    RATERS = raters
    DATA_PATH = "data"
    TRACKING_DATA_PATH = "tracking_data"
    TEST = test
    

    model_name = MODEL.replace(":", "-").replace("/", "-")

    out_file_name = f"synthetic_{DATASET_ID}_{model_name}_"

    if TEST:
        out_file_name = "TEST_" + out_file_name

    out_annotation_file = Path(
        DATA_PATH,
        "synthetic_datasets",
        out_file_name
        + ".csv"
    )

    run_config = {
        "n_raters": RATERS,
        "method": "API calls with Groq",
        "model": MODEL,
        "prompt": TASK_INSTRUCTIONS,
    }

    dataset = Path(DATA_PATH, "human_datasets", DATASET)
    dataset_df = pd.read_csv(dataset, encoding="utf-8")

    if TEST:
        dataset_df = dataset_df[:3]

    checkpoint_file = Path(TRACKING_DATA_PATH, "checkpoint.csv")
    rater_file = Path(TRACKING_DATA_PATH, "current_rater.txt")

    conversation = [
            {"role": "system", "content": TASK_INSTRUCTIONS},
            {"role" : "user", "content": ""}
    ]

    if not checkpoint_file.exists():
        dataset_df.to_csv(checkpoint_file, index = False)
        checkpoint_df = pd.read_csv(checkpoint_file, encoding="utf-8")

        with open(rater_file, "w", encoding = "utf-8") as f:
            f.write(str(1))

    else:
        checkpoint_df = pd.read_csv(checkpoint_file, encoding = "utf-8")

        if checkpoint_df.empty:
            dataset_df.to_csv(checkpoint_file, index = False)
            checkpoint_df = pd.read_csv(checkpoint_file, encoding="utf-8")


            with open(rater_file, "r", encoding = "utf-8") as f:
                previous_rater = int(f.read().strip())
            with open(rater_file, "w", encoding = "utf-8") as f:
                f.write(str(previous_rater + 1))
        else:
            with open(rater_file, "r", encoding = "utf-8") as f:
                rater = f.read().strip()
            with open(Path("conversations", f"rater_{rater}_conversation_" + out_file_name + ".txt"), "r", encoding = "utf-8") as f:
                content = f.read()
                conversation = ast.literal_eval(content)

    with open(rater_file, "r", encoding = "utf-8") as f:
        rater = f.read().strip()

    if int(rater) <= RATERS:

        metaphors_list = checkpoint_df["Metaphor"]
        structures_list = checkpoint_df["Met_structure"]

        for idx, metaphor in list(enumerate(metaphors_list)):
            
            print(rater, idx + 1, "of", len(metaphors_list))
            structure = structures_list[idx]

            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            conversation[-1]["content"] = metaphor
            chat_completion = client.chat.completions.create(
                messages = conversation,
                model = MODEL,
                temperature = 0.8
            )

            reply = chat_completion.choices[0].message.content # content è un attributo dell'oggetto ChatCompletionOutputMessage
            print("output: ", reply)

            checkpoint_df = checkpoint_df[1:]
            checkpoint_df.to_csv(checkpoint_file, index = False)

            conversation.append({"role" : "assistant", "content": reply})
            conversation.append({"role" : "user", "content": ""})

            with open((Path("conversations", f"rater_{rater}_conversation_" + out_file_name + ".txt")), "w", encoding = "utf-8") as f:
                f.write(str(conversation))

            values=reply_to_values(reply)
            print("values: ", values, "\n")

            if DATASET_ID == "MB":

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "metaphor_structure" : structure,
                    "FAMILIARITY_synthetic" : int(values[0]),
                    "MEANINGFULNESS_synthetic" : int(values[1]),
                    "body relatedness" : int(values[2])
                }

            if DATASET_ID == "ME":

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "metaphor_structure" : structure,
                    "FAMILIARITY_synthetic" : int(values[0]),
                    "MEANINGFULNESS_synthetic" : int(values[1]),
                    "DIFFICULTY_synthetic" : int(values[2])
                }

            if DATASET_ID == "MI":

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "metaphor_structure" : structure,
                    "PHISICALITY_synthetic" : int(values[0]),
                    "IMAGEBILITY_synthetic" : int(values[1]),
                }

            if DATASET_ID == "MM":

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "metaphor_structure" : structure,
                    "FAMILIARITY_synthetic" : int(values[0]),
                    "MEANINGFULNESS_synthetic" : int(values[1]),
                }

            write_out(out_annotation_file, row)

            minuto = 60
            time.sleep(1 * minuto)

        print(f"{rater} rated all metaphors\n")

        return False

    else:

        with open(str(out_annotation_file) + "_CONFIG.json", "w") as f:
            json.dump(run_config, f)

        os.remove(checkpoint_file)
        os.remove(rater_file)

        print(f"Metaphor ratings for {DATASET_ID} with {MODEL} completed with success")

        return True
