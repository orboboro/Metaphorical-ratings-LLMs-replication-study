import json
from pathlib import Path
import pandas as pd
import csv
import os
import ast
import time
from huggingface_hub import InferenceClient
import numpy as np

def reply_to_values(response):

    splitter = ","
    if response[1] == ";":
        splitter = ";"
    values_list = response.split(splitter)
    for idx, value in enumerate(values_list):
        values_list[idx] = int("".join([c for c in value if c.isdigit()]))
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

def huggingface_API_calling(dataset, model, raters, temperature, logprobs, memory, test):

    DATASET = str(dataset)
    DATASET_ID = DATASET[-6:-4]
    MODEL = (model).replace(":", "-")
    TASK_INSTRUCTIONS = open(Path("instructions", DATASET_ID + "_task_instructions.txt"), "r", encoding="utf-8").read()
    RATERS = raters
    TEMPERATURE = temperature
    LOGPROBS = logprobs
    DATA_PATH = "data"
    TRACKING_DATA_PATH = "tracking_data"
    MEMORY = memory
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
        "method": "API calls with huggingface_hub",
        "model": MODEL,
        "prompt": TASK_INSTRUCTIONS,
        "memory": MEMORY
    }

    dataset = Path(DATA_PATH, "human_datasets", DATASET)
    dataset_df = pd.read_csv(dataset, encoding="utf-8")

    if TEST:
        dataset_df = dataset_df[:3]

    checkpoint_file = Path(TRACKING_DATA_PATH, "checkpoint.csv")
    rater_file = Path(TRACKING_DATA_PATH, "current_rater.txt")

    
    conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": TASK_INSTRUCTIONS}]
                },
            {
                "role" : "user",
                "content": [{"type": "text", "text": ""}]
                }
    ]

    if not checkpoint_file.exists():

        if MEMORY:
            dataset_df = dataset_df.sample(frac = 1, random_state = 42)
            if not Path("conversations").exists():
                Path("conversations").mkdir()

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
            
            if MEMORY:
                with open(Path("conversations", f"rater_{rater}_conversation_" + out_file_name + ".txt"), "r", encoding = "utf-8") as f:
                    content = f.read()
                    conversation = ast.literal_eval(content)

    with open(rater_file, "r", encoding = "utf-8") as f:
        rater = f.read().strip()

    if int(rater) <= RATERS:

        metaphors_list = checkpoint_df["metaphor"]

        for idx, metaphor in list(enumerate(metaphors_list)):
            print("\n", rater, idx + 1, "of", len(metaphors_list))

            client = InferenceClient(api_key=os.environ["HF_TOKEN_SO41"], provider = "novita")

            if DATASET_ID in ["MB", "BA"]:
                pref = "Espressione: "
            else:
                pref = "Frase: "

            if DATASET_ID in ["MB", "BA", "ME"]:
                max_tokens = 7 # "n1, n2, n3"
            else:
                max_tokens = 4 # "n1, n2"

            if LOGPROBS:
                TOP_LOGPROBS = 3
            else:
                TOP_LOGPROBS = None
                
            conversation[-1]["content"][0]["text"] = pref + '"' + metaphor + '"'

            completion = client.chat.completions.create(
                model = MODEL,
                messages = conversation,
                max_tokens = max_tokens,
                temperature = TEMPERATURE,
                logprobs = LOGPROBS,
                top_logprobs = TOP_LOGPROBS
            )

            reply = completion.choices[0].message.content
            print("output: ", reply)

            if TEMPERATURE == 0.8:
                print("Sto usando temeperatura 0.8")

            if LOGPROBS:
                final_values = list()
                print("Sto usando logprobs")
                
                for i in range(0, max_tokens, 3):

                    top_three_logprobs = completion.choices[0].logprobs.content[i].top_logprobs
                    print("\n", top_three_logprobs)
                    tokens = []
                    logprobs = []

                    for logprob in top_three_logprobs:
                    
                        tokens.append(logprob.token)
                        logprobs.append(logprob.logprob)

                    print ("tokens: ", tokens)
                    print("logprobs: ",logprobs)

                    # Conversione dei logprobs in probabilità lineari
                    linear_probs = np.exp(logprobs)
                    print("linear_probs: ", linear_probs)

                    # Normalizzazione delle probabilità per ottenere i pesi
                    normalized_weights = linear_probs / np.sum(linear_probs)
                    print("normalized_weights: ", normalized_weights)

                    # Calcolo della media ponderata dei token usando le probabilità normalizzate come pesi
                    tokens_as_floats = []
                    for token in tokens:
                        try:
                            tokens_as_floats.append(float(token))
                        except ValueError:
                            print("Il token generato non è numerico")
        
                    numeric_token_weights = [
                        (float(token), weight)
                        for token, weight in zip(tokens, normalized_weights)
                        if token.replace('.', '', 1).isdigit() or (token.startswith('-') and token[1:].replace('.', '', 1).isdigit())
                    ]

                    # Ensure the list is not empty to avoid errors when calculating the weighted mean
                    if numeric_token_weights:
                        values, weights = zip(*numeric_token_weights)
                        weighted_mean_token = np.average(values, weights = weights)
                        final_values.append(round(weighted_mean_token, 9))
                    else:
                        print(f"No valid numeric tokens found for metaphor '{metaphor}'")

                print("weighted_values: ", final_values)
                
            else:
                final_values = reply_to_values(reply)
                print("final values: ", final_values)
                print("NON sto usando logprobs")

            if MEMORY:
                
                conversation.append({"role" : "assistant", "content": [{"type": "text", "text": reply}]})
                if len(conversation) > 11:
                    del conversation[1:3]
                conversation.append({"role" : "user", "content": [{"type": "text", "text": ""}]})

                with open((Path("conversations", f"rater_{rater}_conversation_" + out_file_name + ".txt")), "w", encoding = "utf-8") as f:
                    f.write(str(conversation))

            if DATASET_ID == "MB":

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "FAMILIARITY_synthetic" : final_values[0],
                    "MEANINGFULNESS_synthetic" : final_values[1],
                    "BODY_RELATEDNESS_synthetic" : final_values[2]
                }

            if DATASET_ID == "ME":

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "FAMILIARITY_synthetic" : final_values[0],
                    "MEANINGFULNESS_synthetic" : final_values[1],
                    "DIFFICULTY_synthetic" : final_values[2]
                }

            if DATASET_ID == "MI":

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "PHISICALITY_synthetic" : final_values[0],
                    "IMAGEABILITY_synthetic" : final_values[1],
                }

            if DATASET_ID == "MM":

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "FAMILIARITY_synthetic" : final_values[0],
                    "MEANINGFULNESS_synthetic" : final_values[1],
                }

            if DATASET_ID == "BA":

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "FAMILIARITY_synthetic" : final_values[0],
                    "DIFFICULTY_synthetic" : final_values[1],
                    "MEANINGFULNESS_synthetic" : final_values[2],
                }

            checkpoint_df = checkpoint_df[1:]
            checkpoint_df.to_csv(checkpoint_file, index = False)
            
            write_out(out_annotation_file, row)

            minuto = 60
            time.sleep(0.5 * minuto)

        print(f"{rater} rated all metaphors\n")

        return False

    else:

        with open(str(out_annotation_file) + "_CONFIG.json", "w") as f:
            json.dump(run_config, f)

        os.remove(checkpoint_file)
        os.remove(rater_file)

        print(f"Metaphor ratings for {DATASET_ID} with {MODEL} completed with success")

        return True
