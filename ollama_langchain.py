# Example usage: python langchain_bws.py --model mistral:instruct --prompt specificity_task_instructions.txt --raters 12
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain.output_parsers import CommaSeparatedListOutputParser
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import csv
import argparse


def promptize_tuple(word_tuple):
    # return f"[INST]\n1. {word_tuple[0].upper()}\n2. {word_tuple[1].upper()}\n3. {word_tuple[2].upper()}\n4. {word_tuple[3].upper()}\n[/INST]"
    return f"0 - {word_tuple[0].upper()},\n1 - {word_tuple[1].upper()},\n2 - {word_tuple[2].upper()},\n3 - {word_tuple[3].upper()}\n"


def annotate(tup, history=False):
    prompt_tuple = promptize_tuple(tup)
    # print(prompt_tuple)
    if history:
        request = {
            "chat_history": chat_history,
            "input": prompt_tuple,
        }
    else:
        request = {
            "input": prompt_tuple,
        }

    response = chain.invoke(request)

    return response


def index_to_words(tup, reply):
    try:
        idx_b, idx_w = reply.split(",", 1)
        idx_b = int(idx_b.strip()[-1])
        idx_w = int(idx_w.strip()[0])
        best = tup[idx_b]
        worst = tup[idx_w]
    except:
        idx_b = None
        idx_w = None
        best = None
        worst = None

    return best, worst, idx_b, idx_w


def write_out(out_file_name, results_dict):
    out_annotation_file = Path(str(out_file_name.absolute()) + ".csv")
    if not out_annotation_file.exists():
        with out_annotation_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_dict.keys())
            writer.writeheader()
            writer.writerow(results_dict)
    else:
        with out_annotation_file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_dict.keys())
            writer.writerow(results_dict)


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser(
        description="BWS Annotation Script with llms, langchain and Ollama API",
        usage="python langchain_bws.py --model 'mistral:instruct' --prompt specificity_task_instructions_1.txt [--raters 1 --history --testing]",
    )

    parser.add_argument(
        "--metapors_file",
        type=Path,
        help="study to replicate"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="mistral:instruct",
        help="Model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Path to data",
    )

    parser.add_argument("--history", action="store_true", help="Keep history")

    parser.add_argument(
        "--raters",
        type=int,
        default=1,
        help="number of raters to annotate each tuple",
    )

    parser.add_argument(
        "--testing",
        action="store_true",
        help="Run in testing mode",
    )

    args = parser.parse_args()

    MODEL = args.model

    if args.history:
        KEEP_HISTORY = True
    else:
        KEEP_HISTORY = False

    TASK_INSTRUCTIONS = open(args.prompt, "r").read()

    if args.testing:
        TESTING = True
    else:
        TESTING = False

    DATA_PATH = "../data"
    RATERS = args.raters

    model_name = MODEL.replace(":", "-")
    if TESTING:
        out_file_name = "_TESTING_bws_llm-langchain_"
    else:
        out_file_name = "bws_llm-langchain_"

    if KEEP_HISTORY:
        out_file_name += "keep-history_"
    else:
        out_file_name += "no-history_"
    out_annotation_file = Path(
        DATA_PATH,
        "synthetic_annotations",
        out_file_name
        + model_name
        + "_"
        + str(start_time.isoformat().replace(":", "-").split(".")[0]),
        # + ".csv",
    )

    print(out_annotation_file)

    run_config = {
        "time": str(start_time.isoformat().replace(":", "-").split(".")[0]),
        "n_raters": RATERS,
        "method": "langchain-ollama",
        "model": MODEL,
        "keep_history": KEEP_HISTORY,
        "prompt": TASK_INSTRUCTIONS,
    }

    # INIZIA MODIFICA DA QUI

    metaphors_file = Path(DATA_PATH, args.metaphors_file)
    with open(metaphors_file, "r") as metaphors:
        word_tuples = json.load(word_tuples)

    for tuple_list_type in word_tuples.keys():
        for tuple_list_number in list(word_tuples[tuple_list_type].keys()):
            list_name = "_".join([tuple_list_type, tuple_list_number])
            tuple_list = word_tuples[tuple_list_type][tuple_list_number]
            print(list_name.upper(), "TOTAL TUPLES:", len(tuple_list))

            for n in range(RATERS):
                rater_time = datetime.now()
                rater = f"{list_name}_rater_{n + 1}"
                
                print(rater)

                llm = Ollama(model=MODEL, num_predict=48)  # , temperature=0.2)
                if KEEP_HISTORY:
                    chat_history = []
                    prompt_template = ChatPromptTemplate.from_messages(
                        [
                            ("system", TASK_INSTRUCTIONS),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("user", "{input}"),
                        ]
                    )
                else:
                    prompt_template = ChatPromptTemplate.from_messages(
                        [
                            ("system", TASK_INSTRUCTIONS),
                            ("user", "{input}"),
                        ]
                    )
                chain = prompt_template | llm

                for idx, tup in list(enumerate(tuple_list)):
                    print(rater, idx + 1, "of", len(tuple_list))

                    reply = annotate(tup, history=KEEP_HISTORY)
                    best, worst, idx_b, idx_w = index_to_words(tup, reply)
                    # print(best, worst)

                    row = {
                        "list": list_name,
                        "annotator": rater,
                        "Item1": tup[0],
                        "Item2": tup[1],
                        "Item3": tup[2],
                        "Item4": tup[3],
                        "BestItem": best,
                        "WorstItem": worst,
                        "explanation": reply,
                        "model": MODEL,
                    }
                    if KEEP_HISTORY and best is not None and worst is not None:
                        chat_history.append(HumanMessage(content=promptize_tuple(tup)))
                        chat_history.append(AIMessage(content=reply))

                    write_out(out_annotation_file, row)

                print(f"{rater} completed in: {datetime.now() - rater_time}")

    with open(str(out_annotation_file.absolute()) + "_CONFIG.json", "w") as f:
        json.dump(run_config, f)

    print("BWS completed in: {}".format(datetime.now() - start_time))