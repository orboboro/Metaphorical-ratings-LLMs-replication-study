from pathlib import Path
import argparse
from paced_huggingface_API_calls import reply_to_values, write_out, huggingface_API_calling

def main():

    parser = argparse.ArgumentParser(
        description="Metaphors Ratings Script with llms using Huggingface API",
        usage="python paced_groq_API_calls.py --model llama-3.3-70b-versatile --dataset human_MB.csv"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        help="Target study for replication"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
    )

    parser.add_argument(
        "--raters",
        type=int,
        default=1,
        help="Number of raters to annotate each metaphor",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in testing mode",
    )

    args = parser.parse_args()
    end = False

    while not end:
    
        end = huggingface_API_calling(args.dataset, args.model, args.raters, args.test)

if __name__ == "__main__": # La variabile speciale __name__ viene inizializzata uguale a "__main__" quando un file python viene eseguito
    main()                 # direttamente. Dunque la condizione __name__ == "__main__ è rispettata e quindi il contenuto delle funzione
                            # main viene eseguito. invece, se il file .py viene importato in un altro file, il suo contenuto non verrà
                            # eseguito, perché dal momento che il file non è eseguito direttamente, __name__ non sarà uguale alla stringa
                            # "__main__", ma al nome stesso del file .py. Insomma questa condizione serve a far sì che una funzione
                            # contenuta in un file venga eseguita solo quando è chiamata firettamente da terminale e nonquando è importata
                            # come modulo da altri file. 
                            # Reference: https://www.youtube.com/watch?v=sugvnHA7ElY
    