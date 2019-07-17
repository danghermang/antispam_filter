import json
import argparse
import generate_detection
import os
from collections import Counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that detects whether an email is spam or not.")
    parser.add_argument("-info", help="Get informations about the script written in a file.", type=str)
    parser.add_argument("-scan", help="Parse a folder of emails and write detections to a given file.",
                        type=str, nargs=2)
    args = parser.parse_args()
    with open('only_clean.json', 'r') as f:
        only_clean_json = Counter(json.load(f))
    with open('only_spam.json', 'r') as f:
        only_spam_json = Counter(json.load(f))
    if args.info:
        with open(args.info, 'w') as f:
            result = "Biltru AntisBam\nGherman Dan-Gabriel\nBenis\n9.00"
            f.write(result)
    if args.scan:
        with open(args.scan[1],'w') as f:
            for fi in os.listdir(args.scan[0]):
                fisier = os.path.join(args.scan[0], fi)
                document = generate_detection.parse_mail(fisier)
                if generate_detection.get_score(document, only_clean_json,only_spam_json) >= 0:
                    f.write(os.path.basename(fisier)+'|cln\n')
                else:
                    f.write(os.path.basename(fisier)+'|inf\n')
