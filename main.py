import os
import argparse
import pickle

import generate_detection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that detects whether an email is spam or not.")
    parser.add_argument("-info", help="Get informations about the script written in a file.", type=str)
    parser.add_argument("-scan", help="Parse a folder of emails and write detections to a given file.",
                        type=str, nargs=2)
    args = parser.parse_args()
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('all_vectorizer.pkl', 'rb') as f:
        all_vectorizer = pickle.load(f)
    if args.info:
        with open(args.info, 'w') as f:
            result = "Biltru AntisBam\nGherman Dan-Gabriel\nBenis\n80085"
            f.write(result)
    if args.scan:
        with open(args.scan[1], 'w') as f:
            files = [os.path.join(args.scan[0], mail_file) for mail_file in os.listdir(args.scan[0])]
            representation = generate_detection.get_representation_list(files, all_vectorizer)
            scores = model.predict_classes(representation)
            for file, score in zip(files, scores):
                if score > 0:
                    f.write(os.path.basename(file) + '|cln\n')
                else:
                    f.write(os.path.basename(file) + '|inf\n')
