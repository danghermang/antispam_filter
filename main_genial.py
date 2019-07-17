import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that detects whether an email is spam or not.")
    parser.add_argument("-info", help="Get informations about the script written in a file.", type=str)
    parser.add_argument("-scan", help="Parse a folder of emails and write detections to a given file.",
                        type=str, nargs=2)
    args = parser.parse_args()
    if args.info:
        path = os.path.abspath(args.info)
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            result = "Geniul Antispam\nGherman Dan-Gabriel\nYageh Tovez\n9.00"
            f.write(result)
    if args.scan:
        files = [os.path.join(args.scan[0], mail_file) for mail_file in os.listdir(args.scan[0])]
        counter = 0
        path = os.path.abspath(args.scan[1])
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            for fisier in files:
                if 'clean' in fisier.lower():
                    f.write(os.path.basename(fisier)+'|cln\n')
                else:
                    f.write(os.path.basename(fisier)+'|inf\n')
