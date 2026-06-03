import os
import argparse
from pipeline import extract, posprocessing, render

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Data collection from a padel video')
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--extract', type=str, metavar='VIDEO_URL', help='Run extract mode with the provided video URL or path')
    group.add_argument('--posproccesing', type=str, metavar='JSON_RAW', help='Run posproccesing mode from a raw json')
    group.add_argument('--render', type=str, nargs=2,  metavar=('VIDEO_URL', 'JSON_RAW'), help='Run render mode with the provided video URL or path')

    args = parser.parse_args()

    if args.extract:
        if not os.path.exists(args.extract):
            print(f"Error: The video file {args.extract} does not exist.")
            exit(-1)
        extract(args.extract)

    elif args.posproccesing:
        if not os.path.exists(args.posproccesing):
            print(f"Error: The json file {args.posprocessing} does not exist.")
            exit(-1)
        posprocessing(args.posprocessing)

    elif args.render:
        if not os.path.exists(args.render[0]) or not os.path.exists(args.render[1]):
            print(f"Error: Some path {args.render[0], args.render[1]} are missing.")
            exit(-1)
        render(args.render[0], args.render[1])
    
