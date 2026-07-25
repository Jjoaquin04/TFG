import os
import argparse
from pipeline import extract, postprocessing, render

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Data collection from a padel video')
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--extract', type=str, metavar='VIDEO_URL', help='Run extract mode with the provided video URL or path')
    group.add_argument('--postprocessing', type=str, nargs=2, metavar=('JSON_RAW', 'VIDEO_URL'), help='Run postprocessing mode from a raw json and video')
    group.add_argument('--render', type=str, nargs=2,  metavar=('VIDEO_URL', 'JSON_RAW'), help='Run render mode with the provided video URL or path')

    args = parser.parse_args()

    if args.extract:
        if not os.path.exists(args.extract):
            print(f"Error: The video file {args.extract} does not exist.")
            exit(-1)
        extract(args.extract)

    elif args.postprocessing:
        if not os.path.exists(args.postprocessing[0]):
            print(f"Error: The json file {args.postprocessing[0]} does not exist.")
            exit(-1)
        if not os.path.exists(args.postprocessing[1]):
            print(f"Error: The video file {args.postprocessing[1]} does not exist.")
            exit(-1)
        postprocessing(args.postprocessing[0], args.postprocessing[1])

    elif args.render:
        if not os.path.exists(args.render[0]) or not os.path.exists(args.render[1]):
            print(f"Error: Some path {args.render[0], args.render[1]} are missing.")
            exit(-1)
        render(args.render[0], args.render[1])
    
