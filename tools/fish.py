import json
import cv2
import torch
import os
import numpy as np

from argparse import ArgumentParser
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

def main(args):
    video = cv2.VideoCapture(r'E:\Work\data\fish\videos\A000001_R.avi')
    regions = None

    # Read the regions from file
    with open(os.path.join(os.path.dirname(__file__), './regions.json'), 'r') as f:
        raw = json.load(f)
        regions = [ None ] * len(raw) 

        for i, r in enumerate(raw):
            a = r['shape_attributes']
            regions[i] = [ int(a['x']), int(a['y']), int(a['width']), int(a['height']) ]

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # frame info
    startFrame = 45631
    totalFrames = 200
    results = [ [ None ] * len(regions) for i in range(totalFrames) ]

    # brute force track all the regions
    for regionIndex, region in enumerate(regions):
        # create model
        model = ModelBuilder()

        # load model
        model.load_state_dict(torch.load(args.snapshot,
            map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

        # build and initialise tracker
        video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        _, frame = video.read()
        tracker = build_tracker(model)
        tracker.init(frame, region)

        # first result is always a perfect match...
        results[0][regionIndex] = [ region, 1.0 ]

        for frameIndex in range(1, totalFrames):
            _, frame = video.read()
            outputs = tracker.track(frame)

            bbox = list(map(int, outputs['bbox']))
            results[frameIndex][regionIndex] = [ bbox, float(outputs['best_score']) ]

    # TODO: Make output/result file paths config-able
    # dump results
    with open('./frame-results.json', 'w') as f:
        json.dump(results, f)

    # compile the results
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter('./output.avi', cv2.VideoWriter_fourcc(*'XVID'), int(fps / 4), (width, height))

    video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

    for frameResults in results:
        _, frame = video.read()

        for output in frameResults:
            bbox, bestScore = output
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 3)
            cv2.putText(frame, str(bestScore), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        writer.write(frame)


    writer.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = ArgumentParser(description='fish tracking demo')
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--snapshot', type=str, help='model name')

    args = parser.parse_args()
    main(args)

