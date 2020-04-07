import json
import os
import re
import torch
import time

from cv2 import cv2
from track import track
from argparse import ArgumentParser
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder

def shape_to_box(shape):
    return (int(shape['x']), int(shape['y']), int(shape['width']), int(shape['height']))

def main(args):
    metadata = None

    with open(args.metadata, 'r') as fd:
        metadata = json.load(fd)

    # ensure output directory exists
    output = os.path.abspath(args.output)

    if not os.path.exists(output):
        os.makedirs(output)

    for meta in metadata:
        match = re.search(r'^(?P<video>.*?)(?P<frame>\d+)\.\w+$', meta['filename'])

        if match is None:
            print('unable to parse filename "%s"'%meta['filename'])
            continue

        vfile = os.path.join(args.videos, match['video'])
        start_frame = int(match['frame'])
        name = '%s_%s.%d'%(args.attempt, os.path.splitext(match['video'])[0], start_frame)

        if not os.path.exists(vfile):
            print('unable to find "%s" on disk'%match['video'])
            continue

        # track time for each run
        start_time = time.perf_counter()
        print('started %s'%name)

        # convert shapes to box tuples
        boxes = [ shape_to_box(r['shape_attributes']) for r in meta['regions'] ]

        # initialise the reader
        video = cv2.VideoCapture(vfile)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # initialise the writer to match the input video dimensions / FPS
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(os.path.join(output, name + '.avi'), cv2.VideoWriter_fourcc(*'XVID'), int(fps * args.speed), (width, height))

        # load config
        cfg.merge_from_file(args.config)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create model once
        model = ModelBuilder()

        # load model
        model.load_state_dict(torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

        results = track(video, writer, model, boxes, args.frames)

        with open(os.path.join(output, name+'.json'), 'w') as f:
            json.dump(results, f)

        writer.release()
        video.release()

        print('completed %s in %0.2f seconds'%(name, time.perf_counter() - start_time))
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = ArgumentParser(description='track bounding boxes through frames')

    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--metadata', type=str, help='file containing metadata about crop data')
    parser.add_argument('--videos', type=str, help='directory containing video files referenced in metadata')
    parser.add_argument('--frames', type=int, help='number of frames to track', default=200)
    parser.add_argument('--output', type=str, help='output directory', default='.')
    parser.add_argument('--speed', type=float, help='speed of output relative to input (1.0 is 100%)', default=0.25)
    parser.add_argument('attempt', help='name of the attempt')

    # argparse terminates the process if parse_args() encounters an error
    args = parser.parse_args()
    main(args)
