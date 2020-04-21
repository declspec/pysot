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
#from .tracker import FishTracker

def shape_to_box(shape):
    return (int(shape['x']), int(shape['y']), int(shape['width']), int(shape['height']))

def extract_regions(metadata, min_area):
    labels = []
    boxes = []

    for region in metadata['regions']:
        sattr = region['shape_attributes']

        if (sattr['width'] * sattr['height']) >= min_area:
            labels.append(region['region_attributes']['label'])
            boxes.append(shape_to_box(sattr))

    return labels, boxes

    
#def create_trackers(model, regions, min_area):
#    trackers = []
#
#    for region in regions:
#        sattr = region['shape_attributes']
#        box = map(int, (sattr['x'], sattr['y'], sattr['width'], sattr['height']))
#        label = region['region_attributes']['label']
#
#        if (box[2] * box[3]) >= min_area:
#            trackers.append(FishTracker(label, box, build_tracker(model))), label, box))
#
#    return trackers

def draw_results(labels, results, reader, writer):
    frame_index = 0

    while True:
        _, frame = reader.read()
        total_drawn = 0

        for index, result in enumerate(results):
            label = labels[index]

            if len(result) > frame_index:
                bbox, score = result[frame_index]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 3)
                cv2.putText(frame, '%s (%.5f)' % (label, score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                total_drawn += 1

        writer.write(frame)
        frame_index += 1

        if total_drawn == 0:
            break

def main(args):
    metadata = None

    with open(args.metadata, 'r') as fd:
        metadata = json.load(fd)

    # ensure output directory exists
    output = os.path.abspath(args.output)

    if not os.path.exists(output):
        os.makedirs(output)

    # load config and create model
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    model = ModelBuilder()
    model.load_state_dict(torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    def get_model():
        # return a shared model
        return model

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

        labels, boxes = extract_regions(meta, 300)

        # initialise the reader
        video = cv2.VideoCapture(vfile)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # initialise the writer to match the input video dimensions / FPS
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        # run the tracker
        results = track(video, get_model, boxes, args.frames)
        print('drawing results')

        with open(os.path.join(output, name+'.json'), 'w') as f:
            json.dump(results, f)

        # draw the results to a new file
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(os.path.join(output, name + '.avi'), cv2.VideoWriter_fourcc(*'XVID'), int(fps * args.speed), (width, height))
        draw_results(labels, results, video, writer)
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
