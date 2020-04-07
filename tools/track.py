from cv2 import cv2
from pysot.tracker.tracker_builder import build_tracker

def track(reader, writer, model, boxes, total_frames):
    # In the current implementation, each Tracker instance re-intialises 
    # the model with its own template, meaning the Trackers cannot currently
    # be run in parallel and must each be run against all the frames before moving
    # to the next one
    results = [ [ None ] * len(boxes) for i in range(total_frames) ]
    frames = [ None ] * total_frames

    # read the frames into a buffer to re-use for each tracker
    for i in range(total_frames):
        _, frame = reader.read()
        frames[i] = frame
        
    # for each input box, initialise and run a unique tracker over the frames
    for box_index, box in enumerate(boxes):
        tracker = build_tracker(model)
        tracker.init(frames[0], box)
        results[0][box_index] = [ box, 1.0 ] # first frame result is always a perfect match

        for frame_index in range(1, len(frames)):
            outputs = tracker.track(frames[frame_index])
            bbox = list(map(int, outputs['bbox']))
            score = float(outputs['best_score'])
            results[frame_index][box_index] = [ bbox, score ]

    # after tracking is complete, step through the frame buffer and draw the results
    for frame_index, frame in enumerate(frames):
        for [ bbox, score ] in results[frame_index]:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 3)
            cv2.putText(frame, str(score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        writer.write(frame)

    return results
    