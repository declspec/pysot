from cv2 import cv2
from pysot.tracker.tracker_builder import build_tracker

def is_valid_result(current, previous):
    return current[1] > 0.2

def track(reader, get_model, boxes, total_frames):
    trackers = [ None ] * len(boxes)
    results = [ None ] * len(trackers)

    # read the first frame to initialise the trackers
    _, frame = reader.read()
    frame_index = 0

    # initialise the trackers and give the boxes an initial score of 1.0
    for tracker_index in range(len(trackers)):
        tracker = trackers[tracker_index] = build_tracker(get_model())
        tracker.init(frame, boxes[tracker_index])
        results[tracker_index] = [[ boxes[tracker_index], 1.0 ]]

    # run until no more trackers exist or we run out of frames
    for _ in range(total_frames):
        res, frame = reader.read()
        total_trackers = 0

        if not res:
            break

        for tracker_index in range(len(trackers)):
            tracker = trackers[tracker_index]
            tracker_results = results[tracker_index]

            if tracker is None:
                continue

            outputs = tracker.track(frame)
            result = [ list(map(int, outputs['bbox'])), float(outputs['best_score']) ]

            if not is_valid_result(result, tracker_results):
                trackers[tracker_index] = None
            else:
                tracker_results.append(result)
                total_trackers += 1

        if total_trackers == 0:
            break

    return results

    # In the current implementation, each Tracker instance re-intialises 
    # the model with its own template, meaning the Trackers cannot currently
    # be run in parallel and must each be run against all the frames before moving
    # to the next one
    results = [ [ None ] * len(boxes) for i in range(total_frames) ]
    frames = [ None ] * total_frames
    nobox = [ 0, 0, 0, 0 ] # re-usable rectangle representing no bounding box
    tolerance = 0.1 # min confidence

    # read the frames into a buffer to re-use for each tracker
    for i in range(total_frames):
        _, frame = reader.read()
        frames[i] = frame
        
    # for each input box, initialise and run a unique tracker over the frames
    for box_index, box in enumerate(boxes):
        tracker = build_tracker(get_model())
        tracker.init(frames[0], box)
        results[0][box_index] = [ box, 1.0 ] # first frame result is always a perfect match

        for frame_index in range(1, len(frames)):
            #if frame_index > 1 and ((results[frame_index - 2][box_index][0] + results[frame_index - 1][box_index][0]) / 2) < tolerance:
            # check the previous frame result to see if we should keep tracking
            #if results[frame_index - 1][box_index][1] < tolerance:
            #    results[frame_index][box_index] = [ nobox, 0.0 ]
            #else:
            outputs = tracker.track(frames[frame_index])
            bbox = list(map(int, outputs['bbox']))
            score = float(outputs['best_score'])
            results[frame_index][box_index] = [ bbox, score ]

    # after tracking is complete, step through the frame buffer and draw the results
    for frame_index, frame in enumerate(frames):
        for [ bbox, score ] in results[frame_index]:
            color = (0, 255, 0) if score >= tolerance else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 3)
            cv2.putText(frame, str(score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)

    return results