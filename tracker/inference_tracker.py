#Non Maximum Supression

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)


def suppress_enclosing_or_overlapping_boxes(draw_tracks, iou_thresh=0.7, area_ratio_thresh=1.3):
    def is_enclosing(boxA, boxB):
        return (boxA[0] <= boxB[0] and boxA[1] <= boxB[1] and
                boxA[2] >= boxB[2] and boxA[3] >= boxB[3])

    filtered = []
    for i, t1 in enumerate(draw_tracks):
        keep = True
        for j, t2 in enumerate(draw_tracks):
            if i == j:
                continue
            iou = compute_iou(t1["bbox"], t2["bbox"])
            area_ratio = t1["area"] / (t2["area"] + 1e-5)
            if is_enclosing(t1["bbox"], t2["bbox"]) and area_ratio > area_ratio_thresh:
                keep = False
                break
            if iou > iou_thresh and area_ratio > area_ratio_thresh:
                keep = False
                break
        if keep:
            filtered.append(t1)
    return filtered


class YOLODeepSORTTracker:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(
            max_age=10,
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.4,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True
        )

    def track_video_frame(self, frame):
        results = self.model.predict(source=frame, conf=0.51, save=False, verbose=False)[0]

        bboxes = results.boxes.xyxy.cpu()
        scores = results.boxes.conf.cpu()
        classes = results.boxes.cls.cpu().int()

        if bboxes.shape[0] == 0:
            return frame, []

        nms_indices = nms(bboxes, scores, iou_threshold=0.6)

        filtered_boxes = bboxes[nms_indices]
        filtered_scores = scores[nms_indices]
        filtered_classes = classes[nms_indices]

        final_indices = []
        for i in range(len(filtered_boxes)):
            overlap_count = 0
            for j in range(len(filtered_boxes)):
                if i == j:
                    continue
                iou = compute_iou(filtered_boxes[i].tolist(), filtered_boxes[j].tolist())
                if iou > 0.6:
                    overlap_count += 1
            if overlap_count < 2:
                final_indices.append(i)

        final_boxes = filtered_boxes[final_indices]
        final_scores = filtered_scores[final_indices]
        final_classes = filtered_classes[final_indices]

        MIN_AREA = 1000
        detections = []
        for box, score, cls in zip(final_boxes, final_scores, final_classes):
            x1, y1, x2, y2 = box.tolist()
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_AREA:
                continue
            detections.append(([x1, y1, x2 - x1, y2 - y1], float(score), int(cls)))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        draw_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            draw_tracks.append({
                "track": track,
                "bbox": [x1, y1, x2, y2],
                "area": (x2 - x1) * (y2 - y1)
            })

        # 🚫 Remove larger tracks that enclose or significantly overlap others
        filtered_tracks = suppress_enclosing_or_overlapping_boxes(draw_tracks)

        vehicle_count = len(filtered_tracks)

        for t in filtered_tracks:
            track = t["track"]
            x1, y1, x2, y2 = t["bbox"]
            label = f"ID {track.track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame, [t["track"] for t in filtered_tracks]
