from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import torch

def track_video(video_path, output_path, model_path):
    detection_model = AutoDetectionModel.from_pretrained(
        model_path=model_path,
        confidence_threshold=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    classes_included = {"person", "boat", "bird", "surfboard"}

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            result = get_sliced_prediction(
                frame,
                detection_model,
                slice_height=256,
                slice_width=256,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )

            annotated_frame = frame.copy()

            for prediction in result.object_prediction_list:
                if prediction.category.name in classes_included:
                    bbox = prediction.bbox.to_xyxy()
                    x1, y1, x2, y2 = map(int, bbox)
                    label = f"{prediction.category.name} {prediction.score.value:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

            out.write(annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_path
