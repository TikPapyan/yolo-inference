from track_video import track_video

model_path = "model/yolov10n.pt"
input_path = 'video/new.mp4'
output_path = "video/output.mp4"

video = track_video(input_path, output_path, model_path)
