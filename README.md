# Video Object Tracking with AutoDetectionModel

This project demonstrates how to track specific objects in a video using the AutoDetectionModel from the sahi library. The script track_video.py uses computer vision techniques to detect and annotate objects such as persons, boats, birds, and surfboards in a video.

## Usage

### Prerequisites

- Python 3.x
- PyTorch
- OpenCV

### Installation

Clone the repository:

```
git clone <repository_url>
cd <repository_directory>
```

### Install dependencies:

```
pip install -r requirements.txt
```

### Running the Script

To track objects in a video, modify `main.py` with your video path and run it:

```
from track_video import track_video

model_path = "model/yolov10n.pt"  # Path to your trained model
input_path = 'video/new.mp4'      # Path to input video
output_path = "video/output.mp4"  # Output path for annotated video

video = track_video(input_path, output_path, model_path)
```