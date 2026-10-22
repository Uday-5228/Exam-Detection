# Smart CCTV Exam Cheating Detection

Prototype for reviewing suspicious activity in staged exam-hall video. Alerts are intended to support a human invigilator; they are not conclusions that cheating occurred.

## Current implementation

Phase 0 setup and the Phase 1 single-video pipeline are implemented. The project loads configuration from `config.yaml`, reads a file, RTSP stream, or webcam, runs pretrained YOLO person/phone detection and ByteTrack plus pose estimation, and writes an annotated video. Multi-stream serving, behavior alerts, persistence, and dashboard work belong to later phases.

## Install

Use Python 3.10 or 3.11, then from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place a demo clip at `data/clips/hall_a.mp4` or update the stream source in `config.yaml`. Ultralytics downloads the configured pretrained weights on first use unless they are already available locally. The weights are intentionally excluded from version control.

## Run

Check and print the parsed configuration:

```powershell
python -c "from src.config import load; print(load())"
```

Run the first configured video and write `outputs/annotated.mp4`:

```powershell
python scripts/run_offline.py
```

Choose a configured stream and/or override its input and output:

```powershell
python scripts/run_offline.py --stream cam1 --source data/clips/hall_a.mp4 --output outputs/hall_a_annotated.mp4
```

The console reports processed-frame FPS. Boxes show person track IDs and class labels; visible pose keypoints are drawn for people.

## Troubleshooting

- **Python is not found:** install Python 3.10/3.11 and use the Windows `py -3.11` launcher or activate the virtual environment.
- **Cannot open video:** verify the configured path from the project root, or pass `--source` with a valid file, RTSP URL, or webcam index.
- **Model download fails:** download `yolov8n.pt` and `yolov8n-pose.pt` before the demo and set their local paths under `model` in `config.yaml`.
- **CPU is slow:** set `process_every_n_frames` higher or use smaller input videos. The detector reuses the most recent detections on skipped frames.
- **No phones appear:** the pretrained COCO phone class can miss small or occluded devices; tune the configured confidence and use clear staged footage.

## Data and privacy

Keep recordings in `data/clips/`; that directory and generated snapshots, database files, outputs, and model weights are ignored by Git. This prototype does not intentionally retain video frames other than the annotated output explicitly requested by the offline runner.
