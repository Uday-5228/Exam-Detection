# Smart CCTV Exam Cheating Detection

Prototype for reviewing suspicious activity in staged exam-hall video. Alerts are intended to support a human invigilator; they are not conclusions that cheating occurred.

## Current implementation

Phases 0 through 4 are implemented. The project loads configuration from `config.yaml`, reads file, RTSP, or webcam sources, runs pretrained YOLO person/phone detection and ByteTrack plus pose estimation, evaluates PEEKING and DEVICE_USE rules, saves cooldown-limited snapshots and SQLite events, and serves a live monitoring dashboard through FastAPI. The UI includes a responsive feed grid, real-time alerts, event filters, review controls, and live performance summaries.

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

Run the first configured video, write `outputs/annotated.mp4`, and save alert snapshots and events to the configured locations:

```powershell
python scripts/run_offline.py
```

Choose a configured stream and/or override its input and output:

```powershell
python scripts/run_offline.py --stream cam1 --source data/clips/hall_a.mp4 --output outputs/hall_a_annotated.mp4
```

The console reports processed-frame FPS and any generated alerts. Boxes show person track IDs and class labels; visible pose keypoints are drawn for people. Alert text is neutral and describes suspicious activity for human review.

Run the multi-stream dashboard from the project root:

```powershell
python -m src.main
```

Then open `http://127.0.0.1:8000`. The server binds to localhost so camera feeds, snapshots, and event data are not exposed on other network interfaces by default. Configure each feed in `config.yaml`; file sources loop according to their `loop` setting, while RTSP sources reconnect after interruptions.

## Troubleshooting

- **Python is not found:** install Python 3.10/3.11 and use the Windows `py -3.11` launcher or activate the virtual environment.
- **Cannot open video:** verify the configured path from the project root, or pass `--source` with a valid file, RTSP URL, or webcam index.
- **Model download fails:** download `yolov8n.pt` and `yolov8n-pose.pt` before the demo and set their local paths under `model` in `config.yaml`.
- **CPU is slow:** set `process_every_n_frames` higher or use smaller input videos. The detector reuses the most recent detections on skipped frames.
- **No phones appear:** the pretrained COCO phone class can miss small or occluded devices; tune the configured confidence and use clear staged footage.

## Data and privacy

Keep recordings in `data/clips/`; that directory and generated snapshots, database files, outputs, and model weights are ignored by Git. This prototype does not intentionally retain video frames other than the annotated output explicitly requested by the offline runner.
