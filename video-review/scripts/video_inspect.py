#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_inspect.py - 视频诊断前置提取工具（零外部 Python 依赖）

功能：
  1. 读取视频元数据（时长/分辨率/帧率/码率）-> meta.json
  2. 按固定间隔抽帧 -> frame_XX.jpg（宽 272px，省 token）
  3. 提取音频 -> audio.mp3（16kHz 单声道，供 ASR 转写）

ffmpeg 探测顺序：
  imageio_ffmpeg 包自带的 ffmpeg -> 系统 PATH 里的 ffmpeg -> OpenCV 兜底（只能抽帧，不能提音频）

用法：
  python video_inspect.py <视频路径> [--outdir 输出目录] [--interval 抽帧间隔秒数，默认12]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys


def find_ffmpeg():
    try:
        import imageio_ffmpeg  # noqa
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    return None


def probe_with_ffmpeg(ffmpeg, src):
    p = subprocess.run([ffmpeg, "-i", src], capture_output=True, text=True, encoding="utf-8", errors="replace")
    info = p.stderr or ""
    meta = {"duration_sec": None, "width": None, "height": None, "fps": None}
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", info)
    if m:
        meta["duration_sec"] = round(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 2)
    v = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", info)
    if v:
        meta["width"], meta["height"] = int(v.group(1)), int(v.group(2))
    f = re.search(r"([\d.]+)\s*fps", info)
    if f:
        meta["fps"] = float(f.group(1))
    return meta


def run_ffmpeg_extract(ffmpeg, src, outdir, interval):
    meta = probe_with_ffmpeg(ffmpeg, src)
    # 抽帧
    frame_tpl = os.path.join(outdir, "frame_%03d.jpg")
    subprocess.run(
        [ffmpeg, "-y", "-i", src, "-vf", f"fps=1/{interval},scale=272:-1", frame_tpl, "-loglevel", "error"],
        check=True,
    )
    # 提音频
    audio_path = os.path.join(outdir, "audio.mp3")
    subprocess.run(
        [ffmpeg, "-y", "-i", src, "-vn", "-ar", "16000", "-ac", "1", audio_path, "-loglevel", "error"],
        check=True,
    )
    return meta, audio_path


def run_cv2_extract(src, outdir, interval):
    import cv2

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("OpenCV 打不开这个视频文件")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {
        "duration_sec": round(total / fps, 2),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(fps, 2),
    }
    step = int(fps * interval)
    idx, saved = 0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = 272 / w
            frame = cv2.resize(frame, (272, int(h * scale)))
            saved += 1
            cv2.imwrite(os.path.join(outdir, f"frame_{saved:03d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        idx += 1
    cap.release()
    return meta, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="本地视频文件路径")
    ap.add_argument("--outdir", default="./video_review_frames", help="输出目录")
    ap.add_argument("--interval", type=float, default=12, help="抽帧间隔（秒），默认12")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(json.dumps({"error": f"文件不存在: {args.video}"}, ensure_ascii=False))
        sys.exit(2)

    os.makedirs(args.outdir, exist_ok=True)
    ffmpeg = find_ffmpeg()

    try:
        if ffmpeg:
            meta, audio_path = run_ffmpeg_extract(ffmpeg, args.video, args.outdir, args.interval)
            engine = "ffmpeg"
        else:
            print("[提示] 未找到 ffmpeg，尝试 OpenCV 兜底（只抽帧，不提音频）", file=sys.stderr)
            meta, audio_path = run_cv2_extract(args.video, args.outdir, args.interval)
            engine = "opencv"
    except Exception as e:
        print(json.dumps({"error": str(e), "hint": "请安装 ffmpeg（加入 PATH）或 pip install imageio-ffmpeg"}, ensure_ascii=False))
        sys.exit(1)

    frames = sorted(f for f in os.listdir(args.outdir) if f.startswith("frame_") and f.endswith(".jpg"))
    result = {
        "engine": engine,
        "meta": meta,
        "frame_count": len(frames),
        "frames_dir": os.path.abspath(args.outdir),
        "audio": os.path.abspath(audio_path) if audio_path else None,
        "audio_note": None if audio_path else "OpenCV 模式无音频，请安装 ffmpeg 或用其他方式转写",
    }
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
