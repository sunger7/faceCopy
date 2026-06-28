"""
Face Swap Pipeline — 命令行版本
用法:
    python face_swap.py -s photo.png -v video.mp4 -o output.mp4
    python face_swap.py -s photo.png -v video.mp4 -o output.mp4 --no-gfpgan --no-color-match
"""

# ── torchvision 兼容补丁（必须在所有其他 import 之前）──────────────────────
import sys, types
import torchvision.transforms.functional as _F
_ft = types.ModuleType("torchvision.transforms.functional_tensor")
_ft.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _ft
# ────────────────────────────────────────────────────────────────────────────

import argparse
import os
import shutil
import subprocess
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="人脸替换流水线")
    project_root = Path(__file__).resolve().parent
    parser.add_argument("-s", "--source", required=True, help="源人脸图片路径")
    parser.add_argument("-v", "--video",  required=True, help="目标视频路径")
    parser.add_argument("-o", "--output", required=True, help="输出视频文件名")
    parser.add_argument("--model",    default=os.path.expanduser("models/inswapper_128.onnx"),
                        help="inswapper_128.onnx 路径 (默认: models/inswapper_128.onnx)")
    parser.add_argument("--gfpgan-model", default="models/GFPGANv1.3.pth",
                        help="GFPGANv1.3.pth 路径 (默认: models/GFPGANv1.3.pth)")
    parser.add_argument("--workdir",  default="workspace", help="中间文件目录 (默认: workspace)")
    parser.add_argument("--upscale",  type=int, default=1, help="GFPGAN 放大倍数 (默认: 1)")
    parser.add_argument("--no-gfpgan",      action="store_true", help="禁用 GFPGAN 人脸修复")
    parser.add_argument("--no-color-match", action="store_true", help="禁用颜色匹配")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU (CUDA ctx_id=0)")
    parser.add_argument("--keep-frames", action="store_true", help="保留中间帧文件")
    parser.add_argument("--ffmpeg-path", default=str(project_root / "ffmpeg"),
                        help="ffmpeg 可执行文件路径 (默认: 当前目录/ffmpeg)")
    parser.add_argument("--ffprobe-path", default=str(project_root / "ffprobe"),
                        help="ffprobe 可执行文件路径 (默认: 当前目录/ffprobe)")
    return parser.parse_args()


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def resolve_tool_path(path_str: str, tool_name: str) -> str:
    tool_path = Path(path_str).expanduser()
    if tool_path.is_file():
        return str(tool_path)
    raise FileNotFoundError(
        f"未找到 {tool_name}: {tool_path}\n"
        f"请将 {tool_name} 放到项目当前目录，或使用 --{tool_name}-path 指定路径。"
    )


def extract_frames(video_path: str, output_dir: Path, ffprobe_bin: str, ffmpeg_bin: str) -> dict:
    probe_cmd = [
        ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        video_path,
    ]
    out = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    parts = out.stdout.strip().split(",")
    width, height = int(parts[0]), int(parts[1])
    fps_num, fps_den = map(int, parts[2].split("/"))
    fps = fps_num / fps_den

    subprocess.run(
        [ffmpeg_bin, "-y", "-i", video_path, str(output_dir / "%06d.png")],
        check=True, capture_output=True,
    )

    frame_paths = sorted(output_dir.glob("*.png"))
    print(f"视频信息: {width}x{height} @ {fps:.3f} fps，共 {len(frame_paths)} 帧")
    return {"width": width, "height": height, "fps": fps, "frame_paths": frame_paths}


def color_match_lab(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src.copy()
    for c in range(3):
        s_mean, s_std = src[:, :, c].mean(), src[:, :, c].std()
        t_mean, t_std = tgt[:, :, c].mean(), tgt[:, :, c].std()
        if s_std < 1e-6:
            continue
        out[:, :, c] = (src[:, :, c] - s_mean) * (t_std / s_std) + t_mean
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def face_bbox_padded(frame: np.ndarray, face, pad: int = 20):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    return max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)


def get_source_face(face_app, image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取: {image_path}")
    faces = face_app.get(img)
    if not faces:
        raise ValueError("源图片中未检测到人脸")
    if len(faces) > 1:
        print(f"⚠ 检测到 {len(faces)} 张人脸，使用面积最大的一张")
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    print(f"✓ 源人脸提取成功，置信度: {faces[0].det_score:.3f}")
    return faces[0]


def process_frame(frame, src_face, face_app, swapper, gfpgan_restorer,
                  enable_color_match: bool) -> np.ndarray:
    target_faces = face_app.get(frame)
    if not target_faces:
        return frame

    result = frame.copy()
    for tgt_face in target_faces:
        result = swapper.get(result, tgt_face, src_face, paste_back=True)

        if enable_color_match:
            x1, y1, x2, y2 = face_bbox_padded(frame, tgt_face)
            ref_roi     = frame[y1:y2, x1:x2]
            swapped_roi = result[y1:y2, x1:x2]
            if ref_roi.size and swapped_roi.size:
                matched = color_match_lab(swapped_roi, ref_roi)
                result[y1:y2, x1:x2] = cv2.addWeighted(matched, 0.6, swapped_roi, 0.4, 0)

    if gfpgan_restorer is not None:
        _, _, result = gfpgan_restorer.enhance(
            result, has_aligned=False, only_center_face=False, paste_back=True
        )

    return result


def encode_video(frames_dir: Path, audio_source: str, output: str,
                 fps: float, work_dir: Path, ffmpeg_bin: str,
                 crf: int = 18, preset: str = "slow"):
    tmp = str(work_dir / "_tmp.mp4")
    subprocess.run([
        ffmpeg_bin, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", "libx264", "-crf", str(crf),
        "-preset", preset, "-pix_fmt", "yuv420p",
        tmp,
    ], check=True, capture_output=True)
    print("视频编码完成")

    subprocess.run([
        ffmpeg_bin, "-y",
        "-i", tmp, "-i", audio_source,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output,
    ], check=True, capture_output=True)
    print("音频合并完成")

    os.remove(tmp)
    print(f"✓ 最终视频: {output}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    enable_gfpgan      = not args.no_gfpgan
    enable_color_match = not args.no_color_match
    ctx_id = 0 if args.gpu else -1

    output_dir = Path(args.workdir)
    frames_dir = output_dir / "frames"
    result_dir = output_dir / "result_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"源图片    : {args.source}")
    print(f"目标视频  : {args.video}")
    print(f"输出文件  : {args.output}")
    print(f"GFPGAN   : {enable_gfpgan}")
    print(f"颜色匹配  : {enable_color_match}")

    ffmpeg_bin = resolve_tool_path(args.ffmpeg_path, "ffmpeg")
    ffprobe_bin = resolve_tool_path(args.ffprobe_path, "ffprobe")
    print(f"ffmpeg   : {ffmpeg_bin}")
    print(f"ffprobe  : {ffprobe_bin}")

    # 1. 抽帧
    meta = extract_frames(args.video, frames_dir, ffprobe_bin, ffmpeg_bin)

    # 2. 加载模型
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    print("\n加载 InsightFace buffalo_l ...")
    face_app = FaceAnalysis(name="buffalo_l", root="./")
    face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    print(f"加载 Inswapper: {args.model}")
    swapper = get_model(args.model, download=False, download_zip=False)

    gfpgan_restorer = None
    if enable_gfpgan:
        from gfpgan import GFPGANer
        print(f"加载 GFPGAN: {args.gfpgan_model}")
        gfpgan_restorer = GFPGANer(
            model_path=args.gfpgan_model,
            upscale=args.upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )

    print("\n✓ 所有模型加载完毕\n")

    # 3. 提取源人脸
    source_face = get_source_face(face_app, args.source)

    # 4. 逐帧换脸
    failed = []
    for fp in tqdm(meta["frame_paths"], desc="换脸进度", unit="frame"):
        frame = cv2.imread(str(fp))
        if frame is None:
            failed.append(fp.name)
            continue
        try:
            processed = process_frame(frame, source_face, face_app, swapper,
                                      gfpgan_restorer, enable_color_match)
        except Exception as e:
            print(f"\n⚠ {fp.name} 失败: {e}，保留原帧")
            processed = frame
            failed.append(fp.name)
        cv2.imwrite(str(result_dir / fp.name), processed)

    print(f"\n✓ 处理完成，共 {len(meta['frame_paths'])} 帧")
    if failed:
        print(f"⚠ 失败帧 ({len(failed)}): {failed[:5]}")

    # 5. 编码输出
    encode_video(result_dir, args.video, args.output, meta["fps"], output_dir, ffmpeg_bin)

    # 6. 清理中间文件
    if not args.keep_frames:
        shutil.rmtree(str(frames_dir))
        shutil.rmtree(str(result_dir))
        print("✓ 中间帧已清理")

    print("\n=== 完成 ===")
    print(f"  输出: {args.output}  ({meta['width']}x{meta['height']} @ {meta['fps']:.3f}fps)")


if __name__ == "__main__":
    main()
