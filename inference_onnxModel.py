import os
import subprocess
import numpy as np
import cv2
import argparse
import audio
import shutil
import librosa
from tqdm import tqdm
from scipy.io.wavfile import write
import gc
import sys

import onnxruntime

onnxruntime.set_default_logger_severity(3)

# face detection and alignment
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head_256

detector = RetinaFace(
    "utils/scrfd_2.5g_bnkps.onnx",
    provider=[
        ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
        "CPUExecutionProvider",
    ],
    session_options=None,
)

# specific face selector
from faceID.faceID import FaceRecognition

recognition = FaceRecognition("faceID/recognition.onnx")


# arguments
parser = argparse.ArgumentParser(
    description="Inference code to lip-sync videos in the wild using Wav2Lip models"
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Name of saved checkpoint to load weights from",
    required=True,
)
parser.add_argument(
    "--face",
    type=str,
    help="Filepath of video/image that contains faces to use",
    required=True,
)
parser.add_argument(
    "--audio",
    type=str,
    help="Filepath of video/audio file to use as raw audio source",
    required=True,
)
parser.add_argument(
    "--denoise",
    default=False,
    action="store_true",
    help="Denoise input audio to avoid unwanted lipmovement",
)
parser.add_argument(
    "--outfile",
    type=str,
    help="Video path to save result. See default for an e.g.",
    default="results/result_voice.mp4",
)
parser.add_argument("--hq_output", default=False, action="store_true", help="HQ output")

parser.add_argument(
    "--static",
    default=False,
    action="store_true",
    help="If True, then use only first video frame for inference",
)
parser.add_argument(
    "--pingpong",
    default=False,
    action="store_true",
    help="pingpong loop if audio is longer than video",
)

parser.add_argument("--cut_in", type=int, default=0, help="Frame to start inference")
parser.add_argument("--cut_out", type=int, default=0, help="Frame to end inference")
parser.add_argument("--fade", action="store_true", help="Fade in/out")

parser.add_argument(
    "--fps",
    type=float,
    help="Can be specified only if input is a static image (default: 25)",
    default=25.0,
    required=False,
)
parser.add_argument(
    "--resize_factor",
    default=1,
    type=int,
    help="Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p",
)

parser.add_argument(
    "--enhancer",
    default="none",
    choices=["none", "gpen", "gfpgan", "codeformer", "restoreformer"],
)
parser.add_argument(
    "--blending",
    default=10,
    type=float,
    help="Amount of face enhancement blending 1 - 10",
)
parser.add_argument(
    "--sharpen",
    default=False,
    action="store_true",
    help="Slightly sharpen swapped face",
)
parser.add_argument("--frame_enhancer", action="store_true", help="Use frame enhancer")

parser.add_argument("--face_mask", action="store_true", help="Use face mask")
parser.add_argument(
    "--face_occluder", action="store_true", help="Use x-seg occluder face mask"
)

parser.add_argument(
    "--pads",
    type=int,
    default=4,
    help="Padding top, bottom to adjust best mouth position, move crop up/down, between -15 to 15",
)  # pos value mov synced mouth up
parser.add_argument(
    "--face_mode",
    type=int,
    default=0,
    help="Face crop mode, 0 or 1, rect or square, affects mouth opening",
)

parser.add_argument(
    "--preview", default=False, action="store_true", help="Preview during inference"
)

args = parser.parse_args()

if (
    args.checkpoint_path == "checkpoints\wav2lip_384.onnx"
    or args.checkpoint_path == "checkpoints\wav2lip_384_fp16.onnx"
):
    args.img_size = 384
else:
    args.img_size = 96

mel_step_size = 16
padY = max(-15, min(args.pads, 15))

device = "cpu"
if onnxruntime.get_device() == "GPU":
    device = "cuda"
print("Running on " + device)


if args.enhancer == "gpen":
    from enhancers.GPEN.GPEN import GPEN

    enhancer = GPEN(
        model_path="enhancers/GPEN/GPEN-BFR-256-sim.onnx", device=device
    )  # GPEN-BFR-256-sim

if args.enhancer == "codeformer":
    from enhancers.Codeformer.Codeformer import CodeFormer

    enhancer = CodeFormer(
        model_path="enhancers/Codeformer/codeformerfixed.onnx", device=device
    )

if args.enhancer == "restoreformer":
    from enhancers.restoreformer.restoreformer16 import RestoreFormer

    enhancer = RestoreFormer(
        model_path="enhancers/restoreformer/restoreformer16.onnx", device=device
    )

if args.enhancer == "gfpgan":
    from enhancers.GFPGAN.GFPGAN import GFPGAN

    enhancer = GFPGAN(model_path="enhancers/GFPGAN/GFPGANv1.4.onnx", device=device)

if args.frame_enhancer:
    from enhancers.RealEsrgan.esrganONNX import RealESRGAN_ONNX

    frame_enhancer = RealESRGAN_ONNX(
        model_path="enhancers/RealEsrgan/clear_reality_x4.onnx", device=device
    )

if args.face_mask:
    from blendmasker.blendmask import BLENDMASK

    masker = BLENDMASK(model_path="blendmasker/blendmasker.onnx", device=device)

if args.face_occluder:
    from xseg.xseg import MASK

    occluder = MASK(model_path="xseg/xseg.onnx", device=device)

if args.denoise:
    from resemble_denoiser.resemble_denoiser import ResembleDenoiser

    denoiser = ResembleDenoiser(
        model_path="resemble_denoiser/denoiser.onnx", device=device
    )

if os.path.isfile(args.face) and args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
    args.static: args.static = True


def load_model(device):
    model_path = args.checkpoint_path
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            "CPUExecutionProvider",
        ]

    session = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=providers
    )

    return session


def select_specific_face(model, spec_img, size, crop_scale=1.0):

    # select face:
    h, w = spec_img.shape[:-1]
    roi = (0, 0, w, h)
    cropped_roi = spec_img[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]

    bboxes, kpss = model.detect(cropped_roi, input_size=(320, 320), det_thresh=0.3)
    assert len(kpss) != 0, "No face detected"

    target_face, mat = get_cropped_head_256(
        cropped_roi, kpss[0], size=size, scale=crop_scale
    )
    target_face = cv2.resize(target_face, (112, 112))
    target_id = recognition(target_face)[0].flatten()

    return target_id


def fallback_passthrough_segment(video_path, audio_path, output_path):
    """
    Fallback when no face is detected: Mux original video with aligned audio and exit.
    Also removes any temp files for this segment.
    """
    print("[WARN] No face detected — falling back to original video with new audio.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define common FFmpeg flags (must match the rest of your pipeline)
    common_ffmpeg_flags = [
        "-shortest",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-acodec",
        "aac",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-b:a",
        "128k",
    ]

    # Mux original video with new audio
    command = (
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
        ]
        + common_ffmpeg_flags
        + [output_path]
    )

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[INFO] Fallback output written to {output_path}")
    except subprocess.CalledProcessError as e:
        print("[ERROR] Fallback muxing failed!")
        print("STDERR:\n", e.stderr.decode())
        raise

    # Clean up any temp files
    for temp_file in ["temp/temp.wav", "temp/temp.mp4"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    if os.path.exists("hq_temp"):
        shutil.rmtree("hq_temp")


def estimate_yaw_pitch(kps):
    # Assume kps = np.array with shape (5, 2), typical from RetinaFace
    left_eye, right_eye, nose, left_mouth, right_mouth = kps
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    yaw = np.degrees(np.arctan2(dy, dx))

    # Rough vertical angle estimation
    mid_eyes = (left_eye + right_eye) / 2
    d_nose = nose[1] - mid_eyes[1]
    pitch = np.degrees(np.arctan2(d_nose, dx))  # dx as approx distance for scale

    return yaw, pitch


def process_video_specific(model, img, size, target_id, crop_scale=1.0):
    ori_img = img
    bboxes, kpss = model.detect(ori_img, input_size=(320, 320), det_thresh=0.3)

    assert len(kpss) != 0, "No face detected"

    best_score = -float("inf")
    best_aimg = None
    best_mat = None

    for kps in kpss:
        yaw, pitch = estimate_yaw_pitch(kps)
        if abs(yaw) > 60 or abs(pitch) > 30:
            continue  # skip faces with extreme pose

        aimg, mat = get_cropped_head_256(ori_img, kps, size=size, scale=crop_scale)
        face = aimg.copy()
        face = cv2.resize(face, (112, 112))
        face_id = recognition(face)[0].flatten()
        score = target_id @ face_id

        if score > best_score:
            best_score = score
            best_aimg = aimg
            best_mat = mat

    if best_score < 0.4 or best_aimg is None:
        return None, None

    return best_aimg, best_mat


def face_detect(images, target_id):
    print("Detecting face and generating data...")

    sub_faces = []
    crop_faces = []
    matrix = []
    face_error = []

    for i in tqdm(range(len(images))):
        try:
            crop_face, M = process_video_specific(
                detector, images[i], 256, target_id, crop_scale=1.0
            )

            if crop_face is None or M is None:
                raise ValueError("Face too rotated or not found")

            if args.face_mode == 0:
                sub_face = crop_face[65 - padY : 241 - padY, 62:194]
            else:
                sub_face = crop_face[65 - padY : 241 - padY, 42:214]

            sub_face = cv2.resize(sub_face, (args.img_size, args.img_size))

            sub_faces.append(sub_face)
            crop_faces.append(crop_face)
            matrix.append(M)
            face_error.append(0)

        except:
            # fallback: use empty face, and mark as no-face
            crop_face = np.zeros((256, 256), dtype=np.uint8)
            crop_face = cv2.cvtColor(crop_face, cv2.COLOR_GRAY2RGB) / 255
            sub_face = crop_face[65 - padY : 241 - padY, 62:194]
            sub_face = cv2.resize(sub_face, (args.img_size, args.img_size))
            M = np.float32([[1, 0, 0], [0, 1, 0]])

            sub_faces.append(sub_face)
            crop_faces.append(crop_face)
            matrix.append(M)
            face_error.append(-1)

    return crop_faces, sub_faces, matrix, face_error


def datagen(frames, mels):

    img_batch, mel_batch, frame_batch = [], [], []

    for i, m in enumerate(mels):

        idx = 0 if args.static else i % len(frames)

        frame_to_save = frames[idx].copy()
        frame_batch.append(frame_to_save)

        img_batch.append(frames[idx])
        mel_batch.append(m)

        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2 :] = 0
        # img_masked[:, :, args.img_size // 2:, :] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        yield img_batch, mel_batch, frame_batch
        img_batch, mel_batch, frame_batch = [], [], []


def main():
    """
    Main function for running Wav2Lip-ONNX HQ inference.
    Handles:
    - Frame extraction from input video or image
    - Face mask generation
    - Face detection on initial frame
    """
    if args.hq_output and not os.path.exists("hq_temp"):
        os.mkdir("hq_temp")

    # HQ processing config
    preset = "medium"
    blend = args.blending / 10

    # Create static face mask
    static_face_mask = np.zeros((224, 224), dtype=np.uint8)
    static_face_mask = cv2.ellipse(
        static_face_mask, (112, 162), (62, 54), 0, 0, 360, (255, 255, 255), -1
    )
    static_face_mask = cv2.ellipse(
        static_face_mask, (112, 122), (46, 23), 0, 0, 360, (0, 0, 0), -1
    )
    static_face_mask = cv2.resize(static_face_mask, (256, 256))
    static_face_mask = cv2.rectangle(
        static_face_mask, (0, 246), (246, 246), (0, 0, 0), -1
    )
    static_face_mask = cv2.cvtColor(static_face_mask, cv2.COLOR_GRAY2RGB) / 255
    static_face_mask = cv2.GaussianBlur(static_face_mask, (19, 19), cv2.BORDER_DEFAULT)

    # Create sub face mask
    sub_face_mask = np.zeros((256, 256), dtype=np.uint8)
    sub_face_mask = cv2.rectangle(
        sub_face_mask, (42, 65 - padY), (214, 249), (255, 255, 255), -1
    )
    sub_face_mask = cv2.GaussianBlur(
        sub_face_mask.astype(np.uint8), (29, 29), cv2.BORDER_DEFAULT
    )
    sub_face_mask = cv2.cvtColor(sub_face_mask, cv2.COLOR_GRAY2RGB)
    sub_face_mask = sub_face_mask / 255

    # Validate face path
    if not os.path.isfile(args.face):
        raise ValueError("--face argument must be a valid path to video/image file")

    # Process static image input
    if args.face.split(".")[-1] in ["jpg", "png", "jpeg", "bmp"]:
        orig_frame = cv2.imread(args.face)
        orig_frame = cv2.resize(
            orig_frame,
            (
                orig_frame.shape[1] // args.resize_factor,
                orig_frame.shape[0] // args.resize_factor,
            ),
        )
        orig_frames = [orig_frame]
        fps = args.fps

        h, w = orig_frame.shape[:-1]
        roi = (0, 0, w, h)
        cropped_roi = orig_frame[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
        full_frames = [cropped_roi]
        orig_h, orig_w = cropped_roi.shape[:-1]

        try:
            target_id = select_specific_face(detector, cropped_roi, 256, crop_scale=1)
        except ValueError:
            fallback_passthrough_segment(args.face, args.audio, args.outfile)
            sys.exit(0)

    # Process video input
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        video_stream.set(1, args.cut_in)

        print("Reading video frames...")

        if args.cut_out == 0:
            args.cut_out = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

        new_duration = args.cut_out - args.cut_in
        if args.static:
            new_duration = 1

        full_frames = []
        orig_frames = []

        for l in range(new_duration):
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            if args.resize_factor > 1:
                frame = cv2.resize(
                    frame,
                    (
                        frame.shape[1] // args.resize_factor,
                        frame.shape[0] // args.resize_factor,
                    ),
                )

            if l == 0:
                h, w = frame.shape[:-1]
                roi = (0, 0, w, h)
                cropped_roi = frame[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
                target_id = select_specific_face(
                    detector, cropped_roi, 256, crop_scale=1
                )
                orig_h, orig_w = cropped_roi.shape[:-1]
                print("Reading frames....")

            print(f"\r{l}", end=" ", flush=True)

            cropped_roi = frame[
                int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
            ]
            full_frames.append(cropped_roi)
            orig_frames.append(cropped_roi)

    # Report memory usage of loaded frames
    memory_usage_bytes = sum(frame.nbytes for frame in full_frames)
    memory_usage_mb = memory_usage_bytes / (1024**2)
    print(
        f"Number of frames used for inference: {len(full_frames)} (~{int(memory_usage_mb)} MB)"
    )

    # Convert input audio to mono WAV format
    print("Extracting raw audio...")
    os.makedirs("temp", exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            args.audio,
            "-ac",
            "1",
            "-strict",
            "-2",
            "temp/temp.wav",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.system("cls")  # Clear screen for next stage
    print("Raw audio extracted")

    # Denoise audio if requested
    if args.denoise:
        print("Denoising audio...")
        wav, sr = librosa.load("temp/temp.wav", sr=44100, mono=True)
        wav_denoised, new_sr = denoiser.denoise(wav, sr, batch_process_chunks=False)
        write("temp/temp.wav", new_sr, (wav_denoised * 32767).astype(np.int16))
        if hasattr(denoiser, "session"):
            del denoiser.session
            gc.collect()

    # Convert audio to mel spectrogram
    wav = audio.load_wav("temp/temp.wav", 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            "Mel contains NaNs! Using a TTS voice? Add a small epsilon noise to the wav file and try again."
        )

    # Slice mel into chunks for inference
    mel_chunks = []
    mel_idx_multiplier = 80.0 / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > mel.shape[1]:
            mel_chunks.append(mel[:, -mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print(f"Length of mel chunks: {len(mel_chunks)}")

    # Truncate video frames to match mel chunks
    full_frames = full_frames[: len(mel_chunks)]

    # Run face detection
    aligned_faces, sub_faces, matrix, no_face = face_detect(full_frames, target_id)

    # Handle pingpong augmentation
    if args.pingpong:
        orig_frames *= 2
        full_frames += full_frames[::-1]
        aligned_faces += aligned_faces[::-1]
        sub_faces += sub_faces[::-1]
        matrix += matrix[::-1]
        no_face += no_face[::-1]

    # Initialize inference data generator
    gen = datagen(sub_faces.copy(), mel_chunks)
    fc = 0

    # Load ONNX model
    model = load_model(device)

    frame_h, frame_w = full_frames[0].shape[:2]
    out = cv2.VideoWriter(
        "temp/temp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    )

    os.system("cls")
    print(f"Running on {onnxruntime.get_device()}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Resize factor: {args.resize_factor}")
    if args.pingpong:
        print("Using pingpong mode")
    if args.enhancer != "none":
        print(f"Using enhancer: {args.enhancer}")
    if args.face_mask:
        print("Using face mask")
    if args.face_occluder:
        print("Using occlusion mask")
    print("")

    # Configure fade effects
    fade_in, bright_in = 11, 0
    total_frames = int(np.ceil(len(mel_chunks)))
    fade_out, bright_out = total_frames - 11, 0

    # Loop through data batches
    for i, (img_batch, mel_batch, frames) in enumerate(tqdm(gen, total=total_frames)):
        if fc == len(full_frames):
            fc = 0

        face_err = no_face[fc]

        img_batch = img_batch.transpose((0, 3, 1, 2)).astype(np.float32)
        mel_batch = mel_batch.transpose((0, 3, 1, 2)).astype(np.float32)

        # ONNX inference
        pred = model.run(
            None, {"mel_spectrogram": mel_batch, "video_frames": img_batch}
        )[0][0]

        pred = (
            (pred.transpose(1, 2, 0) * 255)
            .astype(np.uint8)
            .reshape((1, args.img_size, args.img_size, 3))
        )

        mat = matrix[fc]
        mat_rev = cv2.invertAffineTransform(mat)
        aligned_face = aligned_faces[fc].copy()
        aligned_face_orig = aligned_faces[fc].copy()
        full_frame = full_frames[fc]
        final = orig_frames[fc]

        for p, f in zip(pred, frames):
            if not args.static:
                fc += 1

            # Resize prediction to match face mode
            target_size = (132, 176) if args.face_mode == 0 else (172, 176)
            p = cv2.resize(p, target_size)

            # Insert prediction into aligned face
            y1, y2 = 65 - padY, 241 - padY
            x1, x2 = (62, 194) if args.face_mode == 0 else (42, 214)
            aligned_face[y1:y2, x1:x2] = p

            # Blend prediction into original aligned face using mask
            aligned_face = (
                sub_face_mask * aligned_face + (1 - sub_face_mask) * aligned_face_orig
            ).astype(np.uint8)

            # Handle fallback for failed face detection
            if face_err != 0:
                res = full_frame
                face_err = 0
            else:
                # Apply enhancer
                if args.enhancer != "none":
                    enhanced = enhancer.enhance(aligned_face)
                    enhanced = cv2.resize(enhanced, (256, 256))
                    aligned_face = cv2.addWeighted(
                        enhanced.astype(np.float32),
                        blend,
                        aligned_face.astype(np.float32),
                        1.0 - blend,
                        0.0,
                    )

                # Generate appropriate face mask
                if args.face_mask:
                    seg_mask = masker.mask(aligned_face)
                    seg_mask = cv2.blur(seg_mask, (5, 5)) / 255.0
                    mask = cv2.warpAffine(seg_mask, mat_rev, (frame_w, frame_h))

                elif args.face_occluder:
                    try:
                        seg_mask = occluder.mask(aligned_face_orig)
                    except:
                        seg_mask = occluder.mask(aligned_face)
                    seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
                    mask = cv2.warpAffine(seg_mask, mat_rev, (frame_w, frame_h))

                else:
                    mask = cv2.warpAffine(static_face_mask, mat_rev, (frame_w, frame_h))

                # Optional sharpening
                if args.sharpen:
                    aligned_face = cv2.detailEnhance(
                        aligned_face, sigma_s=1.3, sigma_r=0.15
                    )

                # De-align prediction and blend with full frame
                dealigned_face = cv2.warpAffine(
                    aligned_face, mat_rev, (frame_w, frame_h)
                )
                res = (mask * dealigned_face + (1 - mask) * full_frame).astype(np.uint8)

            final = res

        # Optional frame enhancement and resizing
        if args.frame_enhancer:
            final = frame_enhancer.enhance(final)
            final = cv2.resize(final, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

        # Apply fade in/out effects
        if args.fade:
            if i < fade_in:
                final = cv2.convertScaleAbs(final, alpha=0.1 * bright_in, beta=0)
                bright_in += 1
            elif i > fade_out:
                final = cv2.convertScaleAbs(final, alpha=1 - 0.1 * bright_out, beta=0)
                bright_out += 1

        # Save output
        if args.hq_output:
            cv2.imwrite(os.path.join("hq_temp", f"{i:07d}.png"), final)
        else:
            out.write(final)

        # Optional real-time preview
        if args.preview:
            cv2.imshow("Result - press ESC to stop and save", final)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                out.release()
                break
            elif k == ord("s"):
                args.sharpen = not args.sharpen
                print(f"\nSharpen = {args.sharpen}")

    out.release()

    # Build ffmpeg command for final output
    # Ensure uniform encoding for all segments
    common_ffmpeg_flags = [
        "-shortest",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-acodec",
        "aac",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-b:a",
        "128k",
    ]

    if args.hq_output:
        command = (
            [
                "ffmpeg",
                "-y",
                "-i",
                args.audio,
                "-r",
                str(fps),
                "-f",
                "image2",
                "-i",
                "./hq_temp/%07d.png",
            ]
            + common_ffmpeg_flags
            + [args.outfile]
        )
    else:
        command = (
            [
                "ffmpeg",
                "-y",
                "-i",
                args.audio,
                "-i",
                "temp/temp.mp4",
            ]
            + common_ffmpeg_flags
            + [args.outfile]
        )

    # Execute ffmpeg command
    try:
        print("Running ffmpeg command:")
        print(" ".join(command))
        result = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("ffmpeg completed successfully.")
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed:")
        print("STDOUT:\n", e.stdout.decode())
        print("STDERR:\n", e.stderr.decode())
        raise

    # Cleanup temporary files
    for path in ["temp/temp.mp4", "temp/temp.wav"]:
        if os.path.exists(path):
            os.remove(path)

    if os.path.exists("hq_temp"):
        shutil.rmtree("hq_temp")


if __name__ == "__main__":
    main()
