"""
Visualize model predictions — Nano version.

The ONNX/TRT models are exported with decode_in_inference=False,
so this script applies the YOLOX grid decode in Python after inference.
This avoids TRT op-compatibility issues on older Jetson TRT versions.

Usage:
    python nano_visualize_predictions.py
"""
import os, glob, colorsys, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import einops
from yolox.model import create_yolox_m
from yolox.handle_weights import load_pretrained_weights
from data_utils.metrics import post_process_img

NUM_CLASSES = 8
INPUT_SIZE = 640
CONF_TH = 0.25
IOU_TH = 0.50

CLASS_NAMES = [
    "Ascomycetes/Ascospores",        # 0
    "Basidiomycetes/Basidiospores",  # 1
    "Cladosporium",                  # 2
    "Debris",                        # 3
    "Hyphal fragments",              # 4
    "Non-specified Spore",           # 5
    "Penicillium/Aspergillus-like",  # 6
    "Rusts/Smuts/Periconia/Myxomycetes",  # 7
]

# Generate distinct colors per class (BGR)
CLASS_COLORS = []
for i in range(len(CLASS_NAMES)):
    hue = i / len(CLASS_NAMES)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    CLASS_COLORS.append((int(b * 255), int(g * 255), int(r * 255)))


def process_frame(frame, device='cuda', output_size=640):
    img = einops.rearrange(frame, 'h w c -> c h w')
    img = torch.from_numpy(img).float().to(device)

    height, width = img.shape[1:]
    scale = min(output_size / width, output_size / height)
    new_width, new_height = int(width * scale), int(height * scale)
    img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width),
                        mode='bilinear', align_corners=False)

    pad_top = (output_size - new_height) // 2
    pad_bottom = output_size - new_height - pad_top
    pad_left = (output_size - new_width) // 2
    pad_right = output_size - new_width - pad_left

    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=114.0)
    return img, scale, (pad_top, pad_left)


# ---------------------------------------------------------------------------
# Grid decode (replaces the decode_outputs that runs INSIDE the model on the
# regular export but is stripped out for the Nano export)
# ---------------------------------------------------------------------------
def decode_yolox_output(output, input_size=640, strides=[8, 16, 32]):
    """Decode raw YOLOX output (exported with decode_in_inference=False).

    obj and class probs already have sigmoid applied by the model.
    Box coords are still raw: this function applies
        cx = (raw_x + grid_x) * stride
        cy = (raw_y + grid_y) * stride
        w  = exp(raw_w) * stride
        h  = exp(raw_h) * stride

    Args:
        output: (batch, 8400, 5+num_classes)
    Returns:
        decoded: same shape, with pixel-coordinate boxes
    """
    grids = []
    strides_t = []
    for s in strides:
        hsize = input_size // s
        wsize = input_size // s
        yv, xv = torch.meshgrid(
            torch.arange(hsize), torch.arange(wsize), indexing='ij'
        )
        grid = torch.stack((xv, yv), 2).view(1, -1, 2).float()
        grids.append(grid)
        strides_t.append(torch.full((1, hsize * wsize, 1), s, dtype=torch.float32))

    grids = torch.cat(grids, dim=1).to(output.device)
    strides_t = torch.cat(strides_t, dim=1).to(output.device)

    decoded = output.clone()
    decoded[..., 0:2] = (output[..., 0:2] + grids) * strides_t
    decoded[..., 2:4] = torch.exp(output[..., 2:4]) * strides_t
    # indices 4+ (obj, class probs) already have sigmoid — pass through
    return decoded


def draw_boxes(img, boxes, title, show_scores=False):
    """Draw bounding boxes on the image.
    boxes: list of (class_id, x1, y1, x2, y2[, score])
    """
    cv2.putText(img, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    for box in boxes:
        c = int(box[0])
        x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
        if c < 0 or c >= len(CLASS_NAMES):
            continue
        color = CLASS_COLORS[c]
        if show_scores and len(box) > 5:
            text = f"{CLASS_NAMES[c]} {box[5]:.2f}"
        else:
            text = CLASS_NAMES[c]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def load_gt_boxes(label_path, img_w, img_h):
    """Load YOLO format labels and convert to pixel coords."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            c = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            boxes.append((c, x1, y1, x2, y2))
    return boxes


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_preds_to_gt(pred_boxes, gt_boxes, iou_thresh=0.5):
    """Greedy matching: for each pred, find best IoU GT with same class.
    Returns per-class true positive counts."""
    matched_gt = set()
    tp_counts = {}
    scored = [(i, b) for i, b in enumerate(pred_boxes) if len(b) > 5]
    scored.sort(key=lambda x: -x[1][5])
    for _, pb in scored:
        pc = int(pb[0])
        best_iou = 0.0
        best_gt = -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            if int(gb[0]) != pc:
                continue
            iou = compute_iou(pb[1:5], gb[1:5])
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_gt >= 0 and best_iou >= iou_thresh:
            matched_gt.add(best_gt)
            tp_counts[pc] = tp_counts.get(pc, 0) + 1
    return tp_counts


def run_evaluation(image_paths, label_dir, output_dir, infer_fn, backend_name, device="cpu"):
    """Run inference + evaluation using the provided inference function.

    infer_fn(img_tensor) -> output tensor of shape (1, 8400, 5+num_classes)
        (already decoded to pixel coords)
    """
    os.makedirs(output_dir, exist_ok=True)

    gt_counts = [0] * NUM_CLASSES
    pred_counts = [0] * NUM_CLASSES
    tp_counts = [0] * NUM_CLASSES

    t0 = time.time()
    processed = 0
    with torch.no_grad():
        for img_path in image_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"WARNING: could not read {img_path}, skipping")
                continue

            h, w = frame.shape[:2]
            img, scale, (pad_top, pad_left) = process_frame(frame, device=device, output_size=INPUT_SIZE)

            # Inference (backend-specific, returns decoded output)
            output = infer_fn(img)

            # Post-process
            preds = post_process_img(output[0], confidence_threshold=CONF_TH, iou_threshold=IOU_TH)
            preds = preds.cpu().numpy()

            # Rescale predictions from padded 640x640 back to original image coords
            pred_boxes = []
            if preds.size > 0:
                for p in preds:
                    c, x1, y1, x2, y2, s = p
                    x1 = (x1 - pad_left) / scale
                    y1 = (y1 - pad_top) / scale
                    x2 = (x2 - pad_left) / scale
                    y2 = (y2 - pad_top) / scale
                    pred_boxes.append((c, x1, y1, x2, y2, s))
                    c_int = int(c)
                    if 0 <= c_int < NUM_CLASSES:
                        pred_counts[c_int] += 1

            # Load ground truth
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, stem + ".txt")
            gt_boxes = load_gt_boxes(label_path, w, h)
            for box in gt_boxes:
                c_int = int(box[0])
                if 0 <= c_int < NUM_CLASSES:
                    gt_counts[c_int] += 1

            # Match predictions to GT for true positive counting
            img_tp = match_preds_to_gt(pred_boxes, gt_boxes, iou_thresh=IOU_TH)
            for cls_id, count in img_tp.items():
                if 0 <= cls_id < NUM_CLASSES:
                    tp_counts[cls_id] += count

            # Draw side by side with divider
            gt_vis = draw_boxes(frame.copy(), gt_boxes, "Ground Truth")
            pred_vis = draw_boxes(frame.copy(), pred_boxes, f"Predictions ({backend_name})", show_scores=True)
            divider = np.full((frame.shape[0], 4, 3), (0, 0, 255), dtype=np.uint8)
            combined = np.hstack([gt_vis, divider, pred_vis])

            out_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, combined)
            processed += 1

    # Print summary statistics
    DEBRIS_IDX = CLASS_NAMES.index("Debris")
    spore_indices = [i for i in range(NUM_CLASSES) if i != DEBRIS_IDX]

    def err_str(gt, pred):
        if gt > 0:
            return f"{abs(pred - gt) / gt * 100:>.1f}%"
        return "   N/A"

    def miss_rate_str(tp, gt):
        if gt > 0:
            return f"{(1- (tp / gt)) * 100:.1f}%"
        return "  N/A"

    W = 107
    print(f"\n{'='*W}")
    print(f"  {backend_name} Results")
    print(f"{'='*W}")
    print(f"{'Class':<40} {'Actual':>8} {'Predicted':>10} {'Error %':>9} {'Correctly Detected':>20} {'Miss Rate %':>13}")
    print(f"{'='*W}")
    for i in spore_indices:
        print(f"{CLASS_NAMES[i]:<40} {gt_counts[i]:>8} {pred_counts[i]:>10} {err_str(gt_counts[i], pred_counts[i]):>9} {tp_counts[i]:>20} {miss_rate_str(tp_counts[i], gt_counts[i]):>13}")
    print(f"{'-'*W}")
    spore_gt   = sum(gt_counts[i]   for i in spore_indices)
    spore_pred = sum(pred_counts[i] for i in spore_indices)
    spore_tp   = sum(tp_counts[i]   for i in spore_indices)
    print(f"{'SPORE TOTAL':<40} {spore_gt:>8} {spore_pred:>10} {err_str(spore_gt, spore_pred):>9} {spore_tp:>20} {miss_rate_str(spore_tp, spore_gt):>13}")
    print(f"{'='*W}")
    debris_gt   = gt_counts[DEBRIS_IDX]
    debris_pred = pred_counts[DEBRIS_IDX]
    debris_tp   = tp_counts[DEBRIS_IDX]
    print(f"{'Debris (excluded from spore total)':<40} {debris_gt:>8} {debris_pred:>10} {err_str(debris_gt, debris_pred):>9} {debris_tp:>20} {miss_rate_str(debris_tp, debris_gt):>13}")
    print(f"{'='*W}")
    total_time = time.time() - t0
    per_tile = total_time / processed if processed > 0 else 0
    slide_estimate = per_tile * 1116
    print(f"\nProcessed {processed} tiles in {total_time:.1f}s ({per_tile:.2f}s per tile)")
    print(f"Estimated full slide time (~1116 tiles): {slide_estimate:.0f}s ({slide_estimate/60:.1f} min)")
    print(f"Done! Results saved to {output_dir}/")


def main():
    # ---- Configuration -------------------------------------------------------- #
    IMAGE_DIR  = "data/images/val"
    LABEL_DIR  = "data/labels/val"
    CHECKPOINT = "yolox_m_uaFalse_transformsTrue_dn(train.txt)_nc8_ep400_bs128_lr1e-04_wd5e-04_03-09_13.pth"
    ONNX_PATH  = "onnx/yolox_m_nano.onnx"     # exported with decode_in_inference=False
    TRT_PATH   = "onnx/yolox_m_nano.trt"       # built from the no-decode ONNX
    # --------------------------------------------------------------------------- #

    # Gather image paths
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if os.path.isdir(IMAGE_DIR):
        image_paths = sorted(
            p for p in glob.glob(os.path.join(IMAGE_DIR, "*"))
            if os.path.splitext(p)[1].lower() in IMG_EXTS
        )
    else:
        image_paths = [IMAGE_DIR]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    print(f"Found {len(image_paths)} image(s)\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ===== 1) PyTorch (.pth) backend ========================================= #
    print(f"{'#'*60}")
    print(f"  PyTorch Backend  —  {CHECKPOINT}")
    print(f"{'#'*60}")
    model = create_yolox_m(NUM_CLASSES)
    model = load_pretrained_weights(model, CHECKPOINT, NUM_CLASSES, remap=False)
    model.to(device).eval()

    pth_output_dir = f"Test_Visualize_PTH_{CHECKPOINT.split('/')[-1].replace('.pth', '')}"

    def pth_infer(img_tensor):
        # PyTorch model has decode_in_inference=True by default — output is already decoded
        return model(img_tensor)

    run_evaluation(image_paths, LABEL_DIR, pth_output_dir, pth_infer, "PyTorch (.pth)", device=device)

    # --- Capture raw PyTorch output (decode=False) for TRT comparison ------- #
    model.head.decode_in_inference = False
    diag_frame = cv2.imread(image_paths[0])
    diag_img, _, _ = process_frame(diag_frame, device=device, output_size=INPUT_SIZE)
    with torch.no_grad():
        pth_raw = model(diag_img)  # raw: (1, 8400, 13), no grid decode
    print(f"\n  [Diagnostic] Saved PyTorch raw output for TRT comparison (shape {pth_raw.shape})")

    # Free PyTorch model memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ===== 2) TensorRT backend =============================================== #
    # (ONNX Runtime skipped — no CUDA EP on Jetson ARM, CPU is too slow)
    if os.path.exists(TRT_PATH):
        import tensorrt as trt
        print(f"\n\n{'#'*60}")
        print(f"  TensorRT Backend  —  {TRT_PATH}")
        print(f"{'#'*60}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(TRT_PATH, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            trt_engine = runtime.deserialize_cuda_engine(f.read())
        trt_context = trt_engine.create_execution_context()

        # The no-decode model outputs (1, 8400, 5+num_classes) — same shape,
        # but box coords are raw (not yet grid-decoded)
        input_shape  = (1, 3, 640, 640)
        output_shape = (1, 8400, 5 + NUM_CLASSES)
        trt_input_buf  = torch.empty(input_shape,  dtype=torch.float32, device="cuda")
        trt_output_buf = torch.empty(output_shape, dtype=torch.float32, device="cuda")

        trt_context.set_tensor_address("input",  trt_input_buf.data_ptr())
        trt_context.set_tensor_address("output", trt_output_buf.data_ptr())

        trt_stream = torch.cuda.Stream()

        # ---- Diagnostic: compare PyTorch raw vs TRT raw on first image ----- #
        print("\n--- TRT vs PyTorch DIAGNOSTIC (first image) ---")
        torch.cuda.current_stream().synchronize()  # ensure diag_img is ready
        trt_input_buf.copy_(diag_img)
        torch.cuda.current_stream().synchronize()  # ensure copy done
        trt_context.execute_async_v3(stream_handle=trt_stream.cuda_stream)
        trt_stream.synchronize()
        trt_raw = trt_output_buf.clone()

        # Compare raw outputs (both should be undecoded)
        diff = (pth_raw - trt_raw).abs()
        print(f"  PyTorch raw shape: {pth_raw.shape}")
        print(f"  TRT     raw shape: {trt_raw.shape}")
        print(f"  Max |PT - TRT| overall:   {diff.max().item():.6f}")
        print(f"  Max |PT - TRT| xy (0:2):  {diff[..., 0:2].max().item():.6f}")
        print(f"  Max |PT - TRT| wh (2:4):  {diff[..., 2:4].max().item():.6f}")
        print(f"  Max |PT - TRT| obj (4):   {diff[..., 4].max().item():.6f}")
        print(f"  Max |PT - TRT| cls (5:):  {diff[..., 5:].max().item():.6f}")
        # Show anchor with highest PyTorch confidence for sanity check
        top_obj_idx = pth_raw[0, :, 4].argmax().item()
        print(f"\n  Anchor with highest PyTorch obj (idx={top_obj_idx}):")
        print(f"    PyTorch: {pth_raw[0, top_obj_idx, :6].tolist()}")
        print(f"    TRT:     {trt_raw[0, top_obj_idx, :6].tolist()}")
        if diff.max().item() > 1.0:
            print(f"\n  ⚠️  LARGE DIFFERENCE — TRT is NOT matching PyTorch!")
            print(f"  ⚠️  Likely cause: stale ONNX/TRT from different weights or old export.")
            print(f"  ⚠️  FIX: Delete {ONNX_PATH} and {TRT_PATH}, re-run nano_onnx_export.py")
        elif diff.max().item() > 0.01:
            print(f"\n  ⚠️  Moderate difference — possible FP precision issue.")
        else:
            print(f"\n  ✅ TRT raw output matches PyTorch closely.")
        print("--- END DIAGNOSTIC ---\n")

        def trt_infer(img_tensor):
            trt_input_buf.copy_(img_tensor)
            torch.cuda.current_stream().synchronize()  # ensure copy before TRT reads
            trt_context.execute_async_v3(stream_handle=trt_stream.cuda_stream)
            trt_stream.synchronize()
            return decode_yolox_output(trt_output_buf.clone())

        trt_output_dir = f"Test_Visualize_TRT_{os.path.splitext(os.path.basename(TRT_PATH))[0]}"
        run_evaluation(image_paths, LABEL_DIR, trt_output_dir, trt_infer, "TensorRT", device=device)
    else:
        print(f"\n[TRT] Engine not found at {TRT_PATH}, skipping. Run python nano_onnx_export.py to build it.")


if __name__ == "__main__":
    main()
