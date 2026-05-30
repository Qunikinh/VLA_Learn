import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageDraw # 用于图像处理和在占位符上绘制文本
import matplotlib.colors
import cv2 # OpenCV 用于高斯模糊
from tqdm import tqdm

# 控制是否打印详细日志（默认关闭）
VERBOSE = False

def set_verbose(v: bool):
    global VERBOSE
    VERBOSE = bool(v)

#todo尝试导入YOLO模型库
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: 'ultralytics' library not found. YOLO model functionality will be unavailable, falling back to simulated heatmap.")
    print("Please install the required libraries with 'pip install ultralytics opencv-python'.")
    YOLO_AVAILABLE = False

#todo --- 辅助函数：生成模拟热力图 (用作占位符或回退) ---
def generate_simulated_heatmap(image_width, image_height, object_center_x, object_center_y, object_width, object_height, max_intensity=255, falloff_rate=0.0005):
    """
    Generates a simulated heatmap for an object.
    Intensity is highest at the object's center and falls off.
    This function serves as a placeholder for actual model output or as a fallback.
    占位图像，真实输出不可用时将输出这个，中心强度最高的无效热力图。
    """
    y, x = np.ogrid[:image_height, :image_width]
    std_x = object_width / 2
    std_y = object_height / 2
    std_x = max(std_x, 1) # Avoid division by zero
    std_y = max(std_y, 1) # Avoid division by zero

    dist_sq = (((x - object_center_x)**2) / (2 * std_x**2)) + \
              (((y - object_center_y)**2) / (2 * std_y**2))
    heatmap = max_intensity * np.exp(-dist_sq * falloff_rate * 10)
    return np.clip(heatmap, 0, max_intensity)

#todo --- 函数：从真实模型获取热力图 ---
def get_heatmap_from_actual_model(image_np, model_type='detection', object_class_name='cup', model=None, model_name='yolov8s.pt'):
    """
    Attempts to get a heatmap from a real model.
    Uses YOLOv10x if available for object detection and heatmap generation.
    Otherwise, falls back to a simulated heatmap.
    获取模型输出的热力图，目前仅支持检测模型。
    Args:
        image_np (numpy.ndarray): Input image as a NumPy array (H, W, C).图像
        model_type (str): Currently only 'detection' is supported.模型
        object_class_name (str): Target class name for detection (e.g., 'cup').目标
        model (YOLO, optional): Preloaded YOLO model instance for batch inference.
        model_name (str): Model file name to load if `model` is not provided.

    Returns:
        numpy.ndarray: Generated heatmap (2D array).返回图像数组
    """
    if VERBOSE:
        print(f"Attempting to generate heatmap using '{model_type}' model approach.")
    image_height, image_width = image_np.shape[:2]

    if model_type == 'detection' and YOLO_AVAILABLE:
        try:
            if VERBOSE:
                print(f"  Step: Loading pre-trained {model_name} model.")
            if model is None:
                model = YOLO(model_name)
            if VERBOSE:
                print("  Step: Preprocessing image and performing inference.")
            # 将模型推理降为非 verbose，避免大量日志
            results = model(image_np, verbose=False, conf=0.25)
            #*输入图像序列，和设置置信度阈值，verbose输出详细日志，推理结果传递至results
            heatmap = np.zeros((image_height, image_width), dtype=np.float32)
            detections_found = 0 #  初始化检测到的目标数量计数器
            #*从模型内查找是否有无目标，并匹配id
            if VERBOSE:
                print(f"  Step: Filtering for '{object_class_name}' class detections.")
            target_cls_id = -1
            if hasattr(model, 'names') and isinstance(model.names, dict):
                for cls_id, name_val in model.names.items(): # Renamed 'name' to 'name_val' to avoid conflict
                    if name_val == object_class_name:
                        target_cls_id = cls_id
                        break
            else:
                if VERBOSE:
                    print(f"  Warning: Model class names (model.names) not available in the expected format. Cannot map '{object_class_name}' to class ID.")


            if target_cls_id == -1:
                if VERBOSE:
                    print(f"  Warning: Class '{object_class_name}' not found in model's classes or model.names not accessible. Will display an empty heatmap.")
            else:
                if VERBOSE:
                    print(f"  Class ID for '{object_class_name}': {target_cls_id}")
            #*从模型输出寻找目标id，获取目标框并计数，使用opencv绘制热力图
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        if cls == target_cls_id:
                            detections_found += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # 使用置信度作为热度值填充矩形
                            cv2.rectangle(heatmap, (x1, y1), (x2, y2), conf, thickness=cv2.FILLED)
            #*高斯模糊处理热力图，并return
                if detections_found > 0:
                    if VERBOSE:
                        print(f"  Found {detections_found} '{object_class_name}' detection(s).")
                    # 调整高斯模糊的核大小，可以根据效果调整
                    # 较大的核会产生更模糊（弥散）的热力图
                    blur_kernel_size = (101, 101) # 可以尝试减小如 (51,51) 或增大
                    heatmap = cv2.GaussianBlur(heatmap, blur_kernel_size, 0)
                    if heatmap.max() > 0:
                        heatmap = (heatmap / heatmap.max()) * 255 # 归一化到0-255
                    if VERBOSE:
                        print("  Step: Heatmap generated based on detections.")
                    return heatmap.astype(np.uint8)
                else:
                    if VERBOSE:
                        print(f"  No '{object_class_name}' detections found with current settings. Will display an empty heatmap.")
                    return heatmap # Return empty heatmap

        except Exception as e:
            if VERBOSE:
                print(f"  Error during YOLO model operation: {e}")
                print("  Fallback: Using simulated heatmap.")
            # Fallthrough to simulated heatmap generation

    #*回退至占用图 ----- Fallback to simulated heatmap if model is unavailable or an error occurs -----
    if VERBOSE:
        print("  Fallback: Using simulated heatmap as a placeholder.")
    center_x_ratio = 0.47
    center_y_ratio = 0.45
    width_ratio = 0.20
    height_ratio = 0.30

    obj_center_x_abs = int(center_x_ratio * image_width)
    obj_center_y_abs = int(center_y_ratio * image_height)
    obj_width_abs = int(width_ratio * image_width)
    obj_height_abs = int(height_ratio * image_height)

    simulated_heatmap = generate_simulated_heatmap(
        image_width, image_height,
        obj_center_x_abs, obj_center_y_abs,
        obj_width_abs, obj_height_abs
    )
    return simulated_heatmap


def _save_image_array(image_np, path):
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    Image.fromarray(image_np).save(path)


def _save_heatmap_image(heatmap, path):
    heatmap_img = Image.fromarray(np.clip(heatmap, 0, 255).astype(np.uint8))
    heatmap_img.save(path)


def _create_overlay(image_np, heatmap, alpha=0.5):
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    heatmap_uint8 = np.clip(heatmap, 0, 255).astype(np.uint8)
    if heatmap_uint8.ndim == 3 and heatmap_uint8.shape[2] == 3:
        heat_color = heatmap_uint8
    else:
        heat_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    image_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
    overlay = cv2.addWeighted(image_uint8, 1.0 - alpha, heat_color, alpha, 0)
    return overlay


def process_episode_frames(
    frames,
    episode_id,
    model_type='detection',
    object_class_name='cup',
    model_name='yolov8s.pt',
    save_overlay=True,
    alpha=0.5,
    save_disk=False,
    output_dir='yolo_results',
):
    """Process a batch of episode frames and return the processed frame list."""
    if not frames:
        print(f"No frames to process for episode {episode_id}.")
        return []

    if save_disk:
        episode_dir = os.path.join(output_dir, f'episode_{episode_id:03d}')
        heatmap_dir = os.path.join(episode_dir, 'heatmaps')
        overlay_dir = os.path.join(episode_dir, 'overlays')
        os.makedirs(heatmap_dir, exist_ok=True)
        if save_overlay:
            os.makedirs(overlay_dir, exist_ok=True)

    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            if VERBOSE:
                print(f"Loading batch YOLO model '{model_name}' for episode {episode_id}.")
            yolo_model = YOLO(model_name)
        except Exception as e:
            print(f"Warning: Failed to load YOLO model '{model_name}': {e}")
            yolo_model = None

    processed_frames = []
    # 使用 tqdm 显示 episode 级进度条（每帧为一个单位）
    frame_iter = tqdm(frames, desc=f"YOLO ep {episode_id}", unit='frame')
    for idx, frame in enumerate(frame_iter):
        processed = frame.copy()
        for view_name in ['agent']:
            key = f"{view_name}_image"
            if key not in frame:
                continue

            image_np = frame[key]
            heatmap = get_heatmap_from_actual_model(
                image_np,
                model_type=model_type,
                object_class_name=object_class_name,
                model=yolo_model,
                model_name=model_name,
            )
            overlay = _create_overlay(image_np, heatmap, alpha=alpha)
            processed[key] = overlay

            if save_disk:
                frame_tag = f'episode_{episode_id:03d}_frame_{idx:04d}_{view_name}'
                heatmap_path = os.path.join(heatmap_dir, f'{frame_tag}_heatmap.png')
                _save_heatmap_image(heatmap, heatmap_path)
                if save_overlay:
                    overlay_path = os.path.join(overlay_dir, f'{frame_tag}_overlay.png')
                    _save_image_array(overlay, overlay_path)

        processed_frames.append(processed)

    if save_disk:
        if VERBOSE:
            print(f"YOLO post-processing finished for episode {episode_id}. Results saved to {episode_dir}.")
    return processed_frames


def process_episode_images(
    frames,
    episode_id,
    output_dir='yolo_results',
    model_type='detection',
    object_class_name='cup',
    model_name='yolov8s.pt',
    save_overlay=True,
    alpha=0.5,
):
    """Process frames and save heatmap/overlay images to disk."""
    return process_episode_frames(
        frames,
        episode_id=episode_id,
        model_type=model_type,
        object_class_name=object_class_name,
        model_name=model_name,
        save_overlay=save_overlay,
        alpha=alpha,
        save_disk=True,
        output_dir=output_dir,
    )


def plot_image_with_heatmap(image_path, heatmap_data, title="Object Detection Heatmap", alpha=0.6, cmap_name='inferno'):
    """
    Overlays a heatmap on an image and displays it. All plot text is in English.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}.")
        img = Image.new('RGB', (500, 500), color = (128, 128, 128))
        draw = ImageDraw.Draw(img)
        draw.text((50, 230), "Image not found.\nPlease use a valid path.", fill=(255,0,0))
        heatmap_data = np.zeros((500, 500))
        print("Displaying placeholder image and empty heatmap.")

    img_np = np.array(img)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_np)

    if heatmap_data.max() > 0:
        if heatmap_data.shape[0] != img_np.shape[0] or heatmap_data.shape[1] != img_np.shape[1]:
            print(f"Warning: Heatmap dimensions ({heatmap_data.shape}) differ from image dimensions ({img_np.shape[:2]}). Resizing heatmap.")
            heatmap_pil = Image.fromarray(heatmap_data.astype(np.uint8))
            heatmap_resized_pil = heatmap_pil.resize((img_np.shape[1], img_np.shape[0]), Image.BICUBIC)
            heatmap_data_resized = np.array(heatmap_resized_pil)
            cax = ax.imshow(heatmap_data_resized, cmap=plt.get_cmap(cmap_name), alpha=alpha, extent=(0, img_np.shape[1], img_np.shape[0], 0))
        else:
            cax = ax.imshow(heatmap_data, cmap=plt.get_cmap(cmap_name), alpha=alpha, extent=(0, img_np.shape[1], img_np.shape[0], 0))

        cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Heatmap Intensity (Model-derived or Simulated)', rotation=270, labelpad=15)
    else:
        print("Heatmap is empty (no detections or model not run), not overlaying.")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X-coordinate (pixels)", fontsize=12)
    ax.set_ylabel("Y-coordinate (pixels)", fontsize=12)
    ax.axis('on')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- Configuration ---
    image_file_path = 'pic/cup.jpg' # 默认使用提到识别有困难的俯视图图像
    # image_file_path = 'image_2d8ceb.png' # 之前可以识别的图像
    # image_file_path = 'image_2d208d.jpg' # 另一张测试图像

    target_object_name = 'cup'

    # --- 加载图像 ---
    try:
        img_for_model = Image.open(image_file_path).convert('RGB')
        img_np_for_model = np.array(img_for_model)
        img_height, img_width = img_np_for_model.shape[:2]
        print(f"Preparing to generate heatmap for image: {image_file_path} (Dimensions: {img_width}x{img_height})")
    except FileNotFoundError:
        print(f"Fatal Error: Image file '{image_file_path}' not found. Cannot proceed.")
        img_np_for_model = np.zeros((500, 500, 3), dtype=np.uint8)
        img_width, img_height = 500, 500


    # --- Generate Heatmap ---
    heatmap_output = get_heatmap_from_actual_model(
        img_np_for_model,
        model_type='detection',
        object_class_name=target_object_name
    )

    # --- Plot Image with Heatmap ---
    plot_title = f"Heatmap for '{target_object_name}' (YOLOv10x or Simulated)"
    plot_image_with_heatmap(
        image_path=image_file_path,
        heatmap_data=heatmap_output,
        title=plot_title,
        alpha=0.5,
        cmap_name='inferno'
    )

    if not YOLO_AVAILABLE:
        print("\nReminder: To use the actual YOLO model for heatmap generation, ensure 'ultralytics' and 'opencv-python' are installed.")
        print("You can install them via 'pip install ultralytics opencv-python'.")
        print("Currently displaying a simulated heatmap.")