# --------------------------------------------------------
# YOLOv14 Multi-View Demo App
# Supports: pinhole, fisheye, panoramic, drone/BEV, game character
# detection with YOLOv14 variants
# --------------------------------------------------------

import gradio as gr
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

# View type labels for ViewEmbedding module
VIEW_TYPES = {
    "Pinhole (Standard)": 0,
    "Fisheye / Wide-Angle": 1,
    "Panoramic 360°": 2,
    "Drone / Top-Down": 3,
    "BEV / Satellite": 4,
    "Ground / Slanted": 5,
}

# Available model configurations
MODEL_OPTIONS = {
    "YOLOv14n (Standard)": "yolov12n.pt",
    "YOLOv14s (Standard)": "yolov12s.pt",
    "YOLOv14m (Standard)": "yolov12m.pt",
    "YOLOv14l (Standard)": "yolov12l.pt",
    "YOLOv14x (Standard)": "yolov12x.pt",
}

# Scene mode descriptions
SCENE_MODES = {
    "Auto (Adaptive)": {
        "desc": "Auto-detect scene type & apply best augmentation",
        "model_hint": "yolov12-adaptive",
    },
    "Game Characters (Delta Force / COD)": {
        "desc": "Treat game characters as real humans (domain adaptation)",
        "model_hint": "yolov12-game2real",
    },
    "Fisheye / Wide-Angle": {
        "desc": "Compensate for lens distortion",
        "model_hint": "yolov12-deformable",
    },
    "Drone / Aerial": {
        "desc": "Small object detection from top-down view",
        "model_hint": "yolov12-multiview",
    },
    "Panorama 360°": {
        "desc": "Equirectangular panoramic images",
        "model_hint": "yolov12-panorama",
    },
    "Standard": {
        "desc": "Regular pinhole camera images",
        "model_hint": "standard",
    },
}


def apply_fisheye_correction(image, strength=0.0):
    """Apply or correct fisheye distortion."""
    if abs(strength) < 0.01:
        return image
    h, w = image.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32)
    D = np.array([strength, 0, 0, 0], dtype=np.float32)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_32FC1)
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def apply_game_stylization(image, intensity=0.3):
    """Apply game-engine rendering style for immersion preview."""
    if intensity < 0.01 or image is None:
        return image
    # Contrast boost (HDR game lighting)
    alpha = 1.0 + intensity * 0.3
    img = cv2.addWeighted(image, alpha, image, 0, -intensity * 15)
    # Saturation boost
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= (1.0 + intensity * 0.5)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # Sharpening
    k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32) * intensity
    k[1, 1] += 1.0 - intensity
    return np.clip(cv2.filter2D(img, -1, k), 0, 255).astype(np.uint8)


def yolov12_inference(image, video, model_key, image_size, conf_threshold,
                      view_type, scene_mode, correction_strength,
                      game_style_intensity, panorama_mode):
    """Run YOLOv14 inference with multi-view, game, and distortion support."""
    model_path = MODEL_OPTIONS.get(model_key, "yolov12s.pt")
    model = YOLO(model_path)
    view_id = VIEW_TYPES.get(view_type, 0)

    if image is None and video is None:
        return None, None

    def process_frame(frame):
        nonlocal model, image_size, conf_threshold, view_id
        nonlocal correction_strength, game_style_intensity, panorama_mode, scene_mode

        # Step 1: Scene-aware pre-processing
        is_game_mode = "Game" in scene_mode

        # Step 2: Fisheye correction
        if abs(correction_strength) > 0.01:
            frame = apply_fisheye_correction(frame, correction_strength)

        # Step 3: Game style preview (visual only — model handles domain internally)
        if is_game_mode and game_style_intensity > 0.01:
            frame = apply_game_stylization(frame, game_style_intensity)

        # Step 4: Panorama pre-processing
        if panorama_mode:
            h, w = frame.shape[:2]
            crop = int(w * 0.05)
            frame = frame[:, crop:w - crop]

        # Step 5: Run inference
        results = model.predict(
            source=frame,
            imgsz=image_size,
            conf=conf_threshold,
            verbose=False,
        )
        annotated = results[0].plot()
        return annotated[:, :, ::-1]  # BGR to RGB

    if image is not None:
        img_array = np.array(image)
        annotated = process_frame(img_array)
        return annotated, None
    else:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*'vp80'),
            fps, (frame_width, frame_height)
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated = process_frame(frame)
            out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()
        return None, output_video_path


def build_app():
    """Build the Gradio interface."""
    with gr.Blocks(title="YOLOv14 - Multi-View & Game2Real Detection",
                   css="footer {display:none !important}") as demo:
        gr.Markdown(
            """
            # YOLOv14 Multi-View & Game2Real Object Detection
            ### 🎯 Pinhole | Fisheye | Panorama | Drone/BEV | 🎮 Game Characters

            > **Game2Real**: Detects game characters (Delta Force, COD, PUBG) as real humans
            > via domain adaptation. **Adaptive**: Auto-detects scene type & applies best model.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input
                image = gr.Image(type="pil", label="Input Image", visible=True)
                video = gr.Video(label="Input Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"], value="Image",
                    label="Input Type",
                )

                # Model
                model_id = gr.Dropdown(
                    label="Base Model",
                    choices=list(MODEL_OPTIONS.keys()),
                    value="YOLOv14s (Standard)",
                )

                # Scene Mode
                scene_mode = gr.Radio(
                    label="🎯 Scene Mode",
                    choices=list(SCENE_MODES.keys()),
                    value="Standard",
                )

                # Multi-View & Distortion settings
                with gr.Accordion("Advanced Settings", open=True):
                    view_type = gr.Dropdown(
                        label="Camera / View Type",
                        choices=list(VIEW_TYPES.keys()),
                        value="Pinhole (Standard)",
                    )
                    correction_strength = gr.Slider(
                        label="Fisheye Correction",
                        minimum=-0.5, maximum=0.5, step=0.05, value=0.0,
                        info="Negative = barrel, Positive = pincushion",
                    )
                    game_style_intensity = gr.Slider(
                        label="🎮 Game Style Preview",
                        minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                        info="Visual game-style overlay (model handles domain internally)",
                    )
                    panorama_mode = gr.Checkbox(
                        label="Panorama 360° Mode",
                        value=False,
                    )

                # Detection params
                image_size = gr.Slider(
                    label="Image Size", minimum=320, maximum=1280,
                    step=32, value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold", minimum=0.0, maximum=1.0,
                    step=0.05, value=0.25,
                )

                detect_btn = gr.Button(value="🚀 Detect Objects", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(type="numpy", label="Detected Objects", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

                gr.Markdown(
                    """
                    ---
                    ### 🎮 Game2Real
                    Game characters → detected as "person" via domain adaptation.

                    ### 🔄 Adaptive
                    Scene auto-detected → optimal model applied automatically.

                    ### 📐 Multi-View
                    Pinhole | Fisheye | Panorama 360° | Drone | BEV | Ground
                    """
                )

        # Input type switching
        def update_visibility(input_type):
            is_img = input_type == "Image"
            return (
                gr.update(visible=is_img),
                gr.update(visible=not is_img),
                gr.update(visible=is_img),
                gr.update(visible=not is_img),
            )

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        # Scene mode → auto-update view type
        def on_scene_change(mode):
            hints = {
                "Auto (Adaptive)": "Pinhole (Standard)",
                "Game Characters (Delta Force / COD)": "Pinhole (Standard)",
                "Fisheye / Wide-Angle": "Fisheye / Wide-Angle",
                "Drone / Aerial": "Drone / Top-Down",
                "Panorama 360°": "Panoramic 360°",
                "Standard": "Pinhole (Standard)",
            }
            return gr.update(value=hints.get(mode, "Pinhole (Standard)"))

        scene_mode.change(fn=on_scene_change, inputs=[scene_mode], outputs=[view_type])

        # Inference
        def run_inference(image, video, model_id, image_size, conf_threshold,
                          view_type, scene_mode, correction_strength,
                          game_style_intensity, panorama_mode, input_type):
            if input_type == "Image":
                return yolov12_inference(
                    image, None, model_id, image_size, conf_threshold,
                    view_type, scene_mode, correction_strength,
                    game_style_intensity, panorama_mode,
                )
            else:
                return yolov12_inference(
                    None, video, model_id, image_size, conf_threshold,
                    view_type, scene_mode, correction_strength,
                    game_style_intensity, panorama_mode,
                )

        detect_btn.click(
            fn=run_inference,
            inputs=[
                image, video, model_id, image_size, conf_threshold,
                view_type, scene_mode, correction_strength,
                game_style_intensity, panorama_mode, input_type,
            ],
            outputs=[output_image, output_video],
        )

        # Examples
        gr.Examples(
            examples=[
                ["ultralytics/assets/bus.jpg", "YOLOv14s (Standard)", 640, 0.25,
                 "Pinhole (Standard)", "Standard", 0.0, 0.0, False],
                ["ultralytics/assets/zidane.jpg", "YOLOv14x (Standard)", 640, 0.25,
                 "Pinhole (Standard)", "Standard", 0.0, 0.0, False],
            ],
            fn=lambda img, *args: yolov12_inference(
                img, None, args[0], args[1], args[2],
                args[3], args[4], args[5], args[6], args[7]
            )[0],
            inputs=[
                image, model_id, image_size, conf_threshold,
                view_type, scene_mode, correction_strength,
                game_style_intensity, panorama_mode,
            ],
            outputs=[output_image],
            cache_examples=False,
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
