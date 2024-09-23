import os
import cv2
import torch
import numpy as np
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import matplotlib.pyplot as plt

TEXT_PROMPT = "clothing"
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_models():
    """Initialize the SAM2 and Grounding DINO models."""
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    grounding_model = load_model(model_config_path=GROUNDING_DINO_CONFIG, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT, device=DEVICE)
    return sam2_predictor, grounding_model

def segment_image(img_path):
    """Segment the image using SAM2 and Grounding DINO."""
    sam2_predictor, grounding_model = initialize_models()
    
    image_source, image = load_image(img_path)

    sam2_predictor.set_image(image_source)

    # Use Grounding DINO to find objects
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    # Process the boxes for SAM2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # Use SAM2 to get masks
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    segmented_img = cv2.bitwise_and(image_source, image_source, mask=combined_mask)

    return segmented_img  # Return the segmented image


