import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error

# Import FastReID components
from fastreid.config import get_cfg  # pylint: disable=import-error
from fastreid.modeling.meta_arch import build_model  # pylint: disable=import-error
from fastreid.utils.checkpoint import Checkpointer  # pylint: disable=import-error

# Force import all backbone modules to ensure they are registered
try:
    import fastreid.modeling.backbones.resnet
    import fastreid.modeling.backbones.resnest  # This is crucial!

except ImportError as e:
    print(f"Warning: Some backbone modules could not be imported: {e}")

# from torch.backends import cudnn
# cudnn.benchmark = True


def setup_cfg(config_file, opts):
    """
    Load config from file and command-line arguments.
    
    Args:
        config_file: Path to configuration file
        opts: List of configuration options to override
        
    Returns:
        Configuration object
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    return cfg


def postprocess(features):
    """
    Normalize features to compute cosine distance.
    
    Args:
        features: Raw feature tensor
        
    Returns:
        Normalized feature array
    """
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def preprocess(image, input_size):
    """
    Preprocess image for model input.
    
    Args:
        image: Input image
        input_size: Target size (width, height)
        
    Returns:
        Tuple of (preprocessed_image, resize_ratio)
    """
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


class FastReIDInterface:
    """
    Interface for FastReID person re-identification model.
    """
    
    def __init__(self, config_file, weights_path, device, batch_size=8):
        """
        Initialize FastReID interface.
        
        Args:
            config_file: Path to FastReID config file
            weights_path: Path to model weights
            device: Device to run inference on
            batch_size: Batch size for inference
        """
        super(FastReIDInterface, self).__init__()
        
        # Set device
        if device != "cpu":
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.batch_size = batch_size

        # Setup configuration
        self.cfg = setup_cfg(config_file, ["MODEL.WEIGHTS", weights_path])

        # Verify that required backbone is available
        from fastreid.modeling.backbones.build import BACKBONE_REGISTRY
        backbone_name = self.cfg.MODEL.BACKBONE.NAME
        if backbone_name not in BACKBONE_REGISTRY._obj_map:
            available_backbones = list(BACKBONE_REGISTRY._obj_map.keys())
            raise KeyError(
                f"Backbone '{backbone_name}' not found in registry. "
                f"Available backbones: {available_backbones}"
            )

        # Build and load model
        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(weights_path)

        # Move to device and set precision
        if self.device != "cpu":
            self.model = self.model.eval().to(device="cuda").half()
        else:
            self.model = self.model.eval()

        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def inference(self, image, detections):
        """
        Extract features from detected regions.
        
        Args:
            image: Input image
            detections: Detection boxes in format [x1, y1, x2, y2, ...]
            
        Returns:
            Feature embeddings for each detection
        """
        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)

        batch_patches = []
        patches = []
        
        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            patch = image[tlbr[1] : tlbr[3], tlbr[0] : tlbr[2], :]

            # Model expects RGB inputs
            patch = patch[:, :, ::-1]

            # Apply pre-processing to image
            patch, _ = preprocess(patch, (self.pW, self.pH))

            # Convert to tensor and normalize
            patch = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0)
            patch = patch / 255.0
            patches.append(patch)

            if len(patches) == self.batch_size or d == np.size(detections, 0) - 1:
                if len(patches) > 1:
                    batch_patches = torch.cat(patches, dim=0)
                else:
                    batch_patches = patches[0]

                if self.device != "cpu":
                    batch_patches = batch_patches.to(device="cuda").half()

                # Extract features
                with torch.no_grad():
                    features = self.model(batch_patches)
                    features = postprocess(features)

                if len(batch_patches) == 1:
                    batch_features = [features]
                else:
                    batch_features = features

                # Store or return features as needed
                patches = []

        return batch_features