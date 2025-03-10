import torch
import PIL
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import random
import datasets.transforms as T
from util.slconfig import SLConfig
from util.misc import nested_tensor_from_tensor_list
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import io
import os
import logging
import util.misc as utils
import torchvision.ops

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CONF_THRESH = 0.1
PATCH_SIZE = 224
OVERLAP = 32
DEFAULT_IMAGE_SIZE = 800
MAX_IMAGE_SIZE = 800

class ObjectCounter:
    def _create_nested_tensor(self, tensor):
        """Create nested tensor with proper format"""
        mask = torch.zeros((tensor.shape[0], tensor.shape[2], tensor.shape[3]),
                        dtype=torch.bool, device=self.device)
        return utils.NestedTensor(tensor, mask)

    def _transform_image(self, patch):
        """Transform an image patch with proper target handling"""
        try:
            if self.transform is None:
                self.transform = self._build_transforms()
                
            # Create target dictionary with empty exemplars
            target = {
                "exemplars": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.long)
            }
            
            # Transform patch
            transformed_patch, _ = self.transform(patch, target)
            return transformed_patch
            
        except Exception as e:
            logger.error(f"Error in transform_image: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def process_patch(self, patch, x, y, text, exemplar_boxes):
        """Process a single patch with correct tensor handling"""
        try:
            # Transform patch
            input_patch = self._transform_image(patch)
            if input_patch is None:
                logger.warning(f"Failed to transform patch at ({x}, {y})")
                return None
                
            # Move to device and ensure proper dimensions
            input_patch = input_patch.to(self.device)
            if input_patch.dim() == 3:
                input_patch = input_patch.unsqueeze(0)
                
            # Create nested tensor directly
            input_patch_nested = self._create_nested_tensor(input_patch)
            
            # Prepare exemplar boxes if provided
            if exemplar_boxes is None or len(exemplar_boxes) == 0:
                exemplar_boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
            else:
                exemplar_boxes = exemplar_boxes.to(self.device)
                
            # Ensure text format matches original code
            if not text.endswith('.'):
                text = text + ' in the image.'
                
            # Prepare targets
            targets = [{
                "caption": text,
                "exemplars": exemplar_boxes,
                "labels": torch.zeros(exemplar_boxes.size(0), dtype=torch.long, device=self.device)
            }]

            # Get predictions
            with torch.no_grad():
                predictions = self.model(
                    input_patch_nested,
                    [t["exemplars"] for t in targets],
                    [t["labels"] for t in targets],
                    captions=[t["caption"] for t in targets]
                )
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error processing patch: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _build_transforms(self):
        """Build transform pipeline exactly matching original code"""
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return T.Compose([
            T.RandomResize([DEFAULT_IMAGE_SIZE], max_size=MAX_IMAGE_SIZE),
            normalize,
        ])

    def __init__(self, config_path=None, checkpoint_path=None):
        """Initialize the object counter with config and checkpoint paths"""
        self.config_path = config_path or "config/cfg_fsc147_vit_b.py"
        self.checkpoint_path = checkpoint_path or "train8.pth"
        self.device = self._get_device()
        self.model = None
        self.transform = None  # Will be initialized in _transform_image

    def _get_device(self):
        """Get the appropriate device (GPU/CPU)"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        return device


    def initialize(self):
        """Initialize the model and transforms"""
        try:
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Config file not found: {self.config_path}")

            cfg = SLConfig.fromfile(self.config_path)
            cfg.device = str(self.device)

            from models.registry import MODULE_BUILD_FUNCS
            build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
            if build_func is None:
                raise ValueError(f"Model {cfg.modelname} not found in registry")

            self.model, _, _ = build_func(cfg)

            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)["model"]
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)

            logger.info("Model initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            return False



    def process_image(self, image_path, text, exemplar_boxes=None, output_dir=None):
        """Process an image using patch-based approach"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            if self.model is None:
                raise RuntimeError("Model not initialized. Call initialize() first.")
            
            # Load and validate image
            image = Image.open(image_path)
            image = image.resize((800,800))

            if image.mode != 'RGB':
                image = image.convert('RGB')
            logger.info(f"Loaded image {image_path} with size {image.size}")
            
            # Initialize detection map
            det_map = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
            all_boxes = []
            
            # Get patches
            patches = self._get_patches(image)
            logger.info(f"Generated {len(patches)} patches")
            
            # Process each patch
            for idx, (patch, (x, y)) in enumerate(patches):
                logger.info(f"Processing patch {idx+1}/{len(patches)} at position ({x}, {y})")
                predictions = self.process_patch(patch, x, y, text, exemplar_boxes)
                
                if predictions is None:
                    continue
                    
                # Process predictions
                pred_logits = predictions["pred_logits"].sigmoid()[0]
                pred_boxes = predictions["pred_boxes"][0]
                
                # Get confidence scores
                scores = pred_logits.max(dim=-1).values
                
                # Filter by confidence
                keep_idx = scores > CONF_THRESH
                filtered_boxes = pred_boxes[keep_idx]
                
                # Convert and scale coordinates
                if filtered_boxes.numel() > 0:
                    boxes_np = filtered_boxes.cpu().numpy()
                    
                    # Scale coordinates relative to patch size and add offset
                    patch_w, patch_h = patch.size
                    
                    # Scale x coordinates
                    boxes_np[:, [0, 2]] = boxes_np[:, [0, 2]] * patch_w
                    boxes_np[:, [0, 2]] += x
                    
                    # Scale y coordinates
                    boxes_np[:, [1, 3]] = boxes_np[:, [1, 3]] * patch_h
                    boxes_np[:, [1, 3]] += y
                    
                    # Normalize to image size for final output
                    boxes_np[:, [0, 2]] /= image.size[0]
                    boxes_np[:, [1, 3]] /= image.size[1]
                    
                    # Extract centers
                    centers = np.stack([
                        (boxes_np[:, 0] + boxes_np[:, 2]) / 2,
                        (boxes_np[:, 1] + boxes_np[:, 3]) / 2
                    ], axis=1)
                    
                    # Add to all boxes
                    all_boxes.extend(centers.tolist())
            
            all_boxes = np.array(all_boxes)
            # Non-maximum suppression to remove duplicate detections
            if len(all_boxes) > 0:
                # Convert to x1,y1,x2,y2 format for NMS
                boxes_for_nms = np.zeros((len(all_boxes), 4))
                radius = 0.02  # Adjust this value based on your needs
                boxes_for_nms[:, 0] = all_boxes[:, 0] - radius  # x1
                boxes_for_nms[:, 1] = all_boxes[:, 1] - radius  # y1
                boxes_for_nms[:, 2] = all_boxes[:, 0] + radius  # x2
                boxes_for_nms[:, 3] = all_boxes[:, 1] + radius  # y2
                
                # Convert to torch tensors for NMS
                boxes_t = torch.from_numpy(boxes_for_nms).float().to(self.device)
                scores_t = torch.ones(len(boxes_t)).to(self.device)  # Assume equal confidence
                
                # Perform NMS
                keep_idx = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.3)
                
                # Update all_boxes with NMS results
                all_boxes = all_boxes[keep_idx.cpu().numpy()]
            
            # Create visualization
            vis_path = self._create_visualization(image, all_boxes, det_map, image_path, output_dir)
            
            count = len(all_boxes)
            logger.info(f"Detected {count} objects")
            
            return count, vis_path
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _get_patches(self, image):
        """Generate overlapping patches from the image"""
        width, height = image.size
        patches = []
        
        # Calculate steps with overlap
        x_stride = PATCH_SIZE - OVERLAP
        y_stride = PATCH_SIZE - OVERLAP
        
        # Ensure we cover the entire image
        y_positions = list(range(0, height - PATCH_SIZE + 1, y_stride))
        x_positions = list(range(0, width - PATCH_SIZE + 1, x_stride))
        
        # Add final positions if needed
        if height > PATCH_SIZE and y_positions[-1] + PATCH_SIZE < height:
            y_positions.append(height - PATCH_SIZE)
        if width > PATCH_SIZE and x_positions[-1] + PATCH_SIZE < width:
            x_positions.append(width - PATCH_SIZE)
            
        for y in y_positions:
            for x in x_positions:
                # Extract patch
                patch = image.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                if patch.mode != 'RGB':
                    patch = patch.convert('RGB')
                patches.append((patch, (x, y)))
                
        logger.info(f"Generated {len(patches)} patches with size {PATCH_SIZE}x{PATCH_SIZE} and overlap {OVERLAP}")
        return patches


    def _get_ind_to_filter(self, text, word_ids, keywords=""):
        """Get indices to filter predictions"""
        try:
            if not word_ids or len(word_ids) == 0:
                return []
                
            if len(keywords) <= 0:
                return list(range(len(word_ids)))
            
            input_words = text.lower().split()
            keywords = [k.strip().lower() for k in keywords.split(",")]
            
            word_inds = []
            for keyword in keywords:
                if keyword in input_words:
                    if len(word_inds) <= 0:
                        word_inds.append(input_words.index(keyword))
                    else:
                        word_inds.append(input_words.index(keyword, word_inds[-1]))
                else:
                    logger.warning(f"Keyword '{keyword}' not found in input text")
                    continue
                    
            return [i for i, word_id in enumerate(word_ids) if word_id in word_inds]
            
        except Exception as e:
            logger.error(f"Error in _get_ind_to_filter: {str(e)}")
            return []


    def _create_visualization(self, image, boxes, det_map, image_path, output_dir=None):
        """
        Create and save visualization of detections with heat map
        Args:
            image: PIL Image object
            boxes: numpy array of box coordinates (N x 2) with normalized coordinates
            det_map: numpy array of detection map (unused, kept for compatibility)
            image_path: path to original image
            output_dir: directory to save visualization
        """
        try:
            # Get image dimensions
            w, h = image.size
            
            # Create fresh detection map
            det_map = np.zeros((h, w))
            print("det map:", det_map.shape)

            print(boxes)
            
            if len(boxes) > 0:
                # Convert normalized coordinates to pixel coordinates
                pixel_coords = np.zeros_like(boxes)
                #pixel_coords[:, 0] = ((boxes[:, 0]/0.51)-0.1)  * w  # x coordinates
                #pixel_coords[:, 1] = ((boxes[:, 1]/0.51)-0.1 )* h  # y coordinates
                
                pixel_coords[:, 0] = boxes[:, 0] * w  # x coordinates
                pixel_coords[:, 1] = boxes[:, 1] * h# y coordinates
                
                # Round to nearest integer
                pixel_coords = np.round(pixel_coords).astype(int)
                
                # Filter out-of-bounds coordinates
                valid_mask = (
                    (pixel_coords[:, 0] >= 0) & 
                    (pixel_coords[:, 0] < w) & 
                    (pixel_coords[:, 1] >= 0) & 
                    (pixel_coords[:, 1] < h)
                )
                pixel_coords = pixel_coords[valid_mask]
                
                # Place detections in map
                for x, y in pixel_coords:
                    det_map[y, x] = 1
            
            # Apply Gaussian smoothing
            sigma = w / 200.0  # Scale sigma with image width as in original code
            det_map = ndimage.gaussian_filter(det_map, sigma=(sigma, sigma), order=0)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Show original image
            plt.imshow(image)
            
            # Overlay detection heatmap
            # Transpose to (H, W, 1) to match original code
            plt.imshow(det_map[None, :].transpose(1, 2, 0), 
                    cmap='jet', 
                    interpolation='none', 
                    alpha=0.7)
            
            plt.axis('off')
            
            # Save or display
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                input_filename = os.path.basename(image_path)
                output_filename = f"counted_{input_filename.split('.')[0]}.png"
                output_path = os.path.join(output_dir, output_filename)
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                logger.info(f"Saved visualization to: {output_path}")
                plt.close()
                return output_path
            else:
                plt.show()
                plt.close()
                return None
                
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def main():
    """Main function to demonstrate usage"""
    counter = ObjectCounter(
        config_path="C:\\Users\\syash\\Desktop\\CountGD\\config\\cfg_fsc147_vit_b.py",
        checkpoint_path="C:\\Users\\syash\\Desktop\\CountGD\\newtrain1.pth"
        #checkpoint_path="C:\\Users\\syash\\Desktop\\CountGD\\checkpoints\\checkpoint_fsc147_best.pth"
    )
    
    if not counter.initialize():
        logger.error("Failed to initialize counter")
        return

    try:
        image_path = "C:\\Users\\syash\\Desktop\\CountGD\\FSC147_384_V2\\images_384_VarV2\\35.jpg"
        text = "apple"
        output_dir = "C:\\Users\\syash\\Desktop\\CountGD\\predimage1"
        
        count, vis_path = counter.process_image(
            image_path=image_path,
            text=text,
            output_dir=output_dir
        )
        
        print(f"\nResults:")
        print(f"Detected count: {count}")
        if vis_path:
            print(f"Visualization saved to: {vis_path}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()