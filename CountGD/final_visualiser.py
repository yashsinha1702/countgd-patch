import torch
import PIL
from PIL import Image
import torchvision.transforms.functional as F
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CONF_THRESH = 0.012
DEFAULT_IMAGE_SIZE = 800
MAX_IMAGE_SIZE = 1333

class ObjectCounter:
    def __init__(self, config_path=None, checkpoint_path=None):
        """
        Initialize the object counter with config and checkpoint paths
        """
        self.config_path = config_path or "config/cfg_fsc147_vit_b.py"
        self.checkpoint_path = checkpoint_path or "train8.pth"
        self.device = self._get_device()
        self.model = None
        self.transform = None

    def _get_device(self):
        """Get the appropriate device (GPU/CPU)"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        return device

    def _build_transforms(self):
        """Build image transformation pipeline"""
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return T.Compose([
            T.RandomResize([DEFAULT_IMAGE_SIZE], max_size=MAX_IMAGE_SIZE),
            normalize,
        ])

    def initialize(self):
        """Initialize the model and transforms"""
        try:
            # Set random seeds
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

            # Check config file
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Config file not found: {self.config_path}")

            # Load config
            cfg = SLConfig.fromfile(self.config_path)
            cfg.device = str(self.device)

            # Build model
            from models.registry import MODULE_BUILD_FUNCS
            build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
            if build_func is None:
                raise ValueError(f"Model {cfg.modelname} not found in registry")

            # Initialize model
            self.model, _, _ = build_func(cfg)

            # Check checkpoint file
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)["model"]
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)

            # Initialize transforms
            self.transform = self._build_transforms()

            logger.info("Model initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            return False

    def process_image(self, image_path, text, exemplar_boxes=None, output_dir=None):
        """
        Process an image to count and visualize objects
        
        Args:
            image_path (str): Path to input image
            text (str): Description of objects to count
            exemplar_boxes (list): Optional list of example box coordinates
            output_dir (str): Optional directory to save visualization
            
        Returns:
            tuple: (count, visualization_path)
        """
        try:
            # Validate inputs
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            if self.model is None:
                raise RuntimeError("Model not initialized. Call initialize() first.")

            # Load and prepare image
            image = Image.open(image_path)
            logger.info(f"Loaded image {image_path} with size {image.size}")

            # Prepare exemplar boxes
            if exemplar_boxes is None or len(exemplar_boxes) == 0:
                exemplar_boxes = torch.zeros((0, 4), dtype=torch.float32)
            else:
                exemplar_boxes = torch.tensor(exemplar_boxes, dtype=torch.float32)

            # Transform image
            input_image, _ = self.transform(image, {"exemplars": exemplar_boxes})
            input_image = input_image.to(self.device)
            input_image_nested = nested_tensor_from_tensor_list([input_image])

            # Prepare text prompt
            if not text.endswith('.'):
                text = text + ' in the image.'
            logger.info(f"Using text prompt: {text}")

            # Prepare targets
            targets = [{
                "caption": text,
                "exemplars": exemplar_boxes.to(self.device),
                "labels": torch.zeros(exemplar_boxes.size(0), dtype=torch.long, device=self.device)
            }]

            # Get predictions
            with torch.no_grad():
                predictions = self.model(
                    input_image_nested,
                    [t["exemplars"] for t in targets],
                    [t["labels"] for t in targets],
                    captions=[t["caption"] for t in targets]
                )

            # Process predictions
            ind_to_filter = self._get_ind_to_filter(text, predictions["token"][0].word_ids)
            logits = predictions["pred_logits"].sigmoid()[0][:, ind_to_filter]
            boxes = predictions["pred_boxes"][0]

            # Log confidence scores
            confidence_scores = logits.max(dim=-1).values
            logger.info(f"Max confidence: {confidence_scores.max().item():.3f}")
            logger.info(f"Mean confidence: {confidence_scores.mean().item():.3f}")

            # Filter boxes by confidence
            box_mask = confidence_scores > CONF_THRESH
            boxes = boxes[box_mask, :].cpu().numpy()
            count = boxes.shape[0]
            logger.info(f"Detected {count} objects")

            # Create visualization
            vis_path = self._create_visualization(
                image, boxes, image_path, output_dir)

            return count, vis_path

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def _get_ind_to_filter(self, text, word_ids, keywords=""):
        """Get indices to filter predictions"""
        if len(keywords) <= 0:
            return list(range(len(word_ids)))
        
        input_words = text.split()
        keywords = [k.strip() for k in keywords.split(",")]
        
        word_inds = []
        for keyword in keywords:
            if keyword in input_words:
                if len(word_inds) <= 0:
                    word_inds.append(input_words.index(keyword))
                else:
                    word_inds.append(input_words.index(keyword, word_inds[-1]))
            else:
                raise Exception("Keywords must be present in input text")
                
        return [i for i, word_id in enumerate(word_ids) if word_id in word_inds]

    def _create_visualization(self, image, boxes, image_path, output_dir=None):
        """Create and save visualization of detections"""
        w, h = image.size
        det_map = np.zeros((h, w))
        det_map[(h * boxes[:, 1]).astype(int), (w * boxes[:, 0]).astype(int)] = 1
        det_map = ndimage.gaussian_filter(det_map, sigma=(w // 200, w // 200), order=0)

        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.imshow(det_map[None, :].transpose(1, 2, 0), 'jet', 
                  interpolation='none', alpha=0.7)
        plt.axis('off')

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            input_filename = os.path.basename(image_path)
            output_filename = f"counted_{input_filename.split('.')[0]}.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved visualization to: {output_path}")
        else:
            output_path = None

        plt.close()
        return output_path

def main():
    """Main function to demonstrate usage"""
    # Initialize counter
    counter = ObjectCounter(
        config_path="C:\\Users\\syash\\Desktop\\CountGD\\config\\cfg_fsc147_vit_b.py",
        checkpoint_path="/DATA/yash22590/CountGD/t2/checkpoint.pth"
        #checkpoint_path="C:\\Users\\syash\\Desktop\\CountGD\\checkpoints\\checkpoint_fsc147_best.pth"
        
    )
    
    if not counter.initialize():
        logger.error("Failed to initialize counter")
        return

    try:
        # Process image
        image_path = "C:\\Users\\syash\\Desktop\\CountGD\\FSC147_384_V2\\images_384_VarV2\\35.jpg"
        #image_path="C:\\Users\\syash\\Desktop\\CountGD\FSC147_384_V2\\images_384_VarV2\\2524.jpg"
        text = "apple"
        output_dir = "C:\\Users\\syash\\Desktop\\CountGD\\fullpred"
        
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