# CountGD-Patch: Training Open-World Object Counting on an 11 GB GPU

[CountGD](https://github.com/niki-amini-naieni/CountGD) (NeurIPS 2024) counts objects of any class. You describe the objects with a text prompt, with a few exemplar boxes, or with both. It reaches state-of-the-art accuracy on FSC-147, but it does not fit in memory on a single 11 GB card. Training and running the full-resolution model needs **more than 12 GB of GPU memory**, so it fails with an out-of-memory error on an **RTX 2080 Ti**.

This repository is my modified copy of CountGD from my B.Tech project at IIIT-Delhi, *"Fractional Batch Training and Active Memory Governance for Ultra-High-Resolution Open-World Object Counting"*. The core idea is to avoid putting the full image through the network in one pass. Instead, the image is cut into small overlapping patches. The patches are processed in fractional batches, and gradients accumulate across them. GPU memory is released between images. Predictions are then merged back to full-image coordinates. The paper extends this to multi-scale patches and to a top-down FPN fusion of features from the different scales.

Related repo: **[fsc147-vlm-annotation-refinement](https://github.com/yashsinha1702/fsc147-vlm-annotation-refinement)**. It is the second half of the same project. It starts from this model's high-recall predictions and uses them to find and fix missing FSC-147 annotations, with a Gemini judge and human review.

---

## Results

Results on FSC-147, **reported in the B.Tech paper (in preparation)**. They were not re-run from this snapshot. See [What is not in this snapshot](#what-is-not-in-this-snapshot).

| Model (FSC-147) | MAE ↓ | RMSE ↓ | Peak GPU memory |
|---|---|---|---|
| Original CountGD (baseline) | 7.10 | 24.56 | > 12 GB, out of memory on an 11 GB RTX 2080 Ti |
| CountGD-Patch, 2-scale | 11.39 | – | 5.2 GB |
| CountGD-Patch, 4-scale | 7.71 | 28.10 | 10.2 GB |
| **CountGD-Patch, 4-scale + top-down FPN** | **7.58** | **26.64** | runs on an RTX 2080 Ti |

With 4 scales and the FPN fusion, CountGD-Patch comes within 0.5 MAE of the original model and stays within the 11 GB budget. The 2-scale version needs less than half that memory, but it is clearly less accurate.

---

## What changed compared with upstream CountGD

This snapshot is based on upstream CountGD at commit [`5522c25`](https://github.com/niki-amini-naieni/CountGD/tree/5522c25) (July 2024). That was before upstream published its own training code. All paths below are inside `CountGD/`.

**Patch-based training: `engine_inference.py`** (the main change)

- `create_patches()` cuts each image into **224 × 224 patches with 32 px overlap** (stride 192), together with the padding mask.
- `adjust_targets_for_patch()` maps the ground-truth boxes into each patch's frame and drops boxes that fall outside the patch.
- In `train_one_epoch()`, all patches of one image form one forward and backward pass. The image's caption and exemplars are repeated for every patch. Gradients **accumulate across the images in a batch**, followed by a single optimizer step, gradient clipping and an optional AMP scaler. `torch.cuda.empty_cache()` runs after every image, so peak memory depends on the patch batch and not on the full image.
- With `--skip_empty_patches`, an image is skipped when none of its patches contains an object.
- The evaluation crop path no longer crashes on edge cases. These are images that produce no crops, crops with no detections, and captions with no `.` separator token.

**Single-image inference and visualisation**

- `patch_visualise.py` runs inference on one image patch by patch. It tiles the image (224 px patches, 32 px overlap) and runs CountGD on each patch with a text prompt and optional exemplars. It maps the predicted object centres back to global coordinates and **removes duplicates in the overlap regions with NMS** (IoU 0.3). The result is saved as a density-style visualisation.
- `final_visualiser.py` produces the same visualisation on the whole image, as the baseline for comparison.

**Entry point and fixes**

- `main_inference.py` adds `--patch_size` (default 224), `--patch_overlap` (default 32) and `--skip_empty_patches`. Two changes behave differently from upstream:
  - `--eval` is inverted (`store_false`). Evaluation runs by default, and passing `--eval` starts **training**.
  - `--pretrain_model_path` is always loaded, even when `--resume` is set.
- `engine_inference_original.py` is the upstream engine with small fixes:
  - the SAM mask is resized to the image size before test-time normalisation;
  - the code uses the newer `torch.amp` API;
  - the CUDA cache is cleared before training.
- `datasets_inference/odvg.py` converts `neg_labels` to a list before `random.sample`, which no longer accepts sets in newer Python versions.
- `download_bert.py` forces a fresh download of `bert-base-uncased`.

**Earlier experiments, kept for reference**

| File | Idea |
|---|---|
| `engine_inference_cropped.py` | Train on adaptive, object-centred crops whose size depends on the mean box size |
| `engine_inference_correcttargets.py` | Same crops, with the targets re-mapped into each crop |
| `engine_inference_adjustedexemandtarg.py` | 4 adaptive crops per image, with both exemplars and targets re-mapped |
| `engine_inference_patches.py`, `engine_inference_AWpatches.py` | First patch-training versions, with one forward pass per patch |

**Other files**

- `t50/`, `t52/`: configs and logs from two training runs (March 2025).
- `test.py`: a small checkpoint inspector.
- `fullpred/`, `predimage1/`, `mymodel/`, `output.png`: example visualisations.

Everything else (`config/`, `datasets/`, `util/`, `tools/`, `engine.py`, `main.py`) is unchanged from upstream.

---

## What is not in this snapshot

The `.gitignore` excludes the model code in `CountGD/models_inference/GroundingDINO/` and `CountGD/groundingdino/`. The training model package `CountGD/models/` was also never uploaded. As a result, this repository contains the **training and inference loop changes but not the model-side changes**. The logs in `t50/info.txt` and `t52/info.txt` show that the trained model had modules that upstream does not have (`patch_gd.feat_proj`, `patch_gd.simple_cnn` / `patch_gd.light_cnn`, `global_proj`). The multi-scale and top-down FPN variants in the results table were built on that model code.

To run this snapshot, copy the missing folders from upstream. This gives you the **original** CountGD model, which the patch-based loop above can train and evaluate:

```bash
git clone https://github.com/niki-amini-naieni/CountGD.git /tmp/CountGD-upstream
git -C /tmp/CountGD-upstream checkout 5522c25
cp -r /tmp/CountGD-upstream/models /tmp/CountGD-upstream/groundingdino CountGD/
cp -r /tmp/CountGD-upstream/models_inference/GroundingDINO CountGD/models_inference/
```

---

## Usage

### 1. Set up CountGD

Follow the *Preparation* section of [`CountGD/README.md`](CountGD/README.md). In short:

- get the FSC-147 dataset;
- use GCC 11;
- create a conda env with Python 3.9.19 and run `pip install -r requirements.txt`;
- build the MultiScaleDeformableAttention ops in `models/GroundingDINO/ops`;
- install `segment-anything`;
- download BERT (`python download_bert.py`), the Swin-B GroundingDINO weights and the SAM ViT-H weights into `CountGD/checkpoints/`.

Copy the missing model folders first (see the section above). Edit the image and annotation paths in `config/datasets_fsc147*.json`.

### 2. Pick the engine

`main_inference.py` chooses the training and evaluation engine through its imports (lines 22–23). The committed version imports `engine_inference_original`. For patch-based training, change it to:

```python
from engine_inference import train_one_epoch
from engine_inference import evaluate
```

### 3. Train with patches

Remember that `--eval` starts **training** in this fork:

```bash
cd CountGD
python -u main_inference.py --output_dir ./run_patch -c config/cfg_fsc147_vit_b.py --eval \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path checkpoints/<starting_checkpoint>.pth \
  --options text_encoder_type=checkpoints/bert-base-uncased \
  --crop --sam_tt_norm --remove_bad_exemplar \
  --patch_size 224 --patch_overlap 32        # optional: --skip_empty_patches --amp
```

This matches the commands recorded in `t50/info.txt` and `t52/info.txt`.

### 4. Evaluate on FSC-147

Leave out `--eval`:

```bash
python -u main_inference.py --output_dir ./countgd_val -c config/cfg_fsc147_val.py \
  --datasets config/datasets_fsc147_val.json \
  --pretrain_model_path checkpoints/<checkpoint>.pth \
  --options text_encoder_type=checkpoints/bert-base-uncased \
  --crop --sam_tt_norm --remove_bad_exemplar
```

Use `config/cfg_fsc147_test.py` and `config/datasets_fsc147_test.json` for the test split. Drop `--sam_tt_norm` to skip SAM test-time normalisation. That is faster, uses less memory and is slightly less accurate. The script prints MAE and RMSE at the end.

### 5. Count objects in a single image

`patch_visualise.py` (patch-based) and `final_visualiser.py` (whole image) hard-code the config, checkpoint, image path, prompt and output directory in `main()`. Edit those values, then run:

```bash
python patch_visualise.py
```

`PATCH_SIZE`, `OVERLAP` and `CONF_THRESH` are constants at the top of the file.

---

## Hardware notes

- Target GPU: **NVIDIA RTX 2080 Ti (11 GB)**. The memory figures in the results table come from the paper.
- The FSC-147 config (`cfg_fsc147_vit_b.py`) keeps upstream's gradient checkpointing (`use_checkpoint`, `use_transformer_ckpt`) and a batch size of 4. With patching, each image becomes a patch batch of size N. An 800 × 1200 input gives 4 × 6 = 24 patches.
- `--amp` turns on mixed precision and cuts memory further.
- SAM ViT-H test-time normalisation (`--sam_tt_norm`) loads a second large model during evaluation. Leave it out if memory is tight.

---

## Credit and license

This is a modified copy of **CountGD** by Niki Amini-Naieni, Tengda Han and Andrew Zisserman (Visual Geometry Group, University of Oxford): <https://github.com/niki-amini-naieni/CountGD>.

- CountGD is released under the **MIT License**, copyright (c) 2024 Niki Amini-Naieni. The original license is kept unchanged in [`CountGD/LICENSE`](CountGD/LICENSE) and applies to the CountGD code in this repository.
- Some files come from GroundingDINO, DINO, DETR, MDETR and related projects. They keep their original copyright headers, several of which are Apache 2.0.
- The FSC-147 dataset, the GroundingDINO, BERT and SAM weights, and the CountGD checkpoints are **not** included. They belong to their authors and keep their own licenses.

```bibtex
@InProceedings{AminiNaieni24,
  author    = "Amini-Naieni, N. and Han, T. and Zisserman, A.",
  title     = "CountGD: Multi-Modal Open-World Counting",
  booktitle = "Advances in Neural Information Processing Systems (NeurIPS)",
  year      = "2024",
}
```

**References**

- CountGD: N. Amini-Naieni, T. Han, A. Zisserman. *CountGD: Multi-Modal Open-World Counting.* NeurIPS 2024.
- FSC-147: V. Ranjan, U. Sharma, T. Nguyen, M. Hoai. *Learning To Count Everything.* CVPR 2021.
- PatchGD: D. K. Gupta et al. *Patch Gradient Descent: Training Neural Networks on Very Large Images.* arXiv:2301.13817.

This is a B.Tech project (BTP) at **IIIT-Delhi**. The accompanying paper, *"Fractional Batch Training and Active Memory Governance for Ultra-High-Resolution Open-World Object Counting"*, is in preparation.
