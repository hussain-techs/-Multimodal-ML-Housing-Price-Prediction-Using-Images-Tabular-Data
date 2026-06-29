# Task 3: Multimodal ML — Housing Price Prediction Using Images + Tabular Data

Predicts house sale price by fusing **structured tabular features** (sqft, bedrooms,
location, condition, etc.) with **visual features extracted by a CNN** from house
photos.

## ⚠️ About the dataset

No dataset or images were uploaded, and this environment has no access to Kaggle or
similar dataset hosts (network access is restricted to package registries). So this
project **generates its own realistic synthetic dataset**:

- `src/01_generate_tabular_data.py` — 1,200 houses with a realistic generative price
  model (size, rooms, lot, age, renovation, garage, school rating, waterfront,
  neighborhood premium, noise) — structurally the same kind of schema as the
  Kaggle "House Sales in King County" dataset.
- `src/02_generate_images.py` — procedurally renders a 128×128 "photo" for every
  house. Critically, the images are **not random** — visual properties (lawn health,
  wall freshness/weathering, footprint size, water view, number of stories, disrepair
  cracks) are driven by the *same* underlying features that drive price
  (`condition_score`, `sqft_living`, `waterfront`, `age`, `floors`), plus per-image
  noise. This gives the CNN genuine, learnable visual signal, the same way real
  listing photos correlate with home value.

**To use your own data:** replace `data/housing_tabular.csv` (keep the `house_id`
column) and drop matching files into `data/images/house_<house_id>.jpg`. Update
`FEATURE_COLS` in `03_multimodal_pipeline.py` if your schema differs.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `01_generate_tabular_data.py` | Synthesizes tabular housing data |
| 2 | `02_generate_images.py` | Synthesizes matching house images |
| 3 | `03_multimodal_pipeline.py` | Trains the **CNN + tabular fusion model**, evaluates MAE/RMSE/R² |
| 4 | `04_baseline_comparisons.py` | Trains **image-only** and **tabular-only** ablations for comparison |

Run in order from `housing_multimodal/`:
```bash
python3 src/01_generate_tabular_data.py
python3 src/02_generate_images.py
python3 src/03_multimodal_pipeline.py
python3 src/04_baseline_comparisons.py
```

## Methodology

**Preprocessing**
- Numeric features standardized (`StandardScaler`, fit on train only); categorical
  `neighborhood` one-hot encoded.
- Target (`price`) log-transformed (`log1p`) then standardized for stable training;
  predictions are inverted back to dollars before computing metrics.
- Images resized to 64×64, normalized to `[-1, 1]`; random horizontal flip on train only.
- Split: 70% train / 15% val / 15% test (840 / 180 / 180 houses), fixed seed.

**Architecture (`FusionModel`)**
- **Image branch (CNN):** 4 conv blocks (16→32→64→128 channels, BatchNorm, ReLU,
  MaxPool) → global average pool → FC → 64-dim image embedding.
- **Tabular branch (MLP):** 2 fully-connected layers (64 → 32) with dropout → 32-dim
  tabular embedding.
- **Fusion head:** concatenate the two embeddings (96-dim) → FC(64) → FC(16) → FC(1)
  regression output (standardized log-price).
- ~116K parameters total. Trained with Adam (lr 5e-4), MSE loss, gradient clipping,
  `ReduceLROnPlateau`, and early stopping on validation MAE (patience 10).

**Evaluation metrics:** MAE and RMSE (in dollars, after inverting log/standardization)
and R² for context, computed on a held-out test set never seen during training.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Image-only (CNN) | $67,024 | $87,609 | 0.774 |
| **Tabular-only (MLP)** | **$26,190** | **$36,313** | **0.961** |
| **Multimodal (Fusion)** | $29,342 | $40,924 | 0.951 |

(Mean test price ≈ $423,000, so the multimodal model's MAE is ≈6.9% of mean price.)

![Model comparison](outputs/model_comparison.png)
![Training curve and predictions](outputs/training_and_predictions.png)

### Honest discussion of the result

The multimodal fusion model performs strongly (R² = 0.95) and far outperforms the
image-only model, confirming the CNN is learning real visual signal (it recovers most
of the price variance from pixels alone, R² = 0.77). However, **the tabular-only model
edges out the fusion model** on this dataset. This is an expected and informative
artifact of how the *synthetic* data was constructed: the images were procedurally
*derived from* a subset of the tabular features (`condition_score`, `sqft_living`,
`waterfront`, `age`) plus rendering noise — so the image branch contributes mostly
redundant, noisier versions of information the tabular branch already has in exact
numeric form. Adding the CNN branch adds parameters/variance without adding new
information, so it slightly hurts generalization on a dataset this size (1,200 rows).

**In a real-world dataset**, this typically reverses: listing photos carry information
that is *not* captured in the structured fields at all — interior finish quality,
staging, natural light, curb appeal, view quality, renovation details not logged in
public records. In that setting the image branch contributes genuinely novel signal
and fusion reliably beats tabular-only. This project's ablation setup
(`04_baseline_comparisons.py`) is exactly the tool you'd use to verify that on real
data: if fusion doesn't beat tabular-only there either, it's a sign the photos aren't
adding information beyond what's already tabulated — itself a useful finding.

## Files produced

```
outputs/
├── fusion_model.pt                  # trained multimodal model weights
├── metrics_multimodal.json          # test MAE/RMSE/R2 for the fusion model
├── model_comparison.csv             # MAE/RMSE/R2 for all 3 models
├── model_comparison.png             # bar chart comparison
├── training_and_predictions.png     # loss curves + predicted vs actual scatter
├── feature_cols.json                # tabular feature column order (for inference)
├── scaler_mean.npy / scaler_scale.npy  # StandardScaler params (for inference)
```

## Skills demonstrated

- **CNNs** for image feature extraction (custom conv architecture, no pretrained
  weights needed given the small 64×64 input domain)
- **Feature fusion**: late fusion via embedding concatenation from two branches
- **Multimodal regression**: joint end-to-end training of both branches against a
  single target
- **Evaluation**: MAE, RMSE, R², train/val/test methodology, early stopping, and a
  proper ablation study (image-only vs. tabular-only vs. fused) to quantify what each
  modality actually contributes
