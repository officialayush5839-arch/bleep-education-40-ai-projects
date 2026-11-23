# Fashion-MNIST Dataset Specification (DL-09)

## 1. Overview & Data Provenance
Fashion-MNIST is an enterprise-standard benchmark dataset developed by Zalando Research to serve as a direct drop-in replacement for the original MNIST digit dataset. It consists of 70,000 $28 \times 28$ grayscale images across 10 distinct apparel product classes:
- Training Split: 60,000 examples
- Test Split: 10,000 examples

## 2. Apparel Class Dictionary
The dataset maps integer labels $0 \dots 9$ to specific clothing items:

| Class ID | Garment Category | Key Morphological / Silhouette Markers |
| :---: | :--- | :--- |
| `0` | **T-shirt/top** | Short sleeves, crew or v-neck collar, boxy or tapered torso profile |
| `1` | **Trouser** | Two elongated parallel vertical leg columns with waistband |
| `2` | **Pullover** | Long sleeves, ribbed cuffs and bottom hem, closed neck |
| `3` | **Dress** | Continuous single-piece garment with flared skirt profile |
| `4` | **Coat** | Long outerwear profile, open front lapels/slit, thicker silhouette |
| `5` | **Sandal** | Open-foot footwear with thin horizontal and ankle straps |
| `6` | **Shirt** | Structured collar, vertical front button placket, long or folded sleeves |
| `7` | **Sneaker** | Flat cushioned athletic sole, curved toe bumper, low heel collar |
| `8` | **Bag** | Rectangular or trapezoidal carryall body with top handles |
| `9` | **Ankle boot** | Elevated high-collar ankle shaft, thick heel/sole structure |

## 3. Data Schema & Pixel Properties
- **Image Dimensions**: $28 \times 28$ pixels, single-channel grayscale (`uint8` in $[0, 255]$).
- **Normalized Representation**: Scaled to float32 $[0.0, 1.0]$, then standardized using canonical Fashion-MNIST statistics:
  - $\mu = 0.2860$
  - $\sigma = 0.3530$
- **Channel Format**: PyTorch tensor shape `(Batch_Size, 1, 28, 28)`.

## 4. Apparel Tops Sub-Cluster Ambiguity
The primary technical challenge in Fashion-MNIST involves Class 0 (`T-shirt/top`), Class 2 (`Pullover`), Class 4 (`Coat`), and Class 6 (`Shirt`). At $28 \times 28$ pixel resolution, low-contrast button plackets and cuff seams create an inherent ambiguity boundary where subtle sleeve and neckline differences distinguish garments.

## 5. Offline Deterministic Synthetic Generator
To ensure 100% hermetic offline test reproducibility and rapid verification without network connectivity:
- Generator function: `src.data_loader.generate_synthetic_fashion_dataset(n_samples, seed)`
- Generates structurally distinct silhouettes with class-specific geometric patterns (neck cutouts, sleeve blocks, leg splits, and sole contours) combined with bounded uniform noise.
