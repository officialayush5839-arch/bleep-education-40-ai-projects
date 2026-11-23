# Dataset Documentation: Garbage Classification (DL-01)

## Overview & Provenance
- **Dataset Title**: Garbage Classification Dataset
- **Domain**: Environmental Technology / Municipal Solid Waste Sorting
- **Target Variable**: Categorical waste material class (6 discrete categories)
- **Primary Source**: Open-source garbage classification corpus (Gary Thung, Mindy Yang; Stanford CS229)
- **Image Resolution**: Original images standard RGB color space, normalized and standardized to $224 \times 224$ pixels.

## Class Schema & Data Dictionary

| Class Index | Class Name | Description | Disposal / Recycling Destination |
|:---:|:---:|:---|:---|
| 0 | `cardboard` | Corrugated boxes, packaging cartons, paperboard | Blue Recycling Bin (Dry, Flattened) |
| 1 | `glass` | Clear, brown, green glass beverage bottles, jars | Green/Amber Glass Bin (Rinsed, No caps) |
| 2 | `metal` | Tin cans, aluminum soda cans, foil, aerosols | Blue/Yellow Recycling Bin (Cleaned) |
| 3 | `paper` | Office sheets, newspapers, magazines, circulars | Blue Recycling Bin (Dry, Unsoiled) |
| 4 | `plastic` | Beverage bottles (PET), food tubs (HDPE/PP) | Yellow Recycling Bin (Rinsed, Check code) |
| 5 | `trash` | Non-recyclable mixed materials, composite plastics, soiled goods | Landfill / General Waste Bin |

## Dataset Structure & Layout
The dataset is structured following the standard PyTorch `ImageFolder` convention:
```
data/
├── README.md
└── raw/
    ├── cardboard/
    │   ├── img_0001.png
    │   └── ...
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

## Synthetic Generator Fallback
To enable 100% offline, hermetic, and automated testing without requiring network access to download external multi-gigabyte archives:
- The data loader module (`src/data_loader.py`) implements a deterministic synthetic generator.
- Generates structurally authentic synthetic RGB images with class-separable textures, colors, and geometries.
- Deterministic random seed control ensures reproducible evaluation across CPU environments.
