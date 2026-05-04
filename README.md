# TAP-Vid Feature Tracking Demo (constructing)

This repository contains a lightweight pipeline for extracting SVD features and evaluating point tracking on TAP-Vid style datasets. The main scripts cover feature extraction, trajectory evaluation, and visualization output.

## Project Structure

- `extract_features.py`: extract features and query points from TAP-Vid datasets.
- `evaluate.py`: evaluate extracted features and generate tracking visualizations.
- `feature_svd.py`: SVD feature utilities.
- `data_utils.py`: dataset loading and TAP-Vid metric helpers.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Extract features:

```bash
python extract_features.py \
  --data_path data/tapvid_davis/tapvid_davis.pkl \
  --output_dir output/davis
```

Evaluate and visualize:

```bash
python evaluate.py \
  --feat_dir output/davis \
  --output_dir result/vis
```

## Demo Gallery

The examples below are from DAVIS. Videos are shown in slow motion for better visualization.

### Demo 01

https://github.com/user-attachments/assets/8fcba820-6041-484a-bf5a-16a9c0432ff4

### Demo 02

https://github.com/user-attachments/assets/689b6c68-06d6-4bb3-9790-3a500a255544

### Demo 03

https://github.com/user-attachments/assets/a0fe0a5c-a280-45d3-ba60-76a2566ebebd

### Demo 04

https://github.com/user-attachments/assets/be9e80e9-62de-4066-997b-017b10a45733

### Demo 05

https://github.com/user-attachments/assets/afc58057-8c14-4cec-9243-b2d7afe1ff61

### Demo 06

https://github.com/user-attachments/assets/3272adc0-0018-4c51-90c6-0cbb1b9e40bf

### Demo 07

https://github.com/user-attachments/assets/f3f4e5d2-3c72-4ba0-9532-4f132676a3d0

### Demo 08

https://github.com/user-attachments/assets/dd934777-213c-4e90-8185-6e1f259aaf46

### Demo 09

https://github.com/user-attachments/assets/efa871b5-ad2c-4a35-951b-43252d9d7bf1

### Demo 10

https://github.com/user-attachments/assets/63cea561-0ae2-4076-b097-cb4eb670ee8d

### Demo 11

https://github.com/user-attachments/assets/a935a914-c18c-4510-92b2-b2ba74fe1e88

### Demo 12

https://github.com/user-attachments/assets/1b46dfe1-6151-43ba-849f-f4b05b3f999d
