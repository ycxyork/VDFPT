# TAP-Vid Feature Tracking Demo (constrcuting)

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

<table>
  <tr>
    <td align="center" width="33%">
      <video src="assets/1.mp4" controls width="100%"></video>
      <br><sub>Demo 01</sub>
    </td>
    <td align="center" width="33%">
      <video src="assets/2.mp4" controls width="100%"></video>
      <br><sub>Demo 02</sub>
    </td>
    <td align="center" width="33%">
      <video src="assets/3.mp4" controls width="100%"></video>
      <br><sub>Demo 03</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <video src="assets/4.mp4" controls width="100%"></video>
      <br><sub>Demo 04</sub>
    </td>
    <td align="center" width="33%">
      <video src="assets/5.mp4" controls width="100%"></video>
      <br><sub>Demo 05</sub>
    </td>
    <td align="center" width="33%">
      <video src="assets/6.mp4" controls width="100%"></video>
      <br><sub>Demo 06</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <video src="assets/7.mp4" controls width="100%"></video>
      <br><sub>Demo 07</sub>
    </td>
    <td align="center" width="33%">
      <video src="assets/8.mp4" controls width="100%"></video>
      <br><sub>Demo 08</sub>
    </td>
    <td align="center" width="33%">
      <video src="assets/9.mp4" controls width="100%"></video>
      <br><sub>Demo 09</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <video src="assets/10.mp4" controls width="100%"></video>
      <br><sub>Demo 10</sub>
    </td>
    <td align="center" width="33%">
      <video src="assets/11.mp4" controls width="100%"></video>
      <br><sub>Demo 11</sub>
    </td>
    <td align="center" width="33%">
      <video src="assets/12.mp4" controls width="100%"></video>
      <br><sub>Demo 12</sub>
    </td>
  </tr>
</table>
