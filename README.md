# Wafer Map Defect Classification Using CDADA

This repository contains the implementation of the **Classifier-Constrained Deep Adversarial Domain Adaptation (CDADA)** framework, designed for wafer map defect classification in semiconductor manufacturing. The framework addresses domain shifts and class imbalances to classify complex mixed-defect patterns effectively.

---

## Features
- **Domain Adaptation:** Aligns features from the source and target domains using adversarial learning.
  
---

## Data Setup

1. Download the dataset files from the following Google Drive link:  
   [Google Drive - Wafer Map Data](https://drive.google.com/drive/folders/1bHZ-v63jMAc35fXZQhebl_M58xHvO6m9?usp=sharing)

2. Place the downloaded files into a folder named `data` in the root directory of this repository. Your project directory should look like this:
   ```
   .
   ├── data
   │   ├── <your-data-files>
   ├── Robust_wafer_detection.ipynb
   ├── requirements.txt
   ├── README.md
   ├── .gitignore
   ```

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/wafer-map-net.git
   cd wafer-map-net
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have Jupyter Notebook installed:
   ```bash
   pip install notebook
   ```

---

## Running the Framework

1. Open the `Robust_wafer_detection.ipynb` file in Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook Robust_wafer_detection.ipynb
   ```

2. Run the cells sequentially to execute the framework. This notebook contains the entire pipeline, including data preprocessing, model training, and evaluation.

---

## Evaluation Results

The CDADA framework was evaluated on the target domain dataset with the following results:
- **Loss:** 0.3
- **Accuracy:** 67%

These metrics demonstrate the framework's capability to handle cross-domain defect classification tasks effectively.

---

## Repository Structure

- `Robust_wafer_detection.ipynb`: The Jupyter Notebook containing the full implementation of the framework.
- `data/`: Directory to store the downloaded datasets (user must add).
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: Documentation for setup and usage instructions.
- `requirements.txt`: Lists the Python dependencies required to run the project.

---

## Requirements

- **Python >= 3.7**
- **Jupyter Notebook or Jupyter Lab**
- Libraries:
  - `numpy`
  - `pandas`
  - `torch`
  - `torchvision`
  - `matplotlib`
  - `opencv-python`
  - `scikit-learn`

Install all dependencies using the provided `requirements.txt` file.

---

## Questions or Support

For any questions or issues, please open an issue on the repository or contact the maintainers.

---
