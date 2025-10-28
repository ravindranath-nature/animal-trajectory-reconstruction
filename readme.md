# Animal Trajectory Reconstruction


## 🔧 Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment tool (recommended: venv or conda)

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```

### Step 2: Create a Virtual Environment (Recommended)

#### Using venv:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Using conda:
```bash
# Create conda environment
conda create -n yourproject python=3.8
conda activate yourproject
```

### Step 3: Install Required Packages

Install each package individually:

```bash
# Install NumPy - Fundamental package for scientific computing
pip install numpy

# Install pandas - Data manipulation and analysis library
pip install pandas

# Install TensorFlow - Deep learning framework
pip install tensorflow

# Install seaborn - Statistical data visualization
pip install seaborn

# Install scikit-learn - Machine learning library
pip install scikit-learn
```

#### Option 3: Install all at once
```bash
pip install numpy pandas tensorflow seaborn scikit-learn
```

### Verify Installation

You can verify that all packages are installed correctly by running:

```python
python -c "import numpy, pandas, tensorflow, seaborn, sklearn; print('All packages installed successfully!')"
```

### Input data

standardized_gps_data.csv is provided with this repository. Please add your data in this format.

run model_v3.py

```bash
python model_v3.py
```