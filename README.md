# anode-defect-detector
AI-powered inspection system for detecting cracks in GSYuasa anode CT scan images

## 🛠️ Installation

```bash
# Create a new conda environment
conda create -n annode_venv python=3.9 -y
conda activate annode_venv

# Install Python requirements
pip install -r requirements.txt

# Install FAISS with GPU support via Conda
conda install -c pytorch faiss-gpu
