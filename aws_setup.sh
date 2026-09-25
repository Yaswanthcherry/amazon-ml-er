#!/bin/bash
# =============================================================
# AWS EC2 Setup Script for Amazon ML Entity Resolution Pipeline
# =============================================================
# Run this on a fresh EC2 instance (Ubuntu 22.04 recommended)
# Instance type recommendation: r6i.4xlarge (128GB RAM, 16 vCPUs)
# Storage: at least 100 GB EBS volume
#
# Usage:
#   chmod +x aws_setup.sh
#   ./aws_setup.sh
# =============================================================

set -e  # exit on any error

echo "======================================================"
echo " Step 1: System update & Python install"
echo "======================================================"
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git wget unzip tmux htop

echo "======================================================"
echo " Step 2: Clone the repository"
echo "======================================================"
# Replace with your actual GitHub repo URL
REPO_URL="https://github.com/YOUR_USERNAME/amazon-ml-er.git"

if [ -d "amazon-ml-er" ]; then
    echo "Repo already cloned, pulling latest..."
    cd amazon-ml-er && git pull && cd ..
else
    git clone "$REPO_URL"
fi

cd amazon-ml-er

echo "======================================================"
echo " Step 3: Create virtual environment & install deps"
echo "======================================================"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "======================================================"
echo " Step 4: Create data directories"
echo "======================================================"
mkdir -p data/train data/test models outputs

echo "======================================================"
echo " Step 5: Upload data files"
echo "======================================================"
echo ""
echo ">>> You need to upload your TSV data files to this EC2 instance."
echo "    Choose ONE of the options below:"
echo ""
echo "  OPTION A — Copy from your local machine (run on your LOCAL machine):"
echo "    scp -i your-key.pem data/train/*.tsv ubuntu@<EC2-IP>:~/amazon-ml-er/data/train/"
echo "    scp -i your-key.pem data/test/*.tsv  ubuntu@<EC2-IP>:~/amazon-ml-er/data/test/"
echo ""
echo "  OPTION B — Download from S3 (if you uploaded data to S3 first):"
echo "    aws s3 cp s3://your-bucket/data/train/ data/train/ --recursive"
echo "    aws s3 cp s3://your-bucket/data/test/  data/test/  --recursive"
echo ""
echo "  >>> After uploading data, run the pipeline with:"
echo "      source .venv/bin/activate"
echo "      tmux new -s pipeline        # run inside tmux so it survives disconnection"
echo "      python run.py"
echo ""

echo "Setup complete!"
