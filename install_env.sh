
ENV_NAME="checkthis"
PYTHON_VERSION="3.10"

install_miniconda() {
    echo "Conda not found. Installing Miniconda..."
    OS_TYPE=$(uname)
    if [[ "$OS_TYPE" == "Linux" ]]; then
        MINICONDA_SCRIPT="Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        MINICONDA_SCRIPT="Miniconda3-latest-MacOSX-x86_64.sh"
    else
        echo " Unsupported OS: $OS_TYPE"
        exit 1
    fi

    curl -LO "https://repo.anaconda.com/miniconda/$MINICONDA_SCRIPT"
    bash "$MINICONDA_SCRIPT" -b -p "$HOME/miniconda"
    rm "$MINICONDA_SCRIPT"

    export PATH="$HOME/miniconda/bin:$PATH"
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    conda init
    echo "Miniconda installed."
}

if ! command -v conda &> /dev/null; then
    install_miniconda
else
    echo "Conda is already installed."
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "Activating environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "Installing pip packages from requirements.txt..."
pip install -r requirements.txt

conda install -c pytorch faiss-gpu -y

echo "Setup complete! Dropping into environment '$ENV_NAME'..."
exec bash --rcfile <(echo "source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_NAME")

