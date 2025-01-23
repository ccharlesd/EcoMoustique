# EcoMoustique - Insect Recognition CNN

## Project Overview
Convolutional Neural Network (CNN) for binary classification of insect images, focusing on identifying tiger mosquitoes.

## Prerequisites

### System Requirements
- Python 3.8+
- Git
- pip or conda

### Installation

#### Clone the Repository
```bash
git clone https://github.com/yourusername/EcoMoustique.git
cd EcoMoustique
```

#### Create Virtual Environment
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Or using conda
conda create -n ecomoustique python=3.8
conda activate ecomoustique
```

#### Install Dependencies
```bash
# Using pip
pip install -r requirements.txt

# Or manual installation
pip install numpy tensorflow keras matplotlib scikit-learn
```

## Dataset Preparation
1. Ensure `data.zip` is in the project root
2. The notebook will automatically extract the dataset

## Model Training
Run the Jupyter notebook:
```bash
jupyter notebook cnn_develop_load.ipynb
```

## Model Architecture
- Input: 128x128x3 color images
- Layers:
  - 3 Convolutional layers
  - Max pooling
  - Dropout for regularization
- Binary classification (tiger mosquito vs. other)

## Outputs
- Trained model: `insect_recognition_cnn_model.h5`
- Training visualization graphs
- Test prediction samples

## Troubleshooting
- Ensure all dependencies are installed
- Check data integrity in `data.zip`
- Verify Python and library versions

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License
[Add your license information]