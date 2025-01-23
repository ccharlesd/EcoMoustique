# EcoMoustique - Insect Recognition CNN

## Project Overview
Convolutional Neural Network (CNN) for binary classification of insect images, focusing on identifying tiger mosquitoes.

## Prerequisites

### System Requirements
- Python 3.8+
- Git
- pip or conda
- [Git LFS](https://git-lfs.com)
  
  Download and install the Git command line extension. Once downloaded and installed, set up Git LFS for your user account by running:
  ```bash
  git lfs install
  ```
  You only need to run this once per user account.

### Installation

#### Clone the [Repository](https://github.com/ccharlesd/EcoMoustique/tree/main)
```bash
git clone https://github.com/ccharlesd/EcoMoustique.git
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

## Testing the Model

### Automated Tests
The project includes automated tests to validate the model and its predictions:

1. **Test Model Loading**: Verifies that the trained model (`insect_recognition_cnn_model.h5`) can be successfully loaded.
2. **Test Model Predictions**: Confirms that the model can generate predictions for sample data, while visualizing the results.

#### Run the Tests
```bash
python test_model.py
```

#### Visualize Predictions
The `test_model_prediction` function displays a sample of test images along with their true and predicted labels:
- True labels are mapped to readable names: `'Tiger'` for tiger mosquitoes and `'Misc'` for other insects.
- Predictions are displayed alongside the true labels for comparison.

Sample output (image grid):
```
True: Tiger
Pred: Misc
```

Make sure `X_test.npy` and `y_test.npy` files are present in the project root to run the tests.

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
Nothing for the moment

