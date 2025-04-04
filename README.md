# Project Overview

This project implements a simple neural network using PyTorch to model the relationship between `y` and `x` defined by the equation:

\[ y = \sin(3x) + \exp(\cos(x) + 0.1) \]

The project includes data generation, model definition, training, and visualization components.

## Project Structure

```
pytorch-simple-nn
├── src
│   ├── data
│   │   └── dataset.py       # Contains Dataset class for generating data
│   ├── models
│   │   └── network.py       # Defines the SimpleNN class
│   ├── train.py             # Training script for the neural network
│   ├── utils
│   │   └── visualization.py  # Visualization functions for training
│   └── config.py            # Configuration settings
├── notebooks
│   └── exploration.ipynb     # Jupyter notebook for data exploration
├── requirements.txt          # Project dependencies
├── .gitignore                # Git ignore file
└── README.md                 # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd pytorch-simple-nn
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```
   python src/train.py
   ```

## Usage

- Use the `exploration.ipynb` notebook for exploratory data analysis.
- The training process will log metrics to Weights & Biases and TensorBoard for visualization.

## License

This project is licensed under the MIT License.