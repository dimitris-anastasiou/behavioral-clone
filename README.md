# Navigation Agent for a 2D Maze Environment

## Objective
To develop and enhance a navigation agent capable of maneuvering through a 2D maze to reach a goal location while avoiding obstacles, using both classical machine learning and neural network techniques.

## Project Structure
- **data/**: Contains training and testing data, as well as maze images.
  - **images/**: Contains the maze images used in the project.
    - `maze1.png`
    - `maze2.png`
    - `maze3.png`
  - **observations/**: Contains the training and testing data in CSV format.
    - `train_data.csv`
    - `test_data.csv`
- **src/**: Contains source code for classical machine learning and neural network models.
  - **classical_ml/**: Contains Python files for the classical machine learning models.
    - `position_regressor.py`: Implementation of the `PositionRegressor` class.
    - `posbc_robot.py`: Implementation of the `POSBCRobot` class for low-dimensional data.
    - `rgbbc_robot.py`: Implementation of the `RGBBCRobot` class for RGB images.
  - **neural_networks/**: Contains Python files for the neural network models.
    - `dnn_position_predictor.py`: Implementation of the DNN model for positional data.
    - `cnn_navigation_agent_single_map.py`: Implementation of the CNN model for single map images.
    - `cnn_navigation_agent_multi_map.py`: Implementation of the CNN model for multiple map images.
  - **utils/**: Contains utility scripts for data preprocessing.
    - `data_preprocessing.py`: Functions for data cleaning and preparation.
- **results/**: Contains evaluation videos and performance metrics plots.
  - **evaluation_videos/**: Contains videos of the evaluation results.
    - `classical_ml_evaluation.mp4`: Evaluation video for the classical ML models.
    - `nn_evaluation_part1.mp4`: Evaluation video for the DNN model.
    - `nn_evaluation_part2.mp4`: Evaluation video for the CNN model with a single map.
    - `nn_evaluation_part3.mp4`: Evaluation video for the CNN model with multiple maps.
  - **plots/**: Contains performance metrics plots and other visual results.
    - `performance_metrics.png`: Performance metrics of different models.
- **notebooks/**: Contains Jupyter notebooks for running and analyzing the models.
  - `classical_ml_notebook.ipynb`: Jupyter notebook for the classical machine learning models.
  - `neural_networks_notebook.ipynb`: Jupyter notebook for the neural network models.

## Setup Instructions
1. Clone the repository:
   ```sh
   git clone https://github.com/username/navigation-agent.git

2. Navigate to the project direcotyr:
   ```sh
   cd behavioral-clone
