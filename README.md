# Navigation Agent (Behavioral Clone) for a 2D Maze Environment

## Objective
To develop and enhance a navigation agent capable of maneuvering through a 2D maze to reach a goal location while avoiding obstacles, using both classical machine learning and neural network techniques.

[(https://img.youtube.com/vi/7HpZpNuYIjc/0.jpg)](https://www.youtube.com/shorts/7HpZpNuYIjc)

## Project Structure
- **src/**: Contains source code for classical machine learning and neural network models.
  - **classical_ml/**: Contains Python files for the classical machine learning models.
    - `position_regressor.py`: Implementation of the `PositionRegressor` class.
    - `posbc_robot.py`: Implementation of the `POSBCRobot` class for low-dimensional data.
    - `rgbbc_robot.py`: Implementation of the `RGBBCRobot` class for RGB images.
  - **neural_networks/**: Contains Python files for the neural network models.
    - `dnn_navigation_agent.py`: Implementation of the DNN model for positional data.
    - `cnn_navigation_agent_single_map.py`: Implementation of the CNN model for single map images.
    - `cnn_navigation_agent_multi_map.py`: Implementation of the CNN model for multiple map images.

## Setup Instructions
1. Clone the repository:
   ```sh
   git clone https://github.com/dimitris-anastasiou/behavioral-clone.git
    ```
2. Navigate to the project direcotyr:
   ```sh
   cd behavioral-clone
   ```
