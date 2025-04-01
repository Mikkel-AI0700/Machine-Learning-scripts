# Machine Learning Workflow Automation

## Overview
This repository automates various aspects of the machine learning workflow, making model development more efficient and streamlined. The automation script is designed to handle multiple preprocessing steps, hyperparameter tuning, and dataset management dynamically. This allows machine learning practitioners to focus more on model experimentation rather than repetitive tasks.

## Features
The script provides the following automation functionalities:

- **Dynamic Module Importing**: Automatically imports only the required modules based on the current project's needs.
- **Dataset Management**:
  - Loads datasets automatically.
  - Creates three dataset versions:
    - Main pandas DataFrame.
    - A copy of the main dataset.
    - A NumPy representation for numerical operations.
- **Hyperparameter Tuning**:
  - Automates **GridSearchCV** and **RandomizedSearchCV** for hyperparameter optimization.
  - Returns all attributes from the hyperparameter search results.
- **Preprocessing Automation**:
  - **Scaling and Encoding**:
    - Organizes multiple scaling and encoding techniques into their respective classes.
    - Automatically applies the appropriate technique based on the dataset’s characteristics.
- **Exploratory Data Analysis (EDA)**: Performs key statistical and graphical analyses to understand dataset characteristics.
- **Model Visualization**: Uses Scikit-Learn’s visualization API to provide graphical representations of machine learning models.

## Branches
This repository contains four branches:

- **master**: The stable and production-ready branch.
- **test-branch**: Used for testing new features before merging into `master`.
- **version_1.0**: An earlier version of the automation script.
- **version_2.0**: The latest iteration with improvements and optimizations.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/repo-name.git
   ```
2. Navigate into the directory:
   ```bash
   cd repo-name
   ```
3. Checkout the desired branch:
   ```bash
   git checkout branch-name
   ```
4. Run the automation script:
   ```bash
   python script.py
   ```

## Contributing
Feel free to fork this repository and make improvements. Contributions are always welcome! If you have a feature request or encounter any issues, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

---

Happy coding! 🚀

