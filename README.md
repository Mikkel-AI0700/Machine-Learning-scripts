# Machine-Learning-scripts

This repository is a personal collection of **custom automation scripts** designed to simplify and centralize the **machine learning pipeline**. From data preprocessing to model tuning and visualization, this toolkit helps you focus more on building better models by reducing repetitive coding.

## 📌 Purpose

The main goal of this repository is to give you a **copy-paste ready toolbox** of Python scripts that handle common machine learning tasks. It’s especially helpful when you want to:

* Avoid rewriting boilerplate code
* Keep all your ML utilities in one place
* Quickly test and deploy models using clean, modular code

## 📁 Directory Overview

Here's a breakdown of each directory and what it does:

### 1. `data-preprocessing-scripts/`

Scripts that automate key preprocessing steps such as:

* **Encoding** categorical features
* **Scaling** numerical columns
* **Handling missing values** (nulls)
* **Removing outliers**

All scripts expect a **dictionary of parameters** for flexibility and ease of use. Just update the dictionary when you want to tweak behavior.

---

### 2. `dependency_importer/`

A script that allows **dynamic importing** of any Python libraries you need for your ML workflow. Simply pass a list of module names, and it imports them automatically—great for flexible, plug-and-play workflows.

---

### 3. `dataset-loader/`

Automates the loading of datasets:

* **From local files** (for local or Google Colab environments)
* **From the UCI Machine Learning Repository**

Regardless of the source, the loader returns a **dictionary** with:

* A `pandas.DataFrame` (main dataset)
* A copy of the main dataset
* A `NumPy` version of the dataset

---

### 4. `hyperparameter-tuning-scripts/`

Provides scripts that use:

* `GridSearchCV`
* `RandomizedSearchCV`

But with an upgrade!
These scripts **return all attributes** of the search object (e.g. best estimator, scores, CV results) **plus predictions**, so you don’t miss anything useful.

---

### 5. `plot-visualizations/`

Scripts for:

* **Dataset visualization**: relational, distributional, and categorical plots
* **Model visualization** using Scikit-Learn’s built-in tools

Everything is automated—just plug in your data and go.

---

## 🚀 How to Use

Every script is designed to work with **dictionaries**.

Example usage:

```python
params = {
    'data': your_dataframe,
    'columns': ['feature1', 'feature2'],
    'strategy': 'mean',
    # other configuration...
}

# Example transform function
transform(**params)
```

**Why use dictionaries?**
So you only need to change values in one place. It makes the code more readable, flexible, and user-friendly.

---

## 💠 Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Seaborn / Matplotlib** (for plotting)

---

## 💡 Contribution

This is a personal utility repo, but feel free to fork or suggest improvements via pull requests or issues.

---

## 📜 License

This project is open-source. You can use it freely under the terms of the **MIT License**.

---

## 🧠 Summary

* A collection of reusable scripts for common ML tasks
* Focused on automation, modularity, and ease of use
* Powered by dictionaries for clean parameter handling

This is your **go-to toolbox** for rapidly building, tuning, and visualizing machine learning models.
