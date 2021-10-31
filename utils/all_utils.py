import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib.colors import ListedColormap
import os 
import joblib
import logging


def prepare_data(df):
  """It takes in a dataset and returns the predictors and target values

  Args:
      df (pd.dataFrame): It is a dataset in the form of panadas DataFrame 

  Returns:
      tuple: Returns a tuple of predictors and target 
  """
  logging.info("preparing data by seperating dependent and independent variables")
  X = df.iloc[:, :-1]
  y = df.iloc[:, -1]
  return X, y


def save_model(model, filename):
  """It saves the trained model to a file with the given path.

  Args:
      model (python object): It is a trained algorithm which can predict values based on inputs, weights
      filename (string): Using this, a path is created where the model is saved to
  """   
  logging.info("Saving the trained model")
  model_dir = "models"
  os.makedirs(model_dir, exist_ok=True)
  filepath = os.path.join(model_dir, filename)
  joblib.dump(model, filepath)
  logging.info(f"saving the model at: {filepath}")


def predict_model(model, X):
  """It predicts the output based on given inputs 

  Args:
      model (python object): It is a trained algorithm which can predict values based on inputs, weights
      X (array): predictors

  Returns:
      array: Returns an array of output values
  """
  logging.info("Predicting the output")
  return model.predict(X)


def save_plot(df, file_name, model):
  """Saves the created plot

  Args:
      df (pd.dataFrame): It is a dataset in the form of panadas DataFrame 
      file_name (string): Using this, a path is created where the plot is saved to
      model (python object): It is a trained algorithm which can predict values based on inputs, weights
  """
  def _create_base_plot(df):
    logging.info("Creating a base plot")
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    figure = plt.gcf()
    figure.set_size_inches(10, 8)

  def _plot_decision_regions(X, y, classfier, resolution=0.02):
    logging.info("Plotting the decision regions")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    X = X.values # as a array
    x1 = X[:, 0] 
    x2 = X[:, 1]
    x1_min, x1_max = x1.min() -1 , x1.max() + 1
    x2_min, x2_max = x2.min() -1 , x2.max() + 1  

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))

    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()



  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)
  logging.info(f"saving the plot at: {plotPath}")
