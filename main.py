# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files,
# tool windows, actions, and settings.
# IMPORT
# WEB APP
import streamlit as st
# LOAD DATA
from ase.io import read as read_ase
# CREATE DESCRIPTORS
from dscribe.descriptors import CoulombMatrix, SOAP, ACSF, MBTR
import pandas as pd
# TRAIN TEST SPLIT
import numpy as np
from sklearn.model_selection import train_test_split
# BUILD NN
import tensorflow as tf
# PLOT
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.utils.vis_utils import plot_model
# REGRESSION
from sklearn import linear_model
from scipy import stats
# HYPERPARAMETER TUNING
from tensorboard.plugins.hparams import api as hp
# SAVE & LOAD
import zipfile
import tempfile
import os
# SUPPORT
import time
from typing import List
from ase.atoms import Atoms

@st.experimental_memo
def read_data(xyz_data_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(xyz_data_file.read())
    return read_ase(tfile.name, index=":")


def find_all_atoms(structures: list):
    species = set()
    for structure in structures:
        species.update(structure.get_chemical_symbols())
    return species


def find_max_atoms(structures: list):
    return max([structure.get_global_number_of_atoms()
                for structure in structures])


def get_positions(structures: list):
    return [1] * len(structures)


def get_property_names(structures: list):
    return structures[0].info.keys()


def get_properties(structures: list, property_name: str):
    return [float(struct.info[property_name]) for struct in structures]


#@st.cache(suppress_st_warning=True)
@st.experimental_memo
def make_feature_vectors(_structures: List[Atoms], descriptor_name: str):
    if descriptor_name == "soap":
      soap = SOAP(
                  species=find_all_atoms(_structures),
                  periodic=False,
                  rcut=6,
                  nmax=6,
                  lmax=6,
                  average='outer',
                  sparse=False,
              )
      fv = soap.create(_structures, n_jobs=-1)
    elif descriptor_name ==  "cm":
      cm = CoulombMatrix(
          n_atoms_max=find_max_atoms(_structures)
      )
      fv = cm.create(_structures, n_jobs=-1)
    elif descriptor_name ==  "acsf":
      pos = get_positions(_structures)
      acsf = ACSF(
          species=find_all_atoms(_structures),
          periodic=False,
          rcut=7,
          g2_params=[[1, 1], [1, 2], [1, 3]],
          g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
      )
      fv = acsf.create(_structures, n_jobs=-1, positions=pos).reshape(
          5000, -1)
    elif descriptor_name ==  "mbtr":
      mbtr = MBTR(
          species=find_all_atoms(_structures),
          k1={
              "geometry": {"function": "atomic_number"},
              "grid": {"min": 0, "max": 8, "n": 100, "sigma": 0.1},
          },
          k2={
              "geometry": {"function": "inverse_distance"},
              "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
              "weighting": {"function": "exp", "scale": 0.5,
                            "threshold": 1e-3},
          },
          periodic=False,
          normalization="l2_each",
          flatten=True
      )
      fv = mbtr.create(_structures, n_jobs=-1)
    else:
        raise ValueError("Could not find descriptor")
        
    return fv


def tt_split(features_array: np.ndarray, property_list: list):
    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.10

    x = np.asarray(features_array)
    y = np.asarray(property_list)

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, train_size=train_ratio, random_state=1)
    x_val, x_test, y_val, y_test = \
        train_test_split(x_test, y_test, train_size=val_ratio /
                         (test_ratio + val_ratio),
                         random_state=1)
    return x_train, x_test, x_val, y_train, y_test, y_val


#@st.cache(hash_funcs={tf.keras.Sequential})
def compile_model(loss, metrics):
  # define model structure
  model = tf.keras.models.Sequential([
      #tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          32,
          kernel_initializer = tf.keras.initializers.RandomUniform(seed=None),
          activation='softplus',
          use_bias=True),
      tf.keras.layers.Dense(
          1,
          kernel_initializer = tf.keras.initializers.RandomUniform(seed=None),
          activation=None,
          use_bias=True),
      ]
  )

  opt = tf.keras.optimizers.Nadam(learning_rate = 0.0005)
  model.compile(loss=loss, optimizer=opt, metrics=metrics)

  return model


@st.experimental_singleton
def plot_performance(_history, loss, metrics):
  num_plots = 1 + len(metrics)
  num_epochs = len(history.history["loss"])

  fig = make_subplots(rows=num_plots, cols=1,
                      shared_xaxes=True,
                      vertical_spacing=0.02)
  
  fig.add_trace(go.Scatter(y=history.history["loss"],
                           line=dict(color='blue'),
                           name="train", legendgroup="train",),
                row=1, col=1)

  fig.add_trace(go.Scatter(y=history.history["val_"+"loss"],
                           line=dict(color='red'), 
                           name="validation", legendgroup="validation",),
                row=1, col=1)
  
  fig.update_yaxes(title_text=loss, type="log", row=1, col=1)

  for num, metric in enumerate(metrics):

    fig.add_trace(go.Scatter(y=history.history[metric],
                             line=dict(color='blue'), 
                             legendgroup="train",
                             showlegend=False),
                  row=num+2, col=1)
    
    fig.add_trace(go.Scatter(y=history.history["val_"+metric],
                             line=dict(color='red'),
                             legendgroup="validation",
                             showlegend=False),
                  row=num+2, col=1)
    
    fig.update_yaxes(title_text=metric, type="log", row=num+2, col=1)
  

  fig.update_layout(height=600, width=700,
                    title_text="Performance Evaluation")
  #fig.show()
  return fig


@st.experimental_singleton
def plot_regression(inputs, predicted, expected):
  mse = model.evaluate(inputs, expected)
  mse = round(mse[0],3)

  # sklearn linear regression
  regr = linear_model.LinearRegression()
  regr.fit(expected.reshape(-1,1), predicted)

  # SciPy linear regression
  predicted = np.reshape(predicted,-1)

  slope, intercept, R, p_value, std_err = stats.linregress(expected, predicted)
  R2 = R*R
  R2= round(R2, 3)
  slope = round(slope, 3)
  intercept = round(intercept, 3)

  if intercept < 0:
      lin_r = "y = "+str(slope)+"x"+str(intercept)
  else:
      lin_r = "y = "+str(slope)+"x+"+str(intercept)

  fig = go.Figure()
  txt = "{} R^2: {} mse: {}".format(lin_r, R2, mse)

  fig.add_trace(go.Scatter(x=expected, y=predicted,
                           line=dict(color='blue'),
                           name="data", legendgroup="data",
                           mode="markers"),
            )
  
  fig.add_trace(go.Scatter(x=expected, y=expected,
                           line=dict(color='green'),
                           name="diagonal", legendgroup="diagonal",),
            )
  
  fig.add_trace(go.Scatter(x=expected, y=regr.predict(expected.reshape(-1,1)).reshape(1,-1)[0],
                           line=dict(color='red'),
                           name="regression", legendgroup="regression",),
            )
  
  fig.add_annotation(text=txt,
                  xref="paper", yref="paper",
                  x=1, y=0, showarrow=False)

  fig.update_layout(height=350, width=700,
                    title_text="Performance Evaluation")
  #fig.show()
  return fig


def save_model(model):
  model.save('my_model.h5')
  zipObj = zipfile.ZipFile('my_model.h5.zip', 'w')
  zipObj.write('my_model.h5')
  zipObj.close()
  return


def load_model(model_file):
  myzipfile = zipfile.ZipFile(model_file)
  with tempfile.TemporaryDirectory() as tmp_dir:
    myzipfile.extractall(tmp_dir)
    root_folder = myzipfile.namelist()[0]
    model_dir = os.path.join(tmp_dir, root_folder)
    model = tf.keras.models.load_model(model_dir)
  return model


# WEB APP
# -----------------------------------------------------------------------------
PAGE_CONFIG = {"page_title":"ChemPropPrediction.io",
               "page_icon":":atom_symbol:",
               "layout":"centered",
               "initial_sidebar_state":"collapsed",
               "menu_items":{'Get Help': 'https://github.com/MarekWadinger/chem_prediction',
                            'Report a bug': "https://github.com/MarekWadinger/chem_prediction/issues",
                            'About': "# Prediction of chemical properties using regression analysis"
                              }
              }
st.set_page_config(**PAGE_CONFIG)

st.title("""Prediction of chemical properties using regression analysis""")


col1, col2 = st.columns(2)
with col1:
  st.subheader('Data')
  data_file = st.file_uploader('Import data as .xyz file. Contains your sample molecules with properties.', '.xyz')
with col2:
  st.subheader('Pretrained model')
  model_file = st.file_uploader('Import pretrained model as json file. Without pretrained model app trains a new one.', '.zip')

st.subheader('Parameters')
with st.form(key='Model Parameters'):
    descriptors_list = ["soap", "cm", "acsf", "mbtr"]
    descriptor = st.selectbox("Select Descriptor", descriptors_list,
                              index=0)
    box1 = st.empty()
    box1.selectbox("Select Property", [""], index=0)
    submit_button = st.form_submit_button(label='Submit')

p1 = st.empty()
button1 = st.empty()

with st.sidebar:
    st.title('Specify parameters')

if data_file:
    structures_list = read_data(data_file)
    p1.success("Number of systems in set: {}".format(len(structures_list)))
    property_name = box1.selectbox("Select Property", get_property_names(
                                   structures_list), index=0)

    if submit_button:
        p1.info("Creating feature vectors")
        feature_vectors = make_feature_vectors(structures_list, descriptor)
        p1.success("Created {} features".format(feature_vectors.shape[1]))
        chem_property = get_properties(structures_list, property_name)
        X_train, X_test, X_val, Y_train, Y_test, Y_val = tt_split(
            feature_vectors, chem_property)
        n_epoch = 10; loss = "mse"; metrics = ["logcosh", "mae"]

        if model_file:
          model = load_model(model_file)
          p1.success('Model Loaded Successfully')
        else:
          model = compile_model(loss, metrics)
          p1.info("Training model")
          history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
                      epochs=n_epoch, verbose=0)
        st.subheader('Evaluate')
        st.table(pd.DataFrame(columns=[loss] + metrics,
             index=["train", "test", "validate"], 
             data=[model.evaluate(X_train, Y_train, verbose=0),
                   model.evaluate(X_test,  Y_test,  verbose=0),
                   model.evaluate(X_val,   Y_val,   verbose=0)]))
        p1.success('Success!')
        st.plotly_chart(plot_performance(model.history, loss, metrics))
        st.plotly_chart(plot_regression(inputs=X_train, predicted=model.predict(X_train), expected=Y_train))
        save_model(model)
        with open('my_model.h5.zip', 'rb') as f:
          button1.download_button('Download Model', f, 'my_model.h5.zip', "application/zip")

tf.keras.backend.clear_session()
time.sleep(1.2)
p1.empty()
