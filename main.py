# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files,
# tool windows, actions, and settings.
# IMPORT
# Web App
import streamlit as st
# Data Processing
from ase.io import read as ase_read
from dscribe.descriptors import CoulombMatrix, SOAP, ACSF, MBTR

# Support
import numpy as np
from sklearn.model_selection import train_test_split
import time


@st.cache(suppress_st_warning=True)
def read_data(xyz_data_file):
    return ase_read(xyz_data_file.name, index=":")


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


@st.cache(suppress_st_warning=True)
def make_feature_vectors(structures: list, descriptor_name: str):
    match descriptor_name:
        case "soap":
            soap = SOAP(
                species=find_all_atoms(structures),
                periodic=False,
                rcut=6,
                nmax=6,
                lmax=6,
                average='outer',
                sparse=False,
            )
            fv = soap.create(structures, n_jobs=-1)
        case "cm":
            cm = CoulombMatrix(
                n_atoms_max=find_max_atoms(structures)
            )
            fv = cm.create(structures, n_jobs=-1)
        case "acsf":
            pos = get_positions(structures)
            acsf = ACSF(
                species=find_all_atoms(structures),
                periodic=False,
                rcut=7,
                g2_params=[[1, 1], [1, 2], [1, 3]],
                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
            )
            fv = acsf.create(structures, n_jobs=-1, positions=pos).reshape(
                5000, -1)
        case "mbtr":
            mbtr = MBTR(
                species=find_all_atoms(structures),
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
            fv = mbtr.create(structures, n_jobs=-1)
        case _:
            raise ValueError("Could not find descriptor")

    return fv


@st.cache(suppress_st_warning=True)
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


def compile_model(loss, metrics):
  # define model structure
  model = tf.keras.models.Sequential([
      #tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          32,
          kernel_initializer = tf.keras.initializers.RandomUniform(seed=None),
          activation='relu',
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


# WEB APP
# -----------------------------------------------------------------------------
st.title("""Prediction of chemical properties using regression analysis""")

data_file = st.file_uploader('Import data as .xyz file')
with st.form(key='Model Parameters'):
    descriptors_list = ["soap", "cm", "acsf", "mbtr"]
    descriptor = st.selectbox('Select Descriptor', descriptors_list,
                              index=0)
    box1 = st.empty()
    submit_button = st.form_submit_button(label='Submit')
p1 = st.empty()

with st.sidebar:
    st.title('Specify parameters')

if data_file:
    structures_list = read_data(data_file)
    p1.success("Number of systems in set: {}".format(len(structures_list)))
    property_name = box1.selectbox('Select Property', get_property_names(
                                   structures_list), index=0)

    if submit_button:
        p1.info("Creating feature vectors")
        feature_vectors = make_feature_vectors(structures_list, descriptor)
        p1.success("Created {} features".format(feature_vectors.shape[1]))
        chem_property = get_properties(structures_list, property_name)
        X_train, X_test, X_val, Y_train, Y_test, Y_val = tt_split(
            feature_vectors, chem_property)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
