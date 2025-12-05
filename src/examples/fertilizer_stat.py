#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical analysis of IFA Plant Nutrition data

Input:
  - IFADATA Plant Nutrition query - 28-Nov-2025_03.01.xlsx

Outputs:
  - fertilizer_stats_per_country.csv
  - fertilizer_pca_components.csv
  - fertilizer_clusters.csv
  - fertilizer_pca_scatter.png
  - fertilizer_trend_distribution.png
  - fertilizer_pca_variance.png
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import linregress

import matplotlib.pyplot as plt
import seaborn as sns



import numpy as np
import torch
import sys
import os
import scipy.io as sio

cwd = os.getcwd()
#sys.path.append(cwd + '/my_directory')
sys.path.append(cwd)


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir
root_dir = setting_directory(2)


import matplotlib.pyplot as plt

# General imports
import numpy as np
import torch

# DeePyMoD imports
from GNSINDy.src.deepymod import DeepMoD
from GNSINDy.src.deepymod.data import Dataset, get_train_test_loader
from GNSINDy.src.deepymod.data.samples import Subsample_random
from GNSINDy.src.deepymod.data.burgers import burgers_delta, burgers_delta_org
from GNSINDy.src.deepymod.data.burgers import burgers_delta
from GNSINDy.src.deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from GNSINDy.src.deepymod.model.func_approx import NN
from GNSINDy.src.deepymod.model.library import Library1D
from GNSINDy.src.deepymod.model.sparse_estimators import Threshold, STRidge
from GNSINDy.src.deepymod.training import train
#from deepymod.training.training_2 import train
from GNSINDy.src.deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic
#from deepymod.data.data_set_preparation import DatasetPDE, pde_data_loader
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
from GNSINDy.src.deepymod.data.DEIM_class import DEIM
import shutil

from GNSINDy.src.deepymod.utils.utilities import create_or_reset_directory

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(50)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




# -------------------------------------------------------------
# 0. Load Excel file (header row is the 3rd row)
# -------------------------------------------------------------
DATA_FILE = root_dir + "/src/data/IFADATA Plant Nutrition query - 28-Nov-2025_03.01.xlsx"
file_path = Path(DATA_FILE)

# Row 1: comment; Row 2: blank; Row 3: real header
df_raw = pd.read_excel(file_path, header=2)

# Clean column names
df_raw.columns = [c.strip() for c in df_raw.columns]
print("Loaded columns:", df_raw.columns)

# -------------------------------------------------------------
# 1. Keep relevant columns and basic cleaning
# -------------------------------------------------------------
cols_needed = ["Product", "Country", "Year", "Consumption"]
df = df_raw[cols_needed].copy()

# Drop rows with missing consumption
df = df.dropna(subset=["Consumption"])

# Ensure numeric year and consumption
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Consumption"] = pd.to_numeric(df["Consumption"], errors="coerce")
df = df.dropna(subset=["Year", "Consumption"])

# -------------------------------------------------------------
# 2. Aggregate total fertilizer consumption per country & year
# -------------------------------------------------------------
df_total = (
    df.groupby(["Country", "Year"])["Consumption"]
      .sum()
      .reset_index()
)

# Pivot: countries × years
df_pivot = df_total.pivot(index="Country", columns="Year", values="Consumption")

# Remove countries with too many missing years (e.g. <70% coverage)
df_pivot = df_pivot.dropna(thresh=int(0.7 * df_pivot.shape[1]), axis=0)

# Interpolate missing values along time
df_pivot = df_pivot.sort_index(axis=1)          # sort years
df_pivot = df_pivot.interpolate(axis=1)         # linear interpolation
df_pivot = df_pivot.dropna(axis=0)              # drop any remaining NaNs

print("Data after pivot and cleaning:", df_pivot.shape)

# Years as numpy array (for regression)
years = df_pivot.columns.to_numpy(dtype=float)







"""

# -------------------------------------------------------------
# 3. Descriptive statistics per country
# -------------------------------------------------------------
stats = pd.DataFrame(index=df_pivot.index)
stats["mean"] = df_pivot.mean(axis=1)
stats["std"] = df_pivot.std(axis=1)
stats["min"] = df_pivot.min(axis=1)
stats["max"] = df_pivot.max(axis=1)
stats["cv"] = stats["std"] / stats["mean"]   # coefficient of variation

def calc_trend(row):
    slope, intercept, r, p, stderr = linregress(years, row.to_numpy())
    return slope

stats["trend_slope"] = df_pivot.apply(calc_trend, axis=1)

stats.to_csv("fertilizer_stats_per_country.csv")
print("Saved: fertilizer_stats_per_country.csv")

# -------------------------------------------------------------
# 4. PCA on standardized time series
# -------------------------------------------------------------
X = df_pivot.to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
Z = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": Z[:, 0],
    "PC2": Z[:, 1],
    "PC3": Z[:, 2],
}, index=df_pivot.index)

pca_df.to_csv("fertilizer_pca_components.csv")
print("Saved: fertilizer_pca_components.csv")

# -------------------------------------------------------------
# 5. K-means clustering on first two PCs
# -------------------------------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(Z[:, :2])

pca_df["cluster"] = clusters
pca_df.to_csv("fertilizer_clusters.csv")
print("Saved: fertilizer_clusters.csv")

# -------------------------------------------------------------
# 6. Plots
# -------------------------------------------------------------
sns.set(style="whitegrid")

# PCA scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="tab10"
)
plt.title("PCA of Fertilizer Consumption (Total Nutrients)")
plt.tight_layout()
plt.savefig("fertilizer_pca_scatter.png", dpi=300)

# Trend slope distribution
plt.figure(figsize=(8, 5))
sns.histplot(stats["trend_slope"], kde=True)
plt.xlabel("Trend slope (000 metric tonnes / year)")
plt.title("Distribution of Trend Slopes in Fertilizer Consumption")
plt.tight_layout()
plt.savefig("fertilizer_trend_distribution.png", dpi=300)

# PCA variance explained
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_ * 100,
         marker="o")
plt.xlabel("Principal component")
plt.ylabel("Variance explained (%)")
plt.title("PCA Variance Explained")
plt.tight_layout()
plt.savefig("fertilizer_pca_variance.png", dpi=300)

print("Plots saved. Analysis complete.")



# -------------------------------------------------------------
# NEW ADVANCED ANALYSIS PIPELINE
# -------------------------------------------------------------
# 1) log-transform to reduce magnitude
df_log = np.log1p(df_pivot)

# 2) standardize each country (row-wise)
df_std = df_log.sub(df_log.mean(axis=1), axis=0)
df_std = df_std.div(df_std.std(axis=1), axis=0)

# 3) first differences (trend/volatility)
df_diff = df_std.diff(axis=1).dropna(axis=1)

# 4) PCA on cleaned data
X = df_diff.to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
Z = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": Z[:, 0],
    "PC2": Z[:, 1],
    "PC3": Z[:, 2],
}, index=df_pivot.index)




# -----------------------------------


# -------------------------------------------------------------
# 7. Time-series curves per country
#    (total fertilizer consumption per year)
# -------------------------------------------------------------
out_dir = Path("fertilizer_country_curves")
out_dir.mkdir(exist_ok=True)

# Years (x-axis)
years = df_pivot.columns.to_numpy(dtype=float)

for country in df_pivot.index:
    series = df_pivot.loc[country].to_numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(years, series, marker="o", linewidth=1.5)
    plt.title(f"Total Fertilizer Consumption – {country}")
    plt.xlabel("Year")
    plt.ylabel("Consumption (000 metric tonnes of nutrients)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Safe file name
    safe_name = (
        country.replace(" ", "_")
               .replace("/", "_")
               .replace("(", "")
               .replace(")", "")
    )
    plt.savefig(out_dir / f"fertilizer_curve_{safe_name}.png", dpi=300)
    plt.close()

print(f"Saved individual country curves in: {out_dir.resolve()}")




# -------------------------------------------------------------
# NORMALIZED (Z-SCORE) COMPARISON PLOTS
# -------------------------------------------------------------
norm = df_pivot.sub(df_pivot.mean(axis=1), axis=0)
norm = norm.div(df_pivot.std(axis=1), axis=0)

plt.figure(figsize=(12, 6))
for country in ["Syria", "Myanmar", "India"]:
    plt.plot(years, norm.loc[country], marker="o", label=country)

plt.title("Fertilizer Consumption (Normalized by Country Mean & Std)")
plt.xlabel("Year")
plt.ylabel("Z-score")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("fertilizer_comparison_normalized.png", dpi=300)



plt.figure(figsize=(12, 6))
for country in ["Syria", "Myanmar", "India"]:
    plt.plot(years, df_pivot.loc[country], marker="o", label=country)

plt.yscale("log")
plt.title("Fertilizer Consumption (Log Scale)")
plt.xlabel("Year")
plt.ylabel("Consumption (log scale)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("fertilizer_comparison_logscale.png", dpi=300)

"""




################################################

def load_country_coordinates(countries):
    import geopandas as gpd
    import pandas as pd

    # Load Natural Earth data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Centroids
    world["lon"] = world.centroid.x
    world["lat"] = world.centroid.y

    coords = world[["name", "lon", "lat"]]
    coords.columns = ["Country", "lon", "lat"]

    # Reindex to match your df_pivot ordering
    coords_aligned = coords.set_index("Country").reindex(countries)

    return coords_aligned


from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
import geopandas as gpd

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
import geopandas as gpd

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
import geopandas as gpd
import re

try:
    from unidecode import unidecode
except ImportError:
    # simple fallback if unidecode is not installed
    def unidecode(s):
        return s


def _normalize_name(s: str) -> str:
    """Normalize country names for robust matching."""
    s = unidecode(str(s))          # remove accents
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)  # keep alnum, replace others with space
    s = " ".join(s.split())        # collapse spaces
    return s


def create_data():
    """
    Create dataset for the fertilizer-DEIM/PDE simulation.

    Steps:
      - Load IFA Plant Nutrition Excel
      - Aggregate to Country × Year total consumption
      - Clean and interpolate time series
      - Get country centroids (lon, lat) from Natural Earth (via URL)
      - Map your country names (USA, UK, etc.) to Natural Earth equivalents
      - Embed (lon, lat) -> 1D spatial coordinate via PCA, normalize to [0,1]
      - Normalize field X and time axis t to [0,1]
      - Run DEIM on normalized field
      - Return DEIM sample coords (t,x) and values as torch tensors
    """

    # -------------------------------------------------------------
    # 0. Load Excel file (header row is the 3rd row)
    # -------------------------------------------------------------
    DATA_FILE = root_dir + "/src/data/IFADATA Plant Nutrition query - 28-Nov-2025_03.01.xlsx"
    file_path = Path(DATA_FILE)

    df_raw = pd.read_excel(file_path, header=2)

    # Clean column names
    df_raw.columns = [c.strip() for c in df_raw.columns]
    print("Loaded columns:", df_raw.columns)

    # -------------------------------------------------------------
    # 1. Keep relevant columns and basic cleaning
    # -------------------------------------------------------------
    cols_needed = ["Product", "Country", "Year", "Consumption"]
    df = df_raw[cols_needed].copy()

    # Drop rows with missing consumption
    df = df.dropna(subset=["Consumption"])

    # Ensure numeric year and consumption
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Consumption"] = pd.to_numeric(df["Consumption"], errors="coerce")
    df = df.dropna(subset=["Year", "Consumption"])

    # -------------------------------------------------------------
    # 2. Aggregate total fertilizer consumption per country & year
    # -------------------------------------------------------------
    df_total = (
        df.groupby(["Country", "Year"])["Consumption"]
          .sum()
          .reset_index()
    )

    # Pivot: countries × years
    df_pivot = df_total.pivot(index="Country", columns="Year", values="Consumption")

    # Remove countries with too many missing years (e.g. <70% coverage)
    df_pivot = df_pivot.dropna(thresh=int(0.7 * df_pivot.shape[1]), axis=0)

    # Interpolate missing values along time
    df_pivot = df_pivot.sort_index(axis=1)      # sort years
    df_pivot = df_pivot.interpolate(axis=1)     # linear interpolation
    df_pivot = df_pivot.dropna(axis=0)          # drop any remaining NaNs

    # Ensure deterministic ordering
    df_pivot = df_pivot.sort_index(axis=0)      # sort countries
    df_pivot = df_pivot.sort_index(axis=1)      # sort years

    print("Data after pivot and cleaning:", df_pivot.shape)

    # Years and countries (initial)
    years = df_pivot.columns.to_numpy(dtype=float)
    countries = df_pivot.index.to_numpy()

    # -------------------------------------------------------------
    # 3. Load country coordinates from Natural Earth (via URL)
    # -------------------------------------------------------------
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)

    # Compute centroids (lon, lat)
    world["lon"] = world.geometry.centroid.x
    world["lat"] = world.geometry.centroid.y

    # Natural Earth name column is 'ADMIN'
    coords_world = world[["ADMIN", "lon", "lat"]].copy()
    coords_world.columns = ["Country_NE", "lon", "lat"]

    # -------------------------------------------------------------
    # 3a. Build normalized keys + alias map for robust matching
    # -------------------------------------------------------------
    # Map your country names to Natural Earth names where necessary
    alias_map = {
        "USA": "United States of America",
        "US": "United States of America",
        "UK": "United Kingdom",
        "Taiwan, China": "Taiwan",
        # some possible variants:
        "Côte d'Ivoire": "Cote d'Ivoire",  # just in case
        "Ivory Coast": "Cote d'Ivoire",
        "World": None,
        "Others Africa": None,
        "Others Latin America": None,
        "Others Oceania": None,
    }

    # DataFrame of your countries
    df_countries = pd.DataFrame({"Country": countries})

    # Apply alias mapping (leave unchanged if not in alias_map or mapped to None)
    def apply_alias(name):
        mapped = alias_map.get(name, name)
        return mapped

    df_countries["Country_alias"] = df_countries["Country"].map(apply_alias)

    # Build normalized key
    df_countries["key"] = df_countries["Country_alias"].apply(_normalize_name)

    # Build normalized key for Natural Earth countries
    coords_world["key"] = coords_world["Country_NE"].apply(_normalize_name)

    # Merge on normalized key
    merged = df_countries.merge(
        coords_world[["key", "lon", "lat"]],
        on="key",
        how="left",
        suffixes=("", "_world"),
    )

    # Use original country names as index
    coords_aligned = merged.set_index("Country")[["lon", "lat"]]

    # -------------------------------------------------------------
    # 3b. Drop countries still without coordinates (only real misses)
    # -------------------------------------------------------------
    missing_mask = coords_aligned["lon"].isna() | coords_aligned["lat"].isna()
    if missing_mask.any():
        missing_countries = coords_aligned.index[missing_mask].tolist()
        print("Dropping countries without lon/lat after alias+norm:", missing_countries)
        coords_aligned = coords_aligned[~missing_mask]
        df_pivot = df_pivot.loc[coords_aligned.index]
        countries = df_pivot.index.to_numpy()

    # Recompute X and years after possible filtering
    years = df_pivot.columns.to_numpy(dtype=float)
    X = df_pivot.to_numpy()  # (n_countries, n_years)

    # -------------------------------------------------------------
    # 4. Build 1D spatial coordinate from lon/lat via PCA
    # -------------------------------------------------------------
    lon_lat = coords_aligned[["lon", "lat"]].to_numpy()  # (n_countries, 2)

    pca = PCA(n_components=1)
    x_embed = pca.fit_transform(lon_lat).ravel()         # (n_countries,)

    # Normalize x_embed to [0, 1]
    x_min, x_max = x_embed.min(), x_embed.max()
    if x_max > x_min:
        x_o = (x_embed - x_min) / (x_max - x_min)
    else:
        x_o = np.zeros_like(x_embed)

    # Time axis (raw)
    t_o = years.astype(float)

    # -------------------------------------------------------------
    # 5. Global min-max normalization of X, and normalize t to [0,1]
    # -------------------------------------------------------------
    X_min = np.min(X)
    X_max = np.max(X)
    if X_max > X_min:
        X_norm = (X - X_min) / (X_max - X_min)
    else:
        X_norm = np.zeros_like(X)

    print("Normalized X range:", X_norm.min(), X_norm.max())

    t_min = t_o.min()
    t_max = t_o.max()
    if t_max > t_min:
        t_o_norm = (t_o - t_min) / (t_max - t_min)
    else:
        t_o_norm = np.zeros_like(t_o)

    print("t_o_norm range:", t_o_norm.min(), t_o_norm.max())

    x_o_norm = x_o  # already in [0,1]
    print("x_o_norm range:", x_o_norm.min(), x_o_norm.max())

    # -------------------------------------------------------------
    # 6. Instantiate and run DEIM on normalized field
    # -------------------------------------------------------------
    n_d = 2   # number of temporal segments in your DEIM implementation
    deim_obj = DEIM(
        X=X_norm,
        n_d=n_d,
        t_o=t_o_norm,
        x_o=x_o_norm,
        tolerance=1e-3,
        num_basis=1
    )

    S_s, T_s, U_s = deim_obj.execute()  # S_s: space coords, T_s: time coords, U_s: values

    # -------------------------------------------------------------
    # 7. Build coords and data tensors for your PINN / GN-SINDy
    # -------------------------------------------------------------
    coords_np = np.hstack((T_s, S_s))      # (N, 2): [t, x]
    data_np   = U_s.reshape(-1, 1)         # (N, 1)

    coords = torch.from_numpy(coords_np).float()
    data   = torch.from_numpy(data_np).float()

    return coords, data



################################################



"""
def create_data():
    
        
    # -------------------------------------------------------------
    # 0. Load Excel file (header row is the 3rd row)
    # -------------------------------------------------------------
    DATA_FILE = root_dir + "/src/data/IFADATA Plant Nutrition query - 28-Nov-2025_03.01.xlsx"
    file_path = Path(DATA_FILE)
    
    # Row 1: comment; Row 2: blank; Row 3: real header
    df_raw = pd.read_excel(file_path, header=2)
    
    # Clean column names
    df_raw.columns = [c.strip() for c in df_raw.columns]
    print("Loaded columns:", df_raw.columns)
    
    # -------------------------------------------------------------
    # 1. Keep relevant columns and basic cleaning
    # -------------------------------------------------------------
    cols_needed = ["Product", "Country", "Year", "Consumption"]
    df = df_raw[cols_needed].copy()
    
    # Drop rows with missing consumption
    df = df.dropna(subset=["Consumption"])
    
    # Ensure numeric year and consumption
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Consumption"] = pd.to_numeric(df["Consumption"], errors="coerce")
    df = df.dropna(subset=["Year", "Consumption"])
    
    # -------------------------------------------------------------
    # 2. Aggregate total fertilizer consumption per country & year
    # -------------------------------------------------------------
    df_total = (
        df.groupby(["Country", "Year"])["Consumption"]
          .sum()
          .reset_index()
    )
    
    # Pivot: countries × years
    df_pivot = df_total.pivot(index="Country", columns="Year", values="Consumption")
    
    # Remove countries with too many missing years (e.g. <70% coverage)
    df_pivot = df_pivot.dropna(thresh=int(0.7 * df_pivot.shape[1]), axis=0)
    
    # Interpolate missing values along time
    df_pivot = df_pivot.sort_index(axis=1)          # sort years
    df_pivot = df_pivot.interpolate(axis=1)         # linear interpolation
    df_pivot = df_pivot.dropna(axis=0)              # drop any remaining NaNs
    
    print("Data after pivot and cleaning:", df_pivot.shape)
    
    # Years as numpy array (for regression)
    years = df_pivot.columns.to_numpy(dtype=float)
        
        
    # 1) Ensure df_pivot is sorted consistently
    df_pivot = df_pivot.sort_index(axis=0)   # sort countries
    df_pivot = df_pivot.sort_index(axis=1)   # sort years
    
    countries = df_pivot.index.to_numpy()                 # space labels
    years = df_pivot.columns.to_numpy(dtype=float)        # time labels
    X = df_pivot.to_numpy()                               # (n_countries, n_years)
    
    # 2) Define coordinates for DEIM
    x_o = np.arange(len(countries))       # numeric indices for countries
    t_o = years                           # years as time coordinates
    
    
    
    
    
    # 1) Global min-max normalization to [0,1]
    X_min = np.min(X)
    X_max = np.max(X)
    
    X_norm = (X - X_min) / (X_max - X_min)
    
    print("Normalized X range:", X_norm.min(), X_norm.max())
    
    
    t_min = t_o.min()
    t_max = t_o.max()
    t_o_norm = (t_o - t_min) / (t_max - t_min)
    
    print("t_o_norm range:", t_o_norm.min(), t_o_norm.max())
    
    x_min = x_o.min()
    x_max = x_o.max()
    x_o_norm = (x_o - x_min) / (x_max - x_min)
    
    print("x_o_norm range:", x_o_norm.min(), x_o_norm.max())

    
    
    # 3) Instantiate and run DEIM
    n_d = 5                               # no temporal segmentation (use full time span)
    deim_obj = DEIM(
        X=X_norm,
        n_d=n_d,
        t_o=t_o_norm,
        x_o=x_o_norm,
        tolerance=1e-3,
        num_basis=1
    )
    
    S_s, T_s, U_s = deim_obj.execute()    # S_s: space coords, T_s: time coords, U_s: values
    
    
    
    
    #deim_instance = DEIM(Exact, 5, t_o, x_o, tolerance = 1e-03, num_basis = 1)
    #S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
    
    return coords, data
"""



coords_2, data_2 = create_data()


num_of_samples = 1000


import time

start_time = time.time()

x_t, u = create_data()

end_time = time.time()
print(f"Execution time: {end_time - start_time:.6f} seconds")


#########################
#########################
#########################

dataset = Dataset(
    create_data,
    #load_kwargs=load_kwargs,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,
)

coords = dataset.get_coords().cpu()
data = dataset.get_data().cpu()
fig, ax = plt.subplots()
im = ax.scatter(coords[:,0], coords[:,1], c=data[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)

plt.show()



##########################
##########################

train_dataloader, test_dataloader = get_train_test_loader(
    dataset, train_test_split = 1.00)

##########################
##########################

poly_order = 2
diff_order = 2

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1


network = NN(2, [64, 64, 64, 64], 1)
library = Library1D(poly_order, diff_order)
sparsity_scheduler = TrainTestPeriodic(periodicity=100, patience=1000, delta=1e-3)
constraint = STRidgeCons(tol=0.05)
estimator = STRidge()


#constraint = Ridge()
# Configuration of the sparsity scheduler
model = DeepMoD(network, library, estimator, constraint).to(device)

# Defining optimizer
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername = "./data/deepymod/fertilizer/"



# Example usage:
#directory_path = "/path/to/your/directory"
create_or_reset_directory(foldername)



######################################
######################################
######################################


train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    sparsity_scheduler,
    log_dir= foldername,
    exp_ID="Test",
    write_iterations=100,
    max_iterations=100000,
    delta=1e-3,
    patience=500,
)

model.sparsity_masks

print(model.estimator_coeffs())
print(model.constraint.coeff_vectors[0].detach().cpu())

print(model.constraint.coeff_vectors[0].detach().cpu().round(decimals=2))







from GNSINDy.src.deepymod.utils import plot_config_file
from GNSINDy.src.deepymod.analysis import load_tensorboard
history = load_tensorboard(foldername)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
line_width=2

for history_key in history.keys():
    history_key_parts = history_key.split("_")
      
    if history_key_parts[0] == "estimator":
        if history_key_parts[3] == "0" and len(history_key_parts)>4:
            
            if history_key_parts[5] != "2" and history_key_parts[5] != "4":
                
                axs[0].semilogx(
                history[history_key].loc[100:],
                
                linestyle="--",
                linewidth= line_width
                )
        
            elif history_key_parts[5] == "2":
                    
                axs[0].semilogx(
                    history[history_key].loc[100:],
                    label= "\r $u_{xx}$",
                    
                    linewidth= line_width + 1
                    )
            elif history_key_parts[5] == "4":
            
                axs[0].semilogx(
                    history[history_key].loc[100:],
                    label= "\r $uu_x$",
                    
                    linewidth= line_width + 1
                    ) 
            
            
    
axs[0].set_ylim([-10, 10])
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("Coefficients")


# Add legends to the right of the subplots
#axs[0].legend(loc='center left', bbox_to_anchor=(0.45, 0.9), ncol=2)
#axs[0].legend(loc='center left', bbox_to_anchor=(0.0, 1.05), ncol=2)
#axs[0].grid(True)
axs[0].grid(True)


coords = dataset.get_coords().cpu()
data = dataset.get_data().cpu()

im = axs[1].scatter(coords[:,0], coords[:,1], c=data[:,0], marker="o", 
                    label=r"Greedy samples: \texttt{Q-DEIM}",
                    s=10)
axs[1].set_xlabel(r'$t$')
axs[1].set_ylabel(r'$x$',labelpad=0)
axs[1].set_ylim([0,1])
#axs[1].legend(loc='center left', bbox_to_anchor=(0.0, 1.05), ncol=1)
#fig.colorbar(mappable=im)



fig.legend(
    
    loc="center",  # Change the location to upper center
    ncol=4,
    bbox_to_anchor=(0.51, 1),  # Adjust the coordinates
    bbox_transform=fig.transFigure,
    fontsize=10,
    frameon=True,
)


plt.savefig(foldername +
     "Burgers" + "_coeff_iterations_DEIM_" + f'{num_of_samples}'  
     +"_poly_order_ " + f'{poly_order}'
     +"_diff_order_ " + f'{diff_order}'
     +
     ".png", bbox_inches='tight',
    dpi=600,
)
















