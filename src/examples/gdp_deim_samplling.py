#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:09:32 2025

@author: forootan
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


import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from pathlib import Path
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from sklearn.decomposition import PCA


# ================================================================
# Helper 1: detect Maddison aggregates (non-countries)
# ================================================================
def is_maddison_aggregate(name: str) -> bool:
    if name is None:
        return True
    bad_tokens = [
        "Total", "World", "W. Europe", "E. Europe", "L. America", "Caribbean",
        "Africa", "Asia", "Americas", "OECD", "Western Europe", "Eastern Europe",
        "3 Small Afr", "7 E. Europe", "14 small WEC", "small WEC",
    ]
    if any(tok in name for tok in bad_tokens):
        return True
    # Aggregates often start with digits (e.g. “21 Caribbean”)
    if name and name[0].isdigit():
        return True
    return False


# ================================================================
# Helper 2: GDP-specific alias map
# ================================================================
alias_map_gdp = {
    "USA": "United States of America",
    "US": "United States of America",
    "United States": "United States of America",

    "UK": "United Kingdom",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",

    "Russian Federation": "Russia",

    "Korea (South)": "South Korea",
    "Korea (North)": "North Korea",

    "Czechoslovakia": "Czech Republic",

    # Explicitly drop aggregates
    "World": None,
    "World Total": None,
}


alias_map_gdp.update({
    # simple short forms → full ADMIN name
    "Bosnia": "Bosnia and Herzegovina",
    "Burma": "Myanmar",
    "Cape Verde": "Cabo Verde",
    "Centr. Afr. Rep.": "Central African Republic",
    "Comoro Islands": "Comoros",
    "Congo 'Brazzaville'": "Republic of the Congo",
    "Czech Rep.": "Czech Republic",
    "Haïti": "Haiti",
    "Hong Kong": "Hong Kong",
    "Indonesia (& Timor until '99)": "Indonesia",
    "Macedonia": "North Macedonia",
    "N. Zealand": "New Zealand",
    "S. Korea": "South Korea",
    "S. Tomé & P.": "Sao Tome and Principe",
    "Swaziland": "Eswatini",
    "T. & Tobago": "Trinidad and Tobago",
    "Turk-menistan": "Turkmenistan",
    "UAE": "United Arab Emirates",
    "Zaire (Congo-Kinshasa)": "Democratic Republic of the Congo",

    # keep as themselves (Natural Earth already has these names, we just
    # list them here for clarity – the new _normalize_name helps too)
    "Bahrain": "Bahrain",
    "Mauritius": "Mauritius",
    "Seychelles": "Seychelles",
    "Singapore": "Singapore",
    "Tanzania": "Tanzania",

    # historical / composite entities – choose a representative or drop
    # (here: map to a modern country; if you prefer to drop, set value=None)
    "Czecho-slovakia": "Czech Republic",
    "F. Czecho-slovakia": "Czech Republic",
    "F. USSR": "Russia",
    "F. Yugoslavia": "Serbia",
    "Yugoslavia": "Serbia",
    "Serbia/Montenegro/Kosovo": "Serbia",
    "Eritrea & Ethiopia": "Ethiopia",  # or None if you want to drop
    "W. Bank & Gaza": "Palestine",     # Natural Earth: "Palestine"

    # clearly “regional aggregate” → drop them entirely
    "W. Offshoots": None,
})


























def apply_alias_gdp(name: str) -> str | None:
    if is_maddison_aggregate(name):
        return None
    return alias_map_gdp.get(name, name)


# ================================================================
# Helper 3: normalizer for name matching (same as your fertilizer code)
# ================================================================


"""
def _normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    for ch in ["'", ".", ",", "’"]:
        s = s.replace(ch, "")
    s = s.replace("&", "and")
    s = s.replace("-", " ")
    s = " ".join(s.split())
    return s
"""




import unicodedata

def _normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # strip whitespace and lower
    s = s.strip().lower()
    # remove accents (Côte -> cote, Haïti -> haiti)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # basic punctuation / symbol cleanup
    for ch in ["'", ".", ",", "’"]:
        s = s.replace(ch, "")
    s = s.replace("&", "and")
    s = s.replace("-", " ")
    s = " ".join(s.split())
    return s




FILE = "md2010_vertical.xlsx"
SHEET = "Population"   # or "GDP", "PerCapita GDP"


def load_maddison_vertical(file, sheet):
    raw = pd.read_excel(file, sheet_name=sheet, header=None, dtype=str)

    # ------------------------------------------------------
    # 1. Detect country-name header row
    # ------------------------------------------------------
    header_row_idx = None
    for i in range(5):  # first few rows are metadata/header
        row = raw.iloc[i]
        string_count = row.apply(lambda x: isinstance(x, str) and x.strip() != "").sum()
        integer_like_count = row.apply(
            lambda x: isinstance(x, str) and x.strip().isdigit()
        ).sum()

        # heuristic: header row has many strings (country names), few integers
        if string_count > 10 and integer_like_count < 5:
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError("Cannot detect header row.")

    header_row = raw.iloc[header_row_idx, :]
    countries = header_row.iloc[1:].tolist()
    countries = [c.strip() if isinstance(c, str) else None for c in countries]

    # ------------------------------------------------------
    # 2. Data rows start just after header_row_idx
    # ------------------------------------------------------
    data_start = header_row_idx + 1
    data = raw.iloc[data_start:, : len(countries) + 1].copy()
    data.columns = ["Year"] + countries

    # drop rows with no year
    data = data[data["Year"].notna()]

    # convert year to int
    data["Year"] = pd.to_numeric(data["Year"].str.strip(), errors="coerce")
    data = data[data["Year"].notna()]
    data["Year"] = data["Year"].astype(int)

    # ------------------------------------------------------
    # 3. Robust numeric conversion for all country columns
    # ------------------------------------------------------
    for col in data.columns[1:]:
        # make sure everything is string, remove commas & spaces
        s = (
            data[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("\u00a0", "", regex=False)  # non-breaking space, just in case
            .str.strip()
        )
        data[col] = pd.to_numeric(s, errors="coerce")  # invalid → NaN

    data = data.reset_index(drop=True)

    # ------------------------------------------------------
    # 4. Long (tidy) format
    # ------------------------------------------------------
    value_name = sheet.replace(" ", "_")
    long_df = data.melt(
        id_vars="Year",
        var_name="Country",
        value_name=value_name,
    ).dropna(subset=[value_name])

    return data, long_df




# ================================================================
# 🚀 FINAL FUNCTION: create_data_gdp()
# ================================================================
# ================================================================
# 🚀 FIXED FINAL FUNCTION: create_data_gdp()
#    - Build country metadata AFTER cleaning
#    - Keep raw Maddison name as index for alignment with wide_clean
# ================================================================
def create_data_gdp():
    """
    Create DEIM-ready spacetime GDP field:

        t ∈ [0,1]
        x ∈ [0,1]   (countries embedded via PCA(lon,lat))
        u(t,x) = normalized GDP

    FILE  : Maddison vertical Excel file ("md2010_vertical.xlsx")
    SHEET : "GDP" or "PerCapita GDP" or "Population"
    """
    
    
    FILE= "/home/forootan/Documents/Jochen/datasets/Fertilizer/GNSINDy/src/data/md2010_vertical.xlsx"
    SHEET= "Population"
    
    n_d=5
    tol=1e-10
    num_basis=1
    coverage_thresh=0.7
    
    # ---------------------------------------------------------
    # 1. Load Maddison sheet
    # ---------------------------------------------------------
    wide, long = load_maddison_vertical(FILE, SHEET)
    print("Loaded:", SHEET, wide.shape)
    
    
    

    # All raw country names (Maddison columns)
    countries_raw = [c for c in wide.columns if c != "Year"]
    

    

    # Apply alias map & drop explicit aggregates
    raw_to_alias = {}
    kept_raw = []
    for c in countries_raw:
        alias = apply_alias_gdp(c)  # returns None if aggregate
        if alias is not None:
            raw_to_alias[c] = alias
            kept_raw.append(c)

    # ---------------------------------------------------------
    # 2. Build wide_clean and clean in time
    # ---------------------------------------------------------
    # Keep only non-aggregate countries
    wide_clean = wide[["Year"] + kept_raw].copy()

    # Time on rows → Country on rows
    wide_clean = wide_clean.set_index("Year").T  # (Country_raw × Year)
    
        
        # ------------------------------------------------------
    # Keep only years that have values for ALL countries
    # ------------------------------------------------------
    non_na_mask = wide_clean.notna().all(axis=0)   # True where no NaNs in that year
    wide_common = wide_clean.loc[:, non_na_mask]
    
    print("Common years where all countries have GDP:",
          wide_common.columns.min(), "–", wide_common.columns.max())
    print("Shape after taking common years:", wide_common.shape)
    
    # Use wide_common from here on
    wide_clean = wide_common

    
        
        # optional: might be unnecessary if already all non-NaN
    wide_clean = wide_clean.sort_index(axis=1)  # sort years
    wide_clean = wide_clean.interpolate(axis=1)
    wide_clean = wide_clean.dropna(axis=0)
    wide_clean = wide_clean.sort_index(axis=0)  # countries
    wide_clean = wide_clean.sort_index(axis=1)  # years

    

    # Drop countries with poor coverage
    min_non_nan = int(coverage_thresh * wide_clean.shape[1])
    wide_clean = wide_clean.dropna(thresh=min_non_nan, axis=0)

    # Interpolate over time
    wide_clean = wide_clean.sort_index(axis=1)  # sort years
    wide_clean = wide_clean.interpolate(axis=1)
    wide_clean = wide_clean.dropna(axis=0)

    # Sort consistent ordering
    wide_clean = wide_clean.sort_index(axis=0)  # countries
    wide_clean = wide_clean.sort_index(axis=1)  # years

    print("After cleaning:", wide_clean.shape)
    
    
    

    # Surviving Maddison country names
    final_countries_raw = wide_clean.index.to_numpy()
    years = wide_clean.columns.to_numpy(dtype=float)
    X = wide_clean.to_numpy()  # (n_countries, n_years)
    
    
    

    # ---------------------------------------------------------
    # 3. Build df_countries *after* cleaning, with aliases and keys
    # ---------------------------------------------------------
    aliases = [raw_to_alias[c] for c in final_countries_raw]
    df_countries = pd.DataFrame({
        "Country_raw": final_countries_raw,   # original Maddison name
        "Country_alias": aliases
    })
    df_countries["key"] = df_countries["Country_alias"].apply(_normalize_name)

    # ---------------------------------------------------------
    # 4. Load Natural Earth coordinates and match by normalized name
    # ---------------------------------------------------------
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)

    world["lon"] = world.geometry.centroid.x
    world["lat"] = world.geometry.centroid.y

    coords_world = world[["ADMIN", "lon", "lat"]].copy()
    coords_world.columns = ["Country_NE", "lon", "lat"]
    coords_world["key"] = coords_world["Country_NE"].apply(_normalize_name)

    # Match Maddison aliases to Natural Earth names
    merged = df_countries.merge(
        coords_world[["key", "lon", "lat"]],
        on="key",
        how="left"
    )

    # Report unmatched (for your REGION_KEYWORDS / manual fixes if you want)
    missing_mask = merged["lon"].isna() | merged["lat"].isna()
    if missing_mask.any():
        print("Unmatched GDP countries (no NE coords):",
              merged.loc[missing_mask, "Country_raw"].tolist())

    # Drop unmatched from both coordinates and data
    merged = merged.dropna(subset=["lon", "lat"])

    # Now we want to keep the raw Maddison names that have coordinates
    valid_raw = merged["Country_raw"].to_numpy()
    print("Final GDP countries with coords:", len(valid_raw))

    # Align X with valid countries (no KeyError now)
    wide_valid = wide_clean.loc[valid_raw]
    X = wide_valid.to_numpy()
    final_countries = valid_raw  # for info
    
    

    # ---------------------------------------------------------
    # 5. PCA embed countries → 1D spatial coordinate x_o
    # ---------------------------------------------------------
    lon_lat = merged.set_index("Country_raw").loc[final_countries][["lon", "lat"]].to_numpy()
    pca = PCA(n_components=1)
    x_embed = pca.fit_transform(lon_lat).ravel()

    # Normalize [0,1] spatial coordinate
    x_min, x_max = x_embed.min(), x_embed.max()
    x_o = (x_embed - x_min) / (x_max - x_min)
    
    

    # ---------------------------------------------------------
    # 6. Normalize time and GDP
    # ---------------------------------------------------------
    t_o = years.astype(float)
    t_o_norm = (t_o - t_o.min()) / (t_o.max() - t_o.min())

    X_min, X_max = X.min(), X.max()
    X_norm = (X - X_min) / (X_max - X_min)
    print("X_norm ∈", (X_norm.min(), X_norm.max()))

    # ---------------------------------------------------------
    # 7. Run DEIM on normalized GDP
    # ---------------------------------------------------------
    deim_obj = DEIM(
        X=X_norm,
        n_d=n_d,
        t_o=t_o_norm,
        x_o=x_o,
        tolerance=tol,
        num_basis=num_basis
    )

    S_s, T_s, U_s = deim_obj.execute()

    # ---------------------------------------------------------
    # 8. Convert to torch tensors
    # ---------------------------------------------------------
    coords_np = np.hstack((T_s, S_s))  # [t, x]
    data_np = U_s.reshape(-1, 1)

    coords = torch.from_numpy(coords_np).float()
    data = torch.from_numpy(data_np).float()

    return coords, data



coords_gdp, data_gdp = create_data_gdp()




num_of_samples = 10000


import time

start_time = time.time()

x_t, u = create_data_gdp()

end_time = time.time()
print(f"Execution time: {end_time - start_time:.6f} seconds")


#########################
#########################
#########################

dataset = Dataset(
    create_data_gdp,
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






train_dataloader, test_dataloader = get_train_test_loader(
    dataset, train_test_split = 1.00)

##########################
##########################

poly_order = 3
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

foldername = "./data/deepymod/population/"



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
    max_iterations=25000,
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



#############################################
###########################################################
###########################################################














x_o = torch.linspace(-8, 8, 100)
t_o = torch.linspace(0.5, 10.0, 100)
v = 0.1
A = 1.0
coords_org , Exact = burgers_delta_org( x_o, t_o, v, A)
im = axs[2].scatter(coords_org[:,0], coords_org[:,1], c=Exact[:,:], marker="x", s=10)
axs[2].set_xlabel(r'$t$')
axs[2].set_ylabel(r'$x$', labelpad=0)
axs[2].set_ylim([-8,8])
fig.colorbar(mappable=im)






































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
























