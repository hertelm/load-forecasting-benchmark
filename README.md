# Electrical Load Forecasting Benchmark

This is the repository for our upcoming publication on benchmarking electrical load forecasting models.

## Installation

Create a new virtual environment with Python 3.11. The requirements are installed using uv:

```
pip install uv
uv sync
```

If a specific torch/CUDA version is needed, install it like this:

```
 pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 
```

## Data

### Electricity (UCI_ELD) and FeederBW

The Electricity and FeederBW datasets can be downloaded from this URL: https://bwsyncandshare.kit.edu/s/XPzsTpWtHPPdGSQ

The original data sources are:
- Electricity: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 (published under a CC BY 4.0 license), with weather data from the Copernicus ERA5 reanalysis model (https://cds.climate.copernicus.eu/datasets/sis-energy-derived-reanalysis, published under a CC BY 4.0 license).
- FeederBW: https://zenodo.org/records/17831177 (published under a CC BY 4.0 license).

Run the following commands to preprocess the datasets. This will create pickle files with the preprocessed datasets.

```
python preprocess_feeder_bw.py
python preprocess_uci_eld.py
```

### TransnetBW

Download the TransnetBW load data as annually CSV files from the ENTSO-E Transparency platform (https://transparency.entsoe.eu/load/total/dayAhead)
The ERA5 weather data can be downloaded from this URL: https://bwsyncandshare.kit.edu/s/XPzsTpWtHPPdGSQ

Run the following commands to preprocess the data and create a ground-truth file:

```
python preprocess_transnet_bw_load.py
python create_transnet_bw_dataset.py
python create_ground_truth_transnet.py
```

## Results

Predictions from the best-performing models can be downloaded from this URL for comparisons with future approaches: https://bwsyncandshare.kit.edu/s/XPzsTpWtHPPdGSQ

## Example run

An experiment is started with the following command:

```
python main.py
```

All settings are controlled via config files in the 'config' folder and can be overwritten by command line arguments.
Use the argument `--help` to list all parameters.
Results are logged using the weights and biases platform.