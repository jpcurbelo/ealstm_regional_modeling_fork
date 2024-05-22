import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict

from papercode.metrics import *



JOB_DIR = 'run_2703_1441_seed111'

# Set of evaluation functions
# NSE,Alpha-NSE,Beta-NSE,FHV,FMS,FLV,KGE,Beta-KGE,Pearson-r
EVAL_FUNCS = {
    'NSE': calc_nse, 
    'Alpha_NSE': calc_alpha_nse, 
    'Beta_NSE': calc_beta_nse,
    'FHV': calc_fdc_fhv,  
    'FMS': calc_fdc_fms,
    'FLV': calc_fdc_flv,
    'KGE': calc_kge,
    'Beta_KGE': calc_beta_kge,
    'Pearson-r': calc_pearson_r
}


def main():

    # Get the current (this file) directory
    current_dir = Path(__file__).resolve().parent
    
    # Load results from the run directory
    results, cfg = load_results(current_dir / 'runs' / JOB_DIR)

    # Create DataFrames for observed and simulated discharges
    df_qobs, df_qsim = create_dataframes(results)

def load_results(run_dir: Path) -> pd.DataFrame:
    """Load results from a pickle file.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory.

    Returns
    -------
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge, or None if the file does not exist.
    """
    config = load_config(run_dir)
    
    if config["no_static"]:
        file_name = run_dir / f"lstm_no_static_seed{config['seed']}.p"
    else:
        if config["concat_static"]:
            file_name = run_dir / f"lstm_seed{config['seed']}.p"
        else:
            file_name = run_dir / f"ealstm_seed{config['seed']}.p"

    if not file_name.exists():
        print(f"File {file_name} does not exist.")
        return None

    with file_name.open('rb') as fp:
        results = pickle.load(fp)

    print(f"Successfully loaded results from {file_name}")
    return results, config

def load_config(run_dir: Path) -> Dict:
    """Load configuration from the cfg.json file in the run directory.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory.

    Returns
    -------
    config : Dict
        Dictionary containing the configuration.
    """
    config_path = run_dir / "cfg.json"
    with config_path.open('r') as fp:
        config = json.load(fp)
    return config

def create_dataframes(results: Dict) -> (pd.DataFrame, pd.DataFrame):
    """Create separate DataFrames for observed and simulated discharges.

    Parameters
    ----------
    results : Dict
        Dictionary containing DataFrames for observed and simulated discharges.

    Returns
    -------
    df_qobs : pd.DataFrame
        DataFrame containing observed discharges.
    df_qsim : pd.DataFrame
        DataFrame containing simulated discharges.
    """
    df_qobs_list = []
    df_qsim_list = []

    for station, data in results.items():
        df_qobs_list.append(data[['qobs']].rename(columns={'qobs': station}))
        df_qsim_list.append(data[['qsim']].rename(columns={'qsim': station}))

    df_qobs = pd.concat(df_qobs_list, axis=1)
    df_qsim = pd.concat(df_qsim_list, axis=1)

    return df_qobs, df_qsim



if __name__ == "__main__":

    main()
