"""
This script runs DARTS simulations for CO2 injection scenarios including geomechanics response

It sets up and executes multiple realizations of a DARTS simulation model
for CO2 injection, using provided permeability data.
The script handles:

- Setting up simulation parameters
- Running multiple realizations with different injection rate profiles
- Extracting and saving observations at specified monitoring points
- Printing summary information for each realization

The script uses functions from runnable_darts_geomech.py to set up and run
the simulations. It demonstrates how to use the DARTS framework for
coupled flow and geomechanical simulations in a CO2 storage context.

Usage:
    Import this script and call the run_simulations function with the required parameters.

"""

import os
import sys
import numpy as np
os.environ['OMP_NUM_THREADS'] = '26'
# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
darts_geomech_dir = os.path.join(current_dir, 'darts_geomech_runnable')
sys.path.append(darts_geomech_dir)
darts_geomech_path = '/samoa/data/smrserraoseabr/data-assimilation-with-generative-ML/src/data_assimilation_with_generative_ML/darts_geomech_runnable'
sys.path.insert(0, darts_geomech_path)

from runnable_darts_geomech import run_model, convert_mt_per_year_to_kmol_per_day

def run_simulations(permeability_array, n_realizations=2, case='64x64x1', output_dir='runs'):
    np.random.seed(0)
    physics_case = 'ccus'
    geomech_mode = 'surface'
    dt = 30
    n_time_steps_ = 12 * 1  # 1 year

    rate_inj_ccus = convert_mt_per_year_to_kmol_per_day(0.5)
    bhp_inj_ccus = 113.08 * 2

    monitoring_points = {
        'U_z': [
            {'time': -1, 'X': slice(None), 'Y': 0, 'Z': 0},
            {'time': slice(None), 'X': 0, 'Y': 0, 'Z': 0}
        ],
        'BHP': [
            {'time': slice(None)},
        ]
    }

    for realization in range(n_realizations):
        rate_inj_ccus_array_multipliers = [1] + [np.random.uniform(0, 3) for _ in range(10)]
        print('Rate multipliers:')
        print(rate_inj_ccus_array_multipliers)
        
        case_output_dir = os.path.join(output_dir, f'{physics_case}_{case}')
        os.makedirs(case_output_dir, exist_ok=True)
        case_name = f'DARTS_simulation_realization_{realization}'
        
        perm = permeability_array[realization].flatten()
        
        m, observations = run_model(
            case=case,
            case_name=case_name,
            physics_case=physics_case,
            geomech_mode=geomech_mode,
            geomech_period=1,
            permx=perm,
            dir_out=case_output_dir,
            update_rates=True,
            rate_inj_ccus=rate_inj_ccus,
            rate_inj_ccus_array_multipliers=rate_inj_ccus_array_multipliers,
            bhp_inj_ccus=bhp_inj_ccus,  # this is only the BHP constraint for the injection well - controls are actually by rate
            dt=dt,
            n_time_steps_=n_time_steps_,
            monitoring_points=monitoring_points,
            save_observations=True
        )
        
        print(f"Observations for realization {realization}:")
        for var_name, var_data in observations.items():
            print(f"  {var_name}:")
            print("    Dictionary storage:")
            for point_key, point_data in var_data['dict'].items():
                print(f"      {point_key}:")
                print(f"        Shape: {point_data['data'].shape}")
                print(f"        Coordinates: {point_data['coords']}")
            print("    List of numpy arrays:")
            for i, arr in enumerate(var_data['numpy']):
                print(f"      Point {i} shape: {arr.shape}")
        
        del m

if __name__ == "__main__":
    # Example usage
    # Generate a dummy permeability array for demonstration
    # In practice, you would provide your actual permeability data
    dummy_perm = np.random.lognormal(mean=0, sigma=1, size=(2, 64, 64, 1))
    run_simulations(dummy_perm, n_realizations=2, case='64x64x1', output_dir='runs')