import os
import scipy.io
import mat73
import pandas as pd
import numpy as np
# Paths
input_dir = 'mat'              # Folder containing .mat files
output_dir = 'capno'           # Folder to store output CSVs
os.makedirs(output_dir, exist_ok=True)

# Helper function to load .mat (v7.3 or v7.3+)
def load_mat(filepath):
    try:
        return scipy.io.loadmat(filepath)
    except NotImplementedError:
        return mat73.loadmat(filepath)

# Loop through all .mat files
for filename in os.listdir(input_dir):
    if filename.endswith('.mat'):
        filepath = os.path.join(input_dir, filename)
        data = load_mat(filepath)
        params = data['param']
        reference = data['reference']
        print(reference)
        print("len reference is, ", reference['rr']['co2']['y'].size)
        h = reference['rr']['co2']['y']
        print(h)
        print("len ppg is, ",np.array(data['signal']['pleth']['y']).flatten().size)
        print(reference['rr']['co2']['x'])
        print("*********")
        print(data['signal']['pleth'])
        print(params)
        exit()
        print(f"{filename} — keys: {data.keys()}")

        signals = data['signal']
        print(f"signal keys: {signals.keys()}")
        print(f"pleth type: {type(signals['pleth'])}, co2 type: {type(signals['co2'])}")
        print(signals['pleth'].keys())
        print(signals['co2'].keys())

        print(f"Loaded: {filename}, Keys: {list(data.keys())}")

        try:
            signals = data['signal']
            
            ppg_raw = signals['pleth']
            co2_raw = signals['co2']

            # Convert to NumPy arrays if needed
            ppg = np.array(signals['pleth']['y']).flatten()
            co2 = np.array(signals['co2']['y']).flatten()

            # Check if both are now 1D and have length
            if ppg.size == 0 or co2.size == 0:
                raise ValueError(f"Empty signal(s): PPG shape = {ppg.shape}, CO2 shape = {co2.shape}")
            if ppg.ndim != 1 or co2.ndim != 1:
                raise ValueError(f"Unexpected dimensions: PPG={ppg.shape}, CO2={co2.shape}")

            # Extract sampling rates
            fs_ppg = float(params['samplingrate']['pleth'])
            fs_co2 = float(params['samplingrate']['co2'])



            # Truncate to match lengths (if needed)
            min_len = min(len(ppg), len(co2))
            if len(ppg)!= len(co2):
                print("not equal")
            ppg = ppg[:min_len]
            co2 = co2[:min_len]

            # Get sampling rate
            

            # Prepare output
            data_matrix = np.column_stack((ppg, co2))
            df = pd.DataFrame(data_matrix, columns=['PPG', 'NASAL CANULA'])

            # Write file
            out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.csv')
            with open(out_path, 'w') as f:
                f.write('PPG,NASAL CANULA\n')
                f.write(f'{fs_ppg},{fs_co2}\n')
                df.to_csv(f, index=False, header=False)

            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
