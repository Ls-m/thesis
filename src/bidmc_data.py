import scipy.io
import pandas as pd
import numpy as np
import os

def extract_1d_signal(signal):
    """
    Convert a nested MATLAB signal array to a 1D NumPy array.
    Handles squeeze and flattening safely.
    """
    return np.array(signal).squeeze().flatten()

def convert_bidmc_to_csv(data, output_folder):
    """
    Converts each subject in BIDMC dataset (.mat) to a CSV file
    with columns 'PPG' and 'NASAL CANULA' and a sampling rate row.
    """
    for i in range(53):
        try:
            ppg = extract_1d_signal(data['data'][0][i]['ppg'][0][0][0])
            co2 = extract_1d_signal(data['data'][0][i]['ref']['resp_sig'][0][0][0][0][0][0][0][0])
            df = pd.DataFrame({
                'PPG': ppg,
                'NASAL CANULA': co2
            })

            # Optional: add a sampling rate row at the top
            sampling_rate = 125  # BIDMC sampling rate
            df_with_rate = pd.concat([
                pd.DataFrame([{'PPG': sampling_rate, 'NASAL CANULA': sampling_rate}]),
                df
            ], ignore_index=True)

            # Save to CSV
            filename = f'subject_{i:02d}.csv'
            output_path = os.path.join(output_folder, filename)
            df_with_rate.to_csv(output_path, index=False)

            print(f"Saved {filename} ({df.shape[0]} samples)")

        except Exception as e:
            print(f"Failed to process subject {i}: {e}")


    

            
# === Usage ===
if __name__ == '__main__':
    mat_file_path = 'bidmc_data.mat'  
    data = scipy.io.loadmat(mat_file_path)
    output_csv_folder = 'bidmc'  # Will be created if it doesn't exist
    convert_bidmc_to_csv(data, output_csv_folder)
