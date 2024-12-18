# prediction script for Model-M (normalizes the wav files in the folder/DB)
import os
import pandas as pd
import torch
from datasets import ASTVal
from models import ASTXL
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import utils
import argparse
from datetime import datetime
from transformers import ASTFeatureExtractor


torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the prediction script.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--data_file", type=str, default=None, help="Path to the optional data file")
    args = parser.parse_args()

    output_dir = args.output_dir
    data_dir = args.data_dir
    data_file = args.data_file

    # Debug prints (optional)
    print(f"Output Directory: {output_dir}")
    print(f"Data Directory: {data_dir}")
    print(f"Data File: {data_file}")

    # Set this according to your hardware
    bs = 12
    num_workers = 4
    device = 'cpu'

    db_name = os.path.basename(data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nPrediction begins...")
    print(f"Using device: {device}")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    dtype_dict = {
        'db': str,
        'file_path': str,
        'file_num': float,
        'num_votes': float,
        'con_num': float,
        'con_description': str,
        'ref_file_path': str,
        'mos': float,
        'noi': float,
        'dis': float,
        'col': float,
        'loud': float,
        'mos_std': float,
        'noi_std': float,
        'dis_std': float,
        'col_std': float,
        'loud_std': float,
        'db_mean': float,
        'db_std': float
    }

    # Does data file already exist?
    if data_file:
        # Load the data file
        df = pd.read_csv(data_file, dtype=dtype_dict)
        print(f"\nData file found and loaded: {data_file}.")

        # Check for valid and invalid file paths in the data file
        df['file_exists'] = df['file_path'].apply(lambda path: os.path.exists(path) and path.lower().endswith('.wav'))

        # Separate valid and invalid paths
        xdf = df[df['file_exists']]  # Valid paths
        ydf = df[~df['file_exists']]  # Invalid paths

        print(f"{len(xdf)} valid .wav files listed in data file.")
        if not ydf.empty:
            print(f"{len(ydf)} invalid .wav files listed in data file:")
            print(ydf['file_path'].to_list())

        # Check for unlisted .wav files in the data directory
        data_dir = os.path.dirname(data_file)  # Assuming data_file resides in data_dir
        all_wav_files = {os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.wav')}
        listed_files = set(df['file_path'])

        # Find unlisted .wav files
        unlisted_files = all_wav_files - listed_files
        if unlisted_files:
            print(f"Found {len(unlisted_files)} unlisted .wav files in the data directory:")
            print(list(unlisted_files))
        else:
            print("No unlisted .wav files in the data directory.")

    else:
        df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtype_dict.items()})
        file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]

        df['file_path'] = file_paths
        df['db'] = db_name
        df['file_num'] = range(1, len(file_paths) + 1)

        output_file = os.path.join(output_dir, f"{db_name}_data_file.csv")
        df.to_csv(output_file, index=False)
        print(f"\nNo data file found. Created and saved new one at: {output_file}. {len(df)} wav files found.")

    if df['db_mean'].isnull().all():
        # Calculate database statistics and fill `db_mean` and `db_std`
        feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        feature_extractor.sampling_rate = 48000  # Set to 48 kHz for fullband 48k audio
        feature_extractor.max_length = 2000      # Estimate the longest possible audio is 20 seconds long
        feature_extractor.num_mel_bins = 128     # Keep original as default
        feature_extractor.do_normalize = False
        feature_extractor.mean = None
        feature_extractor.std = None

        db_features = []
        
        print("Calculating database feature statistics...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
            file_path = row['file_path']
            waveform, sr = utils.process_audio_file(file_path)
            waveform = waveform.squeeze(0)  # Remove extra channel dimension if necessary
            feature = feature_extractor(waveform, sampling_rate=48000, return_tensors="pt")["input_values"]

            # Identify valid frames by filtering out the padding (zero values across the frequency dimension)
            valid_frames = feature[feature.abs().sum(dim=-1) > 0]

            # Check valid_frames for NaN or extreme values
            if torch.isnan(valid_frames).any():
                print(f"Warning: NaN values found in valid frames. {db_name}, {file_path}")
        
            db_features.append(valid_frames)

        # Calculate mean and std for valid frames
        if db_features:
            try:
                valid_features = torch.cat(db_features)
                valid_mean = valid_features.mean().item()
                valid_std = valid_features.std().item()
                print(f"Database: {db_name}, Mean: {valid_mean}, Std: {valid_std}")
            except Exception as e:
                print(f"An error occurred during mean/std calculation for database {db_name}: {e}")
                valid_mean, valid_std = None, None
        else:
            print(f"No valid features for database: {db_name}")
            valid_mean, valid_std = None, None

        # Add db_mean and db_std columns to df
        df['db_mean'] = valid_mean
        df['db_std'] = valid_std
        df.to_csv(output_file, index=False)
        print(f"Database mean and standard deviation calculated and saved as data file: {output_file}")

    ds = ASTVal(df, data_dir)

    dl = DataLoader(
        dataset=ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers
    )

    # Load MOS model
    model_mos = ASTXL()
    mos_state_dict = torch.load("/Users/wafaa/Code/psamd/ast_models/ast_model_XL/weights/mos.pth", map_location=torch.device(device), weights_only=True)
    model_mos.load_state_dict(mos_state_dict)
    model_mos.eval()

    # Load NOI model
    model_noi = ASTXL()
    noi_state_dict = torch.load("/Users/wafaa/Code/psamd/ast_models/ast_model_XL/weights/noi.pth", map_location=torch.device(device), weights_only=True)
    model_noi.load_state_dict(noi_state_dict)
    model_noi.eval()

    # Load DIS model
    model_dis = ASTXL()
    dis_state_dict = torch.load("/Users/wafaa/Code/psamd/ast_models/ast_model_XL/weights/dis.pth", map_location=torch.device(device), weights_only=True)
    model_dis.load_state_dict(dis_state_dict)
    model_dis.eval()

    # Load COL model
    model_col = ASTXL()
    col_state_dict = torch.load("/Users/wafaa/Code/psamd/ast_models/ast_model_XL/weights/col.pth", map_location=torch.device(device), weights_only=True)
    model_col.load_state_dict(col_state_dict)
    model_col.eval()

    # Load LOUD model
    model_loud = ASTXL()
    loud_state_dict = torch.load("/Users/wafaa/Code/psamd/ast_models/ast_model_XL/weights/loud.pth", map_location=torch.device(device), weights_only=True)
    model_loud.load_state_dict(loud_state_dict)
    model_loud.eval()

    print(f"\nModel loaded") # don't need this, remove

    y_hat_val = torch.full((len(ds), 5), -0.25, device='cpu') # Stores the validation outputs, later filled into ds_val df

    total_files = len(ds.df)  # Total number of files
    processed_files = 0       # Counter for processed files 

    with torch.no_grad():  # Disable gradient tracking for inference
        print("\nCalculating quality scores...")
        for b, (index, batch_features) in enumerate(dl):

            batch_features = batch_features.float().to(device)

            # Forward pass ---------------------------------------
            mos_pred = model_mos(batch_features)
            noi_pred = model_noi(batch_features)
            dis_pred = model_dis(batch_features)
            col_pred = model_col(batch_features)
            loud_pred = model_loud(batch_features)
            
            # Stack predictions for each dimension
            y_hat_batch = torch.stack([mos_pred, noi_pred, dis_pred, col_pred, loud_pred], dim=1).squeeze().to('cpu')
            y_hat_val[index, :] = y_hat_batch

            # Iterate through current batch to print scores with file paths
            for idx, scores in zip(index, y_hat_batch):
                idx = int(idx)  # Convert PyTorch tensor to native Python integer
                file_path = ds.df.loc[idx, 'file_path']  # Retrieve file path using index

                processed_files += 1
                # Descale predictions for display
                descaled_scores = scores * 4 + 1
                print(f"({processed_files}/{total_files}) {os.path.basename(file_path)} | MOS: {descaled_scores[0]:.2f}, "
                    f"NOI: {descaled_scores[1]:.2f}, DIS: {descaled_scores[2]:.2f}, "
                    f"COL: {descaled_scores[3]:.2f}, LOUD: {descaled_scores[4]:.2f}")

    # Scale predictions once all batches are processed
    y_hat_val_descaled = y_hat_val * 4 + 1 # On CPU
    y_hat_val_descaled = y_hat_val_descaled.detach().numpy() # On CPU

    # Convert predictions into DataFrame columns on CPU
    ds.df['mos_pred'] = y_hat_val_descaled[:, 0]
    ds.df['noi_pred'] = y_hat_val_descaled[:, 1]
    ds.df['dis_pred'] = y_hat_val_descaled[:, 2]
    ds.df['col_pred'] = y_hat_val_descaled[:, 3]
    ds.df['loud_pred'] = y_hat_val_descaled[:, 4]

    filtered_val_df = ds.df.loc[
        (ds.df['mos_pred'] != 0.0) &
        (ds.df['noi_pred'] != 0.0) &
        (ds.df['dis_pred'] != 0.0) &
        (ds.df['col_pred'] != 0.0) &
        (ds.df['loud_pred'] != 0.0)
    ]

    filtered_val_df.to_csv(os.path.join(output_dir, db_name + '_prediction_per_file_' + current_time + '.csv'), index=False)  
    print("Saved predicted scores:", os.path.join(output_dir, db_name + '_prediction_per_file_' + current_time + '.csv'))
