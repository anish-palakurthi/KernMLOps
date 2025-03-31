import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_single_data_source(file_path, source_name):
    """
    Load a single parquet file and return it as a DataFrame
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None

    print(f"Loading {source_name} data from: {file_path}")
    table = pq.read_table(file_path)
    df = table.to_pandas()
    print(f"{source_name} data shape: {df.shape}")
    print(f"{source_name} columns: {df.columns.tolist()}")

    # Convert object columns to numeric if possible
    # for col in df.columns:
    #     if df[col].dtype == 'object':
    #         try:
    #             df[col] = pd.to_numeric(df[col], errors='coerce')
    #             print(f"Converted column {col} from object to numeric")
    #         except:
    #             print(f"Could not convert column {col} to numeric")

    return df

def process_rss_data(rss_df):
    """
    Process RSS data for modeling
    """
    # Convert timestamp from ns to sec for easier handling
    if 'ts_ns' in rss_df.columns:
        rss_df['time_sec'] = rss_df['ts_ns'] / 1e9
        print("Converted timestamp from ns to sec")

    # Sort by timestamp to ensure we can find most recent values later
    rss_df = rss_df.sort_values('time_sec')

    # Identify potential target columns for RSS data
    rss_cols = []
    for col in ['anon', 'file', 'swap', 'shmem']:
        if col in rss_df.columns:
            # Check if it has variance
            if rss_df[col].std() > 0:
                rss_cols.append(col)
                print(f"Found RSS column with variance: {col}")

    # Select target column - anon is often the most interesting
    target_col = None
    target_candidates = ['anon', 'file', 'shmem', 'swap']

    for col in target_candidates:
        if col in rss_cols:
            target_col = col
            break

    if target_col is None and rss_cols:
        target_col = rss_cols[0]

    if target_col:
        print(f"Selected target column: {target_col}")
    else:
        print("Warning: No suitable target column found with variance")
        return rss_df, None, []

    # Feature columns are the other RSS metrics
    feature_cols = [col for col in rss_cols if col != target_col]

    # Convert to MB if values are large (likely in bytes)
    # Check first value to determine scale
    for col in rss_cols:
        mean_val = rss_df[col].mean()
        if mean_val > 10000:  # Likely in bytes or KB
            if mean_val > 1000000:  # Likely bytes
                rss_df[f"{col}_mb"] = rss_df[col] / (1024 * 1024)
                print(f"Converted {col} from bytes to MB")
                # Update target and feature columns
                if col == target_col:
                    target_col = f"{col}_mb"
                if col in feature_cols:
                    feature_cols.remove(col)
                    feature_cols.append(f"{col}_mb")
            else:  # Likely KB
                rss_df[f"{col}_mb"] = rss_df[col] / 1024
                print(f"Converted {col} from KB to MB")
                # Update target and feature columns
                if col == target_col:
                    target_col = f"{col}_mb"
                if col in feature_cols:
                    feature_cols.remove(col)
                    feature_cols.append(f"{col}_mb")

    return rss_df, target_col, feature_cols

def process_tlb_data(tlb_df, tlb_type):
    """
    Process TLB data for modeling
    """
    # Convert timestamp to seconds if needed
    if 'ts_uptime_us' in tlb_df.columns:
        tlb_df['time_sec'] = tlb_df['ts_uptime_us'] / 1e6
        print(f"Converted {tlb_type} timestamp from us to sec")

    # Sort by timestamp
    tlb_df = tlb_df.sort_values('time_sec')

    # Identify TLB miss columns
    tlb_cols = []
    for col in tlb_df.columns:
        col_lower = col.lower()
        if 'miss' in col_lower:
            # Check if it has some variance
            if tlb_df[col].std() > 0:
                tlb_cols.append(col)

    print(f"Found {len(tlb_cols)} {tlb_type} columns with variance")

    # Choose main TLB miss column (prefer cumulative/total)
    main_tlb_col = None
    for col in tlb_cols:
        col_lower = col.lower()
        if 'cumulative' in col_lower or 'total' in col_lower:
            main_tlb_col = col
            break

    # If no specific column found, use the first one
    if main_tlb_col is None and tlb_cols:
        main_tlb_col = tlb_cols[0]

    if main_tlb_col:
        print(f"Selected {tlb_type} column: {main_tlb_col}")
        # Create clearly named column
        tlb_df[f'{tlb_type}_misses'] = tlb_df[main_tlb_col]
        return tlb_df, [f'{tlb_type}_misses']
    else:
        print(f"Warning: No suitable {tlb_type} column found")
        return tlb_df, []

def efficient_data_integration(rss_df, dtlb_df, itlb_df):
    """
    Uses TLB timestamp as reference, finds most recent RSS value at each TLB timestamp
    """
    print("Using TLB-centric data integration (more efficient approach)...")

    # Decide which TLB dataframe to use as reference (prefer one with more data)
    reference_df = None
    reference_name = None

    if dtlb_df is not None and (itlb_df is None or len(dtlb_df) > len(itlb_df)):
        reference_df = dtlb_df
        reference_name = "DTLB"
    elif itlb_df is not None:
        reference_df = itlb_df
        reference_name = "ITLB"
    else:
        print("No TLB data available")
        return None

    print(f"Using {reference_name} as reference with {len(reference_df)} timestamps")

    # Create a function to find the most recent RSS value for a given timestamp
    def find_most_recent_rss(timestamp):
        mask = rss_df['time_sec'] <= timestamp
        if not mask.any():
            return None  # No RSS data before this timestamp

        most_recent_idx = rss_df[mask]['time_sec'].idxmax()
        return rss_df.loc[most_recent_idx]

    # Create integrated dataframe
    integrated_data = []
    skip_count = 0

    print(f"Building integrated dataset from {len(reference_df)} reference points...")

    # Sample at regular intervals if there are too many rows (for performance)
    if len(reference_df) > 10000:
        sample_size = 10000
        step = len(reference_df) // sample_size
        print(f"Sampling every {step}th row from reference dataframe")
        reference_rows = reference_df.iloc[::step]
    else:
        reference_rows = reference_df

    for idx, tlb_row in reference_rows.iterrows():
        # Find most recent RSS data
        rss_row = find_most_recent_rss(tlb_row['time_sec'])

        if rss_row is None:
            skip_count += 1
            continue

        # Create integrated row
        row_data = {'time_sec': tlb_row['time_sec']}

        # Add TLB data
        for col in reference_df.columns:
            if col != 'time_sec' and col != 'ts_uptime_us':
                row_data[f'{reference_name.lower()}_{col}'] = tlb_row[col]

        # Add the other TLB data if available
        if reference_name == "DTLB" and itlb_df is not None:
            # Find closest ITLB timestamp
            closest_itlb_idx = (itlb_df['time_sec'] - tlb_row['time_sec']).abs().idxmin()
            itlb_row = itlb_df.loc[closest_itlb_idx]
            time_diff = abs(itlb_row['time_sec'] - tlb_row['time_sec'])

            # Only add if reasonably close (within 1 second)
            if time_diff < 1.0:
                for col in itlb_df.columns:
                    if col != 'time_sec' and col != 'ts_uptime_us':
                        row_data[f'itlb_{col}'] = itlb_row[col]

        elif reference_name == "ITLB" and dtlb_df is not None:
            # Find closest DTLB timestamp
            closest_dtlb_idx = (dtlb_df['time_sec'] - tlb_row['time_sec']).abs().idxmin()
            dtlb_row = dtlb_df.loc[closest_dtlb_idx]
            time_diff = abs(dtlb_row['time_sec'] - tlb_row['time_sec'])

            # Only add if reasonably close (within 1 second)
            if time_diff < 1.0:
                for col in dtlb_df.columns:
                    if col != 'time_sec' and col != 'ts_uptime_us':
                        row_data[f'dtlb_{col}'] = dtlb_row[col]

        # Add RSS data
        for col in rss_df.columns:
            if col != 'time_sec' and col != 'ts_ns':
                row_data[f'rss_{col}'] = rss_row[col]

        integrated_data.append(row_data)

    print(f"Skipped {skip_count} rows with no matching RSS data")

    # Create dataframe
    integrated_df = pd.DataFrame(integrated_data)
    print(f"Created integrated dataframe with {len(integrated_df)} rows and {len(integrated_df.columns)} columns")

    return integrated_df

def prepare_sequences(df, target_col, feature_cols, seq_length=5):
    """
    Prepare sequences for LSTM model
    """
    # Only use numeric columns
    valid_feature_cols = []
    for col in feature_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # Check if column has variance
            if df[col].std() > 0:
                valid_feature_cols.append(col)
            else:
                print(f"Skipping constant column: {col}")

    print(f"Using {len(valid_feature_cols)} features for sequence creation")
    if valid_feature_cols:
        print("Sample features:", valid_feature_cols[:min(5, len(valid_feature_cols))])

    # Scale data
    scalers = {}
    scaled_data = {}

    for col in valid_feature_cols + [target_col]:
        try:
            scaler = MinMaxScaler()
            scaled_data[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            scalers[col] = scaler
        except Exception as e:
            print(f"Error scaling {col}: {e}")
            if col in valid_feature_cols:
                valid_feature_cols.remove(col)

    # Create sequences
    X, y = [], []
    for i in range(len(df) - seq_length):
        features_seq = []
        for col in valid_feature_cols:
            features_seq.append(scaled_data[col][i:i+seq_length])

        X.append(np.hstack(features_seq))
        y.append(scaled_data[target_col][i+seq_length])

    X, y = np.array(X), np.array(y)

    # Reshape for LSTM [samples, timesteps, features]
    X = X.reshape((X.shape[0], seq_length, len(valid_feature_cols)))

    print(f"Created {len(X)} sequences with shape {X.shape}")

    return X, y, scalers, valid_feature_cols

def build_lstm_model(input_shape):
    """
    Build an LSTM model with proper error handling for input shapes

    Args:
        input_shape: Tuple representing input shape (can be 2D or 3D)

    Returns:
        Compiled Keras model
    """
    print(f"Building model with input shape: {input_shape}")

    # Determine feature count based on input shape
    if len(input_shape) == 3:
        # Input shape is (batch, seq_length, features)
        seq_length = input_shape[1]
        n_features = input_shape[2]
        model_input_shape = (seq_length, n_features)
    elif len(input_shape) == 2:
        # Input shape is (seq_length, features)
        seq_length = input_shape[0]
        n_features = input_shape[1]
        model_input_shape = input_shape
    else:
        raise ValueError(f"Unexpected input shape: {input_shape}")

    # Set unit count based on feature count
    unit_count = min(32, max(8, n_features * 2))
    print(f"Using {unit_count} LSTM units for {n_features} features")


    model = Sequential([
        LSTM(unit_count, activation='relu', input_shape=model_input_shape),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

def main():
    """
    Main function to run the RSS-TLB model
    """
    try:
        # Define file paths
        rss_file = "data/curated/mm_rss_stat/5b943464-162d-41d0-a841-4a89a360daf3.gap.parquet"
        dtlb_file = "data/curated/dtlb_misses/5b943464-162d-41d0-a841-4a89a360daf3.gap.parquet"
        itlb_file = "data/curated/itlb_misses/5b943464-162d-41d0-a841-4a89a360daf3.gap.parquet"

        # 1. Load data sources
        rss_df = load_single_data_source(rss_file, "RSS")
        dtlb_df = load_single_data_source(dtlb_file, "DTLB")
        itlb_df = load_single_data_source(itlb_file, "ITLB")

        if rss_df is None:
            raise ValueError("RSS data is required but not found")

        # 2. Process each data source
        rss_df, target_col, rss_features = process_rss_data(rss_df)

        if target_col is None:
            raise ValueError("No suitable target column found in RSS data")

        tlb_features = []
        if dtlb_df is not None:
            dtlb_df, dtlb_features = process_tlb_data(dtlb_df, "dtlb")
            tlb_features.extend(dtlb_features)

        if itlb_df is not None:
            itlb_df, itlb_features = process_tlb_data(itlb_df, "itlb")
            tlb_features.extend(itlb_features)

        # 3. Integration approach: TLB-centric with most recent RSS
        integrated_df = efficient_data_integration(rss_df, dtlb_df, itlb_df)

        if integrated_df is None or len(integrated_df) < 10:
            raise ValueError("Failed to create sufficient integrated dataset")

        # 4. Update column names for target and features
        target_col = f"rss_{target_col}"
        rss_features = [f"rss_{col}" for col in rss_features]

        # 5. Verify target column exists in integrated data
        if target_col not in integrated_df.columns:
            # Try to find an alternative
            for col in integrated_df.columns:
                if 'rss' in col and ('anon' in col or 'file' in col):
                    if integrated_df[col].std() > 0:
                        target_col = col
                        break

        if not target_col or target_col not in integrated_df.columns:
            raise ValueError("Target column not found in integrated data")

        print(f"Final target column: {target_col}")

        # 6. Prepare feature list
        all_features = []

        # Add RSS features
        for col in integrated_df.columns:
            if 'rss_' in col and col != target_col:
                all_features.append(col)

        # Add TLB features
        for col in integrated_df.columns:
            if 'dtlb_' in col or 'itlb_' in col:
                all_features.append(col)

        print(f"Total feature count: {len(all_features)}")

        # 7. Prepare sequences for LSTM
        seq_length = min(5, len(integrated_df) // 3)
        print(f"Using sequence length of {seq_length}")

        X, y, scalers, valid_features = prepare_sequences(
            integrated_df, target_col, all_features, seq_length=seq_length
        )

        # Skip model training if we don't have enough data
        if len(X) < 10:
            print(f"Not enough data for training (only {len(X)} sequences)")
            return

        # 8. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # 9. Build and train model
        model = build_lstm_model((X.shape[1], X.shape[2]))

        # Use fewer epochs for small datasets
        epochs = min(30, max(10, len(X_train) // 2))
        batch_size = min(8, len(X_train) // 2)

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # 10. Evaluate model
        y_pred = model.predict(X_test)

        # Calculate metrics on scaled data
        mse_scaled = np.mean(np.square(y_test - y_pred))
        rmse_scaled = np.sqrt(mse_scaled)

        print(f"Test MSE (scaled): {mse_scaled:.6f}")
        print(f"Test RMSE (scaled): {rmse_scaled:.6f}")

        # Convert to original scale
        y_test_orig = scalers[target_col].inverse_transform(y_test)
        y_pred_orig = scalers[target_col].inverse_transform(y_pred)

        # Calculate metrics on original scale
        mse = np.mean(np.square(y_test_orig - y_pred_orig))
        rmse = np.sqrt(mse)

        print(f"Test MSE: {mse:.2f}")
        print(f"Test RMSE: {rmse:.2f}")

        # 11. Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_orig, label='Actual', marker='o', markersize=4)
        plt.plot(y_pred_orig, label='Predicted', marker='x', markersize=4)
        plt.title('PageRank RSS Prediction with TLB Misses')
        plt.xlabel('Time Steps')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('rss_tlb_predictions.png')

        # Plot training history
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('rss_tlb_history.png')

        # Save the model
        model.save('rss_tlb_model.h5')
        print("Model saved as rss_tlb_model.h5")

        # 12. Print summary with feature importance
        print("\nFinal Results:")
        print(f"Target: {target_col}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")

        # List TLB and RSS features used
        tlb_features_used = [f for f in valid_features if 'tlb' in f.lower()]
        if tlb_features_used:
            print("\nTLB features included in the model:")
            for feat in tlb_features_used:
                print(f"  - {feat}")

        # Print RSS features as well
        rss_features_used = [f for f in valid_features if 'rss' in f.lower()]
        if rss_features_used:
            print("\nRSS features included in the model:")
            for feat in rss_features_used:
                print(f"  - {feat}")

        return model

    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
