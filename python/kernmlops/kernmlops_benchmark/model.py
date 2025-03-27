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

    # Convert object columns to numeric if possible
    # for col in df.columns:
        # if df[col].dtype == 'object':
            # try:
            #     df[col] = pd.to_numeric(df[col], errors='coerce')
            #     print(f"Converted column {col} from object to numeric")
            # except:
            #     print(f"Could not convert column {col} to numeric")

    return df

def process_memory_data(mem_df):
    """
    Process memory data for modeling
    """
    # Convert memory columns to MB
    memory_cols = []
    for col in mem_df.columns:
        if 'bytes' in col:
            try:
                new_col = f"{col}_mb"
                mem_df[new_col] = mem_df[col] / (1024 * 1024)
                memory_cols.append(new_col)
            except Exception as e:
                print(f"Error converting {col} to MB: {e}")

    print(f"Created {len(memory_cols)} memory columns in MB")

    # Identify target column
    target_col = None
    target_candidates = [
        'mem_available_bytes_mb',
        'mem_free_bytes_mb',
        'cached_bytes_mb'
    ]

    for col in target_candidates:
        if col in mem_df.columns:
            # Check if it has some variance
            if mem_df[col].std() > 0:
                target_col = col
                break

    # If no suitable target found, use the first memory column with non-zero variance
    if target_col is None:
        for col in memory_cols:
            if mem_df[col].std() > 0:
                target_col = col
                break

    if target_col:
        print(f"Selected target column: {target_col}")
    else:
        print("Warning: No suitable target column found with variance")
        # Use the first memory column anyway
        if memory_cols:
            target_col = memory_cols[0]

    # Identify feature columns
    feature_cols = [col for col in memory_cols if col != target_col]

    return mem_df, target_col, feature_cols

def process_tlb_data(tlb_df, tlb_type):
    """
    Process TLB data for modeling
    """
    # Identify the TLB miss column
    tlb_cols = []
    for col in tlb_df.columns:
        col_lower = col.lower()
        if 'miss' in col_lower:
            # Check if it has some variance
            if tlb_df[col].std() > 0:
                tlb_cols.append(col)

    print(f"Found {len(tlb_cols)} {tlb_type} columns with variance")

    # Choose the main TLB miss column
    main_tlb_col = None
    for col in tlb_cols:
        if 'cumulative' in col.lower() or 'total' in col.lower():
            main_tlb_col = col
            break

    # If no specific column found, use the first one
    if main_tlb_col is None and tlb_cols:
        main_tlb_col = tlb_cols[0]

    if main_tlb_col:
        print(f"Selected {tlb_type} column: {main_tlb_col}")
        # Create a more clearly named column
        tlb_df[f'{tlb_type}_misses'] = tlb_df[main_tlb_col]
        return tlb_df, [f'{tlb_type}_misses']
    else:
        print(f"Warning: No suitable {tlb_type} column found")
        return tlb_df, []

def simple_time_align(mem_df, dtlb_df, itlb_df):
    """
    Simple time alignment based on nearest timestamps
    """
    # Ensure all dataframes have timestamps in seconds
    for df, name in [(mem_df, 'memory'), (dtlb_df, 'dtlb'), (itlb_df, 'itlb')]:
        if df is not None and 'ts_uptime_us' in df.columns:
            df['time_sec'] = df['ts_uptime_us'] / 1e6

    # Use memory data as the reference
    aligned_data = []

    # For each memory timestamp, find nearest TLB timestamps
    for idx, mem_row in mem_df.iterrows():
        row_data = {'time_sec': mem_row['time_sec']}

        # Add memory data
        for col in mem_df.columns:
            if col != 'time_sec' and col != 'ts_uptime_us':
                row_data[f'memory_{col}'] = mem_row[col]

        # Add DTLB data if available
        if dtlb_df is not None:
            # Find closest DTLB timestamp
            if 'time_sec' in dtlb_df.columns:
                nearest_idx = (dtlb_df['time_sec'] - mem_row['time_sec']).abs().idxmin()
                dtlb_row = dtlb_df.loc[nearest_idx]

                # Add DTLB columns
                for col in dtlb_df.columns:
                    if col != 'time_sec' and col != 'ts_uptime_us':
                        row_data[f'dtlb_{col}'] = dtlb_row[col]

        # Add ITLB data if available
        if itlb_df is not None:
            # Find closest ITLB timestamp
            if 'time_sec' in itlb_df.columns:
                nearest_idx = (itlb_df['time_sec'] - mem_row['time_sec']).abs().idxmin()
                itlb_row = itlb_df.loc[nearest_idx]

                # Add ITLB columns
                for col in itlb_df.columns:
                    if col != 'time_sec' and col != 'ts_uptime_us':
                        row_data[f'itlb_{col}'] = itlb_row[col]

        aligned_data.append(row_data)

    # Create aligned dataframe
    aligned_df = pd.DataFrame(aligned_data)
    print(f"Created aligned dataframe with {len(aligned_df)} rows and {len(aligned_df.columns)} columns")

    return aligned_df

def simple_lstm_model(input_shape):
    """
    Create a simple LSTM model suitable for small datasets
    """
    model = Sequential([
        LSTM(16, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

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

def main():
    """
    Main function to run the simplified model
    """
    try:
        # Define file paths
        memory_file = "data/curated/memory_usage/7023d8d9-b86c-4de1-804a-a96072c1a360.gap.parquet"
        dtlb_file = "data/curated/dtlb_misses/7023d8d9-b86c-4de1-804a-a96072c1a360.gap.parquet"
        itlb_file = "data/curated/itlb_misses/7023d8d9-b86c-4de1-804a-a96072c1a360.gap.parquet"

        # 1. Load data sources
        mem_df = load_single_data_source(memory_file, "memory")
        dtlb_df = load_single_data_source(dtlb_file, "DTLB")
        itlb_df = load_single_data_source(itlb_file, "ITLB")

        if mem_df is None:
            raise ValueError("Memory data is required but not found")

        # 2. Process each data source
        mem_df, target_col, memory_features = process_memory_data(mem_df)

        tlb_features = []
        if dtlb_df is not None:
            dtlb_df, dtlb_features = process_tlb_data(dtlb_df, "dtlb")
            tlb_features.extend(dtlb_features)

        if itlb_df is not None:
            itlb_df, itlb_features = process_tlb_data(itlb_df, "itlb")
            tlb_features.extend(itlb_features)

        # 3. Align data sources by time
        aligned_df = simple_time_align(mem_df, dtlb_df, itlb_df)

        # Update column names for target and features after alignment
        if target_col:
            target_col = f"memory_{target_col}"

        memory_features = [f"memory_{col}" for col in memory_features]

        # 4. Verify target column exists in aligned data
        if target_col not in aligned_df.columns:
            # Try to find an alternative
            for col in aligned_df.columns:
                if 'memory' in col and ('available' in col or 'free' in col or 'cached' in col):
                    if aligned_df[col].std() > 0:
                        target_col = col
                        break

        if not target_col or target_col not in aligned_df.columns:
            raise ValueError("No suitable target column found in aligned data")

        print(f"Final target column: {target_col}")

        # 5. Prepare feature list
        all_features = memory_features.copy()
        for col in aligned_df.columns:
            if 'dtlb' in col or 'itlb' in col:
                if aligned_df[col].std() > 0:  # Only include columns with variance
                    all_features.append(col)

        print(f"Final feature count: {len(all_features)}")
        print("Sample features:", all_features[:5])

        # 6. Prepare sequences for LSTM
        # Use a smaller sequence length for small datasets
        seq_length = min(5, len(aligned_df) // 3)
        print(f"Using sequence length of {seq_length}")

        X, y, scalers, valid_features = prepare_sequences(
            aligned_df, target_col, all_features, seq_length=seq_length
        )

        # Skip model training if we don't have enough data
        if len(X) < 10:
            print(f"Not enough data for training (only {len(X)} sequences)")
            return

        # 7. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # 8. Build and train model
        model = simple_lstm_model((X.shape[1], X.shape[2]))

        # Use fewer epochs for small datasets
        epochs = min(20, len(X_train))
        batch_size = min(4, len(X_train))

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # 9. Evaluate model
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

        # 10. Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_orig, label='Actual', marker='o', markersize=4)
        plt.plot(y_pred_orig, label='Predicted', marker='x', markersize=4)
        plt.title('PageRank Memory Prediction with TLB Misses')
        plt.xlabel('Time Steps')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('simple_model_predictions.png')

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
        plt.savefig('simple_model_history.png')

        # Save the model
        model.save('simple_pagerank_model.h5')
        print("Model saved as simple_pagerank_model.h5")

        # 11. Print summary with feature importance
        print("\nFinal Results:")
        print(f"Target: {target_col}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")

        # Check if TLB features were included
        tlb_features_used = [f for f in valid_features if 'tlb' in f.lower()]
        if tlb_features_used:
            print("\nTLB features included in the model:")
            for feat in tlb_features_used:
                print(f"  - {feat}")

        return model

    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
