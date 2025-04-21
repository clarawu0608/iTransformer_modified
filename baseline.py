import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.iloc[1:, 1:].astype(float)


def mask_data(data, missing_rate, random_seed=42):
    np.random.seed(random_seed)
    mask = np.random.rand(*data.shape) > missing_rate
    # missing_ratio_per_column = (~mask).sum(axis=0) / mask.shape[0]
    # print("Missing per column:")
    # print(missing_ratio_per_column)
    data_masked = data.copy()
    data_masked[~mask] = np.nan
    return data_masked, mask


def interpolate_data(data_masked):
    return data_masked.interpolate(method='linear', limit_direction='both', limit_area='inside')


def compute_errors(original, imputed, mask):
    missing_mask = ~mask
    mae = mean_absolute_error(original[missing_mask], imputed[missing_mask])
    mse = mean_squared_error(original[missing_mask], imputed[missing_mask])
    return mae, mse


def main(csv_file_path, missing_rate):
    # Step 1: Load data
    original_data = load_data(csv_file_path)

    # Step 2: Mask data
    data_masked, mask = mask_data(original_data, missing_rate)

    # Step 3: Interpolate data
    data_imputed = interpolate_data(data_masked)

    # Step 4: Compute MAE and MSE
    mae, mse = compute_errors(original_data.values, data_imputed.values, mask)

    # Step 5: Write results to a file
    with open("baseline_results.txt", "w") as f:
        f.write(f"Missing Rate: {missing_rate}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    


# Example usage
if __name__ == "__main__":
    main("dataset/traffic/traffic_smaller.csv", missing_rate=0.7)