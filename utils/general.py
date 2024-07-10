import os
import numpy as np

def increment_path(path, exist_ok=False, sep=""):
    """
    Generates an incremented file or directory path if it exists, always making the directory; args: path, exist_ok=False, sep="".

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    if os.path.exists(path) and not exist_ok:
        base, suffix = os.path.splitext(path) if os.path.isfile(path) else (path, "")
        
        for n in range(2, 9999):
            incremented_path = f"{base}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(incremented_path):
                path = incremented_path
                break

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # make directory

    return path

def avg_results(path):
    import polars as pl
    max_paccs = []
    max_accs = []
    dfs = [] 

    # Walk through all subdirectories
    for root, _, files in os.walk(path):
        # Check if 'results' is in the directory path
        if os.path.basename(root) == 'results':
            for filename in files:
                if filename.endswith('.csv') and 'server' in filename:
                    file_path = os.path.join(root, filename)
                    df = pl.read_csv(file_path)
                    dfs.append(df)
                    max_paccs.append(df["test_personal_accs"].max())
                    max_accs.append(df["test_traditional_accs"].max())

    if dfs:
        merged_df = pl.concat(dfs)
        stats = merged_df.describe()
        print(stats)

        # Calculate statistics using polars
        mean_paccs = np.mean(max_paccs)
        std_paccs = np.std(max_paccs)
        mean_accs = np.mean(max_accs)
        std_accs = np.std(max_accs)
        
        # Create a DataFrame for statistics using polars
        stats_df = pl.DataFrame({
            'Metric': ['Mean of max personal accs', 'Std of max personal accs',
                       'Mean of server-side max accs', 'Std of server-side max accs'],
            'Value': [mean_paccs, std_paccs, mean_accs, std_accs]
        })

        # Save the statistics to a CSV file using polars
        stats_df.write_csv(os.path.join(path, 'results.csv'))
        print(stats_df)

        return stats
    else:
        return None