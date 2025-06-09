import pandas as pd

def read_csv(csv_path, header_keyword="time"):
    """with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Find the line that contains the header keyword (e.g., "time")
    for i, line in enumerate(lines):
        if header_keyword.lower() in line.lower():  # case-insensitive match
            header_line = i
            break
    else:
        raise ValueError(f'Header keyword "{header_keyword}" not found in the file.')
    """
    # Read the CSV starting from the header line
    df = pd.read_csv(csv_path)
    return df



def merge_csvs(csv_path1, csv_path2):
    df1 = read_csv(csv_path1, header_keyword="time")
    df2 = read_csv(csv_path2, header_keyword="/jointset/knee_r/knee_angle_r/value")

    
    if len(df1) != len(df2):
        raise ValueError("CSV files have different number of rows and can't be merged by position.")

    merged_df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
    return merged_df

# Example usage
if __name__ == "__main__":
    path1 = 'csvs/time.csv'
    path2 = 'csvs/knee_angle.csv'
    merged = merge_csvs(path1, path2)
    print(merged.head())
    print(merged)