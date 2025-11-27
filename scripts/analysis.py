from pathlib import Path
import pandas as pd
import pingouin as pg

OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v1_2025_11_25.xlsx"


def read_data(excel_path):
    try:
        # Reads ALL sheets into an OrderedDict where keys are sheet names and values are DataFrames
        all_sheets = pd.read_excel(excel_path, sheet_name=None, engine='openpyxl',header=0)

        # Example: Accessing the 'processed_data' sheet
        df_raw = all_sheets['raw_data']
        df_ave = all_sheets['processed_ave_data']
        print(f"Successfully loaded {len(all_sheets)} sheets.")

    except FileNotFoundError:
        print(f"Error: The file {excel_path} was not found.")
    except Exception as e:
        print(f"Error reading file: {e}")

    return df_raw, df_ave


if __name__ == "__main__":
    df_raw, df_ave = read_data(EXCEL_FILE_PATH)
    print(df_raw.head())
    print(df_ave.head())

    total_num_of_landmark = df_ave.shape[0]
    total_num_of_landmark_raw = df_raw.shape[0]/2
    print("Total number of landmarks is: ", total_num_of_landmark)
    print("Total number of landmarks before post processing is: ", total_num_of_landmark_raw)

    # how many landmarks per volunteer in raw and filtered data
    landmark_counts = df_ave.groupby('VL_ID').size()
    print("Number of landmarks per volunteer:\n", landmark_counts)
    print("Total number of volunteers is: ", landmark_counts.shape[0])
    landmark_counts_raw = df_raw.groupby('VL_ID').size()
    print("Total number of volunteers before post processing is: ", landmark_counts_raw.shape[0])

    # Count the number of rows for each unique landmark type
    type_counts = df_ave['Landmark type'].value_counts()
    type_proportions = df_ave['Landmark type'].value_counts(normalize=True) * 100

    # print("--- Counts of Each Landmark Type ---")
    # print(type_counts)
    type_summary_table = pd.DataFrame({
        'N': type_counts,
        'Percentage (%)': type_proportions
    }).reset_index().rename(columns={'index': 'Landmark Type'})

    # Ensure percentages are rounded and display the final table
    type_summary_table['Percentage (%)'] = type_summary_table['Percentage (%)'].round(1)
    total_count = type_summary_table['N'].sum()
    total_percentage_display = type_summary_table['Percentage (%)'].sum()

    total_row = pd.DataFrame({
        'Landmark type': ['Total'],
        'N': [total_count],
        'Percentage (%)': [total_percentage_display]
    })
    final_table = pd.concat([type_summary_table, total_row], ignore_index=True)
    print("--- Landmark Type Distribution Summary Table ---")
    print(final_table)


    row_ids = df_ave['VL_ID']
    vl_ids = df_ave['VL_ID'].unique()

    # 1. Define the distance columns
    DISTANCE_COLS_prone = [
        'Distance to nipple (prone) [mm]',
        'Distance to skin (prone) [mm]',
        'Distance to rib cage (prone) [mm]'
    ]

    #   supine

    # 2. Convert the data from wide format to long format
    df_long = pd.melt(
        df_ave,
        id_vars=['VL_ID', 'Landmark name'],  # Columns to keep for grouping
        value_vars=DISTANCE_COLS,
        var_name='Distance_Type',  # New column for the name of the distance
        value_name='Distance_Value_mm'  # New column for the measured distance
    )

    # 3. Clean up the Distance_Type names for clarity (optional)
    df_long['Distance_Type'] = df_long['Distance_Type'].str.split(' to ').str[1].str.split(' ').str[0]
    # e.g., 'nipple', 'skin', 'rib'

    # Run the Repeated Measures ANOVA
    aov = pg.rm_anova(
        data=df_long,
        dv='Distance_Value_mm',  # Dependent variable (the distance value)
        within='Distance_Type',  # The within-subject factor (the type of distance)
        subject='Landmark name',  # The subject/observation identifier (or a combination of 'VL_ID' and 'Landmark name')
        detailed=True
    )

    print(aov)

    # --- Post-Hoc Test (if the ANOVA result is significant) ---
    if aov['p-unc'][0] < 0.05:
        print("\n--- Running Post-Hoc Pairwise T-tests (Bonferroni corrected) ---")
        post_hoc = pg.pairwise_tests(
            data=df_long,
            dv='Distance_Value_mm',
            within='Distance_Type',
            subject='Landmark name',
            padjust='bonf'  # Apply Bonferroni correction for multiple comparisons
        )
        print(post_hoc)