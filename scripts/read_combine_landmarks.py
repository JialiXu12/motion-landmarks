import json
import os


def combine_landmark_files(file1_path, file2_path, file3_path):
    try:
        # --- 1. Read from file 1 ---
        with open(file1_path, 'r') as f:
            data1 = json.load(f)

        # --- 2. Read from file 2 ---
        with open(file2_path, 'r') as f:
            data2 = json.load(f)

        # --- 3. Extract the required data ---

        # Get data from file 1 dictionary
        nipple_r = data1['bodies']['Ray-Test']['landmarks']['nipple-r']
        nipple_l = data1['bodies']['Ray-Test']['landmarks']['nipple-l']
        sternum_sup = data1['bodies']['Ray-Test']['landmarks']['sternal-superior']

        # Get data from file 2 dictionary
        sternum_inf = data2['bodies']['thorax']['landmarks']['sternal-inferior']

        print(f"--- Successfully extracted 4 landmarks ---")

        # --- 4. Create a new dictionary for file 3 ---
        # This new structure will hold only the 4 landmarks
        # We preserve the original hierarchy for clarity

        data3 = {
            "bodies": {
                "Jiali-test": {
                    "landmarks": {
                        "nipple-l": nipple_l,
                        "nipple-r": nipple_r,
                        "sternal-inferior": sternum_inf,
                        "sternal-superior": sternum_sup,
                    }
                }
            }
        }

        print(f"--- Creating new data structure... ---")

        # --- 5. Write the new data to file 3 ---
        with open(file3_path, 'w') as f:
            # json.dump() writes the new dictionary to 'file3.json'
            # indent=4 makes the file human-readable
            json.dump(data3, f, indent=4)

        print(f"Successfully created '{file3_path}' with combined data.")
        print(f"'{file1_path}' and '{file2_path}' were not modified.")

    except FileNotFoundError:
        print(f"Error: One of the files was not found. Check paths.")
    except KeyError as e:
        print(f"Error: A key was not found in the JSON structure: {e}")
        print("Please check that your JSON files have the correct nested structure.")

if __name__ == '__main__':
    vl_ids = [25, 29, 30, 31, 34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52, 54,56,57,58,59,60,61,63,64,
              65,66,67,68,69,70,71,72,74,75,76,77,78,79,84, 85, 89]
    base_path = r'U:\sandbox\jxu759\volunteer_seg\results\prone\landmarks\combined'

    for vl_id in vl_ids:
        file_name = f'VL{vl_id:05d}_skeleton_data_prone_t2.json'

        file1_path = os.path.join(base_path,'bv_ssm',file_name)
        file2_path = os.path.join(base_path,'bv_xiphoid',file_name)
        file3_path = os.path.join(base_path, file_name)

        combine_landmark_files(file1_path, file2_path, file3_path)

