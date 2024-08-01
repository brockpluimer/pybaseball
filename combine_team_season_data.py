import os
import pandas as pd

def ensure_dir(directory):
    """
    Ensure that the specified directory exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def combine_excel_files(input_directory, output_file):
    """
    Combines multiple Excel files from a directory into a single Excel workbook.
    Each year within these files will be a separate sheet in the output workbook.
    
    :param input_directory: Directory containing the Excel files to be combined.
    :param output_file: Path to the output Excel file.
    """
    # Create a Pandas Excel writer object to write the combined workbook
    with pd.ExcelWriter(output_file) as writer:
        # List all Excel files in the input directory
        for filename in sorted(os.listdir(input_directory)):
            # Ignore temporary files and non-Excel files
            if filename.startswith('~$') or not filename.endswith('.xlsx'):
                print(f"Skipping file: {filename}")
                continue
            
            # Construct the full file path
            file_path = os.path.join(input_directory, filename)
            
            try:
                # Load the Excel file to get sheet names
                xls = pd.ExcelFile(file_path)
                
                # Add each sheet (year) as an individual sheet in the combined workbook
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Check for sheet name collision, and resolve if needed
                    sheet_name_combined = f"{sheet_name}"
                    df.to_excel(writer, sheet_name=sheet_name_combined, index=False)
                    
                    print(f"Added {filename}, sheet {sheet_name} to the combined workbook as '{sheet_name_combined}'.")
            
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")

def main():
    # Directory containing the Excel files to combine
    input_directory = 'season_standings'
    
    # Path to the output combined Excel file
    output_file = 'combined_final_standings.xlsx'
    
    # Combine the Excel files into a single workbook
    combine_excel_files(input_directory, output_file)
    
    print("All files have been combined into", output_file)

if __name__ == "__main__":
    main()