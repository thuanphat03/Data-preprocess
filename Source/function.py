import pandas as pd
import numpy as np
import sys
import math

def add(column1, column2):
    result = column1.copy()  
    for i in range(len(column1)):
        if math.isnan(column1[i]) or math.isnan(column2[i]):
            result[i] = np.nan
        else:
            result[i] += column2[i]

    return result

def subtract(column1, column2):
    result = column1.copy()  
    for i in range(len(column1)):
        if math.isnan(column1[i]) or math.isnan(column2[i]):
            result[i] = np.nan
        else:
            result[i] -= column2[i]

    return result

def multiply(column1, column2):
    result = column1.copy()  
    for i in range(len(column1)):
        if math.isnan(column1[i]) or math.isnan(column2[i]):
            result[i] = np.nan
        else:
            result[i] *= column2[i]

    return result

def divide(column1, column2):
    result = column1.copy() 
    for i in range(len(column1)):
        if math.isnan(column1[i]) or math.isnan(column2[i]):
            result[i] = np.NaN
        elif column2[i] == 0:
            result[i] = np.NaN
        else:
            result[i] /= column2[i]
    
    return result

def calculate_std(data_column):
    mean = sum(data_column) / len(data_column)
    var = sum((x - mean) ** 2 for x in data_column) / len(data_column)
    std = var ** 0.5
    return std

def read_file_csv(path):
    # Read the CSV file
    data = pd.read_csv(path)
    return data


# Extract columns with missing values
def extract_missing_value(data):
    # List to store column names with missing values
    result = []

    # Iterate over columns and check for missing values
    for column in data.columns:
        info = data[column].tolist()
        if any(type(item) != str and math.isnan(item) for item in info):
            result.append(column)

    # Print the column names and data of column
    print(f"Column names: \n{result}. Length: {len(result)}\n")
    print(f"The data contains the columns with missing values: \n{data[result]}\n")
    return data[result]

def count_missing_value_rows(data):
    # Create a list to store the row indices with missing values
    rows_with_missing_values = []
    # Initialize the variable count to 0
    count = 0

    # Iterate over rows and check for missing values
    for index, row in data.iterrows():  # Sử dụng iterrows() để lặp qua từng hàng của DataFrame
        if row.isnull().any():
            rows_with_missing_values.append(index)  # Thêm chỉ mục hàng có giá trị None hoặc NaN vào danh sách
            count += 1

    # Print the count
    print(f"The number of rows with missing values is: {count}")

    data = data.iloc[rows_with_missing_values]

    return data


# Fill in the missing value using mean, median (for numeric properties) and mode (for the categorical attribute)
def fill_missing_value_by_mean_or_median_and_mode(data, desired_column="", command="mean"):
    new_data = pd.DataFrame()
    # Variable to check if desired column existed
    is_existed_column = False
    for column in data.columns:
        # Check column name
        if desired_column != "" and column != desired_column:
            continue
        
         # Print
        print(
            f"Count the old missing value: {len(count_missing_value_rows(data)) if desired_column == '' else len(count_missing_value_rows(data[[desired_column]]))}\n")
    
        info = data[column].tolist()
        is_existed_column = True
        # Check numeric attribute
        if all(type(item) != str for item in info) and all(pd.isna(data[desired_column])) is False:
            # Check array have nan values
            if any(math.isnan(item) for item in info):
                # Extract value different nan
                new_list = [item for item in info if math.isnan(item) == False]
                # Check array if values all nan
                if len(new_list) == 0:
                    if desired_column != "":
                        print("Column has no valid numeric values. Unable to calculate mean or median.")
                        return new_data
                    else:
                        new_data[column] = 0
                        data[column] = 0
                        continue
                # Check type of numeric calculation
                if command == "mean":
                    number = sum(new_list) / len(new_list)
                    print(f"Mean {column}: {number}\n")
                elif command == "median":
                    new_list.sort()
                    mid = len(new_list) // 2
                    number = (new_list[mid] + new_list[~mid]) / 2
                    print(f"Median {column}: {number}\n")
                else:
                    print("The method does not work with this attribute!")
                    return data
                # New array with refill missing value
                new_info = [number if math.isnan(item) else item for item in info]
                # Insert
                new_data[column] = new_info
                data[column] = new_info
            else:
                new_data[column] = info
                data[column] = info
                continue
        # Check nominal
        elif all(pd.isna(data[desired_column])) is False:
            # Check type of nominal calculation
            if command == "mean" or command == "median":
                if desired_column != "":
                    print("The method does not work with this attribute!")
                    return data
                else:
                    continue
            # Extract value different nan
            new_list = [item for item in info if type(item) == str]
            # Find mode
            value = max(new_list, key=new_list.count)
            # Print
            print(f"Mode {column}: {value}\n")
            # New array with refill missing value
            new_info = [value if type(item) != str and math.isnan(item) else item for item in info]
            # Insert
            new_data[column] = new_info
            data[column] = new_info
        else:
            print("Column is NaN.")
            return data
        
    if is_existed_column == False:
        print("Column not existed.")
        return data

    # Print
    print(f"Count the new missing value: {len(count_missing_value_rows(new_data))}\n")
    print(f"Output the column have been filled the missing values: \n {new_data}")
    return data

def delete_row_with_a_num(data, number):
    # List to store the index of rows that need to be deleted
    list_rows_to_delete = []

    # Iterate over rows and check for missing values
    for _, row in enumerate(data.index):
        if data.loc[row].isnull().sum() > number:
            list_rows_to_delete.append(row)
    
    # Delete rows
    data.drop(index=list_rows_to_delete, inplace=True)

    print(f"Output the data have been deleted the rows: \n {data}")

    return data


# Deleting columns containing more than a particular number of missing values
def delete_column_with_a_num(data, number):
    # Store new data deleted
    new_data = pd.DataFrame()

    # Iterate over columns and check for missing values
    for column in data.columns:
        info = data[column].tolist()
        # List of nan values to count
        list_nan = [item for item in info if type(item) != str and math.isnan(item)]
        if len(list_nan) < number:
            new_data[column] = data[column]

    # Print
    print(f"Old length of data: {len(data.columns)}, New length of data: {len(new_data.columns)}")
    print(f"Output the data have been deleted the columns: \n {new_data}")
    return new_data

def delete_duplicate_samples(data):
    # Find and remove duplicate rows
    data.drop_duplicates(keep='first', inplace=True)

    print(f"Output the data have been deleted the duplicate samples: \n {data}")

    return data


#  Normalize a numeric attribute using min-max
def normalize_data_using_min_max(data, column, new_min, new_max):
    is_existed_col = False
    # Iterate over columns
    for col_name in data.columns:
        if column != col_name:
            continue
        # Check numeric attribute
        is_existed_col = True
        info = data[column].tolist()
        # Check numeric column
        valid_values = [value for value in info if not pd.isna(value)]
        if all(type(item) != str for item in info):
            old_max = max(valid_values)
            old_min = min(valid_values)
            range = old_max - old_min
            # If old_max and old_min equal
            if range == 0:
                data[column] = new_min
            else:
                data[column] = ((data[column] - old_min) * (new_max - new_min) / range) + new_min
            # Print
            print(f"Output the column have been normalized with using min_max: \n{data[column]}")
        else:
            print("The attribute is not a numeric property!")

    # Check existed column
    if is_existed_col == False:
        print(f"Column is not existed!")

    return data


# Normalize a numeric attribute using z-score
def normalize_data_using_z_score(data, column):
    is_existed_col = False
    # Iterate over columns
    for col_name in data.columns:
        if column != col_name:
            continue
        # Check numeric attribute
        is_existed_col = True
        info = data[column].tolist()
        # Check numeric column
        if all(type(item) != str for item in info):
            # Extract value different nan
            new_list = [item for item in info if math.isnan(item) == False]
            # Mean
            mean = sum(new_list) / len(new_list)
            # Standard Deviation
            std = calculate_std(new_list)

            if std == 0:
                # Avoid division by zero when std_deviation is zero
                data[column] = 0.0
            else:
                # Normalize the data in the column
                data[column] = (data[column] - mean) / std
            # Print
            print(f"Output the column have been normalized with using z_score: \n{data[column]}")
        else:
            print("The attribute is not a numeric property!")

    # Check existed column
    if is_existed_col == False:
        print(f"Column is not existed!")

    return data

def add_two_numeric_attributes(column1, column2):
    # Check if the column contains numeric data
    if np.issubdtype(column1.dtype, np.number) is False or np.issubdtype(column2.dtype, np.number) is False:
        print("The first attribute or the second attribute may not be a numeric property !!!")
        return
    
    # Addition
    result_add = add(column1, column2)
    
    data_to_export = pd.DataFrame({'No.': range(1, len(result_add) + 1), 'Result': result_add})

    print(f"The result of division: \n {data_to_export}")

    return data_to_export

def subtract_two_numeric_attributes(column1, column2):
    # Check if the column contains numeric data
    if np.issubdtype(column1.dtype, np.number) is False or np.issubdtype(column2.dtype, np.number) is False:
        print("The first attribute or the second attribute may not be a numeric property !!!")
        return
    
    # Subtraction
    result_subtract = subtract(column1, column2)
    
    data_to_export = pd.DataFrame({'No.': range(1, len(result_subtract) + 1), 'Result': result_subtract})

    print(f"The result of division: \n {data_to_export}")

    return data_to_export

def multiply_two_numeric_attributes(column1, column2):
    # Check if the column contains numeric data
    if np.issubdtype(column1.dtype, np.number) is False or np.issubdtype(column2.dtype, np.number) is False:
        print("The first attribute or the second attribute may not be a numeric property !!!")
        return
    
    # Multiplication
    result_multiply = multiply(column1, column2)
    
    data_to_export = pd.DataFrame({'No.': range(1, len(result_multiply) + 1), 'Result': result_multiply})

    print(f"The result of division: \n {data_to_export}")

    return data_to_export

def divive_two_numeric_attributes(column1, column2):
    # Check if the column contains numeric data
    if np.issubdtype(column1.dtype, np.number) is False or np.issubdtype(column2.dtype, np.number) is False:
        print("The first attribute or the second attribute may not be a numeric property !!!")
        return
    
    # Division
    result_div = divide(column1, column2)

    data_to_export = pd.DataFrame({'No.': range(1, len(result_div) + 1), 'Result': result_div})

    print(f"The result of division: \n {data_to_export}")

    return data_to_export

def output_file_csv(data, name_file):
    # Add folder OUTPUT
    name_file = "OUTPUT/" + name_file

    # Open the CSV file for writing
    data.to_csv(name_file, index=False)

    
if __name__ == "__main__":
    # Get the path
    path = sys.argv[1]

    # Get the data
    data = read_file_csv(path)
    # Output the original data
    print(f"##################### The original data ##################### \n {data} ")

    for i in range(2, len(sys.argv)):
        try:
            # Get the data
            data = read_file_csv(path)
            print(f"\n\n##################### No.{i - 1} #####################")
            if sys.argv[i] == 'extract':
                data = extract_missing_value(data)
                output_file_csv(data, 'output_extracted_data.csv')    
            elif sys.argv[i] == 'count':
                data = count_missing_value_rows(data)
                output_file_csv(data, 'output_missing_value_lines.csv') 
            elif sys.argv[i].find('fill_mean_') != -1:
                column = str(sys.argv[i][10:])
                data = fill_missing_value_by_mean_or_median_and_mode(data, column, 'mean')
                output_file_csv(data, 'output_fill_mean.csv')
            elif sys.argv[i].find('fill_median_') != -1:
                column = str(sys.argv[i][12:])
                data = fill_missing_value_by_mean_or_median_and_mode(data, column, 'median')
                output_file_csv(data, 'output_fill_median.csv')
            elif sys.argv[i].find('fill_mode_') != -1:
                column = str(sys.argv[i][10:])
                data = fill_missing_value_by_mean_or_median_and_mode(data, column, 'mode')
                output_file_csv(data, 'output_fill_mode.csv')
            elif sys.argv[i].find('delete_row_') != -1:
                number = int(sys.argv[i][11:])
                data = delete_row_with_a_num(data, number)
                output_file_csv(data, 'output_delete_row.csv')
            elif sys.argv[i].find('delete_column_') != -1:
                number = int(sys.argv[i][14:])
                data = delete_column_with_a_num(data, number)
                output_file_csv(data, 'output_delete_column.csv')
            elif sys.argv[i] == 'delete_duplicate':
                data = delete_duplicate_samples(data)
                output_file_csv(data, 'output_delete_duplicate.csv')
            elif sys.argv[i].find('normalize_z_score_') != -1:
                column = str(sys.argv[i][18:])
                data = normalize_data_using_z_score(data, column)
                output_file_csv(data, 'output_z_score.csv')
            elif sys.argv[i].find('normalize_') != -1:
                pos1 = sys.argv[i].find('_')
                column = str(sys.argv[i][pos1 + 1 : sys.argv[i].find('_', pos1 + 1)])
                pos2 = sys.argv[i].find('_', pos1 + 1)
                new_min = int(sys.argv[i][pos2 + 1 : sys.argv[i].find('_', pos2 + 1)])
                pos3 = sys.argv[i].find('_', pos2 + 1)
                new_max = int(sys.argv[i][pos3 + 1 :])
                data = normalize_data_using_min_max(data, column, new_min, new_max)
                output_file_csv(data, 'output_min_max.csv')
            elif sys.argv[i].find('add') != -1:
                attribute1 = str(sys.argv[i][sys.argv[i].find('_') + 1 : sys.argv[i].find('_', sys.argv[i].find('_') + 1)])
                attribute2 = str(sys.argv[i][sys.argv[i].find('_', sys.argv[i].find('_') + 1) + 1 :])
                data = add_two_numeric_attributes(data[attribute1], data[attribute2])
                output_file_csv(data, 'output_add.csv')
            elif sys.argv[i].find('subtract') != -1:
                attribute1 = str(sys.argv[i][sys.argv[i].find('_') + 1 : sys.argv[i].find('_', sys.argv[i].find('_') + 1)])
                attribute2 = str(sys.argv[i][sys.argv[i].find('_', sys.argv[i].find('_') + 1) + 1 :])
                data = subtract_two_numeric_attributes(data[attribute1], data[attribute2])
                output_file_csv(data, 'output_subtract.csv')
            elif sys.argv[i].find('multiply') != -1:
                attribute1 = str(sys.argv[i][sys.argv[i].find('_') + 1 : sys.argv[i].find('_', sys.argv[i].find('_') + 1)])
                attribute2 = str(sys.argv[i][sys.argv[i].find('_', sys.argv[i].find('_') + 1) + 1 :])
                data = multiply_two_numeric_attributes(data[attribute1], data[attribute2])
                output_file_csv(data, 'output_multiply.csv')
            elif sys.argv[i].find('div') != -1:
                attribute1 = str(sys.argv[i][sys.argv[i].find('_') + 1 : sys.argv[i].find('_', sys.argv[i].find('_') + 1)])
                attribute2 = str(sys.argv[i][sys.argv[i].find('_', sys.argv[i].find('_') + 1) + 1 :])
                data = divive_two_numeric_attributes(data[attribute1], data[attribute2])
                output_file_csv(data, 'output_divive.csv')
            else:
                print("The command does not exist !!!")
            print(f"##################### End #####################")
        except Exception:
            print("The command have something wrong !!!")
            print(f"##################### End #####################")

        
   
