import csv
import re

def process_value(value):
    # Remove brackets and split on spaces
    values = value.replace('[', '').replace(']', '').split()
    # Replace . with .0 for each value
    values = [re.sub(r'\.(?!\d)', '.0', v) for v in values]
    return values

def process_row(row, headers_to_keep):
    # Process each value and flatten the list
    return [v for header, value in row.items() if header in headers_to_keep for v in process_value(value)]

def read_and_write_csv(input_file, output_file, headers_to_keep):
    with open(input_file, 'r') as csv_in, open(output_file, 'w', newline='') as csv_out:
        reader = csv.DictReader(csv_in)
        writer = csv.writer(csv_out)
        for row in reader:
            writer.writerow(process_row(row, headers_to_keep))

# Call the function with your input and output file paths and the headers you want to keep
read_and_write_csv('ekf.csv', 'output.csv', ['mu', 'mu_0', 'mu_1', 'mu_2'])
