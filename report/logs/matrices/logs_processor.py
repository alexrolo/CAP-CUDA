import csv
from collections import defaultdict

def process_file(input_file):
    threads_data = defaultdict(list)

    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile, delimiter=';')

        for row in reader:
            if not row['Matrix size']:
                continue
            
            row['Time(ms)'] = row['Time(ms)'].replace('.', ',') if row['Time(ms)'] else '0,000000'
            row['Time(s)'] = row['Time(s)'].replace('.', ',') if row['Time(s)'] else '0,000000'

            threads_per_block = row['Threads per block']
            threads_data[threads_per_block].append({
                'Matrix size': row['Matrix size'],
                'Threads per block': threads_per_block,
                'Time(ms)': row['Time(ms)'],
                'Time(s)': row['Time(s)']
            })

    for threads_per_block, data in threads_data.items():
        if input_file == '1.log':
            output_file = f'{threads_per_block}_threads_per_block_kt.csv'
        elif input_file == '2.log':
            output_file = f'{threads_per_block}_threads_per_block_all.csv'
        with open(output_file, 'w', newline='') as outfile:
            fieldnames = ['Matrix size', 'Threads per block', 'Time(ms)', 'Time(s)']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            writer.writerows(data)
            if input_file == '1.log':
                outfile.write('\nonly kernel time')
            elif input_file == '2.log':
                outfile.write('\nall time')

input_file = '2.log'

process_file(input_file)