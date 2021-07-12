import csv
import os

def load_csv(path):
    with open(path) as csv_file:
        reader=csv.reader(csv_file)
        csv_header = next(reader)
        data_dict = dict()
        for head in csv_header:
            data_dict[head]=[]
        for line in reader:
            for i, head in enumerate(csv_header):
                data_dict[head].append(line[i])
    for i,path in enumerate(data_dict['image_path']):
        data_dict['image_path'][i] = os.path.join('bokeh_vis/static/', path[2:])
    return data_dict
