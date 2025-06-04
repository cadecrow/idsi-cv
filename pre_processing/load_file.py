import csv

# Load the data of the list of images in the mini-batch
def load_file(fname, list_of_row_nums):
    img_dict = {}
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (row[0] == ""):
                break
            if line_count in list_of_row_nums:
                img_name = row[17]
                if (img_name not in img_dict):
                    img_dict[img_name] = {}
                subtype = row[19]
                uid = row[20]
                xy = row[22]
                img_dict[img_name][uid] =  (convert_subtype(subtype), xy)
            line_count += 1
    return img_dict

def check_num_images(fname):
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        for row in csv_reader:
            line_count += 1
        return line_count

def convert_subtype(subtype):
    if subtype == "no-damage":
        return 0
    elif subtype == "minor-damage":
        return 1
    elif subtype == "major-damage":
        return 2
    elif subtype == "destroyed":
        return 3
    else:
        print("Invalid subtype")
        return -1
