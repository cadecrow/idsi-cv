import csv

def create_post_data_file(fname):
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        newfname = fname.split(".")[0] + "_post.csv"
        with open(newfname, "w", newline = "") as f:
            line = csv.writer(f)
            for row in csv_reader:
                if (row[0] == ""):
                    break
                subtype = row[19] # will be NA if pre disaster image
                if (subtype != "NA"):
                    line.writerow(row)

create_post_data_file("full_data_train.csv")
