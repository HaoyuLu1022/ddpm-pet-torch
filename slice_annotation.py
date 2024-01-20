import os
from sklearn.model_selection import train_test_split

full_path = "../slices/full"
low_path = "../slices/low"

if __name__ == "__main__":
    train_file = open('train_slices.txt', 'w')
    data_list = []
    full_names = sorted(os.listdir(f"{full_path}"))
    low_names = sorted(os.listdir(f"{low_path}"))
    for i in range(len(full_names)):
        if (full_names[i].lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.img', 'npy')) and '68847' not in full_names[i].lower() and '68849' not in full_names[i].lower()):
            data_list.append([os.path.join(os.path.abspath(f"{full_path}"), full_names[i]),os.path.join(os.path.abspath(f"{low_path}"), low_names[i])])
    for data in data_list:
        train_file.write(f"{data[0]} {data[1]}")
        train_file.write('\n')
    train_file.close()

