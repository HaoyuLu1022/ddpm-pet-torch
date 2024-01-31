import os
from sklearn.model_selection import train_test_split

#---------------------------------#
#   指向数据集所在的文件夹
#   默认指向根目录下的datasets
#---------------------------------#
full_path = "../cropped/fulldose"
low_path = "../cropped/lowdose"

if __name__ == "__main__":
    train_file = open('train_lines.txt', 'w')
    test_file = open('test_lines.txt', 'w')
    data_list = []
    for idx in [1, 2]:
        full_names = sorted(os.listdir(full_path+str(idx)))
        low_names = sorted(os.listdir(low_path+str(idx)))
        for i in range(len(full_names)):
            if (full_names[i].lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.img', 'npy')) and '68847' not in full_names[i].lower() and '68849' not in full_names[i].lower()):
                data_list.append([os.path.join(os.path.abspath(full_path+str(idx)), full_names[i]),os.path.join(os.path.abspath(low_path+str(idx)), low_names[i])])
    train_data, test_data = train_test_split(data_list, test_size=len(data_list)-200, random_state=114514)
    for data in train_data:
        train_file.write(f"{data[0]} {data[1]}")
        train_file.write('\n')
    for data in test_data:
        test_file.write(f"{data[0]} {data[1]}")
        test_file.write('\n')
    train_file.close()
    test_file.close()

