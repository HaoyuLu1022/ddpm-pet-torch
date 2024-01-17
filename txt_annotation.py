import os
from sklearn.model_selection import train_test_split

#---------------------------------#
#   指向数据集所在的文件夹
#   默认指向根目录下的datasets
#---------------------------------#
datasets_path   = ["/mnt/haoyu/cropped/fulldose1", "/mnt/haoyu/cropped/fulldose2"]

if __name__ == "__main__":
    train_file       = open('train_lines.txt', 'w')
    test_file = open('test_lines.txt', 'w')
    data_list = []
    for dataset_path in datasets_path:
        photos_names    = os.listdir(dataset_path)
        photos_names    = sorted(photos_names)

        for photo_name in photos_names:
            if(photo_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.img')) and '68847' not in photo_name.lower() and '68849' not in photo_name.lower()):
                data_list.append(os.path.join(os.path.abspath(dataset_path), photo_name))
                # train_file.write(os.path.join(os.path.abspath(dataset_path), photo_name))
                # train_file.write('\n')
    train_data, test_data = train_test_split(data_list, test_size=len(data_list)-200, random_state=114514)
    for data in train_data:
        train_file.write(data)
        train_file.write('\n')
    for data in test_data:
        test_file.write(data)
        test_file.write('\n')
    train_file.close()
    test_file.close()

