import os

#---------------------------------#
#   指向数据集所在的文件夹
#   默认指向根目录下的datasets
#---------------------------------#
datasets_path   = ["/mnt/haoyu/cropped/fulldose1", "/mnt/haoyu/cropped/fulldose2"]

if __name__ == "__main__":
    list_file       = open('train_lines.txt', 'w')
    for dataset_path in datasets_path:
        photos_names    = os.listdir(dataset_path)
        photos_names    = sorted(photos_names)

        for photo_name in photos_names:
            if(photo_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.img')) and '68847' not in photo_name.lower() and '68849' not in photo_name.lower()):
                list_file.write(os.path.join(os.path.abspath(dataset_path), photo_name))
                list_file.write('\n')
    list_file.close()

