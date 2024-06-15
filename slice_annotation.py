import os
from sklearn.model_selection import train_test_split

full_path = "../slices_12s/fulldose"
low_path = "../slices_12s/lowdose"

if __name__ == "__main__":
    train_file = open('train_slices.txt', 'w')
    data_list = []
    full_names = sorted(os.listdir(f"{full_path}"))
    low_names = sorted(os.listdir(f"{low_path}"))
    for i in range(len(full_names)//4):
        if (full_names[i].lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.img', '.npy'))
            and '68847' not in full_names[i].lower() 
            and '68849' not in full_names[i].lower() 
            and '68706' not in full_names[i].lower()
            and '68850' not in full_names[i].lower()
            and '68858' not in full_names[i].lower()
            and '68859' not in full_names[i].lower()
            and '68986' not in full_names[i].lower()
            and not full_names[i].lower().endswith('-0.npy')
            and not full_names[i].lower().endswith('-1.npy')
            and not full_names[i].lower().endswith('-2.npy')
            and not full_names[i].lower().endswith('-3.npy')
            and not full_names[i].lower().endswith('-4.npy')
            and not full_names[i].lower().endswith('-5.npy')
            and not full_names[i].lower().endswith('-6.npy')
            and not full_names[i].lower().endswith('-7.npy')
            and not full_names[i].lower().endswith('-8.npy')
            and not full_names[i].lower().endswith('-9.npy')
            and not full_names[i].lower().endswith('-10.npy')
            and not full_names[i].lower().endswith('-11.npy')
            and not full_names[i].lower().endswith('-12.npy')
            and not full_names[i].lower().endswith('-13.npy')
            and not full_names[i].lower().endswith('-127.npy')
            and not full_names[i].lower().endswith('-126.npy')
            and not full_names[i].lower().endswith('-125.npy')
            and not full_names[i].lower().endswith('-124.npy')
            and not full_names[i].lower().endswith('-123.npy')
            and not full_names[i].lower().endswith('-122.npy')
            and not full_names[i].lower().endswith('-121.npy')
            and not full_names[i].lower().endswith('-120.npy')
            and not full_names[i].lower().endswith('-119.npy')
            and not full_names[i].lower().endswith('-118.npy')
            and not full_names[i].lower().endswith('-117.npy')
            and not full_names[i].lower().endswith('-116.npy')
            and not full_names[i].lower().endswith('-115.npy')
            and not full_names[i].lower().endswith('-114.npy')):
            data_list.append([os.path.join(os.path.abspath(f"{full_path}"), full_names[i]),os.path.join(os.path.abspath(f"{low_path}"), low_names[i])])
    for data in data_list:
        train_file.write(f"{data[0]} {data[1]}")
        train_file.write('\n')
    train_file.close()

