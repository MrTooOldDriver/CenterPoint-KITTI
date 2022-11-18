from glob import glob
from pathlib import Path as P
from tqdm import tqdm
import os


def fix_trunc():
    label_path = '/home/gabriel/mmdetection3d/data/vod/lidar/training/label_2'
    write_path = '/home/gabriel/mmdetection3d/data/vod/lidar/training/new_labels'

    all_gt = sorted(glob(label_path+"/*"))

    for file in tqdm(all_gt):

        gt_num = P(file).stem
        fixed_annos = []
        with open(file,'r') as f:
            lines = f.readlines()
            for line in lines:
                cur_line = line.split()
                cur_line[1] = '0'
                fixed_annos += [" ".join(str(c) for c in cur_line)]
            f.close()

        with open(write_path+f'/{gt_num}.txt','w') as f:
            for l in fixed_annos:
                f.write(f'{l}\n')
        f.close()


def add_zero():

    # folders = ['calib','image_2','label_2','old_label','pose','velodyne']
    folders = ['calib','velodyne']


    label_path = '/workspaces/CenterPoint-KITTI/data/vod/radar/training/'

    print(label_path)
    for folder in folders:
        cur_folder = label_path+folder
        all_files = sorted(glob(cur_folder+"/*"))
        for file in all_files:
            f = P(file)
            # print(f,cur_folder+name)
            name = "/0"+f.name 
            os.rename(file,cur_folder+name)

    print('done')
def pad_imagesets():

    files = ['full.txt','test.txt','train_val.txt','train.txt','val.txt']
    label_path = '/home/gabriel/mmdetection3d/data/vod/lidar/ImageSets/'


    for file in files:
        cur_file = label_path + file

        
        with open(cur_file) as f:
            new = ['0'+ l for l in f.readlines()]
                # new.append("0"+line)
        print(new)
        f.close()
        with open(label_path+"new"+file,"w") as f:
            for l in new:
                f.write(l)

        f.close()




add_zero()
