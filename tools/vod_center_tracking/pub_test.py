from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import copy
import json
import os
import numpy as np
from pub_tracker import PubTracker as Tracker
import json 
import time
import glob
import pathlib
 
def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--det_path", type=str)
    parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--max_age", type=int, default=3)

    args = parser.parse_args()

    return args

def read_frame(path, frame_id):
    final_path = path + '/%s.txt' % frame_id.strip()
    with open(final_path, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15])
        self.v_x = float(label[16])
        self.v_y = float(label[17]) 
        self.track_id = None
        self.track_score = self.score 
        
    def to_kitti_track_format(self):
        assert self.track_id is not None
        assert self.track_score is not None
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %d' \
                    % (self.cls_type ,self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry, self.track_score, int(self.track_id))
        return kitti_str


def main():
    args = parse_args()
    print('Deploy OK')

    tracker = Tracker(max_age=args.max_age, hungarian=args.hungarian)
    val_seq = ['delft_1','delft_10','delft_14','delft_22']
    clips_list = glob.glob('./clips/*.txt')
    
    val_seq_details = {}
    
    for clip in clips_list:
        clip_name = clip.split('/')[-1].split('.')[0]
        if clip_name not in val_seq:
            continue
        with open(clip, 'r') as f:
            lines = f.readlines()
        valid_lines = []
        for line in lines:
            try:
                read_frame(args.det_path, line)
                valid_lines.append(line)
            except FileNotFoundError:
                continue
        val_seq_details[clip_name] = valid_lines
        
    
    print("Begin Tracking\n")
    for clip_name, seq_list in val_seq_details.items():
        output_path = './out/%s' % clip_name
        output_path = pathlib.Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        tracker.reset()
        start = time.time()
        for seq in seq_list:
            preds = read_frame(args.det_path, seq)
            VOD_TIME_LAG = 0.1
            outputs = tracker.step_centertrack(preds, VOD_TIME_LAG)
            print(len(outputs))
            anno = []
            for out in outputs:
                kitti_obj = out['origin_obj']
                kitti_obj.track_id = out['tracking_id']
                kitti_obj.tracking_score = kitti_obj.score
                anno.append(kitti_obj.to_kitti_track_format()+'\n')
            output_path_seq = output_path / ('%s.txt' % seq.strip())
            with open(output_path_seq,'w') as f:
                f.writelines(anno)
        end = time.time()
        second = (end-start) 
        speed= len(seq_list) / second
        print("The speed is {} FPS".format(speed))

if __name__ == '__main__':
    main()
