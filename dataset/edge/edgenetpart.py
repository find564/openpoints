import os
import glob
import h5py
import json
import pickle
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from openpoints.models.layers import fps, furthest_point_sample
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def load_data_partseg(partition, DATA_DIR):
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        dataset = glob.glob(os.path.join(DATA_DIR, "train", "product", "org" ,'*.xyz')) \
                    + glob.glob(os.path.join(DATA_DIR, "val", "product", "org", '*.xyz'))
        segset = glob.glob(os.path.join(DATA_DIR, "train", "product", "seg", '*.xyz')) \
                    + glob.glob(os.path.join(DATA_DIR, "val", "product", "seg", '*.xyz'))
    else:
        dataset = glob.glob(os.path.join(DATA_DIR, "%s" % partition, "product", "org" ,'*.xyz'))
        segset = glob.glob(os.path.join(DATA_DIR,"%s" % partition, "product","seg" ,'*.xyz'))
 

    for xyz_name in dataset:
        data = np.loadtxt(xyz_name, dtype=np.float32)
        all_data.append(data) #(P,3) # consider about N size

        #need data convert
        c = np.zeros(1).astype('int64')
        all_label.append(c) #(1)
        #all_label.append(label)

    for xyz_name in segset:
        seg = np.fromfile(xyz_name, dtype=np.int8).astype('int64')
        all_seg.append(seg) #(P,)

        #    data = f['data'][:].astype('float32')
        #    label = f['label'][:].astype('int64')
        #    seg = f['pid'][:].astype('int64')


    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(
        pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(
        rotation_matrix)  # random rotation (x,z)
    return pointcloud


@DATASETS.register_module()
class EdgeNetPart(Dataset):
    def __init__(self,
                 data_root='data/edge',
                 num_points=2048,
                 split='train',
                 class_choice=None,
                 shape_classes=1, transform=None): #shape_classes were 16 before from original shapenetpart : gbpark
        self.data, self.label, self.seg = load_data_partseg(split, data_root)
        self.cat2id = {'product': 0}
        self.seg_num = [3]
        self.index_start = [0]
        self.num_points = num_points
        self.partition = split
        self.class_choice = class_choice
        self.transform = transform


        #should be more seperated for detailed segmentations, for now just a couple of segmentations exist : gbpark
        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 3
            self.seg_start_index = 0
            self.eye = np.eye(shape_classes)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]

        # this is model-wise one-hot enocoding for 16 categories of shapes
        feat = np.transpose(self.eye[label, ].repeat(pointcloud.shape[0], 0))
        data = {'pos': pointcloud,
                'x': feat, #(P,C) : gbpark
                'y': seg}
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return self.data.shape[0]

@DATASETS.register_module()
class EdgePartNormal(Dataset):

    #those are for class info : gbpark
    l_classes = ['product']
    l_seg_num = [3]
    l_parts = []
    d_cls_parts = {'product': [0, 1, 2]}
    d_part_cls = {}  # {0:product...} : gbpark
    t_cls_partembed = torch.zeros(1, 3)
    
    for i, cls in enumerate(l_classes):
        parts_idx = d_cls_parts[cls]
        l_parts.append(parts_idx)
        t_cls_partembed[i].scatter_(0, torch.LongTensor(parts_idx), 1) # one hot vector gen : gbpark
    
    for cls in d_cls_parts.keys():
        for part in d_cls_parts[cls]:
            d_part_cls[part] = cls

    def __init__(self,
                 data_root='data/edge',
                 num_points=2048,
                 split='train',
                 class_choice=None,
                 use_normal=True,
                 shape_classes=1,
                 presample=False,
                 sampler='fps', 
                 transform=None,
                 multihead=False,
                 **kwargs
                 ):
        self.npoints = num_points
        self.root = data_root
        self.transform = transform
        self.catfile = os.path.join(self.root, 'edge_cat_folder.txt') # categories in this txt file should be ordered by index number as imposed in labelCloud : gbpark 
        self.d_cat_folder = {} 
        self.use_normal = use_normal
        self.presample = presample
        self.sampler = sampler 
        self.split = split
        self.multihead=multihead
        self.part_start = [0]
        if os.path.exists(self.catfile):
            with open(self.catfile, 'r') as f:
                for line in f:
                    ls = line.strip().split()
                    self.d_cat_folder[ls[0]] = ls[1] 
        else :
            self.d_cat_folder['product'] = 'product' #just matched with cat name : gbpark

        self.d_cat_cls = dict(zip(self.d_cat_folder, range(len(self.d_cat_folder))))

        if not class_choice is None:
            self.d_cat_folder = {k: v for k, v in self.d_cat_folder.items() if k in class_choice}

        self.d_cat_datapaths = {}
        self.d_cat_normalpaths = {}
        self.d_cat_segpaths = {}


        self.l_cat_datapaths = [] #this will be used as processed data as final : gbpark
        self.l_cat_normalpaths  = []
        self.l_cat_segpaths = []

        for cat in self.d_cat_folder:
            if split == 'trainval':
                dataset = glob.glob(os.path.join(data_root, "train", self.d_cat_folder[cat], "org" ,'*.xyz')) \
                    + glob.glob(os.path.join(data_root, "val", self.d_cat_folder[cat], "org", '*.xyz'))
                normal = glob.glob(os.path.join(data_root, "train", self.d_cat_folder[cat], "normal", '*.xyz')) \
                    + glob.glob(os.path.join(data_root, "val", self.d_cat_folder[cat], "normal", '*.xyz'))
                segset = glob.glob(os.path.join(data_root, "train", self.d_cat_folder[cat], "seg", '*.xyz')) \
                    + glob.glob(os.path.join(data_root, "val", self.d_cat_folder[cat], "seg", '*.xyz'))

            else:
                dataset = glob.glob(os.path.join(data_root, "%s" % split, self.d_cat_folder[cat], "org" ,'*.xyz'))
                normal = glob.glob(os.path.join(data_root,"%s" % split, self.d_cat_folder[cat],"normal" ,'*.xyz'))
                segset = glob.glob(os.path.join(data_root,"%s" % split, self.d_cat_folder[cat],"seg" ,'*.xyz'))

                c = np.zeros(1).astype('int64')

            self.d_cat_datapaths[cat] = dataset
            self.d_cat_normalpaths[cat] = normal
            self.d_cat_segpaths[cat] = segset

            for filepath in self.d_cat_datapaths[cat]:
                self.l_cat_datapaths.append((cat, filepath)) #listed by glob defualt ascending order : gbpark
            
            for filepath in self.d_cat_normalpaths[cat]:
                self.l_cat_normalpaths.append((cat, filepath))

            for filepath in self.d_cat_segpaths[cat]:
                self.l_cat_segpaths.append((cat, filepath))
            
        if transform is None:
            self.eye = np.eye(shape_classes)
        else:
            self.eye = torch.eye(shape_classes)

        # in the testing, using the uniform sampled 2048 points as input
        # presample
        filename = os.path.join(data_root, 'processed',
                                f'{split}_{num_points}_fps.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data, self.seg, self.normal, self.cls, self.fps_idx = [], [], [], [], [] 

            l_npoints = []
            for cat, filepath in tqdm(self.l_cat_datapaths, desc=f'Sample ShapeNetPart {split} org split'):
                cls = self.d_cat_cls[cat]
                cls = np.array([cls]).astype(np.int64)
                data = np.loadtxt(filepath).astype(np.float32)
                l_npoints.append(len(data))

                data = torch.from_numpy(data).to(
                    torch.float32).cuda().unsqueeze(0)
                data, fps_idx = fps(data, num_points)
                data = data.cpu().numpy()[0]
                fps_idx = fps_idx.cpu().numpy()[0]

                self.data.append(data)
                self.fps_idx.append(fps_idx)
                self.cls.append(cls)

            for i, cat, filepath in enumerate(tqdm(self.l_cat_normalpaths, desc=f'Sample ShapeNetPart {split} normal split')):
                normal = np.loadtxt(filepath, dtype=np.float32)
                fps_idx = self.fps_idx[i]
                self.normal.append(normal[fps_idx])
                
            for i, cat, filepath in enumerate(tqdm(self.l_cat_normalpaths, desc=f'Sample ShapeNetPart {split} seg split')):
                seg = np.fromfile(filepath, dtype=np.int8).astype('int64')
                fps_idx = self.fps_idx[i]
                self.normal.append(normal[fps_idx])
            
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f, sampled points' % (
                split, np.median(l_npoints), np.average(l_npoints), np.std(l_npoints)))
            os.makedirs(os.path.join(data_root, 'processed'), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump((self.data, self.seg, self.normal, self.cls, self.fps_idx), f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data, self.seg, self.normal, self.cls, self.fps_idx = pickle.load(f)
                print(f"{filename} load successfully")

    def __getitem__(self, index):
        if not self.presample:
            cat_data, datapath = self.l_cat_datapaths[index]
            cat_normal, normalpath = self.l_cat_normalpaths[index]
            cat_seg, segpath = self.l_cat_segpaths[index]

            if cat_data != cat_normal or cat_seg != cat_normal :
                print("category integrity was broken check data")
                exit(1) 

            cls = self.d_cat_cls[cat_data]
            cls = np.array([cls]).astype(np.int64)
            data = np.loadtxt(datapath).astype(np.float32)
            normal = np.loadtxt(normalpath, dtype=np.float32)
            seg = np.fromfile(segpath, dtype=np.int8).astype('int64')
            
        else:
            data, seg, normal, cls, fps_idx = self.data[index], self.seg[index], self.normal[index], self.cls[index], self.fps_idx[index] #prepared
            

        if 'train' in self.split:
            choice = np.random.choice(len(seg), self.npoints, replace=False)
            data = data[choice]
            normal = normal[choice]
            seg = seg[choice]
        else:
            data = data[:self.npoints]
            normal = normal[:self.npoints]
            seg = seg[:self.npoints]
        if self.multihead:
            seg=seg-self.part_start[cls[0]]

        data = {'pos': data[:, 0:3],
                'x': normal[:, 0:3],
                'cls': cls,
                'y': seg}

        """debug 
        from openpoints.dataset import vis_multi_points
        import copy
        old_data = copy.deepcopy(data)
        if self.transform is not None:
            data = self.transform(data)
        vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
        """
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.l_cat_datapaths)

if __name__ == '__main__':
    train = EdgePartNormal(num_points=2048, split='trainval')
    test = EdgePartNormal(num_points=2048, split='test')
    for dict in train:
        for i in dict:
            print(i, dict[i].shape)