import os
import numpy as np
import h5py
import torch
import torch.utils.data as data


class CreateData(data.Dataset):
    def __init__(self, dataset_dict):
        self.len_dset_dict = len(dataset_dict)
        self.rgb = dataset_dict['rgb']
        self.depth = dataset_dict['depth']
        self.seg_label = dataset_dict['seg_label']

        if self.len_dset_dict > 3:
            self.class_label = dataset_dict['class_label']
            self.use_class = True

    def __getitem__(self, index):
        rgb_img = self.rgb[index]
        depth_img = self.depth[index]
        seg_label = self.seg_label[index]

        rgb_img = torch.from_numpy(rgb_img)
        depth_img = torch.from_numpy(depth_img)

        dataset_list = [rgb_img, depth_img, seg_label]

        if self.len_dset_dict > 3:
            class_label = self.class_label[index]
            dataset_list.append(class_label)
        return dataset_list

    def __len__(self):
        return len(self.seg_label)


def get_data(opt, use_train=True, use_test=True):
    """
    Load NYU_v2 or SUN rgb-d dataset in hdf5 format from disk and prepare
    it for classifiers.
    """
    # Load the chosen datasets path
    if os.path.exists(opt.dataroot):
        path = opt.dataroot
    else:
        raise Exception('Wrong datasets requested. Please choose either "NYU" or "SUN"')
    
    h5file = h5py.File(path, 'r')

    train_dataset_generator = None
    test_dataset_generator = None

    # Create python dicts containing numpy arrays of training samples
    if use_train:
        train_dataset_generator = dataset_generator(h5file, 'train', opt.use_class)
        print('[INFO] Training set generator has been created')

    # Create python dicts containing numpy arrays of test samples
    if use_test:
        test_dataset_generator = dataset_generator(h5file, 'test', opt.use_class)
        print('[INFO] Test set generator has been created')
    h5file.close()
    return train_dataset_generator, test_dataset_generator


def dataset_generator(h5file, dset_type, use_class):
    """
    Move h5 dictionary contents to python dict as numpy arrays and create dataset generator
    """
    dataset_dict = dict()
    # Create numpy arrays of given samples
    dataset_dict['rgb'] = np.array(h5file['rgb_' + dset_type],  dtype=np.float32)
    dataset_dict['depth'] = np.array(h5file['depth_' + dset_type], dtype=np.float32)
    dataset_dict['seg_label'] = np.array(h5file['label_' + dset_type], dtype=np.int64)

    # If classification loss is included in training add the classification labels to the dataset as well
    if use_class:
        dataset_dict['class_label'] = np.array(h5file['class_' + dset_type], dtype=np.int64)
    return CreateData(dataset_dict)
