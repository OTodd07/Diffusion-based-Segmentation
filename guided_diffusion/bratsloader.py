import torch
import torch.nn
import numpy as np
import os
import os.path
import SimpleITK as sitk


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
      
        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    # if test_flag and seqtype == 'seg':
                    #     continue
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            img = sitk.ReadImage(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(sitk.GetArrayFromImage(img)))
        out = torch.stack(out)
        if self.test_flag:
            # image=out
            # image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            # return (image, path)

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]


            label = label.numpy()
            multi_label = np.zeros((3,224,224))
            for i in range(1,4):
                np.putmask(multi_label[i-1,:,:], label == i, i > 0)

            label = torch.Tensor(multi_label)
            
            #label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            return (image, label, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]


            label = label.numpy()
            multi_label = np.zeros((3,224,224))
            for i in range(1,4):
                np.putmask(multi_label[i-1,:,:], label == i, i > 0)

            label = torch.Tensor(multi_label)
            
            #label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            return (image, label)

    def __len__(self):
        return len(self.database)

