import os
import json
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from ldm.util import get_obj_from_str
from omegaconf.listconfig import ListConfig


class DatasetBase(Dataset):
    def __init__(self, json_paths: ListConfig, root_path: str, tgt_transform=None, src_transform=None):
        self.data = []
        for json_path in json_paths:
            with open(json_path, 'rt') as f:
                for line in f:
                    self.data.extend(json.loads(line))

        assert len(self.data) > 0, 'No data found in json files.'
        assert os.path.exists(root_path) and os.path.isdir(
            root_path), 'Root path does not exist or is not a directory.'
        assert 'source' in self.data[0] and 'target' in self.data[0], 'Invalid json file format: no source or target field.'

        self.root = ''
        if root_path not in self.data[0]['source'] and root_path not in self.data[0]['target']:
            self.root = root_path

        if type(tgt_transform) is not ListConfig:
            self.tgt_transform = get_obj_from_str(tgt_transform)
        else:
            self.tgt_transform = []
            for transform in tgt_transform:
                self.tgt_transform.append(get_obj_from_str(transform))

        if type(src_transform) is not ListConfig:
            self.src_transform = get_obj_from_str(src_transform)
        else:
            self.src_transform = []
            for transform in src_transform:
                self.src_transform.append(get_obj_from_str(transform))

        assert self.tgt_transform is None or type(self.tgt_transform) is list or callable(
            self.tgt_transform), 'Invalid tgt_transform.'
        assert self.src_transform is None or type(self.src_transform) is list or callable(
            self.src_transform), 'Invalid src_transform.'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filepath = item['source']
        target_filepath = item['target']

        source_filepath = os.path.join(self.root, source_filepath)
        target_filepath = os.path.join(self.root, target_filepath)

        source = nib.load(source_filepath).get_fdata()
        target = nib.load(target_filepath).get_fdata()

        source = source.astype(np.float32)
        if self.src_transform is not None:
            if type(self.src_transform) is list:
                for transform in self.src_transform:
                    source = transform(source)
            else:
                source = self.src_transform(source)

        target = target.astype(np.float32)
        if self.tgt_transform is not None:
            if type(self.tgt_transform) is list:
                for transform in self.tgt_transform:
                    target = transform(target)
            else:
                target = self.tgt_transform(target)

        return dict(jpg=target, hint=source)


class MyDatasetRefineMask(DatasetBase):
    def __init__(self, json_paths: ListConfig, root_path: str, attn_transform=None, tgt_transform=None, src_transform=None, mask_transform=None):
        self.data = []
        for json_path in json_paths:
            with open(json_path, 'rt') as f:
                for line in f:
                    self.data.extend(json.loads(line))

        assert len(self.data) > 0, 'No data found in json files.'
        # assert os.path.exists(root_path) and os.path.isdir(
        #     root_path), 'Root path does not exist or is not a directory.'
        print("root_path: ", root_path)
        assert 'source' in self.data[0] and 'target' in self.data[0], 'Invalid json file format: no source or target field.'

        self.root = ''
        if root_path not in self.data[0]['source'] and root_path not in self.data[0]['target']:
            self.root = root_path

        if type(attn_transform) is not ListConfig:
            self.attn_transform = get_obj_from_str(attn_transform)
        else:
            self.attn_transform = []
            for transform in attn_transform:
                self.attn_transform.append(get_obj_from_str(transform))

        if type(tgt_transform) is not ListConfig:
            self.tgt_transform = get_obj_from_str(tgt_transform)
        else:
            self.tgt_transform = []
            for transform in tgt_transform:
                self.tgt_transform.append(get_obj_from_str(transform))

        if type(src_transform) is not ListConfig:
            self.src_transform = get_obj_from_str(src_transform)
        else:
            self.src_transform = []
            for transform in src_transform:
                self.src_transform.append(get_obj_from_str(transform))

        if type(mask_transform) is not ListConfig:
            self.mask_transform = get_obj_from_str(mask_transform)
        else:
            self.mask_transform = []
            for transform in mask_transform:
                self.mask_transform.append(get_obj_from_str(transform))

        assert self.tgt_transform is None or type(self.tgt_transform) is list or callable(
            self.tgt_transform), 'Invalid tgt_transform.'
        assert self.src_transform is None or type(self.src_transform) is list or callable(
            self.src_transform), 'Invalid src_transform.'
        assert self.mask_transform is None or type(self.mask_transform) is list or callable(
            self.mask_transform), 'Invalid mask_transform.'
        

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filepath = item['source']
        attn_input_filepath = item['attn']
        target_filepath = item['target']
        mask_filepath = item['mask']
        meta_filepath = item['meta']

        source_filepath = os.path.join(self.root, source_filepath)
        attn_input_filepath = os.path.join(self.root, attn_input_filepath)
        target_filepath = os.path.join(self.root, target_filepath)
        mask_filepath = os.path.join(self.root, mask_filepath)
        meta_filepath = os.path.join(self.root, meta_filepath)

        source = nib.load(source_filepath).get_fdata()
        attn_input = nib.load(attn_input_filepath).get_fdata()
        target = nib.load(target_filepath).get_fdata()
        mask = nib.load(mask_filepath).get_fdata()
        with open(meta_filepath, 'r') as file:
            meta = file.readline().strip()
        meta = meta.split()
        meta = [float(meta[0]), float(meta[1]), float(meta[2]), float(meta[3])]
        bvec = [meta[0] * 1000, meta[1] * 1000, meta[2] * 1000]
        bval = meta[3]
        meta = np.array([meta[0] * 1000, meta[1] * 1000, meta[2] * 1000, meta[3]])

        source = source.astype(np.float32)
        if self.src_transform is not None:
            if type(self.src_transform) is list:
                for transform in self.src_transform:
                    source = transform(source)
            else:
                source = self.src_transform(source)

        attn_input = attn_input.astype(np.float32)
        if self.attn_transform is not None:
            if type(self.attn_transform) is list:
                for transform in self.attn_transform:
                    attn_input = transform(attn_input)
            else:
                attn_input = self.attn_transform(attn_input)

        target = target.astype(np.float32)
        if self.tgt_transform is not None:
            if type(self.tgt_transform) is list:
                for transform in self.tgt_transform:
                    target = transform(target)
            else:
                target = self.tgt_transform(target)

        mask = mask.astype(np.float32)
        if self.mask_transform is not None:
            if type(self.mask_transform) is list:
                for transform in self.mask_transform:
                    mask = transform(mask)
            else:
                mask = self.mask_transform(mask)

        return dict(jpg=target, attn_input=attn_input, hint=source, mask=mask, meta=meta)


class MyDataset45chAE(DatasetBase):
    def __init__(self, json_paths: ListConfig, root_path: str, attn_transform=None, tgt_transform=None, src_transform=None, mask_transform=None):
        self.data = []
        for json_path in json_paths:
            with open(json_path, 'rt') as f:
                for line in f:
                    self.data.extend(json.loads(line))

        self.data = [item for item in self.data if 'volume0' in item['target']]

        assert len(self.data) > 0, 'No data found in json files.'
        # assert os.path.exists(root_path) and os.path.isdir(
        #     root_path), 'Root path does not exist or is not a directory.'
        print("root_path: ", root_path)
        assert 'target' in self.data[0], 'Invalid json file format: no source or target field.'

    def __getitem__(self, idx):
        item = self.data[idx]

        target_filepath = item['target']
        source_filepath = item['source']

        data = []
        for i in range(45):
            temp_path = target_filepath.replace('volume0', f'volume{i}')
            target = nib.load(temp_path).get_fdata()
            data.append(target)
        data = np.array(data).transpose(1, 2, 3, 0).astype(np.float32)

        source = nib.load(source_filepath).get_fdata().astype(np.float32)

        return dict(jpg=data, source=source)


class MyLatent4DFODDataset(DatasetBase):
    def __init__(self, json_paths: ListConfig, root_path: str, attn_transform=None, tgt_transform=None, src_transform=None, mask_transform=None):
        self.data = []
        for json_path in json_paths:
            with open(json_path, 'rt') as f:
                for line in f:
                    self.data.extend(json.loads(line))

        assert len(self.data) > 0, 'No data found in json files.'
        # assert os.path.exists(root_path) and os.path.isdir(
        #     root_path), 'Root path does not exist or is not a directory.'
        print("root_path: ", root_path)
        assert 'source' in self.data[0] and 'target' in self.data[0], 'Invalid json file format: no source or target field.'

        self.root = ''
        if root_path not in self.data[0]['source'] and root_path not in self.data[0]['target']:
            self.root = root_path

        if type(tgt_transform) is not ListConfig:
            self.tgt_transform = get_obj_from_str(tgt_transform)
        else:
            self.tgt_transform = []
            for transform in tgt_transform:
                self.tgt_transform.append(get_obj_from_str(transform))

        if type(src_transform) is not ListConfig:
            self.src_transform = get_obj_from_str(src_transform)
        else:
            self.src_transform = []
            for transform in src_transform:
                self.src_transform.append(get_obj_from_str(transform))

        if type(mask_transform) is not ListConfig:
            self.mask_transform = get_obj_from_str(mask_transform)
        else:
            self.mask_transform = []
            for transform in mask_transform:
                self.mask_transform.append(get_obj_from_str(transform))

        assert self.tgt_transform is None or type(self.tgt_transform) is list or callable(
            self.tgt_transform), 'Invalid tgt_transform.'
        assert self.src_transform is None or type(self.src_transform) is list or callable(
            self.src_transform), 'Invalid src_transform.'
        assert self.mask_transform is None or type(self.mask_transform) is list or callable(
            self.mask_transform), 'Invalid mask_transform.'
        

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filepath = item['source']
        target_filepath = item['target']
        mask_filepath = item['mask']

        source_filepath = os.path.join(self.root, source_filepath)
        target_filepath = os.path.join(self.root, target_filepath)
        mask_filepath = os.path.join(self.root, mask_filepath)

        source = nib.load(source_filepath).get_fdata()
        target = nib.load(target_filepath).get_fdata()
        mask = nib.load(mask_filepath).get_fdata()
        
        source = source.astype(np.float32)
        if self.src_transform is not None:
            if type(self.src_transform) is list:
                for transform in self.src_transform:
                    source = transform(source)
            else:
                source = self.src_transform(source)

        target = target.astype(np.float32)
        if self.tgt_transform is not None:
            if type(self.tgt_transform) is list:
                for transform in self.tgt_transform:
                    target = transform(target)
            else:
                target = self.tgt_transform(target)

        mask = mask.astype(np.float32)
        if self.mask_transform is not None:
            if type(self.mask_transform) is list:
                for transform in self.mask_transform:
                    mask = transform(mask)
            else:
                mask = self.mask_transform(mask)

        return dict(jpg=target, hint=source, mask=mask)


class MyDatasetRefine1(DatasetBase):
    def __init__(self, json_paths: ListConfig, root_path: str, tgt_transform=None, src_transform=None):
        super().__init__(
            json_paths=json_paths,
            root_path=root_path,
            tgt_transform=tgt_transform,
            src_transform=src_transform
        )

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filepath = item['source'].replace('mr', 'sct20')
        target_filepath = item['target']

        source_filepath = os.path.join(self.root, source_filepath)
        target_filepath = os.path.join(self.root, target_filepath)

        source = nib.load(source_filepath).get_fdata()
        target = nib.load(target_filepath).get_fdata()

        source = source.astype(np.float32)
        if self.src_transform is not None:
            if type(self.src_transform) is list:
                for transform in self.src_transform:
                    source = transform(source)
            else:
                source = self.src_transform(source)

        target = target.astype(np.float32)
        if self.tgt_transform is not None:
            if type(self.tgt_transform) is list:
                for transform in self.tgt_transform:
                    target = transform(target)
            else:
                target = self.tgt_transform(target)

        return dict(jpg=target, hint=source)


class MyDatasetRefine2(DatasetBase):
    def __init__(self, json_paths: ListConfig, root_path: str, tgt_transform=None, src_transform=None):
        super().__init__(
            json_paths=json_paths,
            root_path=root_path,
            tgt_transform=tgt_transform,
            src_transform=src_transform
        )

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filepath = item['source']
        mr_filepath = item['source'].replace('refined', 'mr')
        target_filepath = item['target']

        source_filepath = os.path.join(self.root, source_filepath)
        mr_filepath = os.path.join(self.root, mr_filepath)
        target_filepath = os.path.join(self.root, target_filepath)

        source = nib.load(source_filepath).get_fdata()
        mr = nib.load(mr_filepath).get_fdata()
        target = nib.load(target_filepath).get_fdata()

        # source images already in [0, 1].
        source = source.astype(np.float32)
        mr = mr.astype(np.float32)
        source = np.concatenate((source, mr), axis=1)

        if self.src_transform is not None:
            if type(self.src_transform) is list:
                for transform in self.src_transform:
                    source = transform(source)
            else:
                source = self.src_transform(source)

        target = target.astype(np.float32)
        if self.tgt_transform is not None:
            if type(self.tgt_transform) is list:
                for transform in self.tgt_transform:
                    target = transform(target)
            else:
                target = self.tgt_transform(target)

        return dict(jpg=np.transpose(target, (0, 2, 1)), hint=np.transpose(source, (0, 2, 1)))
