import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json

class AlzheimerDataset(Dataset):
    def __init__(self, root, test_envs, max_seq_length=512):
        self.root = root
        assert test_envs in [0, 1], "Only 2 Domains"
        self.test_envs = test_envs
        self.data = []
        self.labels = []
        self.idx2class_name = {0: "EMOTION", 1: "GAMBLING", 2: "LANGUAGE", 3: "MOTOR", 4: "RELATIONAL", 5: "SOCIAL", 6: "WM"}
        self.max_seq_length = max_seq_length

        subfolders = [f for f in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, f))]
        train_folder = subfolders[1 - test_envs]
        print(f'Loading {train_folder}')
        self.process_folder(train_folder)

    def min_max_scale(self, data_array):
        min_val = data_array.min(axis=0)
        max_val = data_array.max(axis=0)
        scale = np.where(max_val - min_val == 0, 1, max_val - min_val)
        scaled_data = (data_array - min_val) / scale
        return scaled_data

    def process_folder(self, folder):
        folder_path = os.path.join(self.root, folder)
        data_dir = os.path.join(folder_path, 'processed_data.npy')
        labels_dir = os.path.join(folder_path, 'processed_labels.npy')
        idx2class_name_path = os.path.join(folder_path, 'idx2class_name.json')
        if os.path.exists(data_dir) and os.path.exists(labels_dir):
            print(f'Loading data from {data_dir} and {labels_dir}')
            self.data = np.load(data_dir, allow_pickle=True)
            self.labels = np.load(labels_dir, allow_pickle=True)
            with open(idx2class_name_path, 'r') as json_file:
                self.idx2class_name = json.load(json_file)
        else:
            print(f'Processing data from {folder_path}')
            for filename in tqdm(os.listdir(folder_path), desc=f'Loading Data'):
                if filename.endswith('.csv'):
                    class_name = filename.split('_')[1]
                    csv_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(csv_path, sep = '\t')
                    data_array = df.values
                    # min-max scaling
                    #data_array = self.min_max_scale(data_array)
                    
                    # padding 
                    padded_data_array = self.pad_sequence(data_array)

                    self.data.append(padded_data_array)
                    label_idx = self.get_class_index(class_name)
                    self.labels.append(label_idx)
            

            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
            np.save(data_dir, self.data)
            np.save(labels_dir, self.labels)
            with open(idx2class_name_path, 'w') as json_file:
                json.dump(self.idx2class_name, json_file)

    def pad_sequence(self, data_array):
        padding_length = self.max_seq_length - data_array.shape[0]
        if padding_length > 0:
            padded_sequence = np.pad(data_array, ((0, padding_length), (0, 0)), 
                                     'constant', constant_values=0)
        else:
            # If no padding is needed, truncate the sequence
            padded_sequence = data_array[:self.max_seq_length]
        return padded_sequence

    def get_class_index(self, class_name):
        return list(self.idx2class_name.keys())[list(self.idx2class_name.values()).index(class_name)]
    
    def get_class_number(self):
        return len(self.idx2class_name)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        # create attention mask
        attention_mask = torch.tensor((self.data[index] != 0).all(axis=1), dtype=torch.long)
        inputs = {"data": data, "attention_mask": attention_mask, "labels": label}
        return inputs
    


class HCPWMDataset(Dataset):
    def __init__(self, root, test_env, max_seq_length = 512, update = False, args = None):
        assert test_env in [0, 1], "Only 2 Domains"
        self.root = root
        self.max_seq_length = max_seq_length
        self.train_folder = 'Scan1' if test_env == 0 else 'Scan2'
        label_folder = 'scan1' if test_env == 0 else 'scan2'
        self.labels_files = pd.read_csv(os.path.join(root, f'timing_{label_folder}.csv'))
        self.data_files = sorted([f for f in os.listdir(os.path.join(root, self.train_folder)) if f.endswith('.csv')])
        self.data = []
        self.labels = []
        # remove 5
        if args.skip_5:
            self.label_mapping = {7: 0, 9: 1, 13: 2, 15: 3, 2: 4, 17: 5, 4: 6, 11: 7}   # 8
        else:
            self.label_mapping = {7: 0, 9: 1, 5: 2, 13: 3, 15: 4, 2: 5, 17: 6, 4: 7, 11: 8}  # 9
        self.args = args
        self.process_folder(self.train_folder, update, args)
        

    def __len__(self):
        return len(self.data)
    
    def pad_sequence(self, data_array):
        padding_length = self.max_seq_length - data_array.shape[0]
        if padding_length > 0:
            padded_sequence = np.pad(data_array, ((0, padding_length), (0, 0)), 
                                     'constant', constant_values=0)
        else:
            # If no padding is needed, truncate the sequence
            print(f'Warning: Sequence length {data_array.shape[0]} is longer than max sequence length {self.max_seq_length}')
            padded_sequence = data_array[:self.max_seq_length]
        return padded_sequence
    
    def get_class_number(self):
        return len(self.label_mapping)

    def max_scale(self, data_array):
        max_val = data_array.max(axis=0)  
        max_val[max_val == 0] = 1
        scaled_data = data_array / max_val  
        return scaled_data


    def process_folder(self, folder, update, args):
        folder_path = os.path.join(self.root, folder)
        if args.skip_5:
            data_dir = os.path.join(folder_path, 'processed_data_skip5.npy')
            labels_dir = os.path.join(folder_path, 'processed_labels_skip5.npy')
        else:
            data_dir = os.path.join(folder_path, 'processed_data.npy')
            labels_dir = os.path.join(folder_path, 'processed_labels.npy')
        if os.path.exists(data_dir) and os.path.exists(labels_dir) and not update:
            print(f'Loading data from {data_dir} and {labels_dir}')
            self.data = np.load(data_dir, allow_pickle=True)
            self.labels = np.load(labels_dir, allow_pickle=True)
        else:
            print(f'Processing data from {folder_path}')
            for filename in tqdm(sorted(os.listdir(folder_path)), desc=f'Loading Data'):
                if filename.endswith('.csv'):
                    csv_file = os.path.join(folder_path, filename)
                    data = pd.read_csv(csv_file, header=None).values.T # DL to LD
                    if np.isnan(data).any() or np.isinf(data).any():
                        print(f'Warning: NaN or Inf found in {filename}')
                        continue
                    # get labels
                    file_index = str(filename.split('.')[0])
                    if file_index not in self.labels_files.columns:
                        #raise KeyError(f'File index {file_index} not found in labels')
                        print(f'File index {file_index} not found in labels')
                        continue
                    sequence_labels = self.labels_files[file_index].values
                    # split the sequqnces according to labels and pad
                    sequences = []
                    labels = []
                    start = 0
                    for timepoint in range(0, len(sequence_labels)):
                        if sequence_labels[timepoint] != sequence_labels[start]:
                            if sequence_labels[start] == 5 and args.skip_5:
                                start = timepoint
                                continue
                            else:
                                sequence = data[start:timepoint, :]
                                # max scaling
                                sequence = self.max_scale(sequence)
                                # pad
                                padded_sequence = self.pad_sequence(sequence)
                                sequences.append(padded_sequence)
                                mapped_label = self.label_mapping[sequence_labels[start]]
                                labels.append(mapped_label)
                                start = timepoint
                    # add the last sequence
                    condition = sequence_labels[start] == 5 and args.skip_5
                    if start < len(sequence_labels) and not condition:
                        sequence = data[start:, :]
                        # max scaling
                        sequence = self.max_scale(sequence)
                        padded_sequence = self.pad_sequence(sequence)
                        sequences.append(padded_sequence)
                        mapped_label = self.label_mapping[sequence_labels[start]]
                        labels.append(mapped_label)

                    self.data.extend(sequences)
                    self.labels.extend(labels)
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
            np.save(data_dir, self.data)
            np.save(labels_dir, self.labels)
            print(f'Saved processed data to {data_dir} and {labels_dir}')
                    
    
    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        # create attention mask
        attention_mask = torch.tensor((self.data[index] != 0).all(axis=1), dtype=torch.long)
        inputs = {"data": data, "attention_mask": attention_mask, "labels": label}
        return inputs
        
                