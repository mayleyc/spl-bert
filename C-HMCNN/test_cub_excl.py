import os
import datetime
import json
from time import perf_counter
import copy
import pickle
import glob

import torch
import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import (
    precision_score, 
    average_precision_score, 
    hamming_loss, 
    jaccard_score
)
from sklearn.model_selection import train_test_split

import json
from timeit import default_timer as timer


# Circuit imports
import sys
sys.path.append(os.path.join(sys.path[0],'hmc-utils'))
sys.path.append(os.path.join(sys.path[0],'hmc-utils', 'pypsdd'))

from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE
from pysdd.sdd import SddManager, Vtree

from sklearn import preprocessing


# misc
from common import *


def log1mexp(x):
        assert(torch.all(x >= 0))
        return torch.where(x < 0.6931471805599453094, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

from torch.utils.data import Dataset
from PIL import Image

#Functions for 
'''class CUB_Dataset(Dataset):
    def __init__(self, image_paths, labels, transform = None, to_eval = True):
        """
        Args:
            image_paths (list): List of image file paths.
            transform (callable, optional): transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.to_eval = to_eval
    def __len__(self):
        """
        Returns dataset size.
        """

        return len(self.image_paths)

    def process_image(self, img_path):
        """
        Load, transform and pad an image.
        """
        # load image
        image_pil = Image.open(img_path).convert("RGB")  # load image

        if self.transform:
            image = self.transform(image_pil)  # 3, h, w
        else:
            transform = T.Compose(
            [
                T.Lambda(lambda img: resize_image(img, height=800, max_width=1333)),  # Resize with fixed height
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
            image = transform(image_pil)  # 3, H, W
        # Get current height and width
        _, H, W = image.shape

        # Compute padding
        pad_w = max(667 - W, 0)  # Only pads if W < 1333
        pad_h = max(400 - H, 0)    # Only pads if H < 800

        # Compute symmetric padding
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Apply padding correctly: (left, top, right, bottom)
        padded_image = F.pad(image, [pad_left, pad_top, pad_right, pad_bottom])  # Padding with 0s

        return image_pil, padded_image, padded_image.shape  #image = tensor
    
    def __getitem__(self, idx):
        """
        Get and process a sample given index `idx`.
        """
        img_path = self.image_paths[idx]
        label_set = self.labels[idx]
        _, image, _ = self.process_image(img_path)

        return image, label_set
'''   
class Embeddings_Dataset(Dataset):
    def __init__(self, embeddings, labels, to_eval = True):
        self.embeddings = embeddings
        self.labels = labels
        self.to_eval = to_eval

    def __len__(self):
        """
        Returns dataset size.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get and process a sample given index `idx`.
        """
        emb = self.embeddings[idx]
        label = self.labels[idx]

        return emb, label


class ConstrainedFFNNModel(nn.Module):
    """ C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss """
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R, dataset):
        super(ConstrainedFFNNModel, self).__init__()
        
        self.nb_layers = hyperparams['num_layers']
        self.R = R
        self.dataset = dataset
        '''
        if "cub" in self.dataset:
            self.conv1 = nn.Conv2d(3, 32, 3)
            self.conv2 = nn.Conv2d(32, 64, 3)

            self.pool = nn.MaxPool2d(2, 2) # Downsampling: reduce each dimension by half

            self.conv3 = nn.Conv2d(64, 128, 3)
            #self.conv4 = nn.Conv2d(128, 256, 3)

            #self.conv5 = nn.Conv2d(256, 512, 3)
            #self.conv6 = nn.Conv2d(512, 512, 3)

            # Adaptive Pooling
            self.global_pool = nn.AdaptiveAvgPool2d((7, 7)) # like resnet-50
            '''

        fc = []
        
        for i in range(self.nb_layers):
            if i == 0:
                '''
                if "cub" in dataset:
                    fc.append(nn.Linear(128 * 7 * 7, hidden_dim))
                else:
                    fc.append(nn.Linear(input_dim, hidden_dim))
                    '''
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers-1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)
        
        self.drop = nn.Dropout(hyperparams['dropout'])
        
        if hyperparams['non_lin'] == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()
        
    def forward(self, x, sigmoid=False, log_sigmoid=False):
        '''
        if "cub" in self.dataset:
            x = self.pool(self.f(self.conv1(x)))
            x = self.pool(self.f(self.conv2(x)))
            
            x = self.pool(self.f(self.conv3(x)))
            #x = self.pool(self.f(self.conv4(x)))
            
            #x = self.pool(self.f(self.conv5(x)))
            #x = self.pool(self.f(self.conv6(x)))

            x = self.global_pool(x)
            x = torch.flatten(x, 1)  # Flatten for fully connected layers
        '''
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                if sigmoid:
                    x = nn.Sigmoid()(self.fc[i](x))
                elif log_sigmoid:
                    x = nn.LogSigmoid()(self.fc[i](x))
                else:
                    x = self.fc[i](x)
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)

        
        if self.R is None:
            return x
        
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out
    
class LeNet5(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R, dataset):
            super().__init__()
            self.nb_layers = hyperparams['num_layers']
            self.R = R
            self.dataset = dataset
            '''
            if "cub" in self.dataset:
                self.layer1 = nn.Sequential(
                    nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
                    nn.BatchNorm2d(6),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2))
                self.layer2 = nn.Sequential(
                    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2))
            if "cub" in self.dataset:
                # Dynamically compute the feature map size after layer2
                sample = torch.rand(1, 3, 400, 667) #dummy sample to calculate size
                out = self.layer2(self.layer1(sample))
                flattened_dim = out.view(1, -1).shape[1]

                self.fc = nn.Linear(flattened_dim, hidden_dim)
            else:
                self.fc = nn.Linear(input_dim, hidden_dim)
            '''
            self.fc = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            
    def forward(self, x, sigmoid=False, log_sigmoid=False):
            '''
            if "cub" in self.dataset:
                x = self.layer1(x)
                x = self.layer2(x)
                x = x.reshape(x.size(0), -1)
            '''
            x = self.fc(x)
            x = self.relu(x)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            if self.R is None:
                return x
            if self.training:
                constrained_out = x
            else:
                constrained_out = get_constr_out(x, self.R)
            return constrained_out
    

def main():

    args = parse_args()

    # Set device
    torch.cuda.set_device(int(args.device))
    #print(torch.cuda.is_available())  # Should print True
    #print(torch.cuda.device_count())  # Should match the number of GPUs
    #print(torch.cuda.get_device_name(0))  # Check which GPU PyTorch is using

    # Load train, val and test set
    dataset_name = args.dataset
    data = dataset_name.split('_')[0]
    spl_bert_datasets = {}
    #ontology = dataset_name.split('_')[1]
    hidden_dim = 1000 #hidden_dims[ontology][data]
    output_dim = 128 #not the number of classes

    num_epochs = args.n_epochs
    
    with open("embeddings_config.json") as f:
        emb_fp = json.load(f)

    #mat_path = mat_path_full
    #csv_path = csv_path_full

    # Set the hyperparameters 
    hyperparams = {
        'num_layers': 3,
        'dropout': 0.2, #0.7
        'non_lin': 'relu',
    }

    # Set seed
    seed_all_rngs(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    emb_model_name = "bert-base-uncased"
    
    '''
    # list of files in cub 2011 and y labels
    if "cub" in args.dataset:
        #Try out with 5 classes in CUB
        # Get all class folder names
        all_classes = sorted(os.listdir(images_dir))  # Sorting ensures consistency
        
        # To use the CUB mini dataset, enter the dataset as cub_mini. For full CUB, use cub_others
        if "mini" in args.dataset:
        # Select all classes or only the first 5 classes (mini csv, mat and image set)
            selected_classes = all_classes[:5]
            csv_path = csv_path_mini
            mat_path = mat_path_mini
        else:
            selected_classes = all_classes
            csv_path = csv_path_full
            mat_path = mat_path_full

        # Get image paths only for selected classes
        image_paths = []
        for cls in selected_classes:
            class_images = glob.glob(os.path.join(images_dir, cls, "*.jpg"))
            image_paths.extend(class_images)

        labels_unprocessed = [os.path.basename(os.path.dirname(path)) for path in image_paths]
        label_species = [label.split('.')[-1] for label in labels_unprocessed]
        label_species = [re.sub('_', ' ', label) for label in label_species] # the species-level label for each image
        # Create one-hot encoding based on species lookup in the csv
        ohe_dict, unique_val_map = get_one_hot_labels(label_species, csv_path)
        
        # Create labels for trials
        ohe_dict_50 = {}
        for index, i in enumerate(ohe_dict.keys()):
            if index < len(ohe_dict)/2:
                ohe_dict_50[i] = np.zeros(ohe_dict[i].shape)
            else:
                ohe_dict_50[i] = ohe_dict[i]
        
        ohe_dict_0 = {}
        for i in ohe_dict.keys():
            ohe_dict_0[i] = np.zeros(ohe_dict[i].shape)
        ohe_dict_1 = {}
        for i in ohe_dict.keys():
            ohe_dict_1[i] = np.ones(ohe_dict[i].shape)
       
        
        #print(ohe_dict)
        #image_labels = [torch.from_numpy(ohe_dict[species]).to(device) for species in label_species]
        image_labels = [torch.from_numpy(ohe_dict[species]).to(device) for species in label_species]
        
        #Define image transform process
        transform = T.Compose(
                [
                    T.Lambda(lambda img: resize_image(img, height=400, max_width=667)),  # Resize with fixed height
                    #T.Resize(800, max_size=1333), # like GroundingDINO
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            '''
    
    if data in {"amazon", "bgc", "wos"}:  
        # load pickle at /embeddings
        #emb_file = find_latest_emb_file(emb_model_name, dataset_name)
        train_emb_file = emb_fp[data]["train"]
        test_emb_file = emb_fp[data]["test"]
        val_emb_file = emb_fp[data]["val"]
        
        all_embeddings = {}
        all_labels = {}
        all_texts = {}

        dataset_files = {
                        'train': train_emb_file,
                        'val': val_emb_file,
                        'test': test_emb_file
                        }
        
        #ohe_df = pd.read_csv(ohe_csv, index_col=0).astype(int)

        for split, emb_file in dataset_files.items():
            with open(emb_file, "rb") as f:
                texts, emb_list, ohe_labels = pickle.load(f)
            
            #label_species = [label[-1] for label in labels_unprocessed]
            
            #ohe_labels = [torch.from_numpy(ohe_df[species]).to(device) for species in label_species]
            
            #Normalize tensors
            emb_tensors = [
                                nn.functional.normalize(torch.tensor(embe, dtype=torch.float32, device=device), p=2.0, dim=-1, eps=1e-12)
                                for embe in emb_list
                                ]
            ohe_labels = [ohe.detach().clone() for ohe in ohe_labels]

            all_embeddings[split] = emb_tensors
            all_labels[split] = ohe_labels
            all_texts[split] = texts
        
        #print(all_labels['train'][0])
        #quit()


        '''for i, emb_file in enumerate([train_emb_file, val_emb_file, test_emb_file]):
        # Load and split dataset into train, val, and test sets
            with open(emb_file, "rb") as f:
                texts, embeddings, labels_unprocessed = pickle.load(f)
            label_species = [label[-1] for label in labels_unprocessed]
            #label_species = [re.sub('_', ' ', label) for label in label_species] # the species-level label for each image
            
            #ohe_dict, _ = get_one_hot_labels(label_species, csv_path)
            ohe_labels = [torch.from_numpy(ohe_df[species]).to(device) for species in label_species]
            
            embeddings_tensor = [torch.tensor(emb) for emb in embeddings]

            if i in dataset_map:
                data_split = dataset_map[i]
                embeddings[data_split] = embeddings_tensor
                labels[data_split] = ohe_labels

            if i == 0: 
                train_emb, train_labels = embeddings_tensor, ohe_labels
            elif i == 1: # no validation set?
                val_emb, val_labels = embeddings_tensor, ohe_labels
            elif i == 2:
                test_emb, test_labels = embeddings_tensor, ohe_labels'''
        
            #train_emb, temp_emb, train_labels, temp_labels = train_test_split(all_embeddings_tensor, ohe_labels, test_size=0.3, random_state=args.seed)
            #val_emb, test_emb, val_labels, test_labels = train_test_split(temp_emb, temp_labels, test_size=0.7, random_state=args.seed)
            
    elif ('others' in args.dataset):
        train, test = initialize_other_dataset(dataset_name, datasets)
        train.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.bool),  torch.tensor(test.to_eval, dtype=torch.bool)
        train.X, valX, train.Y, valY = train_test_split(train.X, train.Y, test_size=0.30, random_state=args.seed)
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)
        train.to_eval, val.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.bool), torch.tensor(val.to_eval, dtype=torch.bool), torch.tensor(test.to_eval, dtype=torch.bool)
        #print(train.Y.shape)

        #Create loaders
    if data in {"amazon", "bgc", "wos"}: 
        # Create datasets for each split: Change labels
        train_dataset = Embeddings_Dataset(all_embeddings["train"], all_labels["train"], to_eval = True)
        #val_dataset = Embeddings_Dataset(val_emb, val_labels, to_eval = True)
        test_dataset = Embeddings_Dataset(all_embeddings["test"], all_labels["test"], to_eval = True)
        val_dataset = Embeddings_Dataset(all_embeddings["val"], all_labels["val"], to_eval = True)

        # convert them into tensors: shape = output_dim + 1
        train_dataset.to_eval, val_dataset.to_eval, test_dataset.to_eval = torch.tensor(train_dataset.to_eval, dtype=torch.bool), torch.tensor(val_dataset.to_eval, dtype=torch.bool), torch.tensor(test_dataset.to_eval, dtype=torch.bool)
        #train_dataset.to_eval, test_dataset.to_eval = torch.tensor(train_dataset.to_eval, dtype=torch.bool), torch.tensor(test_dataset.to_eval, dtype=torch.bool)

    else:
        train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
        if ('others' not in args.dataset):
            val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]
            #for (x, y) in zip(val.X, val.Y):
            #    train_dataset.append((x,y))
        else:
            val_dataset = [(x, y) for (x, y) in zip(valX, valY)]

        test_dataset = [(x, y) for (x, y) in zip(test.X, test.Y)]

    #different_from_0 = torch.tensor(np.array((test.Y.sum(0)!=0), dtype = bool), dtype=torch.bool)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)

    # We do not evaluate the performance of the model on the 'roots' node (https://dtai.cs.kuleuven.be/clus/hmcdatasets/)
    if 'GO' in dataset_name:
        num_to_skip = 4
    else:
        num_to_skip = 1

    # Prepare matrix
    if data in {"amazon", "bgc", "wos"}:     
        mat = np.load(mat_path_dict[data])
    else:
        mat = train.A

    # Prepare circuit: TODO needs cleaning
    if not args.no_constraints:
        #print(mat.shape) #500x500 classes
        #print(np.array(train_labels).shape)
        
        if not os.path.isfile('constraints/' + dataset_name + '_excl' + '.sdd') or not os.path.isfile('constraints/' + dataset_name + '_excl' + '.vtree'):
            # Compute matrix of ancestors R
            # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
            #np.savetxt("foo.csv", mat, delimiter=",") #Check mat
            R = np.zeros(mat.shape)
            np.fill_diagonal(R, 1)
            g = nx.DiGraph(mat)
            layer_map = layer_mapping_BFS(g.reverse(copy=True), num_st_nodes) # 1-indexed # keep original g
            for i in range(len(mat)):
                descendants = list(nx.descendants(g, i))
                if descendants:
                    R[i, descendants] = 1
            R = torch.tensor(R)

            #Transpose to get the ancestors for each node 
            R = R.unsqueeze(0).to(device)

            # Uncomment below to compile the constraint
            R.squeeze_()
            mgr = SddManager(
                var_count=R.size(0),
                auto_gc_and_minimize=True)
            
            max_layer = max(layer_map.values())

            me_layers = layer_map.values() #{l for l in layer_map.values() if l != max_layer} #{max_layer-1, max_layer} #layer_map.values() #
            nz_layers = {1} #bgc has Level 1 annotations :(
            # there is no root in R

            '''alpha = mgr.true()
            alpha.ref()
            for i in range(R.size(0)):

                beta = mgr.true()
                beta.ref()
                for j in range(R.size(0)):

                    if R[i][j] and i != j:
                        old_beta = beta
                        beta = beta & mgr.vars[j+1]
                        beta.ref()
                        old_beta.deref()

                old_beta = beta
                beta = -mgr.vars[i+1] | beta
                beta.ref()
                old_beta.deref()

               #DELTA - ME constraint
                if layer_map[i] not in me_layers:
                    continue
                #make a list of indices of all species under g
                species = [j for j in range(R.size(0)) if R[i][j] and i != j]

                for idx1 in range(len(species)):
                    for idx2 in range(idx1 + 1, len(species)): # all species after s1

                        old_beta = beta
                        beta = beta & (-mgr.vars[idx1+1] | -mgr.vars[idx2+1]) #one clause must be true # sdd count starts at 1
                        beta.ref()
                        old_beta.deref()


                old_alpha = alpha
                alpha = alpha & beta
                alpha.ref()
                old_alpha.deref()
'''

            alpha = mgr.true()
            alpha.ref()
            
            for i in range(R.size(0)):
                beta = mgr.true()
                beta.ref()
                for j in range(R.size(1)):

                    if R[i][j] and i != j: # why not R[j][i]?
                        old_beta = beta
                        beta = beta & mgr.vars[j+1] # conjunction of all of its children: true & j1 & j2 & ...
                        beta.ref()
                        old_beta.deref()

                old_beta = beta
                beta = -mgr.vars[i+1] | beta
                beta.ref()
                old_beta.deref()

                old_alpha = alpha
                alpha = alpha & beta
                alpha.ref()
                old_alpha.deref()

            print("alpha after beta hierarchy:", alpha.is_true(), alpha.is_false(), alpha.model_count())
                
                #initialize delta for ME
            
            for i in range(R.size(0)):    
                delta = mgr.true() 
                delta.ref()
                if layer_map[i] in me_layers:
                    #make a list of indices of all species under g
                    species = [j for j in range(R.size(1)) if R[j][i] and layer_map[j] == layer_map[i] + 1] #i != j]
                    if species:
                        #print(f"row {i}: {species}")
                    
                        for idx1 in range(len(species)):
                            for idx2 in range(idx1 + 1, len(species)): # all species after s1
                                s1 = species[idx1]   # actual node index
                                s2 = species[idx2]   

                                old_delta = delta
                                delta = delta & (-mgr.vars[s1+1] | -mgr.vars[s2+1]) #OR: one clause must be true # sdd count starts at 1                       
                                #delta = delta & (-mgr.vars[idx1+1] | -mgr.vars[idx2+1]) #OR: one clause must be true # sdd count starts at 1
                                delta.ref()
                                old_delta.deref()

                old_delta = delta
                delta = -mgr.vars[i+1] | delta
                delta.ref()
                old_delta.deref()

                old_alpha = alpha
                alpha = alpha & delta
                alpha.ref()
                old_alpha.deref()
                
        
            print("alpha after delta hierarchy:", alpha.is_true(), alpha.is_false(), alpha.model_count())

            
            # NONZERO CONSTRAINTS HERE
            

            for i in range(R.size(0)):
                # initialize for every node
                #zeta = mgr.false() 
                #zeta.ref()
                
                #(R[i] == 1).nonzero(as_tuple=True)[0].tolist() #
                if layer_map[i] in nz_layers:
                    
                    species_nz = [j for j in range(R.size(1)) if R[j][i] and layer_map[j] == layer_map[i] + 1]
                    #species_nz = [j for j in range(R.size(0)) if R[i][j]] #and layer_map[j] == layer_map[i] + 1] #PROBLEM HERE: THIS IS [0] 
                    #print(species_nz) #[0]
                    #quit()

                    '''if species_nz: # RESULTS: alpha after zeta hierarchy: 0 0 702485784
                        #print(f"row {i}: {species_nz}")
                        for s in species_nz: # loop to OR every species
                            old_zeta = zeta
                            # A -> B => -A | A = -A | B
                            #check for 1 single parent-child pair (from beta downwards): should go down by 1
                            #Write down on paper the truth table
                            #without constraints: 2^n + 1
                            zeta = zeta | mgr.vars[s+1] #select one #negative?
                            zeta.ref()
                            old_zeta.deref()'''
                    if not species_nz:
                        continue
                    # RESULTS: alpha after zeta hierarchy: 0 0 479001600
                    #if species_nz:
                    #old_zeta = zeta
                    zeta = mgr.vars[species_nz[0]+1]
                    zeta.ref()
                    #old_zeta.deref()
                    for s in species_nz[1:]: # loop to OR every species
                        old_zeta = zeta
                        # A -> B => -A | A = -A | B
                        #check for 1 single parent-child pair (from beta downwards): should go down by 1
                        #Write down on paper the truth table
                        #without constraints: 2^n + 1
                        
                        zeta = zeta | mgr.vars[s+1]                            
                        zeta.ref()
                        old_zeta.deref()

                    old_zeta = zeta
                    zeta = -mgr.vars[i+1] | zeta
                    zeta.ref()
                    old_zeta.deref()

                    old_alpha = alpha
                    alpha = alpha & zeta
                    alpha.ref()
                    old_alpha.deref()   
                    #zeta.deref()

            #print("zeta (nonzero):", zeta.is_true(), zeta.is_false(), zeta.model_count())      

                # Mutual exclusivity logic
            #try with 1 parent: see if the numbers are correct
            

            '''for i in range(R.size(0)): #for all genera g
                #initialize delta for ME
                delta = mgr.true()
                delta.ref()
                if layer_map[i] not in me_layers:
                    continue
                #make a list of indices of all species under g
                species = [j for j in range(R.size(0)) if R[i][j] and layer_map[j] == layer_map[i] + 1] #i != j]
                if not species:
                    continue

                for idx1 in range(len(species)):
                    for idx2 in range(idx1 + 1, len(species)): # all species after s1
                        s1 = species[idx1]   # actual node index
                        s2 = species[idx2]   

                        old_delta = delta
                        delta = delta & (-mgr.vars[s1+1] | -mgr.vars[s2+1]) #OR: one clause must be true # sdd count starts at 1                       
                        #delta = delta & (-mgr.vars[idx1+1] | -mgr.vars[idx2+1]) #OR: one clause must be true # sdd count starts at 1
                        delta.ref()
                        old_delta.deref()

                old_alpha = alpha
                alpha = alpha & delta
                alpha.ref()
                old_alpha.deref()
                
        
            print("alpha after delta hierarchy:", alpha.is_true(), alpha.is_false(), alpha.model_count())

            # NONZERO CONSTRAINTS HERE
            
            
            for i in range(R.size(0)): 
                zeta = mgr.false()
                zeta.ref()
                if layer_map[i] not in nz_layers:
                    continue
                
                species_nz = [j for j in range(R.size(0)) if R[i][j] and layer_map[j] == layer_map[i] + 1]
                if not species_nz:
                    continue
                for s in species_nz: # loop to OR every species
                    old_zeta = zeta
                    # A -> B => -A | A = -A | B
                    #check for 1 single parent-child pair (from beta downwards): should go down by 1
                    #Write down on paper the truth table
                    #without constraints: 2^n + 1
                    zeta = zeta | mgr.vars[s+1] #select one #negative?
                    zeta.ref()
                    old_zeta.deref()

                old_alpha = alpha
                alpha = alpha & zeta
                alpha.ref()
                old_alpha.deref()   '''            

            
            
            print("alpha after zeta hierarchy:", alpha.is_true(), alpha.is_false(), alpha.model_count())

            print("delta (mutual exclusivity):", delta.is_true(), delta.is_false(), delta.model_count())

            print("zeta (nonzero):", zeta.is_true(), zeta.is_false(), zeta.model_count())
            #quit()

            alpha.save(str.encode('constraints/' + dataset_name + '_excl'+ '.sdd'))
            alpha.vtree().save(str.encode('constraints/' + dataset_name + '_excl'+ '.vtree'))
            


        # Create circuit object
        cmpe = CircuitMPE('constraints/' + dataset_name + '_excl'+ '.vtree', 'constraints/' + dataset_name + '_excl'+ '.sdd')

        if args.S > 0:
            cmpe.overparameterize(S=args.S)
            print("Done overparameterizing")

        # Create gating function
        gate = DenseGatingFunction(cmpe.beta, gate_layers=[128] + [256]*args.gates, num_reps=args.num_reps).to(device)

        R = None


    else:
        # Use fully-factorized sdd
        mgr = SddManager(var_count=mat.shape[0], auto_gc_and_minimize=True)
        alpha = mgr.true()
        vtree = Vtree(var_count = mat.shape[0], var_order=list(range(1, mat.shape[0] + 1)))
        alpha.save(str.encode('ancestry.sdd'))
        vtree.save(str.encode('ancestry.vtree'))
        cmpe = CircuitMPE('ancestry.vtree', 'ancestry.sdd')
        cmpe.overparameterize()

        # Gating function
        gate = DenseGatingFunction(cmpe.beta, gate_layers=[128]).to(device) #changed 462 to 128. why 462?

        R = None


    # Create the model
    # Load train, val and test set
    
    model = ConstrainedFFNNModel(input_dims[data], hidden_dim, output_dim, hyperparams, R, args.dataset) # 1% at 45 ep, learns faster?/better? but accuracy still low, 13%
    #model = LeNet5(input_dims[data], hidden_dim, output_dim, hyperparams, R, args.dataset) #1% accuracy at 80 epochs
    #if args.no_train:
    #best_file, best_loss = find_best_pth_file(model.__class__.__name__)
    pretrained = args.pretrained
    #checkpoint_model = torch.load(best_file)
    #checkpoint_gate = torch.load(re.sub("model", "gate", str(best_file)))
    checkpoint_model = torch.load("models/250825_model-d_wos_me_nz_model.pth", weights_only=False)
    checkpoint_gate = torch.load("models/250825_model-d_wos_me_nz_gate.pth")
    if pretrained == True:
        full_model = checkpoint_model["model_state_dict"]
        backbone_state_dict = {k.replace('backbone.', ''):v for k, v in full_model.items() if k.startswith('backbone.')} # extract only the backbone states
        model.load_state_dict(backbone_state_dict, strict=True)

    else:
        model.load_state_dict(checkpoint_model["model_state_dict"], strict=True)
    gate.load_state_dict(checkpoint_gate, strict=True)
    print("Loaded best weights")
    model.to(device)
    print("Model on gpu", next(model.parameters()).is_cuda)
    # is optimizer reinitializing the weights?
    optimizer = torch.optim.Adam(list(model.parameters()) + list(gate.parameters()), lr=args.lr, weight_decay=args.wd)
    criterion = nn.BCELoss(reduction="none")
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Output path
    if args.exp_id:
        out_path = os.path.join(args.output, args.exp_id)
    else:
        date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(args.output,  '{}_{}_{}_{}_{}_{}'.format(args.dataset, data, date_string, args.batch_size, args.gates, args.lr))
    os.makedirs(out_path, exist_ok=True)

        # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(out_path, "runs"))

    # Dump experiment parameters
    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))

    print("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    best_loss = float('inf')
    valid_loss = float('inf')
    best_model_weights = None
    best_model_path = None
    best_gate_path = None
    patience = 10
    model_save_folder = "models"

    
    '''def evaluate(model):
        test_val_t = perf_counter()
        for i, (x,y) in enumerate(test_loader):

            model.eval()
                    
            x = x.to(device)
            y = y.to(device)

            constrained_output = model(x.float(), sigmoid=True)
            predicted = constrained_output.data > 0.5

            # Total number of labels
            total = y.size(0) * y.size(1)

            # Total correct predictions
            correct = (predicted == y.byte()).sum()
            num_correct = (predicted == y.byte()).all(dim=-1).sum()

            # Move output and label back to cpu to be processed by sklearn
            predicted = predicted.to('cpu')
            cpu_constrained_output = constrained_output.to('cpu')
            y = y.to('cpu')

            if i == 0:
                test_correct = num_correct
                predicted_test = predicted
                constr_test = cpu_constrained_output
                y_test = y
            else:
                test_correct += num_correct
                predicted_test = torch.cat((predicted_test, predicted), dim=0)
                constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
                y_test = torch.cat((y_test, y), dim =0)

        test_val_e = perf_counter()
        avg_score = average_precision_score(y_test[:,test.to_eval], constr_test.data[:,test.to_eval], average='micro')
        jss = jaccard_score(y_test[:,test.to_eval], predicted_test[:,test.to_eval], average='micro')
        print(f"Number of correct: {test_correct}")
        print(f"avg_score: {avg_score}")
        print(f"test micro AP {jss}\t{(test_val_e-test_val_t):.4f}")
    '''
    def evaluate_circuit(model, gate, cmpe, epoch, data_loader, data_split, prefix, diff_data):
        test_val_t = perf_counter()
        if args.separate:
            test_correct = {level: 0 for level in hierarchy_levels[data]}
        else:
            test_correct = 0            
        for i, (x,y) in enumerate(data_loader):

            model.eval()
            gate.eval()
                    
            x = x.to(device)
            y = y.to(device)

            # Parameterize circuit using nn
            emb = model(x.float())
            thetas = gate(emb)

            # negative log likelihood and map
            cmpe.set_params(thetas)
            nll = cmpe.cross_entropy(y, log_space=True).mean()

            cmpe.set_params(thetas)
            pred_y = (cmpe.get_mpe_inst(x.shape[0]) > 0).long()

            pred_y = pred_y.to('cpu')
            y = y.to('cpu')
            
            # compare label and prediction per example (y and pred_y are batches, shape (256, 373))
            for j in range(x.shape[0]):  
                difference = (pred_y[j] - y[j]).numpy()
                diff_data.append(difference)

            if args.species_only:
                #species only: select last 200 columns
                pred_y = pred_y[:, -hierarchy_num[data][-1]:]
                y = y[:, -hierarchy_num[data][-1]:]

            if args.separate:
                pred_y_separate = split_category(pred_y, data) #returns tuple of tensors
                y_separate = split_category(y, data)
                # Initialize a dictionary to hold correct counts per level
                #num_correct = {level: 0 for level in hierarchy_levels}
                #nll_batch = {level: cmpe.cross_entropy(y_part, log_space=True).mean() for level, y_part in zip(hierarchy_levels, y_separate)}

                # Loop over each hierarchical level (order, family, genus, species)
            
                for j, level in enumerate(hierarchy_levels[data]):
                    # Extract predictions and labels for each level
                    pred_level = pred_y_separate[j]
                    label_level = y_separate[j]
                    
                    # Compare predictions and labels
                    correct = (pred_level == label_level.byte()).all(dim=-1).sum().item()
                    
                    # Store the result in the corresponding list
                    test_correct[level] += correct

                if i == 0:
                    predicted_test = pred_y
                    y_test = y
                    #nll_levels = {level: val for level, val in nll_batch.items()}

                else:
                    #test_correct = {level: test_correct[level] + num_correct[level] for level in hierarchy_levels}
                    predicted_test = torch.cat((predicted_test, pred_y), dim=0)
                    y_test = torch.cat((y_test, y), dim=0)
                    #for level in nll_batch:
            
            else:
                num_correct = (pred_y == y.byte()).all(dim=-1).sum()

                if i == 0:
                    test_correct = num_correct
                    predicted_test = pred_y
                    y_test = y
                else:
                    test_correct += num_correct
                    predicted_test = torch.cat((predicted_test, pred_y), dim=0)
                    y_test = torch.cat((y_test, y), dim=0)
                        #    nll_levels[level] += nll_batch[level]
        
        dt = perf_counter() - test_val_t
        y_test = y_test[:,data_split.to_eval]
        predicted_test = predicted_test[:,data_split.to_eval]


        if args.separate:
            
            accuracy_levels = {level: test_correct[level] / len(y_test) for level in hierarchy_levels[data]}
            nll = nll.detach().to("cpu").numpy() / (i+1)

            if data in ["cub", "amazon", "bgc", "wos"]:
                #remove redundant dimension (all True)
                y_test_squeeze = y_test.squeeze(1)
                predicted_test_squeeze = predicted_test.squeeze(1)
                            
                true_split = split_category(y_test_squeeze, data)
                pred_split = split_category(predicted_test_squeeze, data)

                # Ensure correct shape (1D numpy array) & convert to np tensor
                y_test = y_test.squeeze().cpu().numpy()
                predicted_test = predicted_test.squeeze().cpu().numpy()
            
                
                #nll_levels = {level: nll_levels[level].detach().to("cpu").numpy() / (i+1) for level in hierarchy_levels}
                jaccard_levels = {level: jaccard_score(true, pred, average='micro') for level, true, pred in zip(hierarchy_levels[data], true_split, pred_split)}
                hamming_levels = {level: hamming_loss(true, pred) for level, true, pred in zip(hierarchy_levels[data], true_split, pred_split)}

                # Print evaluation metrics for each level
                print(f"Evaluation metrics on {prefix} \t {dt:.4f}")
                #print(test_correct["genus"],  accuracy_levels["genus"], hamming_levels["genus"], jaccard_levels["genus"], nll)
                
                
                print('\n'.join(
                    [f"Level {level.capitalize()}:\tCorrect: {test_correct[level]}\tAccuracy: {accuracy_levels[level]}"
                    f"\tHamming: {hamming_levels[level]}\tJaccard: {jaccard_levels[level]}\tNLL: {nll}"
                    for level in hierarchy_levels[data]]
                ))

                return {
                f"{prefix}/accuracy": ([accuracy_levels[level] for level in hierarchy_levels[data]], epoch, dt),
                f"{prefix}/hamming": ([hamming_levels[level] for level in hierarchy_levels[data]], epoch, dt),
                f"{prefix}/jaccard": ([jaccard_levels[level] for level in hierarchy_levels[data]], epoch, dt),
                f"{prefix}/nll": (nll, epoch, dt),
            }, true_split, pred_split, predicted_test, y_test
            

        else:
            
            
            accuracy = test_correct / len(y_test)
            nll = nll.detach().to("cpu").numpy() / (i+1)

            assert y_test.shape == predicted_test.shape, "Mismatch between y and y_pred lengths!"
            
            if data in ["cub", "amazon", "bgc", "wos"]:
                #remove redundant dimension (all True)
                y_test_squeeze = y_test.squeeze(1)
                predicted_test_squeeze = predicted_test.squeeze(1)

            if args.species_only:
                true_split = split_category_species(y_test_squeeze, data)
                pred_split = split_category_species(predicted_test_squeeze, data)
            else:
                true_split = split_category(y_test_squeeze, data)
                pred_split = split_category(predicted_test_squeeze, data)

            if data in ["cub", "amazon", "bgc", "wos"]:
                # Ensure correct shape (1D numpy array)
                y_test = y_test.squeeze()
                predicted_test = predicted_test.squeeze()
                # Convert to numpy (currently torch.int64)
                y_test = y_test.cpu().numpy()
                predicted_test = predicted_test.cpu().numpy()

            #print(f"Output: {predicted_test[:5]}")  # Print the first 5 model outputs
            #print(f"True Labels: {y_test[:5]}")  # Print the first 5 true labels

            
            jaccard = jaccard_score(y_test, predicted_test, average='micro')
            hamming = hamming_loss(y_test, predicted_test)

            print(f"Evaluation metrics on {prefix} \t {dt:.4f}")
            print(f"Num. correct: {test_correct}")
            print(f"Accuracy: {accuracy}")
            print(f"Hamming Loss: {hamming}")
            print(f"Jaccard Score: {jaccard}")
            print(f"nll: {nll}")


            return {
                    f"{prefix}/accuracy": (accuracy, epoch, dt),
                    f"{prefix}/hamming": (hamming, epoch, dt),
                    f"{prefix}/jaccard": (jaccard, epoch, dt),
                    f"{prefix}/nll": (nll, epoch, dt),
                }, true_split, pred_split, predicted_test, y_test

    
    if data in ["cub", "amazon", "bgc", "wos"]:
        data_split_test = test_dataset
        data_split_val = val_dataset
        data_split_train = train_dataset
    else:
        data_split_test = test
        data_split_train = train
        data_split_val = None

    if args.no_train:
        diff_data_train = []
        diff_data_test = []
        perf_test, true_split_test, pred_split_test, predicted_test, y_test = evaluate_circuit(
            model,
            gate, 
            cmpe,
            epoch=None,
            data_loader=test_loader,
            data_split=data_split_test,
            prefix="param_sdd/test",
            diff_data=diff_data_test,
        )

        '''
        perf_train, true_split_train, pred_split_train = evaluate_circuit(
            model,
            gate,
            cmpe,
            epoch=None,
            data_loader=train_load,
            data_split=data_split_train,
            prefix="param_sdd/train",
            diff_data=diff_data_train
        )
        '''

        # Merge dicts
        perf = {**perf_test} #, **perf_train}
        
        if args.separate:
            # Export metrics
            for perf_name, tupl in perf.items():
                if "nll" in perf_name:
                    score, epoch, dt = tupl
                    writer.add_scalar(perf_name, score, global_step=epoch, walltime=dt)
                else:
                    scores, epoch, dt = tupl
                    for level, score in zip(hierarchy_levels[data], scores):
                        writer.add_scalar(f"{perf_name}/{level}", score, global_step=epoch, walltime=dt)
        else:
            for perf_name, (score, epoch, dt) in perf.items():
                writer.add_scalar(perf_name, score, global_step=epoch, walltime=dt)
        
        writer.flush()

        # Save difference file for test and train sets
        if args.exp_id:
            cm_folder = f"./confusion_matrices/{date_string}_{args.exp_id}"
            pred_y_folder = f"./pred_y/{date_string}_{args.exp_id}"
            diff_folder = f"./diff/{date_string}_{args.exp_id}"

        else:
            cm_folder = f"./confusion_matrices/{date_string}"
            pred_y_folder = f"./pred_y/{date_string}"
            diff_folder = f"./diff/{date_string}"

        os.makedirs(cm_folder, exist_ok=True)
        os.makedirs(pred_y_folder, exist_ok=True)
        os.makedirs(diff_folder, exist_ok=True)

        if args.exp_id:
            #pd.DataFrame(diff_data_train).to_csv(f"difference_train_{model.__class__.__name__}_oe{args.one_each}_{args.exp_id}_{date_string}.csv", index=False, header=False)
            pd.DataFrame(diff_data_test).to_csv(os.path.join(diff_folder, f"difference_test_{data}_oe{args.one_each}_{args.exp_id}_{date_string}.csv"), index=False, header=False)
            pd.DataFrame(predicted_test).to_csv(os.path.join(pred_y_folder, f"predicted_test_{data}_oe{args.one_each}_{args.exp_id}_{date_string}.csv"), index=False, header=False)
            pd.DataFrame(y_test).to_csv(os.path.join(pred_y_folder, f"y_test_{data}_oe{args.one_each}_{args.exp_id}_{date_string}.csv"), index=False, header=False)
        else:
            #pd.DataFrame(diff_data_train).to_csv(f"difference_train_{model.__class__.__name__}_oe{args.one_each}_{date_string}.csv", index=False, header=False)
            pd.DataFrame(diff_data_test).to_csv(os.path.join(diff_folder, f"difference_test_{data}_oe{args.one_each}_{date_string}.csv"), index=False, header=False)
            pd.DataFrame(predicted_test).to_csv(os.path.join(pred_y_folder, f"predicted_test_{data}_oe{args.one_each}_{date_string}.csv"), index=False, header=False)
            pd.DataFrame(y_test).to_csv(os.path.join(pred_y_folder, f"y_test_{data}_oe{args.one_each}_{date_string}.csv"), index=False, header=False)

        #Create confusion matrix per class for test set only
        if args.species_only:
            matrix_names = ["species"]
            num_classes_per_level = [hierarchy_num[data][-1]]
            #print(type(num_classes_per_level))

            
        else:
            matrix_names = hierarchy_levels[data]
            num_classes_per_level = hierarchy_num[data]
        #
        cm_folder = f"./confusion_matrices/{date_string}_{args.exp_id}"

        if args.exp_id:
            cm_folder = f"./confusion_matrices/{date_string}_{args.exp_id}"
        else:
            cm_folder = f"./confusion_matrices/{date_string}"

        os.makedirs(cm_folder, exist_ok=True)

        for t, p, n, num_classes in tqdm(zip(true_split_test, pred_split_test, matrix_names, num_classes_per_level)):
            print(f"\nEvaluating on {n}: ")

            t = convert_ohe_to_1d(t)
            p = convert_ohe_to_1d(p)

            labels = list(range(num_classes))
            #print(f"Labels: {labels}") #correct
            #print("Sample predictions:", p[:10])
            #print("Sample targets:", t[:10])
            #print("Unique predicted labels:", np.unique(p))
            #print("Unique true labels:", np.unique(t)) #there exist -1 labels in predictions
            matrix = confusion_matrix(t, p, labels=[-1] + labels)
            df = pd.DataFrame(matrix)
            plt.figure(figsize=(20, 16))  # optional: adjust size
            sns.heatmap(df, cmap = "crest", fmt="d", linewidths=0.5, linecolor='white')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix") 
            
            plt.savefig(os.path.join(cm_folder, f"{n}_{data}_nc{args.no_constraints}_{date_string}.png"))
            plt.close()

        print("Confusion matrices successfully generated.")        
        
    #testing for a training model    
    else:
        diff_data_test = []
        for epoch in range(num_epochs):

            if epoch % 5 == 0 and epoch != 0:

                print(f"EVAL@{epoch}")
                perf_test, true_split_test, pred_split_test, predicted_test, y_test = evaluate_circuit(
                        model,
                        gate, 
                        cmpe,
                        epoch=epoch,
                        data_loader=test_loader,
                        data_split=data_split_test,
                        prefix="param_sdd/test",
                        diff_data=diff_data_test,
                    )
                perf = {**perf_test}

                for perf_name, (score, epoch, dt) in perf.items():
                    writer.add_scalar(perf_name, score, global_step=epoch, walltime=dt)

                writer.flush()
            
                       
            train_t = perf_counter()

            model.train() #why not change it to eval and load best trained model?
            gate.train()

            tot_loss = 0
            for i, (x, labels) in enumerate(train_loader):

                x = x.to(device)
                labels = labels.to(device)
            
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Use fully-factorized distribution via circuit
                output = model(x.float(), sigmoid=False)
                thetas = gate(output)
                cmpe.set_params(thetas)
                loss = cmpe.cross_entropy(labels, log_space=True).mean()

                tot_loss += loss
                loss.backward()
                optimizer.step()

            train_e = perf_counter()
            avg_loss = tot_loss/(i+1)
            print(f"{epoch+1}/{num_epochs} train loss: {avg_loss}\t {(train_e-train_t):.4f}")

            # Early stopping loop and save best model
            if valid_loss < best_loss: #may change to valid_loss instead of avg_loss (training loss). avg_loss performs slightly better (longer training time)
                best_loss = valid_loss
                best_model_weights = copy.deepcopy(model.state_dict()) #Retrieve best model weights 
                patience = 10  # Reset patience counter
                if args.exp_id:
                    out_path_model = os.path.join(model_save_folder, f"{date_string}_{args.exp_id}_model_finetuned.pth")
            
                    out_path_gate = os.path.join(model_save_folder, f"{date_string}_{args.exp_id}_gate_finetuned.pth")

                else:
                    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    out_path_model = os.path.join(model_save_folder,  
                                                '{}_{}_{}_{}_nc{}_model_finetuned.pth'.format(
                                                    args.dataset, data, date_string, args.batch_size, args.no_constraints
                                                    ))
                    
                    out_path_gate = os.path.join(model_save_folder,  
                                                '{}_{}_{}_{}_nc{}_gate_finetuned.pth'.format(
                                                    args.dataset, data, date_string, args.batch_size, args.no_constraints
                                                    ))
                
                # Remove the previous best model and gate (for each bash run) if exists
                for i in [best_model_path, best_gate_path]:
                    if i and os.path.exists(i):
                        os.remove(i)

                # Save best model and gate into .pth files and update best paths            
                os.makedirs(os.path.dirname(out_path_model), exist_ok=True)
                torch.save(gate.state_dict(), out_path_gate)# save the gate parameters: gate is a dense fc model
                torch.save({
                    'model_state_dict': best_model_weights,
                    'best_loss': best_loss,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr
                    }, out_path_model)
                best_model_path = out_path_model
                best_gate_path = out_path_gate
                    
            # Early stopping    
            else:
                patience -= 1
                if patience == 0:
                    #generate confusion matrix and evaluate at last epoch
            
                    print(f"EVAL@{epoch+1}")
                    perf_test, true_split_test, pred_split_test, predicted_test, y_test = evaluate_circuit(
                            model,
                            gate, 
                            cmpe,
                            epoch=epoch,
                            data_loader=test_loader,
                            data_split=data_split_test,
                            prefix="param_sdd/test",
                            diff_data=diff_data_test
                        )
                    perf = {**perf_test}

                    for perf_name, (score, epoch, dt) in perf.items():
                        writer.add_scalar(perf_name, score, global_step=epoch, walltime=dt)

                    writer.flush()

                    # Save difference file for test and train sets

                    if args.exp_id:
                        pd.DataFrame(diff_data_test).to_csv(f"difference_test_{data}_oe{args.one_each}_{args.exp_id}_{date_string}.csv", index=False)
                    else:
                        pd.DataFrame(diff_data_test).to_csv(f"difference_test_{data}_oe{args.one_each}_{date_string}.csv", index=False)

                    #Create confusion matrix per class for test set only
                    if args.species_only:
                        matrix_names = ["species"]
                        num_classes_per_level = hierarchy_num[data][-1]
                        
                    else:
                        matrix_names = hierarchy_levels[data]
                        num_classes_per_level = hierarchy_num[data]
                    #
                    cm_folder = f"./confusion_matrices/{date_string}_{args.exp_id}"

                    if args.exp_id:
                        cm_folder = f"./confusion_matrices/{date_string}_{args.exp_id}"
                    else:
                        cm_folder = f"./confusion_matrices/{date_string}"

                    os.makedirs(cm_folder, exist_ok=True)

                    for t, p, n, num_classes in tqdm(zip(true_split_test, pred_split_test, matrix_names, num_classes_per_level)):
                        print(f"\nEvaluating on {n}: ")

                        t = convert_ohe_to_1d(t)
                        p = convert_ohe_to_1d(p)

                        labels = [-1] + list(range(num_classes))
                        matrix = confusion_matrix(t, p, labels=labels)
                        df = pd.DataFrame(matrix)
                        plt.figure(figsize=(20, 16))  # optional: adjust size
                        sns.heatmap(df, cmap = "crest", fmt="d", linewidths=0.5, linecolor='white')
                        plt.xlabel("Predicted")
                        plt.ylabel("True")
                        plt.title("Confusion Matrix") 
                        
                        plt.savefig(os.path.join(cm_folder, f"{n}_{model.__class__.__name__}_nc{args.no_constraints}_{date_string}.png"))
                        plt.close()

                    print("Confusion matrices successfully generated.") 
                    print("Patience ran out.")
                    break
        

if __name__ == "__main__":
    main()
