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
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_score, 
    average_precision_score, 
    hamming_loss, 
    jaccard_score
)

# Circuit imports
import sys
sys.path.append(os.path.join(sys.path[0],'hmc-utils'))
sys.path.append(os.path.join(sys.path[0],'hmc-utils', 'pypsdd'))

from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE
from pysdd.sdd import SddManager, Vtree

# misc
from common import *

from torch.utils.data import Dataset
from PIL import Image

def log1mexp(x):
        assert(torch.all(x >= 0))
        return torch.where(x < 0.6931471805599453094, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

'''
class CUB_Dataset(Dataset):
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
class CUB_Dataset_Embeddings(Dataset):
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
    
# custom dataset for one example per class
# Input: a dataset built from CUB_Dataset_Embeddings, e.g. train_dataset
# Output: the same train_dataset but reduced to 1 example per class
class OEDataset(Dataset):
    def __init__(self, dataset, to_eval = True):
        self.dataset = dataset
        self.class_examples = self._get_oe()
        self.to_eval = to_eval
    #for internal use (func inside class)
    def _get_oe(self):
        class_examples = defaultdict(list)
        for i, (x, y) in enumerate(self.dataset):
            class_examples[tuple(y.tolist())].append((x,y)) #fetch all (x, y) tuples by y
        
        selected_examples = []
        for class_id, examples in class_examples.items():
            selected_examples.append(examples[0]) #select the first, else replace with random.choice(examples)

        return selected_examples
    
    def __len__(self):
        return len(self.class_examples)

    def __getitem__(self, idx):
        return self.class_examples[idx]

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

    # Load train, val and test set
    dataset_name = args.dataset
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]
    hidden_dim = hidden_dims[ontology][data]
    output_dim = 128 #not the number of classes

    num_epochs = args.n_epochs

    if "cub" in args.dataset:
        if "mini" in args.dataset:
            mat_path = mat_path_mini
            csv_path = csv_path_mini
        else:
            mat_path = mat_path_full
            csv_path = csv_path_full

    # Set the hyperparameters 
    hyperparams = {
        'num_layers': 3,
        'dropout': 0.7,
        'non_lin': 'relu',
    }

    # Set seed
    seed_all_rngs(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    emb_model_name = "resnet50"

    
    # Load the datasets
    import glob
    from sklearn import preprocessing
    '''
    # list of files in cub 2011 and y labels
    if "cub" in args.dataset:
        #Try out with 5 classes in CUB
        # Get all class folder names
        all_classes = sorted(os.listdir(images_dir))  # Sorting ensures consistency
        
        if "mini" in args.dataset:
        # Select all classes or only the first 5 classes (mini csv, mat and image set)
            selected_classes = all_classes[:5]
            csv_path = csv_path_mini
            mat_path = mat_path_mini
        else:
            selected_classes = all_classes
            csv_path = csv_path_full
            mat_path = mat_path_full

        # getting 1 example from each class
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
            
    
    if "cub" in args.dataset:    
        # Split dataset into train, val, and test sets
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, image_labels, test_size=0.3, random_state=args.seed)
        val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.7, random_state=args.seed)
    ''' 
    if "cub" in args.dataset:    
        # load pickle at /embeddings
        emb_file = find_latest_emb_file(emb_model_name, dataset_name)

        # Load and split dataset into train, val, and test sets
        with open(emb_file, "rb") as f:
            all_paths, all_embeddings, labels_unprocessed = pickle.load(f)
        label_species = [label.split('.')[-1] for label in labels_unprocessed]
        label_species = [re.sub('_', ' ', label) for label in label_species] # the species-level label for each image
        
        ohe_dict, _ = get_one_hot_labels(label_species, csv_path)
        ohe_labels = [torch.from_numpy(ohe_dict[species]).to(device) for species in label_species]
        all_embeddings_tensor = [torch.tensor(emb) for emb in all_embeddings]
        train_emb, temp_emb, train_labels, temp_labels = train_test_split(all_embeddings_tensor, ohe_labels, test_size=0.3, random_state=args.seed)
        val_emb, test_emb, val_labels, test_labels = train_test_split(temp_emb, temp_labels, test_size=0.7, random_state=args.seed)
      
    elif ('others' in args.dataset):
        train, test = initialize_other_dataset(dataset_name, datasets)
        train.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.bool),  torch.tensor(test.to_eval, dtype=torch.bool)
        train.X, valX, train.Y, valY = train_test_split(train.X, train.Y, test_size=0.30, random_state=args.seed)
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)
        train.to_eval, val.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.bool), torch.tensor(val.to_eval, dtype=torch.bool), torch.tensor(test.to_eval, dtype=torch.bool)
        #print(train.Y.shape)

        #Create loaders
    '''
    if "cub" in args.dataset:
        # Create datasets for each split: Change labels
        train_dataset = CUB_Dataset(train_paths, train_labels, transform, to_eval = True)
        val_dataset = CUB_Dataset(val_paths, val_labels, transform, to_eval = True)
        test_dataset = CUB_Dataset(test_paths, test_labels, transform, to_eval = True)
    '''
    if "cub" in args.dataset:
        # Create datasets for each split: Change labels
        train_dataset = CUB_Dataset_Embeddings(train_emb, train_labels, to_eval = True)
        val_dataset = CUB_Dataset_Embeddings(val_emb, val_labels, to_eval = True)
        test_dataset = CUB_Dataset_Embeddings(test_emb, test_labels, to_eval = True)
        # convert them into tensors: shape = output_dim + 1
        train_dataset.to_eval, val_dataset.to_eval, test_dataset.to_eval = torch.tensor(train_dataset.to_eval, dtype=torch.bool), torch.tensor(val_dataset.to_eval, dtype=torch.bool), torch.tensor(test_dataset.to_eval, dtype=torch.bool)
        
        # Create OE (1 example) datasets when needed

        if args.one_each:
            train_dataset_1_each = OEDataset(train_dataset)
            test_dataset_1_each = OEDataset(test_dataset)
            train_dataset_1_each.to_eval, test_dataset_1_each.to_eval = torch.tensor(train_dataset_1_each.to_eval, dtype=torch.bool), torch.tensor(test_dataset_1_each.to_eval, dtype=torch.bool)

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
    if args.one_each:
        train_oe_loader = torch.utils.data.DataLoader(dataset=train_dataset_1_each,
                                            batch_size=args.batch_size,
                                            shuffle=True)
        test_oe_loader = torch.utils.data.DataLoader(dataset=train_dataset_1_each,
                                            batch_size=args.batch_size,
                                            shuffle=True)

    # We do not evaluate the performance of the model on the 'roots' node (https://dtai.cs.kuleuven.be/clus/hmcdatasets/)
    if 'GO' in dataset_name:
        num_to_skip = 4
    else:
        num_to_skip = 1

    if "cub" in args.dataset:
        mat = np.load(mat_path)
    else:
        mat = train.A
    # Prepare circuit: TODO needs cleaning
    if not args.no_constraints:

        if not os.path.isfile('constraints/' + dataset_name + '.sdd') or not os.path.isfile('constraints/' + dataset_name + '.vtree'):
            # Compute matrix of ancestors R
            # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
            R = np.zeros(mat.shape)
            np.fill_diagonal(R, 1)
            g = nx.DiGraph(mat)
            for i in range(len(mat)):
                descendants = list(nx.descendants(g, i))
                if descendants:
                    R[i, descendants] = 1
            R = torch.tensor(R)

            #Transpose to get the ancestors for each node 
            #R = R.transpose(1, 0)
            R = R.unsqueeze(0).to(device)

            # Uncomment below to compile the constraint
            R.squeeze_()
            mgr = SddManager(
                var_count=R.size(0),
                auto_gc_and_minimize=True)

            

            alpha = mgr.true()
            alpha.ref()
            for i in range(R.size(0)):

               beta = mgr.true()
               beta.ref()
               for j in range(R.size(0)):

                   if R[i][j] and i != j: # if j is descendant of i and not in the diagonal
                       old_beta = beta # clone new beta
                       beta = beta & mgr.vars[j+1] #beta = {boolean val of beta} AND class_j_is_active
                       beta.ref() #protect new one
                       old_beta.deref() #remove old_beta

               old_beta = beta
               beta = -mgr.vars[i+1] | beta #if class i is present, then all its descendants must be present
               beta.ref()
               old_beta.deref()

               old_alpha = alpha
               alpha = alpha & beta
               alpha.ref()
               old_alpha.deref()

            # Saving circuit & vtree to disk
            alpha.save(str.encode('constraints/' + dataset_name + '.sdd'))
            alpha.vtree().save(str.encode('constraints/' + dataset_name + '.vtree'))

        # Create circuit object
        cmpe = CircuitMPE('constraints/' + dataset_name + '.vtree', 'constraints/' + dataset_name + '.sdd')

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
        gate = DenseGatingFunction(cmpe.beta, gate_layers=[128]).to(device)
        R = None

    # Output path
    if args.exp_id:
        out_path = os.path.join(args.output, args.exp_id)
    else:
        date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(args.output,  '{}_{}_{}_{}_{}'.format(args.dataset, date_string, args.batch_size, args.gates, args.lr))
    os.makedirs(out_path, exist_ok=True)

    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(out_path, "runs"))

    # Dump experiment parameters
    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))

    print("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    # Create the model
    # Load train, val and test set
    
    model = ConstrainedFFNNModel(input_dims[data], hidden_dim, output_dim, hyperparams, R, args.dataset) # 1% at 45 ep, learns faster?/better? but accuracy still low, 13%
    #model = LeNet5(input_dims[data], hidden_dim, output_dim, hyperparams, R, args.dataset) #1% accuracy at 80 epochs
    #if args.no_train:
    best_file, best_loss = find_best_pth_file(model.__class__.__name__)
    pretrained = False
    #checkpoint_model = torch.load(best_file)
    #checkpoint_gate = torch.load(re.sub("model", "gate", str(best_file)))
    checkpoint_model = torch.load("/mnt/cimec-storage6/users/nguyenanhthu.tran/intern25/spl/SPL/C-HMCNN/models/cub_others_Pretrain_Model_20250422-053938_256_ncFalse_patienceFalse_pretrained_model.pth")
    checkpoint_gate = torch.load("/mnt/cimec-storage6/users/nguyenanhthu.tran/intern25/spl/SPL/C-HMCNN/models/cub_others_ConstrainedFFNNModel_20250405-000154_256_0.001_avgloss_gate.pth")
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
                difference = (pred_y[j] - y[j]).cpu().numpy()
                diff_data.append(difference)
            
            if args.species_only:
                #species only: select last 200 columns
                pred_y = pred_y[:, -200:]
                y = y[:, -200:]

            num_correct = (pred_y == y.byte()).all(dim=-1).sum()

            if i == 0:
                test_correct = num_correct
                predicted_test = pred_y
                y_test = y
            else:
                test_correct += num_correct
                predicted_test = torch.cat((predicted_test, pred_y), dim=0)
                y_test = torch.cat((y_test, y), dim=0)
                

        dt = perf_counter() - test_val_t
        y_test = y_test[:,data_split.to_eval]
        predicted_test = predicted_test[:,data_split.to_eval]
        

        accuracy = test_correct / len(y_test)
        nll = nll.detach().to("cpu").numpy() / (i+1)

        assert y_test.shape == predicted_test.shape, "Mismatch between y and y_pred lengths!"
        
        if "cub" in args.dataset:
            #remove redundant dimension (all True)
            y_test_squeeze = y_test.squeeze(1)
            predicted_test_squeeze = predicted_test.squeeze(1)

        if args.species_only:
            true_split = split_category_species(y_test_squeeze)
            pred_split = split_category_species(predicted_test_squeeze)
        else:
            true_split = split_category(y_test_squeeze)
            pred_split = split_category(predicted_test_squeeze)

        if "cub" in args.dataset:
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
        }, true_split, pred_split
    
    if "cub" in args.dataset:
        if args.one_each:
            data_split_test = test_dataset_1_each
            data_split_train = train_dataset_1_each
            test_load = test_oe_loader
            train_load = train_oe_loader
        else:
            data_split_test = test_dataset
            data_split_train = train_dataset
            test_load = test_loader
            train_load = train_loader
    else:
        data_split_test = test
        data_split_train = train
        test_load = test_loader
        train_load = train_loader

    if args.no_train:
        diff_data_train = []
        diff_data_test = []
        perf_test, true_split_test, pred_split_test = evaluate_circuit(
            model,
            gate, 
            cmpe,
            epoch=None,
            data_loader=test_load,
            data_split=data_split_test,
            prefix="param_sdd/test",
            diff_data=diff_data_test
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
        
        # Export metrics
        for perf_name, (score, epoch, dt) in perf.items():
            writer.add_scalar(perf_name, score, global_step=epoch, walltime=dt)
        writer.flush()

        # Save difference file for test and train sets

        if args.exp_id:
            #pd.DataFrame(diff_data_train).to_csv(f"difference_train_{model.__class__.__name__}_oe{args.one_each}_{args.exp_id}_{date_string}.csv", index=False, header=False)
            pd.DataFrame(diff_data_test).to_csv(f"difference_test_{model.__class__.__name__}_oe{args.one_each}_{args.exp_id}_{date_string}.csv", index=False, header=False)
        else:
            #pd.DataFrame(diff_data_train).to_csv(f"difference_train_{model.__class__.__name__}_oe{args.one_each}_{date_string}.csv", index=False, header=False)
            pd.DataFrame(diff_data_test).to_csv(f"difference_test_{model.__class__.__name__}_oe{args.one_each}_{date_string}.csv", index=False, header=False)

        #Create confusion matrix per class for test set only
        if args.species_only:
            matrix_names = ["species"]
            num_classes_per_level = [200]
            
        else:
            matrix_names = ["order", "family", "genus", "species"]
            num_classes_per_level = [13, 37, 123, 200]
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
            matrix = confusion_matrix(t, p)
            df = pd.DataFrame(matrix)
            plt.figure(figsize=(20, 16))  # optional: adjust size
            sns.heatmap(df, cmap = "crest", fmt="d", linewidths=0.5, linecolor='white')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix") 
            
            plt.savefig(os.path.join(cm_folder, f"{n}_{model.__class__.__name__}_nc{args.no_constraints}_{date_string}.png"))
            plt.close()

        print("Confusion matrices successfully generated.")        
        
    #testing for a training model    
    else:
        diff_data_test = []
        for epoch in range(num_epochs):

            if epoch % 5 == 0 and epoch != 0:

                print(f"EVAL@{epoch}")
                perf_test, true_split_test, pred_split_test = evaluate_circuit(
                        model,
                        gate, 
                        cmpe,
                        epoch=epoch,
                        data_loader=test_load,
                        data_split=data_split_test,
                        prefix="param_sdd/test",
                        diff_data=diff_data_test
                    )
                perf = {**perf_test}

                for perf_name, (score, epoch, dt) in perf.items():
                    writer.add_scalar(perf_name, score, global_step=epoch, walltime=dt)

                writer.flush()
            
                       
            train_t = perf_counter()

            model.train() #why not change it to eval and load best trained model?
            gate.train()

            tot_loss = 0
            for i, (x, labels) in enumerate(train_load):

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
            if avg_loss < best_loss: #may change to valid_loss instead of avg_loss (training loss). avg_loss performs slightly better (longer training time)
                best_loss = avg_loss
                best_model_weights = copy.deepcopy(model.state_dict()) #Retrieve best model weights 
                patience = 10  # Reset patience counter
                if args.exp_id:
                    out_path_model = os.path.join(model_save_folder, f"{date_string}_{args.exp_id}_model_finetuned.pth")
            
                    out_path_gate = os.path.join(model_save_folder, f"{date_string}_{args.exp_id}_gate_finetuned.pth")

                else:
                    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    out_path_model = os.path.join(model_save_folder,  
                                                '{}_{}_{}_{}_nc{}_model_finetuned.pth'.format(
                                                    args.dataset, model.__class__.__name__, date_string, args.batch_size, args.no_constraints
                                                    ))
                    
                    out_path_gate = os.path.join(model_save_folder,  
                                                '{}_{}_{}_{}_nc{}_gate_finetuned.pth'.format(
                                                    args.dataset, model.__class__.__name__, date_string, args.batch_size, args.no_constraints
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
                    perf_test, true_split_test, pred_split_test = evaluate_circuit(
                            model,
                            gate, 
                            cmpe,
                            epoch=epoch,
                            data_loader=test_load,
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
                        pd.DataFrame(diff_data_test).to_csv(f"difference_test_{model.__class__.__name__}_oe{args.one_each}_{args.exp_id}_{date_string}.csv", index=False)
                    else:
                        pd.DataFrame(diff_data_test).to_csv(f"difference_test_{model.__class__.__name__}_oe{args.one_each}_{date_string}.csv", index=False)

                    #Create confusion matrix per class for test set only
                    if args.species_only:
                        matrix_names = ["species"]
                        num_classes_per_level = [200]
                        
                    else:
                        matrix_names = ["order", "family", "genus", "species"]
                        num_classes_per_level = [13, 37, 123, 200]
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
                        matrix = confusion_matrix(t, p)
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
