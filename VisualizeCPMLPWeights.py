import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
from PIL import Image
from scipy.ndimage import filters
from torch import nn
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def visualize_weights(weights, layer_num):
    num_neurons = weights.shape[0]
    num_inputs = weights.shape[1]

    # fig, axes = plt.subplots(num_neurons, num_inputs, figsize=(num_inputs, num_neurons))
    # fig.suptitle(f'Layer {layer_num} Weights Visualization')
 

    plt.imshow(weights, cmap='coolwarm', ) #interpolation='nearest'
    # for i in range(num_neurons):
    #     for j in range(num_inputs):
    #         ax = axes[i, j]
    #         ax.imshow(weights[i, j], cmap='coolwarm', interpolation='nearest')
    #         ax.axis('off')

    #plt.tight_layout()
    plt.show()

# Example usage
# Assuming weights is a 2D numpy array containing the weights of neurons in a hidden layer
# Shape of weights: (num_neurons, num_inputs)
# For example, if weights.shape is (10, 20), it means there are 10 neurons in the layer, each with 20 input weights.

# Define your MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 10)  # One layer with 10 input neurons and 10 output neurons

    def forward(self, x):
        x = self.fc1(x)
        return x

# Create an instance of the model
model = MLP()

# Access weights of the first layer
weights = model.fc1.weight.data.numpy()
biases = model.fc1.bias.data.numpy()

# Access weights of the first layer
weights = model.fc1.weight.data.numpy()


def visualzie_pattern1(weights):
    # Plotting the weights as a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Weights of the MLP Layer')
    plt.xlabel('Output Neurons')
    plt.ylabel('Input Neurons')
    plt.show()

def visualzie_pattern2(weights):
    # Create a directed graph
    G = nx.DiGraph()

    # Add input and output neurons as nodes
    for i in range(10):
        G.add_node('Input {}'.format(i+1), pos=(0, i))
        G.add_node('Output {}'.format(i+1), pos=(1, i))

    # Add edges with weights
    for i in range(10):
        for j in range(10):
            G.add_edge('Input {}'.format(i+1), 'Output {}'.format(j+1), weight=weights[i][j])

    # Define positions for nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title('Connections between Input and Output Neurons in MLP Layer')
    plt.show()


def visualzie_pattern3(weights)
    # Create a directed graph
    G = nx.DiGraph()

    # Add input and output neurons as nodes
    for i in range(10):
        G.add_node('Input {}'.format(i+1), pos=(0, i))
        G.add_node('Output {}'.format(i+1), pos=(1, i))

    # Add edges with weights
    for i in range(10):
        for j in range(10):
            weight = weights[i][j]
            G.add_edge('Input {}'.format(i+1), 'Output {}'.format(j+1), weight=weight)

    # Define positions for nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Get edge weights
    edge_weights = [G[u][v]['weight']*10 for u, v in G.edges()] 

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20,
            width=edge_weights, edge_color='black', alpha=0.7)
    plt.title('Connections between Input and Output Neurons in MLP Layer')
    plt.show()



if __name__ == "__main__":

    #add Core-periphery NN after CLIP
    if('RN' in opt.vision_encoder):
        nodes = nodes_lookup_table['RN']
        cores = round(nodes * opt.ratio)
    elif('ViT' in opt.vision_encoder):
        nodes = nodes_lookup_table['ViT']
        cores = round(nodes * opt.ratio)
    cp_mask = get_cp_mask(nodes, cores)
    cp_network = CP_NN(in_dim=nodes, out_dim=nodes, CP_mask= torch.from_numpy(cp_mask).to(device), device=device)
    cp_network.to(device)

    #@markdown ---
    #@markdown #### CLIP model settings
    vis_model = "RN50" #@param ["RN50", "RN101", "RN50x4", "RN50x16"]
    saliency_layer = "layer4" #@param ["layer4", "layer3", "layer2", "layer1"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(vis_model, device=device, jit=False)
    model_ori, preprocess = clip.load(vis_model, device=device, jit=False)
    
    DataSet = 'CheXpert5x200'
    if(DataSet == 'TMED2'):
        dir_name = DataSet+'labeled'
    else:
        dir_name = DataSet
    #load saved model
    saved_models = os.listdir('./models_saved')
    for saved_model in saved_models:
        if( vis_model in saved_model and 'finetune' in saved_model and DataSet in saved_model and 'CPCLIP' in saved_model \
            and '717' in saved_model and '1024' in saved_model):  #819
            checkpoint = torch.load('./models_saved/'+saved_model)
            model.load_state_dict(checkpoint['network'])
            cp_network.load_state_dict(checkpoint['cp_nn'])
            print(saved_model)
            print('load saved models done!')
    

    #@markdown #### Image & Caption settings
    image_url = 'https://images2.minutemediacdn.com/image/upload/c_crop,h_706,w_1256,x_0,y_64/f_auto,q_auto,w_1100/v1554995050/shape/mentalfloss/516438-istock-637689912.jpg' #@param {type:"string"}

    image_caption = 'collapsed lung' #@param {type:"string"}

    '''
    for name, param in model.named_parameters():
        if param.requires_grad:
            print name, param.data
    '''
    #@markdown ---
    #@markdown #### Visualization settings
    blur = True #@param {type:"boolean"}

    # Download the image from the web.
    image_path = 'image.png'
    urllib.request.urlretrieve(image_url, image_path)

    image_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/'+dir_name+'/3118s1_0.png'
    #/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/TMEDlabeled
    image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_np = load_image(image_path, model.visual.input_resolution)
    text_input = clip.tokenize([image_caption]).to(device)

    attn_map = gradCAM(
        model.visual,
        image_input,
        model.encode_text(text_input).float(),
        getattr(model.visual, saliency_layer)
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()

    #########################################################################
    attn_map_ori = gradCAM(
        model_ori.visual,
        image_input,
        model_ori.encode_text(text_input).float(),
        getattr(model_ori.visual, saliency_layer)
    )
    attn_map_ori = attn_map_ori.squeeze().detach().cpu().numpy()
    #########################################################################

    viz_attn(image_np, attn_map, attn_map_ori, blur)