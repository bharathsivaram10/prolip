import requests
from PIL import Image
import torch
from prolip.model import ProLIPHF
from transformers import CLIPProcessor
from prolip.tokenizer import HFTokenizer
from transformers import AutoProcessor, CLIPModel
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np

class LinearWithDropout(nn.Module):
    '''Wrapper that applies dropout after a linear layer'''
    def __init__(self, linear_layer, dropout_p):
        super().__init__()
        self.linear = linear_layer
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x

class CLIPUtil():

    def __init__(self, dropout=0.1):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.model_nodrop = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # Inject dropout after all linear layers in vision model
        self._inject_dropout_to_linear_layers(self.model.vision_model, dropout)

    def check_dropout_layers(self):
        '''Check if model actually has dropout layers'''
        print("\n=== Checking for Dropout Layers ===")
        dropout_layer_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                dropout_layer_count += 1
                # print(f"{name}: {module}")
                # print(f"  p={module.p}, training={module.training}")
        print(f"Linear Dropout Layers: {dropout_layer_count}")

    def _inject_dropout_to_linear_layers(self, module, p, parent_name=''):
        '''Recursively find all Linear layers and wrap them with dropout'''
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            if isinstance(child, nn.Linear):
                # Wrap the linear layer with dropout
                # print(f"Adding dropout to: {full_name}")
                setattr(module, name, LinearWithDropout(child, p))
            else:
                # Recursively process child modules
                self._inject_dropout_to_linear_layers(child, p, full_name)

    def get_cov_trace(self, image, N=100):
        '''Sample the model N times, save image embeddings and compute cov trace'''
        features = []

        self.model.train()

        inputs = self.processor(images=image, return_tensors="pt")

        # Disable gradient calculations for efficiency
        with torch.no_grad():
            for _ in range(N):
                image_features = self.model.get_image_features(**inputs)
                features.append(image_features)

        self.model.eval()

        # 1. Stack the list of tensors into a single tensor
        # The result is a tensor of shape [N, embedding_dim]
        stacked_features = torch.cat(features, dim=0)

        # 2. Compute the covariance matrix
        # torch.cov expects variables as rows, so we transpose our tensor
        # from [N, embedding_dim] to [embedding_dim, N]
        covariance_matrix = torch.cov(stacked_features.T)

        # print(covariance_matrix.shape)

        # 3. Compute the trace of the covariance matrix
        trace = torch.trace(covariance_matrix)

        return trace.item()
    
class ProlipUtil():

    def __init__(self):

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = ProLIPHF.from_pretrained("SanghyukChun/ProLIP-ViT-B-16-DC-1B-12_8B")

    def get_cov_trace(self, image):

        inputs = self.processor(images=image, return_tensors="pt", padding=True)

        outputs = self.model(image=inputs["pixel_values"])

        mean = outputs["image_features"]['mean']

        diag_cov = torch.exp(outputs["image_features"]['std'])

        return diag_cov.sum(dim=-1).item()
    
def trace_comparison(clip_util, prolip_util, image, N):
    # Calculate the uncertainty (trace of the covariance)
    trace_mcdo = clip_util.get_cov_trace(image, N)
    trace_prolip = prolip_util.get_cov_trace(image)
    print(f"Monte Carlo Dropout Uncertainty (Trace): {trace_mcdo}")
    print(f"Prolip Uncertainty (Trace): {trace_prolip}")
    return trace_mcdo, trace_prolip

if __name__ == '__main__':
    DATA_DIR = '../general_testimgs'
    
    # Load images from directory
    images = {}  # category -> list of (image_path, image_object)
    
    for category in os.listdir(DATA_DIR):
        category_path = os.path.join(DATA_DIR, category)
        
        # Skip if not a directory
        if not os.path.isdir(category_path):
            continue
            
        images[category] = []
        
        for image_filename in os.listdir(category_path):
            image_path = os.path.join(category_path, image_filename)
            
            # Skip if not an image file
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            
            try:
                img = Image.open(image_path).convert('RGB')
                images[category].append((image_path, img))
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    
    # Initialize the utility classes
    clip_util = CLIPUtil(dropout=0.1)
    clip_util.check_dropout_layers()
    prolip_util = ProlipUtil()
    
    # Store traces
    traces_prolip = {}  # category -> list of traces
    traces_mcdo = {}    # category -> list of traces
    
    # Process each category
    for category, img_list in images.items():
        print(f"\nProcessing category: {category}")
        traces_prolip[category] = []
        traces_mcdo[category] = []
        
        for image_path, img in img_list:
            print(f"  Processing: {os.path.basename(image_path)}")
            trace_mcdo, trace_prolip = trace_comparison(clip_util, prolip_util, img, N=100)
            traces_prolip[category].append(trace_prolip)
            traces_mcdo[category].append(trace_mcdo)
    
    # After processing all images, before plotting:
    # Flatten all data into single lists
    all_mcdo_traces = []
    all_prolip_traces = []
    index_to_name = {}  # index -> image filename

    idx = 0
    for category, img_list in images.items():
        for i, (image_path, img) in enumerate(img_list):
            if i < len(traces_mcdo.get(category, [])):
                all_mcdo_traces.append(traces_mcdo[category][i])
                all_prolip_traces.append(traces_prolip[category][i])
                index_to_name[idx] = f"{category}/{os.path.basename(image_path)}"
                idx += 1

    # Print the index mapping
    print("\n" + "="*60)
    print("INDEX TO IMAGE MAPPING")
    print("="*60)
    for idx, name in index_to_name.items():
        print(f"{idx}: {name}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.set_title('Monte Carlo Dropout Uncertainty', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Trace (Uncertainty)')

    ax2.set_title('ProLIP Uncertainty', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Image Index')
    ax2.set_ylabel('Trace (Uncertainty)')

    # Simple scatter plots
    x_positions = range(len(all_mcdo_traces))
    ax1.scatter(x_positions, all_mcdo_traces, s=100, alpha=0.7)
    ax2.scatter(x_positions, all_prolip_traces, s=100, alpha=0.7)

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('uncertainty_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total images: {len(all_mcdo_traces)}")
    print(f"MCDO   - Mean: {np.mean(all_mcdo_traces):.4f}, Std: {np.std(all_mcdo_traces):.4f}")
    print(f"ProLIP - Mean: {np.mean(all_prolip_traces):.4f}, Std: {np.std(all_prolip_traces):.4f}")

    # Save the mapping to a file
    with open('index_to_image_mapping.txt', 'w') as f:
        for idx, name in index_to_name.items():
            f.write(f"{idx}: {name}\n")
    print("\nIndex mapping saved to 'index_to_image_mapping.txt'")
