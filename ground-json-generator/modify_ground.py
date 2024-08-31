import os
import json
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pickle
import sys

random_seed = 1337
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
train_ratio = 0.85  # 80% for training, 20% for validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE=64
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS=int(os.cpu_count() / 2) 

#print(NUM_WORKERS)
#print(AVAIL_GPUS)


# LATEST CONDITIONAL
class FNNGenerator(nn.Module):
    def __init__(self, input_size, condition_size, hidden_sizes, output_size, dropout_prob=0.0):
        super(FNNGenerator, self).__init__()
        layers = []
        sizes = [input_size + condition_size] + hidden_sizes  # Add condition size to the input size
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))  # Fully connected layer
            layers.append(nn.BatchNorm1d(sizes[i+1]))  # Batch normalization
            layers.append(nn.LeakyReLU(0.1, inplace=True))  # Leaky ReLU activation with in-place operation
            # layers.append(nn.Dropout(dropout_prob))  # Dropout for regularization
        
        layers.append(nn.Linear(sizes[-1], output_size))  # Final layer to output the generated data
        layers.append(nn.Tanh())  # Tanh activation to keep the output within a certain range
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, condition):
        condition = condition.expand(x.size(0), -1)  # Ensure conditions have correct shape
        x = torch.cat((x, condition), dim=1)  # Concatenate noise with condition
        return self.model(x)
    
class FNNDiscriminator(nn.Module):
    def __init__(self, input_size, condition_size, hidden_sizes, output_size, dropout_prob=0.255):
        super(FNNDiscriminator, self).__init__()
        layers = []
        sizes = [input_size + condition_size] + hidden_sizes  # Add condition size to the input size
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))  # Fully connected layer
            layers.append(nn.BatchNorm1d(sizes[i+1]))  # Batch normalization
            layers.append(nn.ReLU(inplace=True))  # ReLU activation
            layers.append(nn.Dropout(dropout_prob))  # Dropout for regularization
        
        layers.append(nn.Linear(sizes[-1], output_size))  # Final layer to output a single value
        layers.append(nn.Sigmoid())  # Sigmoid activation for binary classification
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, condition):
        condition = condition.expand(x.size(0), -1)  # Ensure conditions have correct shape
        x = torch.cat((x, condition), dim=1)  # Concatenate real/fake data with condition
        return self.model(x)

def normalize_angles(angles, min_angle=-45, max_angle=60):
    # Normalize angles to the range [-1, 1]
    return 2 * ((angles - min_angle) / (max_angle - min_angle)) - 1

def denormalize_angles(normalized_angles, min_angle=-45, max_angle=60):
    # Denormalize angles to the range [min_angle, max_angle]
    return ((normalized_angles + 1) / 2) * (max_angle - min_angle) + min_angle
    
def convert_to_useable_data(data, start_idx, end_idx):
    x = 0
    y = data['segments'][end_idx + 1]['y']
    if(start_idx - 1 >= 0):
        x, y = data['segments'][start_idx - 1]['x'], data['segments'][start_idx - 1]['y']

    for i in range(start_idx, end_idx):
        segment = data['segments'][i]
        length = segment["length"]
        tmpangle = segment["angleToNextVector"]
        
        angle = np.radians(segment["angleToNextVector"])  # Convert angle to radians
        
        new_x = x + length * np.cos(angle)
        new_y = y + length * np.sin(angle)
        segment["x"] = new_x
        segment["y"] = new_y
        
        x, y = new_x, new_y
        
    # sample["grassPositions"].append(int(np.random.choice([-1, 0, 1, 2, 3, 4, -1], p=[0.5, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1])))
    
def modify_level(generator, existing_level, point_of_interest, condition):
    """
    Modify specific points in an existing level around a point of interest using a trained generator.
    
    Parameters:
    - generator: Trained generator model
    - existing_level: The existing level data to be modified (torch.Tensor)
    - point_of_interest: Index around which the level will be modified (int)
    - modification_radius: Radius around the point_of_interest to be modified (int)
    - noise_dim: Dimension of the noise vector (int)
    - condition: The condition to be passed to the generator (torch.Tensor)
    
    Returns:
    - modified_level: The level with modifications (torch.Tensor)
    """
    # Ensure the generator is in evaluation mode
    generator.eval()
    
    modification_radius = 30
    
    segments = existing_level['segments']
    segments_tensor = torch.tensor([[segment['length'], segment['angleToNextVector'], segment['x'], segment['y']] for segment in segments], dtype=torch.float32).to(device)
    
    # Determine the range of indices to be modified
    start_idx = max(0, int(point_of_interest) - modification_radius)
    end_idx = min(len(existing_level['segments']), int(point_of_interest) + modification_radius + 1)
    
    #print("Start index: ", start_idx)
    #print("End index: ", end_idx)
    if start_idx >= end_idx:
        raise ValueError(f"Invalid indices range: start_idx ({start_idx}) >= end_idx ({end_idx}) with point_of_interest ({point_of_interest}) and modification_radius ({modification_radius})")

    
    modification_indices = torch.arange(start_idx, end_idx).to(device)
    #print("Modification indices: ", modification_indices)
    
    def normalize_condition(condition, min_val, max_val):
        return (condition - min_val) / (max_val - min_val)

    min_val = torch.tensor([0.2211]).to(device)  # Replace with actual min_val used in training
    max_val = torch.tensor([0.6012]).to(device)  # Replace with actual max_val used in training
    
    # Generate new values for the specified indices using the generator
    with torch.no_grad():
        # Generate noise vectors for each modification point
        z = torch.randn(1, input_size_generator).to(device)  # Generate random noise
        dummy_condition = torch.tensor([[condition]]).to(device)  # Fill tensor with specific value 
        # Normalize the condition value
        normalized_condition = normalize_condition(condition, min_val, max_val)
        
        dummy_condition = normalized_condition  # Use the normalized condition value
        
        fake_data = generator(z, dummy_condition)  # Pass both noise and condition
        fake_data = fake_data.view(fake_data.size(0), 855, 2)
        
        fake_data[:, :, 1] = denormalize_angles(fake_data[:, :, 1])
    
    # go through fake data tensor and only take the segments that are in the range of the modification_indices
    fake_data = fake_data[0][start_idx:end_idx]
    
    # add a column of zeros for x and y
    fake_data = torch.cat((fake_data, torch.zeros(fake_data.size(0), 2).to(device)), dim=1)
    
    # Replace the specified indices in the existing level with the new values
    modified_segments = segments_tensor.clone()
    
    #print existing level segments 
    # for i in range(start_idx-1, end_idx):
        # print("Existing level segment: ", segments[i])
    
    # print("Existing after end index: ", segments[end_idx])
    
    modified_segments[modification_indices] = fake_data
    
    
    final_level = {"segments": [{"length": seg[0].item(), "angleToNextVector": seg[1].item(), "x": seg[2].item(), "y": seg[3].item()} for seg in modified_segments], "grassPositions": existing_level['grassPositions'], "startingPoint": existing_level['startingPoint']}
        
    convert_to_useable_data(final_level, start_idx, end_idx)
    
    # for i in range(start_idx - 1, end_idx + 1):
        # print("Modified level segment at index "+str(i)+": ", final_level['segments'][i])
    
    num = 0
    # go through the final level and check if x are increasing
    for i in range(1, len(final_level['segments'])):
        if final_level['segments'][i]['x'] < final_level['segments'][i-1]['x']:
            #print("x is not increasing")
            # print("Index: ", i)
            #print("x: ", final_level['segments'][i]['x'])
            #print("x-1: ", final_level['segments'][i-1]['x'])
            num +=1
    
    # print("Number of x not increasing: ", num)
    
    # go through final level and check if y that are modified are not too far from the original y
    diff = 0
    if final_level['segments'][end_idx-1]['y'] - final_level['segments'][end_idx]['y'] > 0.5:
        diff = final_level['segments'][end_idx-1]['y'] - final_level['segments'][end_idx]['y']
        # add the difference to every y after the from start_idx to end_idx
        for i in range(start_idx, end_idx):
            final_level['segments'][i]['y'] -= diff
    if final_level['segments'][start_idx-1]['y'] - final_level['segments'][start_idx]['y'] > 0.5 and start_idx > 0:
        diff = final_level['segments'][start_idx-1]['y'] - final_level['segments'][start_idx]['y']
        # add the difference to every y after the from start_idx to end_idx
        for i in range(start_idx, end_idx):
            final_level['segments'][i]['y'] += diff

    return final_level


# Function to read JSON data from a file
def read_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    
    
if __name__ == "__main__":
    # get groundData for level and generation death data from parameters when running py script
    
    # convert json data to numbers
    conditionHigh = json.loads(sys.argv[1])
    conditionLow = json.loads(sys.argv[2])
    groundData = read_from_file(sys.argv[3])
    safe_point_of_interest = json.loads(sys.argv[4])
    unsafe_point_of_interest = json.loads(sys.argv[5])

    # Define generator hyperparameters
    input_size_discriminator = 1710  # Input size should match the length of each segment for length and angle only
    hidden_sizes_discriminator = [512, 512, 512, 512, 512, 512, 512]
    output_size_discriminator = 1

    input_size_generator = 16
    hidden_sizes_generator = [512, 512, 512, 512, 512, 512, 512]
    output_size_generator = 1710  # Output size should match the size of each segment for length and angle only

    # Initialize generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = FNNGenerator(input_size_generator, 1, hidden_sizes_generator, output_size_generator).to(device)

    generator.load_state_dict(torch.load('../gans/generator-best-test_5.pth', map_location=device))

    # Point of interest and radius
    point_of_interest = safe_point_of_interest['x']

    # Modify the level
    modified_level = modify_level(generator, groundData, point_of_interest, conditionLow)
    
    point_of_interest = unsafe_point_of_interest['x']
    
    # Modify the level
    modified_level = modify_level(generator, modified_level, point_of_interest, conditionHigh)
    
    #for i in range(len(modified_level['segments'])):
        #print("Modified level segment at index "+str(i)+": ", modified_level['segments'][i])
    print(json.dumps(modified_level))
