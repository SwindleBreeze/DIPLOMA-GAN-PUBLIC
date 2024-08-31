import os
import json
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pickle


# random_seed = 1337
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)

BATCH_SIZE=64
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS=int(os.cpu_count() / 2) # Number of workers for data loader

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

def convert_to_useable_data(data):
    for sample in data:
        #print("sample", sample)
        segments = sample["segments"]
        # Starting point x is 0 y is rand value from 10 to 20
        x, y = 0, int(np.random.uniform(10, 15))
        sample["grassPositions"] = []
        sample["startingPoint"] = np.random.uniform(0, 100000)
        for segment in segments:
            length = segment["length"]
            tmpangle = segment["angleToNextVector"]
            #while tmpangle < 0:
            #    tmpangle = tmpangle + 360
            angle = np.radians(segment["angleToNextVector"])  # Convert angle to radians

            new_x = x + length * np.cos(angle)
            new_y = y + length * np.sin(angle) 
            segment["x"] = new_x
            segment["y"] = new_y
            

            # Update the current point
            x, y = new_x, new_y
            
            # add random integer betwwen -1 and 5 into the grassPositions, with 2/3 chance of it being -1
            sample["grassPositions"].append(int(np.random.choice([-1, 0, 1, 2, 3, 4, -1], p=[0.5, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1])))
    # sample["grassPositions"].append(int(np.random.choice([-1, 0, 1, 2, 3, 4, -1], p=[0.5, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1])))


def generate_ground_data():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    input_size_discriminator = 1710  # Input size should match the length of each segment for length and angle only
    hidden_sizes_discriminator = [512, 512, 512, 512, 512, 512, 512]
    output_size_discriminator = 1

    input_size_generator = 16
    hidden_sizes_generator = [512, 512, 512, 512, 512, 512, 512]
    output_size_generator = 1710  # Output size should match the size of each segment for length and angle only

    torch.map_location = device

    # load the generator and discriminator models from .pth files

    new_discriminator = FNNDiscriminator(input_size_discriminator, 1, hidden_sizes_discriminator, output_size_discriminator).to(device)
    new_generator = FNNGenerator(input_size_generator, 1, hidden_sizes_generator, output_size_generator).to(device)

    # print current contents of the directory
    # print(os.listdir('./gans/testsaves'))

    # new_discriminator.load_state_dict(torch.load('../gans/testsaves/best_discriminator_test_1.pth'))
    # new_generator.load_state_dict(torch.load('../gans/testsaves/best_generator_test_1.pth'))
    
    new_discriminator.load_state_dict(torch.load('../gans/discriminator-best-test_5.pth', map_location=device))
    new_generator.load_state_dict(torch.load('../gans/generator-best-test_5.pth', map_location=device))
    
    # new_discriminator.load_state_dict(torch.load('../gans/discriminator-best-so-far-test.pth'))
    # new_generator.load_state_dict(torch.load('../gans/generator-best-so-far-test.pth'))

    # Generate new data after training
    # z = torch.randn(1, input_size_generator).unsqueeze(1).repeat(1, 855, 1).to(device)  # Generate random noise
    new_discriminator.eval()
    new_generator.eval()
    
    def normalize_condition(condition, min_val, max_val):
        return (condition - min_val) / (max_val - min_val)

    min_val = torch.tensor([0.2211]).to(device)  # Replace with actual min_val used in training
    max_val = torch.tensor([0.6012]).to(device)  # Replace with actual max_val used in training
    # Generate new data after training
    with torch.no_grad():  # Disable gradient computation for inference
        z = torch.randn(1, input_size_generator).to(device)  # Generate random noise
        specific_condition_value = 9.15 # Replace with specific condition value
        dummy_condition = torch.tensor([[specific_condition_value]]).to(device)  # Fill tensor with specific value
        
        # Normalize the condition value
        normalized_condition = normalize_condition(specific_condition_value, min_val, max_val)
        
        dummy_condition = normalized_condition  # Use the normalized condition value
        
        fake_data = new_generator(z, dummy_condition)  # Pass both noise and condition
        fake_data = fake_data.view(fake_data.size(0), 855, 2)

    fake_data[:, :, 1] = denormalize_angles(fake_data[:, :, 1])

    generated_data = []

    for sample in fake_data:
        segment_data = []
        for vector in sample:
            segment_data.append({
                "length": vector[0].item(),
                "angleToNextVector": vector[1].item()
            })
        generated_data.append({"segments": segment_data})
        


    convert_to_useable_data(generated_data)
    return generated_data

if __name__ == "__main__":
    generated_data = generate_ground_data()
    print(json.dumps(generated_data))