import os
import json
import random

#import perlin noise library
from perlin_noise import PerlinNoise


segments = []

delta_y = (17-0) / 854
delta_x = 0.5

# use perlin noise to generate y values between 17 and 25
noise = PerlinNoise(octaves=60, seed=random.randint(0, 1000))

def perlin_noise(x):
    return 20 + noise(x) * 12

for x in range(1, 13):
    
    for i in range(854):
        print(noise(i*0.00075))
        segment = {
            "x": delta_x * i,
            "y": perlin_noise(i*0.00075),
            "length":0.5,
            "angleToNextVector":10.0
        }
        segments.append(segment)
        
    with open("../validation-data/perlin-ground.json", "r") as f:
        # read old data into a dictionary and update the segments
        data = json.load(f)
        data["segments"] = segments
        
    with open("../validation-data/perlin-ground-"+str(x)+".json", "w") as f:
        json.dump(data, f)
    
    
print("Done")