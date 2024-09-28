import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import json
import urllib.request
from io import BytesIO

# Load the pre-trained ResNet-18 model
model = models.resnet152(pretrained=True)
model.eval()

# Load ImageNet class labels
url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
response = urllib.request.urlopen(url)
idx_to_class = json.load(response)

# Function to get class name from output index
def get_class_name(idx):
    return idx_to_class[str(idx)][1]

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# URLs of sample images
image_urls = [
    'https://images.unsplash.com/photo-1560807707-8cc77767d783',    # Blenheim_spaniel
    'https://images.unsplash.com/photo-1518020382113-a7e8fc38eac9',  # pug
    'https://images.unsplash.com/photo-1534081333815-ae5019106622',  # snorkel
    'https://images.unsplash.com/photo-1518770660439-4636190af475',  # hard_disc
    'https://images.unsplash.com/photo-1508672019048-805c876b67e2' #lakeside
    ]

# Process and display images
for url in image_urls:
    # Load image from URL
    print("url: ", url)
    try:
        response = urllib.request.urlopen(url)
        img = Image.open(BytesIO(response.read())).convert('RGB')
    except:
        continue
    
    # Preprocess image
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch

    # Move to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Inference
    with torch.no_grad():
        output = model(input_batch)

    # Get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Print the top 1 prediction
    predicted_idx = top5_catid[0].item()
    predicted_class = get_class_name(predicted_idx)
    confidence = top5_prob[0].item()
    print(f"Predicted class: {predicted_class}, Confidence: {100 * confidence:.2f}\n")

    # Print top 5 predictions
    print("Top 5 predictions:")
    for i in range(top5_prob.size(0)):
        idx = top5_catid[i].item()
        cls = get_class_name(idx)
        prob = top5_prob[i].item()
        print(f"{cls}: {100 * prob:.2f}")
    print("-" * 50)



# Preprocessing pipeline (excluding normalization)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

# Load the pre-trained ResNet-18 model
model = models.resnet152(pretrained=True)
model.eval()

# Load ImageNet class labels
url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
response = urllib.request.urlopen(url)
idx_to_class = json.load(response)

# Function to get class name from output index
def get_class_name(idx):
    return idx_to_class[str(idx)][1]

# Preprocessing pipeline (excluding normalization)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

# Transformation to tensor and normalization (applied after adding noise)
to_tensor_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# URLs of sample images
image_urls = [
    'https://images.unsplash.com/photo-1560807707-8cc77767d783',    # Blenheim_spaniel
    'https://images.unsplash.com/photo-1518020382113-a7e8fc38eac9',  # pug
    'https://images.unsplash.com/photo-1534081333815-ae5019106622',  # snorkel
    'https://images.unsplash.com/photo-1518770660439-4636190af475',  # hard_disc
    'https://images.unsplash.com/photo-1508672019048-805c876b67e2' #lakeside
    ]

# Load and preprocess images
images = []
for url in image_urls:
    # Load image from URL
    response = urllib.request.urlopen(url)
    img = Image.open(BytesIO(response.read())).convert('RGB')
    
    # Preprocess image (resize and crop)
    img = preprocess(img)
    images.append(img)

# Noise levels from light to heavy (10 stages)
noise_levels = np.linspace(0, 1, 10)  # Standard deviation from 0 to 0.5

# Load and preprocess images
images = []
for url in image_urls:
    # Load image from URL
    response = urllib.request.urlopen(url)
    img = Image.open(BytesIO(response.read())).convert('RGB')
    
    # Preprocess image (resize and crop)
    img = preprocess(img)
    images.append(img)

# Noise levels from light to heavy (10 stages)
noise_levels = np.linspace(0, 1, 15)  # Standard deviation from 0 to 0.5

# Process each image with increasing noise
for idx, img in enumerate(images):
    print(f"\nProcessing Image {idx + 1} with Increasing Noise Levels")
    num_cols = 5  # Number of images per row
    num_rows = 3  # Total rows needed
    plt.figure(figsize=(20, 10))  # Increase figure size (width, height)
    for i, noise_std in enumerate(noise_levels):
        # Convert image to NumPy array and scale to [0, 1]
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_std, img_np.shape)
        
        # Add noise and clip to [0, 1]
        noisy_img_np = np.clip(img_np + noise, 0, 1)
        
        # Convert back to PIL Image
        noisy_img = Image.fromarray((noisy_img_np * 255).astype(np.uint8))
        
        # Apply tensor conversion and normalization
        input_tensor = to_tensor_normalize(noisy_img)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch

        # Move to GPU if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        # Inference
        with torch.no_grad():
            output = model(input_batch)

        # Get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top prediction
        top_prob, top_catid = torch.topk(probabilities, 1)
        predicted_idx = top_catid[0].item()
        predicted_class = get_class_name(predicted_idx)
        confidence = top_prob[0].item()

        # Display the image
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(noisy_img)
        plt.title(f"Noise: {noise_std:.2f}\n{predicted_class}\nConf: {100 * confidence:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import urllib.request
from io import BytesIO
import torchattacks

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()

# Load ImageNet class labels
url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
response = urllib.request.urlopen(url)
idx_to_class = json.load(response)

# Function to get class name from output index
def get_class_name(idx):
    return idx_to_class[str(idx)][1]

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# De-normalization for visualization
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# URLs of sample images
image_urls = [
    'https://images.unsplash.com/photo-1560807707-8cc77767d783',    # Blenheim_spaniel
    'https://images.unsplash.com/photo-1518020382113-a7e8fc38eac9',  # pug
    'https://images.unsplash.com/photo-1534081333815-ae5019106622',  # snorkel
    'https://images.unsplash.com/photo-1518770660439-4636190af475',  # hard_disc
    'https://images.unsplash.com/photo-1508672019048-805c876b67e2' #lakeside
    ]

# Load and preprocess images
images = []
original_images = []
for url in image_urls:
    # Load image from URL
    response = urllib.request.urlopen(url)
    img = Image.open(BytesIO(response.read())).convert('RGB')
    original_images.append(img)
    
    # Preprocess image
    input_tensor = preprocess(img)
    images.append(input_tensor)

# Stack images into a batch
inputs = torch.stack(images)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = inputs.to(device)

# Get the model's predictions on original images
with torch.no_grad():
    outputs = model(inputs)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top_probs, top_catids = torch.topk(probabilities, 1)
    preds = top_catids.squeeze(1)
    predicted_classes = [get_class_name(idx.item()) for idx in preds]
    confidences = [prob.item() for prob in top_probs.squeeze(1)]

# Generate adversarial examples using FGSM attack
from torchattacks import CW

atk.set_mode_default()
# Create the FGSM attack method
atk = CW(model, steps=200, lr=0.002)  # Adjust eps for different attack strengths

# Generate adversarial images
adv_images = atk(inputs, preds)

adv_images_pil = []
# Process each adversarial image
for idx in range(len(adv_images)):
    adv_image = adv_images[idx]
    # De-normalize for visualization
    adv_image_denorm = inv_normalize(adv_image)
    # Clamp to [0,1] range
    adv_image_denorm = torch.clamp(adv_image_denorm, 0, 1)
    # Convert to NumPy array
    adv_image_np = adv_image_denorm.cpu().numpy()
    # Transpose to (H, W, C)
    adv_image_np = np.transpose(adv_image_np, (1, 2, 0))
    # Convert to PIL Image and ensure mode and size match
    adv_img_pil = Image.fromarray((adv_image_np * 255).astype(np.uint8))
    adv_img_pil = adv_img_pil.convert('RGB')  # Ensure it's in 'RGB' mode
    adv_images_pil.append(adv_img_pil)
    
    # Original image
    orig_image = original_images[idx]
    
    # Inference on adversarial image
    adv_input = adv_images[idx].unsqueeze(0)
    output = model(adv_input)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    adv_predicted_idx = top_catid[0].item()
    adv_predicted_class = get_class_name(adv_predicted_idx)
    adv_confidence = top_prob[0].item()
    
    # Display the original and adversarial images side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_image)
    plt.title(f"Original Image\n{predicted_classes[idx]}\nConfidence: {confidences[idx]:.2f}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(adv_img_pil)
    plt.title(f"Adversarial Image\n{adv_predicted_class}\nConfidence: {adv_confidence:.2f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
