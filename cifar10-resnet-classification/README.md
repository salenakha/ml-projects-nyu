# CIFAR-10 ResNet Classification

Transfer learning with ResNet50V2 for image classification on CIFAR-10.

## Model Details

**Base Model**: ResNet50V2 (pre-trained on ImageNet, frozen)

**Custom Layers**:
```
- Resizing(224, 224)
- GlobalAveragePooling2D
- Dense(128, relu) + Dropout(0.1)
- Dense(32, relu) + Dropout(0.1)
- Dense(32, relu) + Dropout(0.1)
- Dense(10, softmax)
```

**Training**:
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch Size: 128
- Epochs: 11
- Total Parameters: 23,832,586
- Trainable Parameters: 267,786

## Dataset

**CIFAR-10**: 60,000 32×32 color images in 10 classes
- Training: 50,000 images
- Test: 10,000 images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Running the Code

**Jupyter Notebook**:
```bash
jupyter notebook notebooks/CIFAR10_ResNet_Classification.ipynb
```

**Python Script**:
```bash
python src/CIFAR10_ResNet_Classification.py
```

## Key Features

- Uses transfer learning with pre-trained ResNet50V2
- Images scaled from 32×32 to 224×224 for ResNet compatibility
- Data preprocessing: `resnet_v2.preprocess_input()`
- Validation accuracy tracked across epochs

## Results

Visit [presentation](./presentations/CIFAR10%20ResNet%20Classification.pdf) for our fun results!
