# SmartVision — Dataset Plan

## Objective
Build an image classification system that recognizes five known food categories and rejects inputs from categories excluded from training.

## Proposed dataset
Food-101

## Known classes
- Pizza
- Hamburger
- Sushi
- Ice cream
- Pancakes

## Unknown classes
- Donuts
- Waffles
- Tacos

## Data splitting
- Training: known categories only
- Validation: held-out images from known categories
- Testing: unseen known-category images and images from excluded categories

## Preprocessing
- Verify image labels and class counts
- Resize images to the model's expected input size
- Convert images to RGB
- Normalize using the pretrained model's expected normalization
- Apply data augmentation to training images only

## Important safeguards
- Never train the classifier on unknown categories.
- Never use the final test set to choose the rejection threshold.
- Record dataset source, preprocessing choices, and evaluation metrics.
- Keep the original test images separate from training data.
