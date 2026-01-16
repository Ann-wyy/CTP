"""
Example usage of CTP Classification Network

Demonstrates how to create, train, and use the CTP classifier.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from ctp_classification_model import create_ctp_classifier, CTPClassificationNet


def example_basic_usage():
    """Basic example of creating and using the model."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create model for binary classification (stroke vs no stroke)
    model = create_ctp_classifier(
        num_time_points=21,
        num_classes=2,
        resnet_type='resnet50',
        pretrained=False
    )

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model moved to: {device}")

    # Create dummy input data
    batch_size = 4
    ctp_data = torch.randn(batch_size, 512, 512, 32, 21).to(device)
    prior_maps = torch.randn(batch_size, 5, 512, 512).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(ctp_data, prior_maps)
        probabilities = torch.softmax(logits, dim=1)

    print(f"\nInput shapes:")
    print(f"  CTP data: {ctp_data.shape}")
    print(f"  Prior maps: {prior_maps.shape}")
    print(f"\nOutput shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Probabilities: {probabilities.shape}")
    print(f"\nSample predictions:")
    for i in range(batch_size):
        pred_class = torch.argmax(probabilities[i]).item()
        confidence = probabilities[i, pred_class].item()
        print(f"  Sample {i}: Class {pred_class}, Confidence: {confidence:.4f}")


def example_training_loop():
    """Example training loop."""
    print("\n" + "=" * 60)
    print("Example 2: Training Loop")
    print("=" * 60)

    # Create model
    model = create_ctp_classifier(
        num_time_points=21,
        num_classes=2,
        resnet_type='resnet34',  # Smaller model for faster training
        pretrained=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy training data
    num_epochs = 3
    num_batches = 5
    batch_size = 2

    print(f"Training for {num_epochs} epochs with {num_batches} batches per epoch")

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            # Generate dummy data
            ctp_data = torch.randn(batch_size, 512, 512, 32, 21).to(device)
            prior_maps = torch.randn(batch_size, 5, 512, 512).to(device)
            labels = torch.randint(0, 2, (batch_size,)).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(ctp_data, prior_maps)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


def example_feature_extraction():
    """Example of extracting intermediate features."""
    print("\n" + "=" * 60)
    print("Example 3: Feature Extraction")
    print("=" * 60)

    model = create_ctp_classifier(num_time_points=21, num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create dummy input
    ctp_data = torch.randn(1, 512, 512, 32, 21).to(device)
    prior_maps = torch.randn(1, 5, 512, 512).to(device)

    # Extract features
    model.eval()
    with torch.no_grad():
        learned, priors, combined = model.get_feature_maps(ctp_data, prior_maps)

    print(f"Extracted feature maps:")
    print(f"  Learned features: {learned.shape}")
    print(f"  Prior features: {priors.shape}")
    print(f"  Combined features: {combined.shape}")

    # You can save these features for visualization or further analysis
    print(f"\nFeature statistics:")
    print(f"  Learned - mean: {learned.mean():.4f}, std: {learned.std():.4f}")
    print(f"  Prior - mean: {priors.mean():.4f}, std: {priors.std():.4f}")
    print(f"  Combined - mean: {combined.mean():.4f}, std: {combined.std():.4f}")


def example_different_configurations():
    """Example of creating models with different configurations."""
    print("\n" + "=" * 60)
    print("Example 4: Different Model Configurations")
    print("=" * 60)

    configs = [
        {'name': 'ResNet18 - T=20', 'resnet_type': 'resnet18', 'num_time_points': 20},
        {'name': 'ResNet34 - T=21', 'resnet_type': 'resnet34', 'num_time_points': 21},
        {'name': 'ResNet50 - T=21', 'resnet_type': 'resnet50', 'num_time_points': 21},
    ]

    for config in configs:
        model = create_ctp_classifier(
            num_time_points=config['num_time_points'],
            num_classes=2,
            resnet_type=config['resnet_type'],
            pretrained=False
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{config['name']}:")
        print(f"  Total parameters: {total_params:,}")


def example_multi_class_classification():
    """Example for multi-class classification."""
    print("\n" + "=" * 60)
    print("Example 5: Multi-class Classification")
    print("=" * 60)

    # Create model for 3-class classification
    # e.g., Class 0: No stroke, Class 1: Ischemic stroke, Class 2: Hemorrhagic stroke
    num_classes = 3
    model = create_ctp_classifier(
        num_time_points=21,
        num_classes=num_classes,
        resnet_type='resnet50',
        pretrained=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create dummy input
    ctp_data = torch.randn(2, 512, 512, 32, 21).to(device)
    prior_maps = torch.randn(2, 5, 512, 512).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(ctp_data, prior_maps)
        probabilities = torch.softmax(logits, dim=1)

    print(f"Multi-class classification with {num_classes} classes")
    print(f"Output shape: {logits.shape}")
    print(f"\nPredictions:")
    for i in range(logits.shape[0]):
        pred_class = torch.argmax(probabilities[i]).item()
        print(f"  Sample {i}: Predicted class = {pred_class}")
        print(f"    Class probabilities: {probabilities[i].cpu().numpy()}")


def example_model_saving_loading():
    """Example of saving and loading the model."""
    print("\n" + "=" * 60)
    print("Example 6: Model Saving and Loading")
    print("=" * 60)

    # Create and save model
    model = create_ctp_classifier(num_time_points=21, num_classes=2)

    # Save model
    save_path = 'ctp_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Load model
    loaded_model = create_ctp_classifier(num_time_points=21, num_classes=2)
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval()
    print(f"Model loaded from {save_path}")

    # Test loaded model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model = loaded_model.to(device)

    ctp_data = torch.randn(1, 512, 512, 32, 21).to(device)
    prior_maps = torch.randn(1, 5, 512, 512).to(device)

    with torch.no_grad():
        output = loaded_model(ctp_data, prior_maps)

    print(f"Inference successful with loaded model")
    print(f"Output shape: {output.shape}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CTP Classification Network - Usage Examples")
    print("=" * 60 + "\n")

    # Run all examples
    example_basic_usage()
    example_training_loop()
    example_feature_extraction()
    example_different_configurations()
    example_multi_class_classification()
    example_model_saving_loading()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
