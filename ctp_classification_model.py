"""
CTP Classification Network

This module implements a CT Perfusion (CTP) classification network that combines
raw 4D CTP data with perfusion prior maps for stroke classification.

Architecture:
    1. Frontend Encoder: Processes raw CTP 4D data (B, 512, 512, 32, 21/20)
    2. Prior Fusion: Integrates 5 perfusion maps (CBF, CBV, MTT, Tmax, TTP)
    3. ResNet Backbone: Modified ResNet for final classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights,
)
from typing import Optional, Tuple


class FrontendEncoder(nn.Module):
    """
    Frontend encoder that processes raw CTP 4D data using 3D convolutions.

    Converts (B, 512, 512, 32, T) -> (B, 32, T, 512, 512) -> (B, 155, 128, 128)
    using 3D CNN with temporal convolutions + spatial downsampling + adaptive pooling.

    Spatial downsampling (512->256->128) is critical to keep memory usage feasible:
      - Without downsampling: (B, 128, T, 512, 512) ~10 GB at batch_size=4
      - With downsampling:    (B, 128, T, 128, 128)  ~670 MB at batch_size=4

    Supports arbitrary time points (T can be any value: 20, 21, 22, etc.)
    """

    def __init__(self, out_channels: int = 155):
        """
        Args:
            out_channels: Number of output learned features (default: 155)
        """
        super(FrontendEncoder, self).__init__()

        # 3D convolution to learn spatio-temporal features
        # stride=(1,2,2): keeps temporal dim, halves spatial dims 512->256
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        # stride=(1,2,2): keeps temporal dim, halves spatial dims 256->128
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )

        # Adaptive pooling to handle variable time points
        # Pool temporal dimension to fixed size (8 time steps)
        self.temporal_pool = nn.AdaptiveAvgPool3d((8, None, None))

        # 2D convolution for final spatial feature extraction
        self.conv2d = nn.Sequential(
            # Reduce channels: 128*8=1024 -> 256
            nn.Conv2d(128 * 8, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Spatial features: 256 -> out_channels
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input CTP data of shape (B, 512, 512, 32, T) where T can be any positive integer

        Returns:
            Learned features of shape (B, out_channels, 128, 128)
        """
        B, H, W, Z, T = x.shape

        # Reshape: (B, 512, 512, 32, T) -> (B, 32, T, 512, 512)
        x = x.permute(0, 3, 4, 1, 2)  # (B, 32, T, 512, 512)

        # 3D convolutions with spatial downsampling
        x = self.conv3d_1(x)  # (B, 64, T, 256, 256)
        x = self.conv3d_2(x)  # (B, 128, T, 128, 128)

        # Adaptive temporal pooling: handle variable T
        x = self.temporal_pool(x)  # (B, 128, 8, 128, 128)

        # Flatten temporal dimension
        B, C, T_pooled, H, W = x.shape
        x = x.reshape(B, C * T_pooled, H, W)  # (B, 1024, 128, 128)

        # 2D convolutions: final spatial features
        x = self.conv2d(x)  # (B, out_channels, 128, 128)

        return x


class PriorFusionModule(nn.Module):
    """
    Prior fusion module that adapts perfusion prior maps.

    Processes 5 perfusion maps (CBF, CBV, MTT, Tmax, TTP) using Conv + BN.
    Downsamples from 512x512 to 128x128 to match FrontendEncoder output.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 5):
        """
        Args:
            in_channels: Number of prior maps (default: 5 for CBF, CBV, MTT, Tmax, TTP)
            out_channels: Number of output channels (default: 5)
        """
        super(PriorFusionModule, self).__init__()

        self.adapter = nn.Sequential(
            # 1x1 conv to mix channels
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Downsample 512->128 (4x) to match FrontendEncoder spatial output
            nn.AvgPool2d(kernel_size=4, stride=4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Prior maps of shape (B, 5, 512, 512)

        Returns:
            Adapted prior features of shape (B, 5, 128, 128)
        """
        return self.adapter(x)


class ModifiedResNet(nn.Module):
    """
    Modified ResNet backbone with custom first conv layer.

    Modifies conv1 to accept 160 input channels instead of 3.
    """

    def __init__(
        self,
        in_channels: int = 160,
        num_classes: int = 2,
        resnet_type: str = 'resnet50',
        pretrained: bool = False
    ):
        """
        Args:
            in_channels: Number of input channels (default: 160)
            num_classes: Number of output classes (default: 2 for binary classification)
            resnet_type: Type of ResNet ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained: Whether to use pretrained weights (only for non-modified layers)
        """
        super(ModifiedResNet, self).__init__()

        # Load base ResNet model
        resnet_configs = {
            'resnet18':  (resnet18,  ResNet18_Weights.IMAGENET1K_V1),
            'resnet34':  (resnet34,  ResNet34_Weights.IMAGENET1K_V1),
            'resnet50':  (resnet50,  ResNet50_Weights.IMAGENET1K_V1),
            'resnet101': (resnet101, ResNet101_Weights.IMAGENET1K_V1),
            'resnet152': (resnet152, ResNet152_Weights.IMAGENET1K_V1),
        }

        if resnet_type not in resnet_configs:
            raise ValueError(f"resnet_type must be one of {list(resnet_configs.keys())}")

        model_fn, weights = resnet_configs[resnet_type]
        base_model = model_fn(weights=weights if pretrained else None)

        # Create modified conv1 layer with custom input channels
        original_conv1 = base_model.conv1
        self.conv1 = self._create_modified_conv1(original_conv1, in_channels)

        # Initialize conv1 weights from pretrained model if available
        if pretrained:
            self._initialize_conv1_weights(original_conv1.weight.data, in_channels)

        # Keep remaining layers from base model
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # Replace final fully connected layer
        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

    def _create_modified_conv1(self, original_conv1: nn.Conv2d, in_channels: int) -> nn.Conv2d:
        """
        Create a modified conv1 layer with custom input channels.

        Args:
            original_conv1: Original conv1 layer from base ResNet
            in_channels: Desired number of input channels

        Returns:
            Modified conv1 layer
        """
        return nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

    def _initialize_conv1_weights(self, pretrained_weights: torch.Tensor, in_channels: int) -> None:
        """
        Initialize conv1 weights by adapting pretrained 3-channel weights to custom channel count.

        Strategy: Repeat the pretrained weights across channels and normalize to maintain
        similar activation magnitudes.

        Args:
            pretrained_weights: Original weights from pretrained model (shape: [out_ch, 3, H, W])
            in_channels: Target number of input channels
        """
        with torch.no_grad():
            PRETRAINED_CHANNELS = 3

            # Calculate how many times to repeat and remaining channels
            full_repeats = in_channels // PRETRAINED_CHANNELS
            remaining_channels = in_channels % PRETRAINED_CHANNELS

            # Build new weights by repeating and optionally adding partial weights
            if full_repeats > 0:
                # Repeat the full 3-channel weights
                new_weights = pretrained_weights.repeat(1, full_repeats, 1, 1)

                # Add partial weights if we have remaining channels
                if remaining_channels > 0:
                    partial_weights = pretrained_weights[:, :remaining_channels, :, :]
                    new_weights = torch.cat([new_weights, partial_weights], dim=1)
            else:
                # If in_channels < 3, just use subset of pretrained weights
                new_weights = pretrained_weights[:, :in_channels, :, :]

            # Normalize to maintain magnitude (scale by ratio of channel counts)
            scaling_factor = PRETRAINED_CHANNELS / in_channels
            self.conv1.weight.data = new_weights * scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (B, 160, 512, 512)

        Returns:
            Classification logits of shape (B, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class CTPClassificationNet(nn.Module):
    """
    Complete CTP Classification Network.

    Combines raw 4D CTP data with perfusion prior maps for stroke classification.

    Pipeline:
        1. Raw CTP (B, 512, 512, 32, T) -> Frontend Encoder -> (B, 155, 128, 128)
        2. Prior maps (B, 5, 512, 512) -> Prior Fusion -> (B, 5, 128, 128)
        3. Concatenate -> (B, 160, 128, 128)
        4. Modified ResNet -> (B, num_classes)
    """

    def __init__(
        self,
        num_time_points: int = 21,
        num_classes: int = 2,
        resnet_type: str = 'resnet50',
        pretrained: bool = False,
        learned_features: int = 155,
        prior_channels: int = 5
    ):
        """
        Args:
            num_time_points: Number of time points (kept for compatibility, but model handles any T)
            num_classes: Number of classification classes (default: 2)
            resnet_type: Type of ResNet backbone
            pretrained: Whether to use pretrained ResNet weights
            learned_features: Number of learned features from frontend (default: 155)
            prior_channels: Number of prior perfusion maps (default: 5)
        """
        super(CTPClassificationNet, self).__init__()

        self.num_time_points = num_time_points  # For compatibility, actual T is flexible

        # Frontend encoder for raw CTP data (3D convolutions)
        # Now handles arbitrary time points via adaptive pooling
        self.frontend_encoder = FrontendEncoder(
            out_channels=learned_features
        )

        # Prior fusion module for perfusion maps
        self.prior_fusion = PriorFusionModule(
            in_channels=prior_channels,
            out_channels=prior_channels
        )

        # Modified ResNet backbone
        total_channels = learned_features + prior_channels  # 155 + 5 = 160
        self.backbone = ModifiedResNet(
            in_channels=total_channels,
            num_classes=num_classes,
            resnet_type=resnet_type,
            pretrained=pretrained
        )

    def forward(
        self,
        ctp_data: torch.Tensor,
        prior_maps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            ctp_data: Raw CTP 4D data of shape (B, 512, 512, 32, T)
                     where T can be any positive integer (20, 21, 22, ...)
            prior_maps: Perfusion prior maps of shape (B, 5, 512, 512)
                       Channels: [CBF, CBV, MTT, Tmax, TTP]

        Returns:
            Classification logits of shape (B, num_classes)
        """
        # Extract learned features from raw CTP data
        learned_features = self.frontend_encoder(ctp_data)  # (B, 155, 512, 512)

        # Adapt prior perfusion maps
        prior_features = self.prior_fusion(prior_maps)  # (B, 5, 512, 512)

        # Concatenate learned and prior features
        combined_features = torch.cat([learned_features, prior_features], dim=1)  # (B, 160, 512, 512)

        # Pass through modified ResNet for classification
        output = self.backbone(combined_features)  # (B, num_classes)

        return output

    def get_feature_maps(
        self,
        ctp_data: torch.Tensor,
        prior_maps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get intermediate feature maps for visualization/analysis.

        Args:
            ctp_data: Raw CTP 4D data of shape (B, 512, 512, 32, T)
            prior_maps: Perfusion prior maps of shape (B, 5, 512, 512)

        Returns:
            Tuple of (learned_features, prior_features, combined_features)
        """
        learned_features = self.frontend_encoder(ctp_data)
        prior_features = self.prior_fusion(prior_maps)
        combined_features = torch.cat([learned_features, prior_features], dim=1)

        return learned_features, prior_features, combined_features


def create_ctp_classifier(
    num_time_points: int = 21,
    num_classes: int = 2,
    resnet_type: str = 'resnet50',
    pretrained: bool = False
) -> CTPClassificationNet:
    """
    Factory function to create a CTP classification network.

    Args:
        num_time_points: Number of time points in CTP sequence (20 or 21)
        num_classes: Number of classification classes
        resnet_type: Type of ResNet ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        pretrained: Whether to use pretrained ResNet weights

    Returns:
        CTPClassificationNet model

    Example:
        >>> model = create_ctp_classifier(num_time_points=21, num_classes=2)
        >>> ctp = torch.randn(2, 512, 512, 32, 21)
        >>> priors = torch.randn(2, 5, 512, 512)
        >>> output = model(ctp, priors)
        >>> print(output.shape)  # (2, 2)
    """
    return CTPClassificationNet(
        num_time_points=num_time_points,
        num_classes=num_classes,
        resnet_type=resnet_type,
        pretrained=pretrained
    )


if __name__ == '__main__':
    # Test the model
    print("Testing CTP Classification Network...")

    # Create model
    model = create_ctp_classifier(num_time_points=21, num_classes=2, resnet_type='resnet50')

    # Create dummy input
    batch_size = 2
    ctp_data = torch.randn(batch_size, 512, 512, 32, 21)  # Raw CTP data
    prior_maps = torch.randn(batch_size, 5, 512, 512)  # Prior perfusion maps

    # Forward pass
    output = model(ctp_data, prior_maps)

    print(f"Input CTP shape: {ctp_data.shape}")
    print(f"Input prior maps shape: {prior_maps.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output}")

    # Test feature extraction
    learned, priors, combined = model.get_feature_maps(ctp_data, prior_maps)
    print(f"\nFeature map shapes:")
    print(f"Learned features: {learned.shape}")
    print(f"Prior features: {priors.shape}")
    print(f"Combined features: {combined.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
