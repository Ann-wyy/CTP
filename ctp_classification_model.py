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
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from typing import Optional, Tuple


class FrontendEncoder(nn.Module):
    """
    Frontend encoder that processes raw CTP 4D data.

    Converts (B, 512, 512, 32, T) -> (B, 32*T, 512, 512) -> (B, 155, 512, 512)
    using a small CNN with 1x1 Conv + BN + ReLU + 3x3 Conv.
    """

    def __init__(self, in_channels: int = 672, out_channels: int = 155):
        """
        Args:
            in_channels: Number of input channels (32*21=672 or 32*20=640)
            out_channels: Number of output learned features (default: 155)
        """
        super(FrontendEncoder, self).__init__()

        # Small frontend CNN
        self.encoder = nn.Sequential(
            # 1x1 Conv to reduce dimensions
            nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 3x3 Conv for spatial feature extraction
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input CTP data of shape (B, 512, 512, 32, T) where T is 20 or 21

        Returns:
            Learned features of shape (B, 155, 512, 512)
        """
        B, H, W, Z, T = x.shape

        # Reshape: (B, 512, 512, 32, T) -> (B, 32*T, 512, 512)
        x = x.permute(0, 3, 4, 1, 2)  # (B, 32, T, 512, 512)
        x = x.reshape(B, Z * T, H, W)  # (B, 32*T, 512, 512)

        # Apply frontend encoder
        x = self.encoder(x)  # (B, 155, 512, 512)

        return x


class PriorFusionModule(nn.Module):
    """
    Prior fusion module that adapts perfusion prior maps.

    Processes 5 perfusion maps (CBF, CBV, MTT, Tmax, TTP) using 1x1 Conv + BN.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 5):
        """
        Args:
            in_channels: Number of prior maps (default: 5 for CBF, CBV, MTT, Tmax, TTP)
            out_channels: Number of output channels (default: 5)
        """
        super(PriorFusionModule, self).__init__()

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Prior maps of shape (B, 5, 512, 512)

        Returns:
            Adapted prior features of shape (B, 5, 512, 512)
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
        resnet_models = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152
        }

        if resnet_type not in resnet_models:
            raise ValueError(f"resnet_type must be one of {list(resnet_models.keys())}")

        base_model = resnet_models[resnet_type](pretrained=pretrained)

        # Modify conv1 to accept custom input channels
        original_conv1 = base_model.conv1
        self.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

        # If pretrained, initialize new conv1 weights intelligently
        if pretrained:
            with torch.no_grad():
                # Repeat the original weights across new channels
                original_weight = original_conv1.weight.data
                repeat_times = in_channels // 3
                remainder = in_channels % 3

                # Repeat the 3-channel weights
                if repeat_times > 0:
                    repeated_weight = original_weight.repeat(1, repeat_times, 1, 1)
                    if remainder > 0:
                        extra_weight = original_weight[:, :remainder, :, :]
                        self.conv1.weight.data = torch.cat([repeated_weight, extra_weight], dim=1)
                    else:
                        self.conv1.weight.data = repeated_weight
                else:
                    self.conv1.weight.data = original_weight[:, :in_channels, :, :]

                # Normalize to maintain magnitude
                self.conv1.weight.data /= (in_channels / 3)

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
        1. Raw CTP (B, 512, 512, 32, T) -> Frontend Encoder -> (B, 155, 512, 512)
        2. Prior maps (B, 5, 512, 512) -> Prior Fusion -> (B, 5, 512, 512)
        3. Concatenate -> (B, 160, 512, 512)
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
            num_time_points: Number of time points in CTP sequence (20 or 21)
            num_classes: Number of classification classes (default: 2)
            resnet_type: Type of ResNet backbone
            pretrained: Whether to use pretrained ResNet weights
            learned_features: Number of learned features from frontend (default: 155)
            prior_channels: Number of prior perfusion maps (default: 5)
        """
        super(CTPClassificationNet, self).__init__()

        self.num_time_points = num_time_points
        in_channels_frontend = 32 * num_time_points  # 672 for T=21, 640 for T=20

        # Frontend encoder for raw CTP data
        self.frontend_encoder = FrontendEncoder(
            in_channels=in_channels_frontend,
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
                     where T is num_time_points (20 or 21)
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
