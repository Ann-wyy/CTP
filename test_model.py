"""
Unit tests for CTP Classification Network
"""

import torch
import pytest
from ctp_classification_model import (
    FrontendEncoder,
    PriorFusionModule,
    ModifiedResNet,
    CTPClassificationNet,
    create_ctp_classifier
)


def test_frontend_encoder():
    """Test FrontendEncoder with different time points."""
    print("Testing FrontendEncoder...")

    # Test with T=21
    encoder_21 = FrontendEncoder(in_channels=672, out_channels=155)
    x = torch.randn(2, 512, 512, 32, 21)
    out = encoder_21(x)
    assert out.shape == (2, 155, 512, 512), f"Expected shape (2, 155, 512, 512), got {out.shape}"

    # Test with T=20
    encoder_20 = FrontendEncoder(in_channels=640, out_channels=155)
    x = torch.randn(2, 512, 512, 32, 20)
    out = encoder_20(x)
    assert out.shape == (2, 155, 512, 512), f"Expected shape (2, 155, 512, 512), got {out.shape}"

    print("✓ FrontendEncoder tests passed")


def test_prior_fusion_module():
    """Test PriorFusionModule."""
    print("Testing PriorFusionModule...")

    fusion = PriorFusionModule(in_channels=5, out_channels=5)
    x = torch.randn(2, 5, 512, 512)
    out = fusion(x)
    assert out.shape == (2, 5, 512, 512), f"Expected shape (2, 5, 512, 512), got {out.shape}"

    print("✓ PriorFusionModule tests passed")


def test_modified_resnet():
    """Test ModifiedResNet with different configurations."""
    print("Testing ModifiedResNet...")

    resnet_types = ['resnet18', 'resnet34', 'resnet50']

    for resnet_type in resnet_types:
        model = ModifiedResNet(
            in_channels=160,
            num_classes=2,
            resnet_type=resnet_type,
            pretrained=False
        )
        x = torch.randn(2, 160, 512, 512)
        out = model(x)
        assert out.shape == (2, 2), f"Expected shape (2, 2), got {out.shape}"
        print(f"  ✓ {resnet_type} passed")

    print("✓ ModifiedResNet tests passed")


def test_ctp_classification_net():
    """Test complete CTPClassificationNet."""
    print("Testing CTPClassificationNet...")

    # Test with T=21
    model_21 = CTPClassificationNet(num_time_points=21, num_classes=2)
    ctp_data = torch.randn(2, 512, 512, 32, 21)
    prior_maps = torch.randn(2, 5, 512, 512)
    out = model_21(ctp_data, prior_maps)
    assert out.shape == (2, 2), f"Expected shape (2, 2), got {out.shape}"

    # Test with T=20
    model_20 = CTPClassificationNet(num_time_points=20, num_classes=2)
    ctp_data = torch.randn(2, 512, 512, 32, 20)
    out = model_20(ctp_data, prior_maps)
    assert out.shape == (2, 2), f"Expected shape (2, 2), got {out.shape}"

    # Test multi-class
    model_multi = CTPClassificationNet(num_time_points=21, num_classes=5)
    ctp_data = torch.randn(2, 512, 512, 32, 21)
    out = model_multi(ctp_data, prior_maps)
    assert out.shape == (2, 5), f"Expected shape (2, 5), got {out.shape}"

    print("✓ CTPClassificationNet tests passed")


def test_feature_extraction():
    """Test feature extraction method."""
    print("Testing feature extraction...")

    model = CTPClassificationNet(num_time_points=21, num_classes=2)
    ctp_data = torch.randn(1, 512, 512, 32, 21)
    prior_maps = torch.randn(1, 5, 512, 512)

    learned, priors, combined = model.get_feature_maps(ctp_data, prior_maps)

    assert learned.shape == (1, 155, 512, 512), f"Expected learned shape (1, 155, 512, 512), got {learned.shape}"
    assert priors.shape == (1, 5, 512, 512), f"Expected priors shape (1, 5, 512, 512), got {priors.shape}"
    assert combined.shape == (1, 160, 512, 512), f"Expected combined shape (1, 160, 512, 512), got {combined.shape}"

    print("✓ Feature extraction tests passed")


def test_create_ctp_classifier():
    """Test factory function."""
    print("Testing create_ctp_classifier...")

    model = create_ctp_classifier(
        num_time_points=21,
        num_classes=2,
        resnet_type='resnet50',
        pretrained=False
    )

    ctp_data = torch.randn(2, 512, 512, 32, 21)
    prior_maps = torch.randn(2, 5, 512, 512)
    out = model(ctp_data, prior_maps)

    assert out.shape == (2, 2), f"Expected shape (2, 2), got {out.shape}"

    print("✓ create_ctp_classifier tests passed")


def test_gradient_flow():
    """Test that gradients flow properly through the network."""
    print("Testing gradient flow...")

    model = create_ctp_classifier(num_time_points=21, num_classes=2)
    ctp_data = torch.randn(1, 512, 512, 32, 21, requires_grad=True)
    prior_maps = torch.randn(1, 5, 512, 512, requires_grad=True)

    out = model(ctp_data, prior_maps)
    loss = out.sum()
    loss.backward()

    # Check that gradients exist
    assert ctp_data.grad is not None, "No gradient for ctp_data"
    assert prior_maps.grad is not None, "No gradient for prior_maps"

    # Check that model parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"

    print("✓ Gradient flow tests passed")


def test_batch_sizes():
    """Test different batch sizes."""
    print("Testing different batch sizes...")

    model = create_ctp_classifier(num_time_points=21, num_classes=2)
    model.eval()

    batch_sizes = [1, 2, 4, 8]
    for bs in batch_sizes:
        ctp_data = torch.randn(bs, 512, 512, 32, 21)
        prior_maps = torch.randn(bs, 5, 512, 512)

        with torch.no_grad():
            out = model(ctp_data, prior_maps)

        assert out.shape == (bs, 2), f"Expected shape ({bs}, 2), got {out.shape}"

    print("✓ Batch size tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running CTP Classification Network Tests")
    print("=" * 60 + "\n")

    try:
        test_frontend_encoder()
        test_prior_fusion_module()
        test_modified_resnet()
        test_ctp_classification_net()
        test_feature_extraction()
        test_create_ctp_classifier()
        test_gradient_flow()
        test_batch_sizes()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
