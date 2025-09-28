"""
Unit tests for semi-supervised learning components.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent))

from src.cyberpuppy.semi_supervised import (ConsistencyConfig,
                                            ConsistencyRegularizer,
                                            PseudoLabelConfig,
                                            PseudoLabelingPipeline,
                                            SelfTrainingConfig,
                                            SelfTrainingFramework)


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(10, num_classes)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Simple forward pass for testing
        batch_size = input_ids.size(0)
        features = torch.randn(batch_size, 10)
        logits = self.linear(features)
        return type("obj", (object,), {"logits": logits})()


class MockDataset(torch.utils.data.Dataset):
    """Mock dataset for testing."""

    def __init__(self, size=10, seq_len=20, num_classes=3):
        self.size = size
        self.seq_len = seq_len
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(1, 1000, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len),
            "labels": torch.randint(0, self.num_classes, (1,)).item(),
            "text": f"Sample text {idx}",
        }


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    tokenizer = Mock()
    tokenizer.decode.return_value = "mock decoded text"
    tokenizer.return_value = {
        "input_ids": torch.randint(1, 1000, (1, 20)),
        "attention_mask": torch.ones(1, 20),
    }
    return tokenizer


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestPseudoLabelingPipeline:
    """Test pseudo-labeling pipeline."""

    def test_config_creation(self):
        """Test configuration creation."""
        config = PseudoLabelConfig(confidence_threshold=0.9, max_pseudo_samples=1000)
        assert config.confidence_threshold == 0.9
        assert config.max_pseudo_samples == 1000

    def test_pipeline_initialization(self, mock_tokenizer, device):
        """Test pipeline initialization."""
        model = MockModel()
        config = PseudoLabelConfig()
        pipeline = PseudoLabelingPipeline(model, mock_tokenizer, config, device)

        assert pipeline.model == model
        assert pipeline.tokenizer == mock_tokenizer
        assert pipeline.device == device
        assert pipeline.current_threshold == config.confidence_threshold

    def test_predict_with_confidence(self, mock_tokenizer, device):
        """Test prediction with confidence scores."""
        model = MockModel().to(device)
        config = PseudoLabelConfig()
        pipeline = PseudoLabelingPipeline(model, mock_tokenizer, config, device)

        dataset = MockDataset(size=5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        predictions, confidences, logits = pipeline.predict_with_confidence(dataloader)

        assert len(predictions) == 5
        assert len(confidences) == 5
        assert len(logits) == 5
        assert all(0 <= conf <= 1 for conf in confidences)

    def test_select_pseudo_labels(self, mock_tokenizer, device):
        """Test pseudo-label selection."""
        model = MockModel()
        config = PseudoLabelConfig(confidence_threshold=0.8, max_pseudo_samples=3)
        pipeline = PseudoLabelingPipeline(model, mock_tokenizer, config, device)

        texts = ["text1", "text2", "text3", "text4", "text5"]
        predictions = np.array([0, 1, 2, 0, 1])
        confidences = np.array([0.9, 0.7, 0.95, 0.6, 0.85])
        logits = np.random.randn(5, 3)

        selected_texts, selected_labels, selected_confidences = pipeline.select_pseudo_labels(
            texts, predictions, confidences, logits
        )

        # Should select samples with confidence >= 0.8, limited to max_pseudo_samples
        assert len(selected_texts) <= 3
        assert all(conf >= 0.8 for conf in selected_confidences)

    def test_threshold_update(self, mock_tokenizer, device):
        """Test threshold update mechanism."""
        model = MockModel()
        config = PseudoLabelConfig(
            confidence_threshold=0.9, min_confidence_threshold=0.7, max_confidence_threshold=0.95
        )
        pipeline = PseudoLabelingPipeline(model, mock_tokenizer, config, device)

        initial_threshold = pipeline.current_threshold

        # Test threshold increase with good performance
        pipeline.update_threshold(0.95)
        assert pipeline.current_threshold >= initial_threshold

        # Test threshold decrease with poor performance
        pipeline.update_threshold(0.60)
        assert pipeline.current_threshold <= initial_threshold

    def test_early_stopping(self, mock_tokenizer, device):
        """Test early stopping mechanism."""
        model = MockModel()
        config = PseudoLabelConfig(early_stopping_patience=2)
        pipeline = PseudoLabelingPipeline(model, mock_tokenizer, config, device)

        # Simulate poor performance
        for _ in range(3):
            pipeline.update_threshold(0.5)

        assert pipeline.should_stop_early()


class TestSelfTrainingFramework:
    """Test self-training framework."""

    def test_config_creation(self):
        """Test configuration creation."""
        config = SelfTrainingConfig(distillation_temperature=4.0, ema_decay=0.999)
        assert config.distillation_temperature == 4.0
        assert config.ema_decay == 0.999

    def test_teacher_model_creation(self, device):
        """Test teacher model creation."""
        config = SelfTrainingConfig()
        trainer = SelfTrainingFramework(config, device).trainer

        student_model = MockModel().to(device)
        teacher_model = trainer.create_teacher_model(student_model)

        # Teacher should be a copy with frozen parameters
        assert sum(p.numel() for p in teacher_model.parameters()) == sum(
            p.numel() for p in student_model.parameters()
        )
        assert not any(p.requires_grad for p in teacher_model.parameters())

    def test_knowledge_distillation_loss(self, device):
        """Test knowledge distillation loss calculation."""
        config = SelfTrainingConfig(distillation_temperature=3.0)
        trainer = SelfTrainingFramework(config, device).trainer

        student_logits = torch.randn(4, 3).to(device)
        teacher_logits = torch.randn(4, 3).to(device)

        loss = trainer.knowledge_distillation_loss(student_logits, teacher_logits)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_consistency_loss(self, device):
        """Test consistency loss calculation."""
        config = SelfTrainingConfig()
        trainer = SelfTrainingFramework(config, device).trainer

        logits1 = torch.randn(4, 3).to(device)
        logits2 = torch.randn(4, 3).to(device)

        loss = trainer.consistency_loss(logits1, logits2)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_confidence_mask(self, device):
        """Test confidence mask computation."""
        config = SelfTrainingConfig(confidence_threshold=0.8)
        trainer = SelfTrainingFramework(config, device).trainer

        # Create logits with known max probabilities
        logits = torch.tensor(
            [
                [2.0, 0.0, 0.0],  # High confidence
                [0.1, 0.1, 0.1],  # Low confidence
                [3.0, 0.0, 0.0],  # High confidence
            ]
        ).to(device)

        mask = trainer.compute_confidence_mask(logits)

        assert mask.dtype == torch.bool
        assert len(mask) == 3
        # First and third samples should have high confidence
        assert mask[0]
        assert mask[2]

    def test_teacher_model_update(self, device):
        """Test EMA update of teacher model."""
        config = SelfTrainingConfig(ema_decay=0.9)
        trainer = SelfTrainingFramework(config, device).trainer

        student_model = MockModel().to(device)
        teacher_model = trainer.create_teacher_model(student_model)

        # Get original teacher parameters
        original_params = [p.clone() for p in teacher_model.parameters()]

        # Modify student model
        with torch.no_grad():
            for p in student_model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Update teacher
        trainer.update_teacher_model(teacher_model, student_model)

        # Check that teacher parameters changed
        for orig, updated in zip(original_params, teacher_model.parameters()):
            assert not torch.equal(orig, updated)


class TestConsistencyRegularizer:
    """Test consistency regularization."""

    def test_config_creation(self):
        """Test configuration creation."""
        config = ConsistencyConfig(consistency_weight=2.0, augmentation_strength=0.2)
        assert config.consistency_weight == 2.0
        assert config.augmentation_strength == 0.2

    def test_text_augmentation(self, device):
        """Test text augmentation methods."""
        config = ConsistencyConfig(augmentation_strength=0.1)
        regularizer = ConsistencyRegularizer(config, device)

        batch = {
            "input_ids": torch.randint(1, 1000, (2, 20)),
            "attention_mask": torch.ones(2, 20),
            "labels": torch.tensor([0, 1]),
        }

        augmented_batch = regularizer.augmenter.augment_batch(batch)

        assert "input_ids" in augmented_batch
        assert "attention_mask" in augmented_batch
        assert "labels" in augmented_batch
        assert augmented_batch["input_ids"].shape == batch["input_ids"].shape

    def test_token_dropout(self, device):
        """Test token dropout augmentation."""
        config = ConsistencyConfig()
        regularizer = ConsistencyRegularizer(config, device)

        input_ids = torch.randint(10, 1000, (2, 20))
        attention_mask = torch.ones(2, 20)

        aug_input_ids, aug_attention_mask = regularizer.augmenter.token_dropout(
            input_ids, attention_mask, dropout_prob=0.2
        )

        assert aug_input_ids.shape == input_ids.shape
        assert aug_attention_mask.shape == attention_mask.shape
        # Some tokens should have been replaced with mask token (103)
        assert (aug_input_ids == 103).sum() > 0

    def test_consistency_loss_calculation(self, device):
        """Test consistency loss calculation."""
        config = ConsistencyConfig()
        regularizer = ConsistencyRegularizer(config, device)

        original_logits = torch.randn(4, 3).to(device)
        augmented_logits = torch.randn(4, 3).to(device)

        # Test MSE loss
        mse_loss = regularizer.compute_consistency_loss(
            original_logits, augmented_logits, loss_type="mse"
        )
        assert isinstance(mse_loss, torch.Tensor)
        assert mse_loss.item() >= 0

        # Test KL loss
        kl_loss = regularizer.compute_consistency_loss(
            original_logits, augmented_logits, loss_type="kl"
        )
        assert isinstance(kl_loss, torch.Tensor)
        assert kl_loss.item() >= 0

    def test_consistency_weight_rampup(self, device):
        """Test consistency weight ramp-up schedule."""
        config = ConsistencyConfig(consistency_ramp_up_epochs=5, max_consistency_weight=10.0)
        regularizer = ConsistencyRegularizer(config, device)

        # Test ramp-up
        assert regularizer.get_consistency_weight(0) == 0.0
        assert regularizer.get_consistency_weight(2) == 4.0
        assert regularizer.get_consistency_weight(5) == 10.0
        assert regularizer.get_consistency_weight(10) == 10.0

    def test_confidence_masking(self, device):
        """Test confidence-based masking."""
        config = ConsistencyConfig(use_confidence_masking=True, confidence_threshold=0.8)
        regularizer = ConsistencyRegularizer(config, device)

        # Create logits with known confidence levels
        logits = torch.tensor(
            [
                [3.0, 0.0, 0.0],  # High confidence (~0.95)
                [1.0, 1.0, 1.0],  # Low confidence (~0.33)
            ]
        ).to(device)

        confidence_mask = regularizer.compute_confidence_mask(logits)
        assert confidence_mask[0]  # High confidence
        assert not confidence_mask[1]  # Low confidence


def test_integration():
    """Test integration between components."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create components
    model = MockModel().to(device)
    tokenizer = Mock()
    tokenizer.decode.return_value = "mock text"

    pseudo_config = PseudoLabelConfig()
    self_training_config = SelfTrainingConfig()
    consistency_config = ConsistencyConfig()

    pseudo_pipeline = PseudoLabelingPipeline(model, tokenizer, pseudo_config, device)
    self_training = SelfTrainingFramework(self_training_config, device)
    consistency = ConsistencyRegularizer(consistency_config, device)

    # Test that all components can be created without errors
    assert pseudo_pipeline is not None
    assert self_training is not None
    assert consistency is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
