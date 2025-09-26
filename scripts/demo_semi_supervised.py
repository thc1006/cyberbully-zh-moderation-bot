#!/usr/bin/env python3
"""
Demo script for Semi-supervised Learning Framework.

This script demonstrates how to use the pseudo-labeling, self-training,
and consistency regularization components for Chinese cyberbullying detection.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.cyberpuppy.semi_supervised import (
    PseudoLabelingPipeline, PseudoLabelConfig,
    SelfTrainingFramework, SelfTrainingConfig,
    ConsistencyRegularizer, ConsistencyConfig
)


class SimpleClassifier(nn.Module):
    """Simple classifier for demonstration."""

    def __init__(self, model_name: str, num_classes: int = 3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return type('obj', (object,), {'logits': logits})()


class DemoDataset(torch.utils.data.Dataset):
    """Demo dataset with Chinese text samples."""

    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'text': text
            }
        else:
            # Dummy encoding for demonstration
            item = {
                'input_ids': torch.randint(1, 1000, (self.max_length,)),
                'attention_mask': torch.ones(self.max_length),
                'text': text
            }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def create_demo_data():
    """Create demonstration data with Chinese text."""

    # Sample Chinese texts (cyberbullying related)
    labeled_texts = [
        "你真的很聰明，我很佩服你的想法。",  # positive
        "這個想法不錯，但還有改進空間。",      # neutral
        "你就是個廢物，什麼都不會。",        # toxic
        "今天天氣真好，適合出去走走。",      # positive
        "我覺得這個方案可以考慮。",          # neutral
        "你這麼蠢，怎麼可能成功？",          # toxic
        "感謝你的幫助，非常感激。",          # positive
        "報告需要再修改一下。",              # neutral
        "你滾出去，不要再來煩我！",          # toxic
        "期待與你的合作。",                  # positive
    ]

    labeled_labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # 0: positive, 1: neutral, 2: toxic

    unlabeled_texts = [
        "今天的會議很有意義。",
        "需要更多時間來完成這個項目。",
        "你的表現讓人失望。",
        "這是一個很好的學習機會。",
        "請注意截止日期。",
        "我不同意你的觀點。",
        "感謝大家的參與。",
        "這個問題很複雜。",
        "你應該更加努力。",
        "期待下次的討論。",
        "你的能力有限。",
        "讓我們一起解決這個問題。",
        "你不適合這個工作。",
        "這個建議很有價值。",
        "我需要更多的信息。",
    ]

    val_texts = [
        "你的努力值得讚賞。",              # positive
        "這個計劃需要調整。",              # neutral
        "你真的很無能。",                  # toxic
    ]
    val_labels = [0, 1, 2]

    return labeled_texts, labeled_labels, unlabeled_texts, val_texts, val_labels


def demo_pseudo_labeling():
    """Demonstrate pseudo-labeling approach."""
    print("=== Pseudo-labeling Demo ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'hfl/chinese-macbert-base'

    # Create model and tokenizer
    model = SimpleClassifier(model_name, num_classes=3).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create demo data
    labeled_texts, labeled_labels, unlabeled_texts, val_texts, val_labels = create_demo_data()

    # Create datasets
    labeled_dataset = DemoDataset(labeled_texts, labeled_labels, tokenizer)
    unlabeled_dataset = DemoDataset(unlabeled_texts, tokenizer=tokenizer)
    val_dataset = DemoDataset(val_texts, val_labels, tokenizer)

    # Create data loaders
    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=2, shuffle=True)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=2, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Configure pseudo-labeling
    config = PseudoLabelConfig(
        confidence_threshold=0.8,
        max_pseudo_samples=10,
        validation_metric="f1_macro"
    )

    # Create pipeline
    pipeline = PseudoLabelingPipeline(model, tokenizer, config, device)

    # Generate pseudo labels
    pseudo_samples, stats = pipeline.generate_pseudo_labels(model, unlabeled_loader)

    print(f"Generated {len(pseudo_samples)} pseudo-labeled samples")
    print(f"Statistics: {stats}")

    # Show some pseudo-labeled samples
    for i, sample in enumerate(pseudo_samples[:3]):
        print(f"Sample {i+1}:")
        print(f"  Pseudo label: {sample['pseudo_label']}")
        print(f"  Confidence: {sample['confidence']:.3f}")
        print()


def demo_self_training():
    """Demonstrate self-training approach."""
    print("=== Self-training Demo ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'hfl/chinese-macbert-base'

    # Create model
    student_model = SimpleClassifier(model_name, num_classes=3).to(device)

    # Configure self-training
    config = SelfTrainingConfig(
        teacher_update_frequency=5,
        max_epochs=2,
        distillation_temperature=3.0
    )

    # Create framework
    framework = SelfTrainingFramework(config, device)

    # Create teacher model
    trainer = framework.trainer
    teacher_model = trainer.create_teacher_model(student_model)

    print(f"Created teacher model with {sum(p.numel() for p in teacher_model.parameters())} parameters")
    print(f"Teacher model requires_grad: {any(p.requires_grad for p in teacher_model.parameters())}")

    # Demo knowledge distillation loss
    batch_size = 2
    num_classes = 3
    student_logits = torch.randn(batch_size, num_classes).to(device)
    teacher_logits = torch.randn(batch_size, num_classes).to(device)

    kd_loss = trainer.knowledge_distillation_loss(student_logits, teacher_logits)
    print(f"Knowledge distillation loss: {kd_loss.item():.4f}")

    # Demo teacher model update
    original_param = list(teacher_model.parameters())[0].clone()
    trainer.update_teacher_model(teacher_model, student_model)
    updated_param = list(teacher_model.parameters())[0]

    param_change = torch.norm(updated_param - original_param).item()
    print(f"Teacher model parameter change: {param_change:.6f}")


def demo_consistency_regularization():
    """Demonstrate consistency regularization."""
    print("=== Consistency Regularization Demo ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configure consistency training
    config = ConsistencyConfig(
        consistency_weight=1.0,
        augmentation_strength=0.1,
        use_confidence_masking=True
    )

    # Create regularizer
    regularizer = ConsistencyRegularizer(config, device)

    # Demo text augmentation
    sample_batch = {
        'input_ids': torch.randint(1, 1000, (2, 20)),
        'attention_mask': torch.ones(2, 20)
    }

    augmented_batch = regularizer.augmenter.augment_batch(sample_batch)

    print("Original input_ids (first sample):")
    print(sample_batch['input_ids'][0][:10].tolist())
    print("Augmented input_ids (first sample):")
    print(augmented_batch['input_ids'][0][:10].tolist())

    # Demo consistency loss calculation
    original_logits = torch.randn(2, 3).to(device)
    augmented_logits = torch.randn(2, 3).to(device)

    consistency_loss = regularizer.compute_consistency_loss(
        original_logits, augmented_logits, loss_type="mse"
    )
    print(f"Consistency loss (MSE): {consistency_loss.item():.4f}")

    kl_loss = regularizer.compute_consistency_loss(
        original_logits, augmented_logits, loss_type="kl"
    )
    print(f"Consistency loss (KL): {kl_loss.item():.4f}")

    # Demo confidence masking
    confidence_mask = regularizer.compute_confidence_mask(original_logits)
    print(f"Confidence mask: {confidence_mask.tolist()}")
    print(f"High confidence samples: {confidence_mask.sum().item()}/{len(confidence_mask)}")


def main():
    """Run all demonstrations."""
    print("Semi-supervised Learning Framework Demo")
    print("=" * 50)

    try:
        demo_pseudo_labeling()
        print()

        demo_self_training()
        print()

        demo_consistency_regularization()
        print()

        print("All demos completed successfully!")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()