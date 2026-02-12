import sys
import argparse
import torch
import torchvision
from torchvision import transforms
from transformers import ViTForImageClassification
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, '/home/chandan/test_current')
from train_vit_rotational import TrainingConfig
from rotational_pissa_unified import RotationalPiSSAConfig, replace_linear_with_rotational_pissa
import __main__
__main__.TrainingConfig = TrainingConfig


def load_model_from_checkpoint(checkpoint_path, device="cuda"):
    """Load a trained SOARA model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    state_dict = checkpoint['model_state_dict']
    config = checkpoint.get('config', None)
    
    # Create base model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=100,
        ignore_mismatched_sizes=True
    )
    
    # Check for SOARA adapters
    has_soara = any(
        k.endswith('.R_U') or k.endswith('.R_V') or
        k.endswith('.U') or k.endswith('.V') or k.endswith('.S') or
        'butterfly_u' in k or 'butterfly_v' in k or
        'givens_layers' in k or 'current_butterfly' in k
        for k in state_dict.keys()
    )
    
    if has_soara and config:
        print("  Reconstructing SOARA model...")
        method = getattr(config, 'method', 'way0')
        rank = getattr(config, 'pissa_rank', None) or getattr(config, 'rank', 16)
        use_butterfly = getattr(config, 'use_butterfly', False)
        butterfly_sequential = getattr(config, 'butterfly_sequential', False)
        target_modules = getattr(config, 'target_modules', ['query', 'key', 'value', 'dense'])
        
        print(f"    Config: method={method}, rank={rank}, butterfly={use_butterfly}")
        
        pissa_config = RotationalPiSSAConfig(
            r=rank,
            use_butterfly=use_butterfly,
            butterfly_sequential=butterfly_sequential,
            orthogonality_reg_weight=0.0,
            method=method,
        )
        
        replace_linear_with_rotational_pissa(
            model=model,
            pissa_config=pissa_config,
            target_modules=target_modules,
            exclude_modules=['classifier'],
            adapter_name="default",
            freeze_base_model=True
        )
    
    model.load_state_dict(state_dict, strict=True)
    print("  Loaded weights successfully")
    
    return model.to(device)


def find_top_classes_by_ratio(model, data_path="./data", n_classes=5, device="cuda"):
    """Find top N classes with highest inter/intra cluster ratio."""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = torchvision.datasets.FGVCAircraft(
        root=data_path, split='test', transform=transform, download=True
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Extract embeddings per class
    class_embeddings = {i: [] for i in range(100)}
    
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model.vit(images, output_hidden_states=True)
            embeds = outputs.hidden_states[6][:, 0, :]  # Layer 6, CLS token
            
            for emb, label in zip(embeds.cpu(), labels):
                class_embeddings[label.item()].append(emb)
    
    # Compute per-class intra and inter-cluster distances
    class_ratios = {}
    all_class_centroids = {}
    
    for c, embeds in class_embeddings.items():
        if len(embeds) < 2:
            continue
        embeds = torch.stack(embeds)
        centroid = embeds.mean(dim=0)
        all_class_centroids[c] = centroid
        
        # Intra-cluster: mean distance to centroid
        intra = torch.norm(embeds - centroid, dim=1).mean().item()
        class_embeddings[c] = {'embeds': embeds, 'centroid': centroid, 'intra': intra}
    
    # Compute inter-cluster distances (mean distance to other centroids)
    centroids = torch.stack([class_embeddings[c]['centroid'] for c in class_embeddings if isinstance(class_embeddings[c], dict)])
    
    for c in class_embeddings:
        if not isinstance(class_embeddings[c], dict):
            continue
        centroid = class_embeddings[c]['centroid']
        # Distance to all other centroids
        inter = torch.norm(centroids - centroid, dim=1).mean().item()
        intra = class_embeddings[c]['intra']
        ratio = inter / (intra + 1e-8)
        class_ratios[c] = {'intra': intra, 'inter': inter, 'ratio': ratio}
    
    # Sort by ratio
    sorted_classes = sorted(class_ratios.items(), key=lambda x: x[1]['ratio'], reverse=True)
    
    print(f"\nTop {n_classes} classes by inter/intra ratio:")
    print("-" * 60)
    top_classes = []
    for c, metrics in sorted_classes[:n_classes]:
        name = dataset.classes[c]
        print(f"  Class {c:3d}: {name:30s} ratio={metrics['ratio']:.3f} (inter={metrics['inter']:.2f}, intra={metrics['intra']:.2f})")
        top_classes.append(c)
    
    return top_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find top classes by cluster ratio")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--n-classes", type=int, default=5, help="Number of top classes")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to data")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    top_classes = find_top_classes_by_ratio(model, args.data_path, args.n_classes, args.device)