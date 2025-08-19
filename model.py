
import torch
import torchvision
from torch import nn

def create_vit_model(num_classes: int = 101,
                     seed: int = 42,
                     fine_tune_layers: int = 0) -> tuple[torch.nn.Module, torchvision.transforms.Compose]:
    """Creates a ViT-B/16 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of target classes. Defaults to 101.
        seed (int, optional): random seed value for output layer. Defaults to 42.
        fine_tune_layers (int, optional): number of layers to unfreeze for fine-tuning.
                                          Defaults to 0 (only classifier head is trained).

    Returns:
        model (torch.nn.Module): ViT-B/16 feature extractor model.
        transforms (torchvision.transforms): ViT-B/16 image transforms.
    """
    # Create ViT_B_16 pretrained weights, transforms and model
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)

    # Freeze all layers in model initially
    for param in model.parameters():
        param.requires_grad = False

    # # Unfreeze the specified number of layers from the end of the feature extractor
    # if fine_tune_layers > 0:
    #     # Get the list of layers in the encoder
    #     encoder_layers = list(model.encoder.children())
    #     # Determine how many layers to unfreeze from the end
    #     num_encoder_layers = len(encoder_layers)
    #     layers_to_unfreeze = encoder_layers[num_encoder_layers - fine_tune_layers:]

    #     for layer in layers_to_unfreeze:
    #         for param in layer.parameters():
    #             param.requires_grad = True

    # Change classifier head to suit our needs (this will always be trainable)
    torch.manual_seed(seed)
    model.heads = nn.Sequential(nn.Linear(in_features=768, # keep this the same as original model
                                          out_features=num_classes)) # update to reflect target number of classes

    return model, transforms
