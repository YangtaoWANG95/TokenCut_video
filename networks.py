import torch
import torch.nn as nn
import dino.vision_transformer as vits
from torchvision.models.resnet import resnet50

def get_model(arch, patch_size, device):

    # Initialize model with pretraining
    url = None
    if "moco" in arch:
        if arch == "moco_vit_small" and patch_size == 16:
            url = "moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
        elif arch == "moco_vit_base" and patch_size == 16:
            url = "moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
        model = vits.__dict__[arch](num_classes=0)
    elif "mae" in arch:
        if arch == "mae_vit_base" and patch_size == 16:
            url = "mae/visualize/mae_visualize_vit_base.pth"
        model = vits.__dict__[arch](num_classes=0)
    elif "vit" in arch:
        if arch == "vit_small" and patch_size == 16:
            url = "/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif arch == "vit_base" and patch_size == 16:
            url = "/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif arch == "resnet50":
            url = "/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    else:
        raise NotImplementedError 

    for p in model.parameters():
        p.requires_grad = False

    if url is not None:
        print(
            "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
        )
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/" + url
        )
        if "moco" in arch:
            state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif "mae" in arch:
            state_dict = state_dict['model']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('decoder') or k.startswith('mask_token'):
                    # remove prefix
                    #state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=True)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                url, msg
            )
        )
    else:
        print(
            "There is no reference weights available for this model => We use random weights."
        )


    model.eval()
    model.to(device)
    return model


def feature_extractor(arch, model, img_tensor):
    with torch.no_grad():
        # ------------ FORWARD PASS -------------------------------------------
        if "vit"  in arch:
            # Store the outputs of qkv layer from the last attention layer
            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

            # Forward pass in the model
            attentions = model.get_last_selfattention(img_tensor[None, :, :, :])

            # Dimensions
            nh = attentions.shape[1]  # Number of heads
            nb_tokens = attentions.shape[2]  # Number of tokens

            # Extract the k features of the last attention layer
            qkv = (
                feat_out["qkv"]
                .reshape(1, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(1, nb_tokens, -1)

            return k[0, 1:].cpu().numpy() ## get rid of cls token features
        else:
            raise ValueError("Unknown model.")

