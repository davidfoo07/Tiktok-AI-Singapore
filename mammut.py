import torch

# import vision transformer

from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor

vit = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    patch_dropout = 0.5  # https://arxiv.org/abs/2212.00794
)

vit = Extractor(vit, return_embeddings_only = True, detach = False)

# extractor will enable it so the vision transformer returns its embeddings

# import MaMMUT and instantiate it

from mammut_pytorch.mammut_pytorch import MaMMUT

mammut = MaMMUT(
    dim = 512,                     # model dimension
    img_encoder = vit,             # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
    image_dim = 1024,              # image embedding dimension, if not the same as model dimensions
    num_tokens = 20000,            # number of text tokens
    depth = 6,                     # depth of the transformer
    dim_head = 64,                 # dimension per attention head
    heads = 8,                     # number of attention heads
    caption_loss_weight = 1.,      # weight on the autoregressive caption loss
    contrastive_loss_weight = 1.,  # weight on the contrastive loss between image and text CLS embeddings
).cuda()

# mock text and images

text = torch.randint(0, 20000, (4, 512)).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

# train by giving MaMMUT your text and images with `return_loss = True`

loss = mammut(
    text = text,
    images = images,
    return_loss = True  # set this to True to get the full caption + contrastive loss
)

loss.backward()

# do the above for as much text and images...
# then you can get the caption logits as so

logits = mammut(
    text = text,
    images = images
) # (4, 512, 20000)

# and the CLIP-like text and image embeddings as

text_embeds, image_embeds = mammut(
    text = text,
    images = images,
    return_embeddings = True
) # (4, 512), (4, 512)