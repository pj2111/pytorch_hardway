import torch
from t5_pytorch import T5

model = T5(
    dim=768,
    enc_num_tokens=512,
    enc_depth=6,
    enc_heads=12,
    enc_dim_head=64,
    enc_mlp_mult=4,
    dec_num_tokens=512,
    dec_depth=6,
    dec_heads=12,
    dec_dim_head=64,
    dec_mlp_mult=4,
    dropout=0.,
    tie_token_emb=True
)

src = torch.randint(0, 512, (1, 1024))
src_mask = torch.ones_like(src).bool()
tgt = torch.randint(0, 512, (1, 1024))

output = model(src, tgt, mask=src_mask)
print(output.shape)