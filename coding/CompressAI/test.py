import pdb

import torch

def pack_tokens_to_square(feat_257_1024: torch.Tensor) -> torch.Tensor:
    """
    入:  [257, 1024]（第0行是CLS，后256行为patch token）
    出:  [1, 1, 512, 512]  作为 CompressAI 输入
    """
    assert feat_257_1024.shape == (257, 1024)
    tokens = feat_257_1024[1:, :]            # [256, 1024] 丢CLS
    # 16×16 token 网格，1024=32×32
    x = tokens.view(16, 16, 32, 32)          # [H=16, W=16, Ch1=32, Ch2=32]
    # 把通道两个因子铺到空间维： (H, Ch1, W, Ch2) -> (H*Ch1, W*Ch2)
    x = x.permute(0, 2, 1, 3).contiguous()   # [16,32,16,32]
    x = x.view(16*32, 16*32)                 # [512, 512]
    x = x.unsqueeze(0).unsqueeze(0)          # [1, 1, 512, 512]  (B=1,C=1,H=512,W=512)
    return x

def unpack_square_to_tokens(x_1_1_512_512: torch.Tensor) -> torch.Tensor:
    """
    入:  [1, 1, 512, 512]（编解码后的重建）
    出:  [256, 1024]（不含CLS）
    """
    assert x_1_1_512_512.shape[-2:] == (512, 512)
    x = x_1_1_512_512.squeeze(0).squeeze(0)  # [512, 512]
    x = x.view(16, 32, 16, 32)               # 逆向拆成 [H,Ch1,W,Ch2]
    x = x.permute(0, 2, 1, 3).contiguous()   # [16,16,32,32]
    x = x.view(256, 1024)                    # [256,1024]
    return x
feat = torch.arange(257*1024, dtype=torch.float32).view(257,1024)
y = pack_tokens_to_square(feat)
rec = unpack_square_to_tokens(y)
import pdb
pdb.set_trace()
print(torch.testing.assert_close(rec, feat[1:]))  # True
