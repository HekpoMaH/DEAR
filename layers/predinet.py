import torch
import torch.nn as nn
import torch_geometric


class PrediNet(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_heads,
        key_size,
        relations,
        flatten_pooling=torch_geometric.nn.global_max_pool,
    ):
        super(PrediNet, self).__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.key_size = key_size
        self.get_keys = nn.Linear(latent_dim, key_size, bias=False)
        self.flatten_pooling = flatten_pooling

        self.get_Q = nn.ModuleList()
        for i in range(2):
            self.get_Q.append(nn.Linear(latent_dim, num_heads * key_size, bias=False))

        self.embed_entities = nn.Linear(latent_dim, relations, bias=False)
        self.output = nn.Sequential(
            nn.Linear(num_heads * relations, latent_dim), nn.LeakyReLU()
        )

    def forward(self, inp, batch_ids):
        batch_size = batch_ids.max() + 1
        # inp shape is N_TOTAL x H

        inp_flatten = self.flatten_pooling(inp, batch_ids)  # G x H
        inp, mask = torch_geometric.utils.to_dense_batch(
            inp, batch=batch_ids
        )  # G x N_MAX x H
        inp_tiled = inp.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # G x MHA x N_MAX x H
        mask_tiled = mask.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # G x MHA x N_MAX x H

        keys = self.get_keys(inp)  # G x N_MAX x H
        keys_T = (
            keys.unsqueeze(1).repeat(1, self.num_heads, 1, 1).transpose(2, 3)
        )  # G x MHA x N_MAX x H
        embeddings = []
        for i in range(2):
            q_i = self.get_Q[i](inp_flatten)  # G x H
            q_i1 = q_i.unsqueeze(1).repeat(1, self.num_heads, 1)  # G x MHA x H
            q_i2 = q_i1.unsqueeze(2)  # G x MHA x 1 x H
            qkmul = torch.matmul(q_i2, keys_T)  # G x MHA x MHA x H
            qkmul[~mask[:, None, None, :]] = float(-1e9)
            att_i = torch.softmax(qkmul, dim=-1)  # G X MHA X MHA X H
            feature_i = torch.squeeze(torch.matmul(att_i, inp_tiled), 2)  # G X MHA X H
            emb_i = self.embed_entities(feature_i)  # G X MHA X H
            embeddings.append(emb_i)

        dx = embeddings[0] - embeddings[1]
        dx = dx.reshape(batch_size, -1)  # G X (MHAxH)
        return self.output(dx)  # G X H


if __name__ == "__main__":
    pn = PrediNet(1, 5, 3, 4)
    inp = torch.tensor([[-1], [-2], [1], [2]]).float()
    batch_ids = torch.tensor([0, 0, 1, 1]).long()
    print("OUT", pn(inp, batch_ids).shape)
    pass
