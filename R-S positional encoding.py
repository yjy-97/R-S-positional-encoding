import torch.nn as nn
import torch
import math


class Positional_Encoding(nn.Module):
    def __init__(self, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)

    def relative_position_encoding(self, depth, max_length, pad_size=64, max_relative_position=4):
        vocab_size = max_relative_position * 2 + 1
        range_vec = torch.arange(max_length)
        range_vec_n, _ = range_vec.sort(descending=True)
        range_mat = range_vec.repeat(pad_size).view(max_length, pad_size)
        range_mat_n = range_vec_n.repeat(pad_size).view(max_length, pad_size)
        distance_mat = range_mat - range_mat_n

        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position

        embeddings_table = torch.zeros(vocab_size, depth)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))
        embeddings_table[:, 0::2] = torch.sin(position * div_term)
        embeddings_table[:, 1::2] = torch.cos(position * div_term)
        embeddings_table = embeddings_table.unsqueeze(0).transpose(0, 1).squeeze(1)

        flat_relative_positions_matrix = final_mat.view(-1)
        one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                        num_classes=vocab_size).float()
        positions_encoding = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
        my_shape = list(final_mat.size())
        my_shape.append(depth)
        positions_encoding = positions_encoding.view(my_shape)
        return positions_encoding

    def forward(self, x):
        out = x + nn.Parameter(self.relative_position_encoding(100, len(x)), requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


if __name__ == '__main__':
    test_data = torch.rand((32, 64, 100))
    print(test_data)
    dropout, device = 0.5, 'cpu'
    model = Positional_Encoding(dropout, device).forward(test_data)
    print(model.shape)
