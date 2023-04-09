import torch
import torch.nn as nn
import numpy as np
from unittest import TestCase


class TestLayer(TestCase):
    def test_embedding(self):
        torch.manual_seed(1)
        word_to_ix = {"hello": 0, "world": 1}
        embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
        lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
        hello_embed = embeds(lookup_tensor)
        print(hello_embed)

    def test_batchnorm1d(self):
        # m = nn.BatchNorm1d(100)  # With Learnable Parameters
        m = nn.BatchNorm1d(100, affine=False)  # Without Learnable Parameters
        input = torch.randn(20, 100)
        output = m(input)
        input_np = np.array(input)
        output_np = np.array(output)
        print(output)
