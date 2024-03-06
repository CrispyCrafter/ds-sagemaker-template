import torch
import torch.nn as nn

from src.base import BaseNet


class {{cookiecutter.project_slug}}(BaseNet):
    def __init__(self, config, embeddings):
        super().__init__(config, embeddings)

    def forward(self, x, dx, categories):
        categorical_embeddings = torch.cat(
            list(self.embeddings(categories).values()), dim=-1
        )
       


