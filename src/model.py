import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


class SentimentModel(nn.Module):
    def __init__(self, pretrained_dir, num_labels=5, dann=False, load_pretrained=True):
        super().__init__()
        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(pretrained_dir, local_files_only=True,
                use_safetensors=True, add_pooling_layer=False, attn_implementation="sdpa")
        else:
            cfg = AutoConfig.from_pretrained(pretrained_dir, local_files_only=True)
            cfg._attn_implementation = "sdpa"
            self.encoder = AutoModel.from_config(cfg, add_pooling_layer=False)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self.domain_classifier = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 2)) if dann else None

    def forward(self, strength=0.0, **inputs):
        features = self.encoder(**inputs).last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(features))
        domains = self.domain_classifier(ReverseGradient.apply(features, strength)) if self.domain_classifier is not None else None
        return logits, domains
