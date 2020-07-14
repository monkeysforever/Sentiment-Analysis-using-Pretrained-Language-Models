import torch
from transformers import BertModel


class PretrainedClassifier(torch.nn.Module):
    def __init__(self, language_model=BertModel.from_pretrained('bert-base-uncased'),
                 embedding_size=768, num_classes=2):
        super(PretrainedClassifier, self).__init__()
        self.lm = language_model
        self.classifier = torch.nn.Linear(embedding_size, num_classes)

    def forward(self, sequences, attn_masks):
        x, _ = self.lm(sequences, attention_mask=attn_masks)
        x = x[:, 0]
        logits = self.classifier(x)
        return logits

    def freeze_lm(self):
        for p in self.lm.parameters():
            p.requires_grad = False

    def unfreeze_lm(self):
        for p in self.lm.parameters():
            p.requires_grad = True
