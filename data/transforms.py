from transformers import BertTokenizer
import torch


class Tokenize(object):
    def __init__(self, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased').tokenize):
        self.tokenizer = tokenizer

    def __call__(self, x):
        return self.tokenizer(x)


class InsertTokens(object):
    def __init__(self, init_token='[CLS]', eos_token='[SEP]'):
        self.init_token = init_token
        self.eos_token = eos_token

    def __call__(self, x):
        return [self.init_token] + x + [self.eos_token]


class Pad(object):
    def __init__(self, max_length=512, pad_token='[PAD]', eos_token='[SEP]'):
        self.max_length = max_length
        self.pad_token = pad_token
        self.eos_token = eos_token

    def __call__(self, x):
        if len(x) < self.max_length:
            x = x + [self.pad_token for _ in range(self.max_length - len(x))]
        else:
            x = x[:self.max_length - 1] + [self.eos_token]
        return x


class Numericalize(object):
    def __init__(self, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
        self.tokenizer = tokenizer

    def __call__(self, x):
        return self.tokenizer.convert_tokens_to_ids(x)


class ToTensor(object):
    def __call__(self, x):
        return torch.tensor(x)


class Augment(object):
    def __init__(self, augmenter):
        self.augmenter = augmenter

    def __call__(self, x):
        return self.augmenter.augment(x)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x
