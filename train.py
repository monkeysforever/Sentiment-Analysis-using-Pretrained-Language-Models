import argparse
from transformers import BertTokenizer, BertModel
from transformers import AdamW
import torch
import logging
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import dataset, transforms
from models.lm_classifier import PretrainedClassifier
import time
from utils.misc import AverageMeter, get_accuracy, set_seed
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

best_acc = 0


def get_args(parser):
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--model', default='bert',
                        choices=['bert'],
                        help='type of language model')
    parser.add_argument('--tar', default='model.pth.tar', type=str,
                        help='path for saving the model')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='imdb', type=str,
                        choices=['imdb'],
                        help='dataset name')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batch size')
    parser.add_argument('--test-batch-size', default=40, type=int,
                        help='test batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--wdecay', default=0.0, type=float,
                        help='weight decay')        
    parser.add_argument('--out', default='results',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--train', type=str, default='data/IMDB_train.csv',
                        help='The path of train dataset')
    parser.add_argument('--eval', type=str, default='data/IMDB_test.csv',
                        help='The path of test dataset')
    return parser.parse_args()


def train(train_loader, model, optimizer, criterion, args, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    p_bar = None
    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    model.train()
    for batch_idx, (texts, labels) in enumerate(train_loader):
        texts = texts.to(args.device)
        labels = labels.to(args.device)
        masks = []
        for text in texts:
            masks.append([int(token != 0) for token in text])
        masks = torch.FloatTensor(masks).to(args.device)
        data_time.update(time.time() - end)
        logits = model(texts, masks)
        loss = criterion(logits, labels)
        loss.backward()
        losses.update(loss.item())
        optimizer.step()
        model.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_progress:
            if (epoch + 1) % 1 == 0:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. "
                    "Loss: {loss:.4f}. ".format(
                        epoch=epoch + 1,
                        batch=batch_idx + 1,
                        epochs=args.epochs,
                        iter=args.iteration,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg))
                p_bar.update()
    if not args.no_progress:
        p_bar.close()
    return losses.avg


def test(test_loader, model, criterion, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    test_loader = tqdm(test_loader)

    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    outlist = torch.zeros(0, dtype=torch.float, device='cpu')
    model.eval()
    with torch.no_grad():
        for batch_idx, (texts, labels) in enumerate(test_loader):
            texts = texts.to(args.device)
            labels = labels.to(args.device)
            masks = []
            for text in texts:
                masks.append([int(token != 0) for token in text])
            masks = torch.FloatTensor(masks).to(args.device)
            data_time.update(time.time() - end)
            logits = model(texts, masks)
            _, predicted = torch.max(logits.data, 1)
            loss = criterion(logits, labels)
            predlist = torch.cat([predlist, predicted.view(-1).cpu()])
            lbllist = torch.cat([lbllist, labels.view(-1).cpu()])
            outlist = torch.cat([outlist, logits.view(-1).cpu()])
            if len(args.classes) >= 5:
                topk = (1, 5)
            prec = get_accuracy(logits, labels, args.topk)
            losses.update(loss.item(), texts.shape[0])
            top1.update(prec[0].item(), texts.shape[0])
            if len(prec) > 1:
                top5.update(prec[1].item(), texts.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. top1: {top1:.2f}. ".format(
                batch=batch_idx + 1,
                iter=len(test_loader),
                loss=losses.avg,
                top1=top1.avg,
            ))

        test_loader.close()
    logger.info("top-1 acc: {:.2f}".format(top1.avg))

    return losses.avg, top1.avg


def main():
    parser = argparse.ArgumentParser(description='PyTorch MultiCon Text Training')
    args = get_args(parser)
    global best_acc
    text_transforms = None
    writer = None
    args.device = torch.device('cuda', args.gpu_id)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    if args.seed != -1:
        set_seed(args)

    if not args.no_progress:
        os.makedirs(args.out, exist_ok=True)
        writer = SummaryWriter(args.out)

    if args.model == 'bert':
        args.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        args.max_length = 512
        args.pad_token = args.tokenizer.pad_token
        print(args.pad_token)
        args.unk_token = args.tokenizer.unk_token
        args.init_token = args.tokenizer.cls_token
        args.end_token = args.tokenizer.sep_token
        args.lm = BertModel.from_pretrained('bert-base-uncased')
        args.embedding_size = 768
        text_transforms = transforms.Compose([
            transforms.Tokenize(tokenizer=args.tokenizer.tokenize),
            transforms.InsertTokens(init_token=args.init_token, eos_token=args.end_token),
            transforms.Pad(max_length=args.max_length, pad_token=args.pad_token, eos_token=args.end_token),
            transforms.Numericalize(tokenizer=args.tokenizer),
            transforms.ToTensor()
        ])
        

    if args.dataset == 'imdb':
        args.classes = ['neg', 'pos']
        args.topk = (1,)

    train_dataset = dataset.TextCSVDataset(args.train, text_transforms=text_transforms)
    test_dataset = dataset.TextCSVDataset(args.eval, text_transforms=text_transforms)

    model = PretrainedClassifier(language_model=args.lm, embedding_size=args.embedding_size,
                                 num_classes=len(args.classes))

    logger.info("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    model.freeze_lm()
    model.to(args.device)
    optimizer = AdamW(params=model.parameters(), lr = args.lr, weight_decay = args.wdecay)

    criterion = torch.nn.CrossEntropyLoss()

    train_sampler = RandomSampler

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.test_batch_size,
        drop_last=True)

    args.iteration = len(train_dataset) // args.batch_size
    args.total_steps = args.epochs * args.iteration
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")    
    logger.info(f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    test_accs = []
    test_loss = None
    model.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(train_loader, model, optimizer, criterion, args, epoch)
        writer.add_scalar('Loss/train', train_loss, args.epochs - start_epoch)

        if (epoch + 1) % 1 == 0:
            logger.info("Epoch {}. train_loss: {:.4f}."
                        .format(epoch + 1, train_loss))

        test_model = model

        if (epoch + 1) % 1 == 0:
            test_loss, test_acc = test(test_loader, test_model, criterion, args)
            writer.add_scalar('Loss/test', test_loss)
            writer.add_scalar('Acc/test', test_acc)
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            if is_best:
                model_to_save = test_model.module if hasattr(test_model, "module") else test_model
                torch.save({
                    'epoch': epoch + 1,
                    'arch': 'BERT',
                    'state_dict': model_to_save.state_dict(),
                    'best_acc1': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, args.tar)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))

        scheduler.step(test_loss)
    writer.close()


if __name__ == '__main__':
    main()
