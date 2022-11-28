import json
import os
import argparse
from tqdm import tqdm, trange
import numpy as np
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import set_seed, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor

from data_helper import load_raw_dataset, Data_Collator_for_Training, get_tensor_dataset
from utils import get_logger 

class Text2TextForMultiChoice(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='../cache/')

    def forward(self, input_ids, attention_mask, target_ids, labels=None):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        target_ids = target_ids.view(-1, target_ids.size(-1))

        outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target_ids,
                )

        log_probs = - F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), target_ids.view(-1), ignore_index=-100, reduction='none') 
        log_probs = log_probs.view(-1, target_ids.size(-1)).sum(dim=-1)
        seq_lengths = (target_ids != -100).sum(dim=-1) * 1.0
        log_probs /= seq_lengths
        log_probs = log_probs.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(log_probs, labels)
        return ((loss,) + (log_probs,)) if loss is not None else log_probs

def evaluate(dataset, model, args):

    data_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.eval_batch_size)
    model.eval()
    epoch_iterator = tqdm(dataloader, desc="Eval Iteration")

    accuracy = 0.
    output_predictions = []
    output_scores = []
    for step, batch in enumerate(epoch_iterator):

        input_ids, attention_mask, target_ids, labels = tuple(t.to(args.device) for t in batch) 
        with torch.no_grad():
            outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        target_ids=target_ids,
                    )
            logits = outputs

            _, predictions = logits.max(dim=1)
            probs = F.softmax(logits, dim=-1)
            answer_probs = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)

        accuracy += predictions.eq(labels).sum().item()
        output_predictions.extend(predictions.tolist())
        output_scores.extend(answer_probs.tolist())

        if args.debug and step > 10:
            break

    accuracy /= len(dataset)
    return accuracy * 100., output_predictions, output_scores

def main(args, seed):
    # ----------------------------------------------------- #
    # prepare logger
    log_path = os.path.join(args.save_dir, 'train_seed{}.log'.format(seed))
    logger = get_logger("model", log_path)
    logger.info('args: {}'.format(args))

    # ----------------------------------------------------- #
    # model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir='../cache/')
    model = Text2TextForMultiChoice(args.model_name)
    model.to(args.device)

    # ----------------------------------------------------- #
    # data
    train_dataloader_dict = {}

    # all input
    trainset = load_raw_dataset('train', args)
    train_sampler1 = RandomSampler(trainset)
    data_collator_for_training1 = Data_Collator_for_Training(tokenizer, args, mask_inference=False, dropout_context=args.dropout_context)
    train_dataloader1 = DataLoader(trainset, collate_fn=data_collator_for_training1, sampler=train_sampler1, batch_size=args.train_batch_size)
    train_dataloader_dict["all"] = train_dataloader1

    # no inference
    if args.counter_factor > 0:
        trainset3 = load_raw_dataset('train', args)
        train_sampler3 = RandomSampler(trainset3)
        if args.contrast_size > 0:
            data_collator_for_training3 = Data_Collator_for_Contrastive_Learning(tokenizer, args)
        else:
            data_collator_for_training3 = Data_Collator_for_Training(tokenizer, args, mask_inference=True, dropout_context=0)
        train_dataloader_dict["mask"] = DataLoader(trainset3, collate_fn=data_collator_for_training3, sampler=train_sampler3, batch_size=args.train_batch_size)

    devset = get_tensor_dataset(args.dev_split, tokenizer, args)

    # ----------------------------------------------------- #
    # optimization
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=args.learning_rate,
                    weight_decay=0.0,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False
                )

    num_update_steps_per_epoch = len(train_dataloader_dict["all"])
    t_total = num_update_steps_per_epoch // args.grad_step * args.num_epoch
    warmup_steps = int(t_total * args.warmup_ratio)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps) #, num_training_steps=t_total)

    # ----------------------------------------------------- #
    # training loop
    model_ckpt = os.path.join(args.save_dir, 'model_{}.ckpt'.format(seed))
    global_step = 0
    best_dev_acc = 0
    step_nogress = 0
    optimizer.zero_grad()
    if args.debug:
        args.num_epoch = 1
    for epoch in trange(int(args.num_epoch), desc="Epoch"):
        train_loss = 0.0
        epoch_no_inference_loss = 0.
        model.train()
        train_dataloader_list = [train_dataloader_dict["all"]]
        if args.counter_factor > 0:
            train_dataloader_list.append(train_dataloader_dict["mask"])
        epoch_iterator = tqdm(zip(*train_dataloader_list), desc="Train Iteration at Epoch {}".format(epoch), total=num_update_steps_per_epoch)
        for step, batch_list in enumerate(epoch_iterator):

            input_ids, attention_mask, target_ids, labels, _ = tuple(t.to(args.device) for t in batch_list[0]) 

            outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        target_ids=target_ids,
                        labels=labels,
                    )

            loss = outputs[0]

            if args.counter_factor > 0:
                input_ids, attention_mask, target_ids, labels, smoothing_weights = tuple(t.to(args.device) for t in batch_list[1]) 
                mask_outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            target_ids=target_ids,
                            # labels=labels,
                        )
                # label smoothing
                no_inference_loss = F.cross_entropy(mask_outputs, labels, label_smoothing=args.label_smoothing_no_inference)
                loss += args.counter_factor * no_inference_loss

            loss /= args.grad_step
            loss.backward()
            if (global_step + 1) % args.grad_step == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

            train_loss += outputs[0].item()
            step_log = "Epoch {} loss {:.4f}".format(epoch, train_loss / (step + 1))
            if args.counter_factor > 0:
                epoch_no_inference_loss += no_inference_loss.item()
                step_log += " counter {:.4f}".format(epoch_no_inference_loss / (step + 1))
            global_step += 1
            epoch_iterator.set_description(step_log)
            if args.debug and global_step > 10:
                break

        epoch_log = 'Epoch: {:03d} Train loss: {:.4f}'.format(epoch, train_loss / (step + 1))
        if args.counter_factor > 0:
            epoch_log += " smoothing {:.4f}".format(epoch_no_inference_loss / (step + 1))
        logger.info(epoch_log)

        dev_result, _, _ = evaluate(devset, model, args)

        log = 'Epoch: {:03d}, dev accuracy: {:.4f}'
        if dev_result > best_dev_acc:
            torch.save({'ckpt': model.state_dict(), 'args': args}, model_ckpt)
            log += ' best'
            best_dev_acc = dev_result
            step_nogress = 0
        else:
            step_nogress += 1
        logger.info(log.format(epoch, dev_result))
        if step_nogress > 1 and global_step > warmup_steps:
            break

    return_result = {}
    model.load_state_dict(torch.load(model_ckpt)['ckpt'])
    testset = get_tensor_dataset('test', tokenizer, args)
    eval_result, predictions, scores = evaluate(testset, model, args)
    log = 'Epoch: {:03d}, test accuracy: {:.4f}'
    logger.info(log.format(-1, eval_result))
    return_result["test_all"] = eval_result

    with open(os.path.join(args.save_dir, 'predictions_test_seed{}.txt'.format(seed)), 'w') as fw:
        for p, s in zip(predictions, scores):
            fw.write('{}\t{}\n'.format(p, s))

    if not args.test_split == "none":
        for split in args.test_split.split(','):
            testset = get_tensor_dataset(split, tokenizer, args)
            eval_result, predictions, scores = evaluate(testset, model, args)
            log = 'Epoch: {:03d}, {} accuracy: {:.4f}'
            logger.info(log.format(-1, split, eval_result))
            return_result[split] = eval_result

            with open(os.path.join(args.save_dir, 'predictions_{}_seed{}.txt'.format(split, seed)), 'w') as fw:
                for p, s in zip(predictions, scores):
                    fw.write('{}\t{}\n'.format(p, s))
                         
    if not args.save_ckpt:
        os.remove(model_ckpt)
    return return_result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--save_dir', '-o', type=str)
    parser.add_argument('--sample_size', type=int, default=0)
    parser.add_argument("--save_ckpt", action='store_true')
    parser.add_argument("--debug", action='store_true')

    # model
    parser.add_argument('--model_name', '-m', type=str)
    parser.add_argument('--max_enc_length', type=int, default=128)
    parser.add_argument('--max_dec_length', type=int, default=128)
    parser.add_argument("--no_explanation", action='store_true')
    parser.add_argument("--label_smoothing_no_inference", default=0, type=float)
    parser.add_argument("--label_smoothing_shuffle_inference", default=0, type=float)
    parser.add_argument("--dropout_context", default=0, type=float)
    parser.add_argument("--mask_prob", default=1.0, type=float)
    parser.add_argument("--counter_factor", default=1.0, type=float)
    parser.add_argument("--mask_ratio", default=0, type=float)
    parser.add_argument("--replace_ratio", default=0, type=float)
    parser.add_argument("--contrast_size", default=0, type=int)

    # training
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--grad_step', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument('--num_epoch', type=float, default=1000)
    parser.add_argument('--dev_split', type=str, default="dev")

    # inference
    parser.add_argument('--test_split', type=str, default="none")
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument("--overwrite_output", action='store_true')

    # gpu and workers option
    parser.add_argument('--gpu_device', type=str, default='0')

    args = parser.parse_args()

    args.device = torch.device('cuda:{}'.format(args.gpu_device))

    eval_result_all_split = {}
    for seed in range(41, 45):
        set_seed(seed)
        eval_result = main(args, seed)
        for split in eval_result:
            if split not in eval_result_all_split:
                eval_result_all_split[split] = []
            eval_result_all_split[split].append(eval_result[split])
    output_result = {}
    for split in eval_result_all_split:
        output_result[split] = {
                "accuracy_mean": np.mean(eval_result_all_split[split]),
                "accuracy_std": np.std(eval_result_all_split[split]),
        }
    with open(os.path.join(args.save_dir, 'evaluation_results.json'), 'w') as fw:
        json.dump(output_result, fw, indent=4)
