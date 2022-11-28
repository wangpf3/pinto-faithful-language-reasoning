import json
import os 
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional
import random

import torch
from torch.utils.data import Dataset, TensorDataset

@dataclass(frozen=True)
class InputExample:

    context: str
    explanation: List[str]
    choices: List[str]
    answer: int

class TrainingDataset(Dataset):

    features: List[InputExample]

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputExample:
        return self.features[i]

def load_raw_dataset(split, args):
    data_path = os.path.join('./data', args.dataset, '{}.jsonl'.format(split))
    dataset = []

    with open(data_path, 'r') as fr:
        for example_id, line in tqdm(enumerate(fr), desc='processing {}'.format(data_path)):
            example = json.loads(line)

            choices = []
            if "choices" in example:
                choices = example["choices"]
            else:
                if "context" in example:
                    choices = ['false', 'true']
                else:
                    choices = ['no', 'yes']

            if all(isinstance(expl, list) for expl in example["explanation"]):
                for expl_id in range(len(example["explanation"][0])):
                    explanation = []
                    for choice_id in range(len(example["explanation"])):
                        explanation.append(example["explanation"][choice_id][expl_id])
                    dataset.append(
                            InputExample(
                                    context=example["context"] if "context" in example else example["question"],
                                    explanation=explanation,
                                    choices=choices,
                                    answer=example["answer"],
                            )
                    )
            else:
                dataset.append(
                        InputExample(
                                context=example["context"] if "context" in example else example["question"],
                                explanation=example["explanation"],
                                choices=choices,
                                answer=example["answer"],
                        )
                )

    for example in dataset[:2]:
        print("*** Example ***")
        print(example)

    return TrainingDataset(dataset)
    
def get_label_tensor(raw_label, tokenizer, args):
    label_ids = tokenizer.encode(raw_label, add_special_tokens=False)
    label_ids = label_ids[:args.max_dec_length] 
    label_ids += [-100] * (args.max_dec_length - len(label_ids))
    return label_ids

def format_input(question, choices=None):
    input_seq = "Question: {}".format(question.strip())
    # input_seq += " Answer: {}.".format(choice.strip())
    if choices is not None:
        input_seq += " Answer Choices:"
        for choice_id, choice in enumerate(choices):
            input_seq += " ({}) {}".format(chr(ord('a')+choice_id), choice)
        input_seq += '.'
    return input_seq

def format_explanation(explanation):
    input_seq = ' Explanation: ' + explanation.strip()
    return input_seq

class Data_Collator_for_Training(object):
    def __init__(self, tokenizer, args, mask_inference=False, dropout_context=0):
        self.tokenizer = tokenizer
        self.mask_inference = mask_inference
        self.dropout_context = dropout_context 
        self.args = args

    def __call__(self, examples):

        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        decoder_label_tensor = []
        label_tensor = []
        smoothing_tensor = []

        for example_idx, example in enumerate(examples):
            input_ids = []
            attention_mask = []

            context = example.context
            # input_ids += self.tokenizer.encode(context.strip(), add_special_tokens=False)

            choices_input_ids = [input_ids.copy() for choice in example.choices]
            choices_attention_mask = [attention_mask.copy() for ids in choices_input_ids]
            for choice_id in range(len(example.choices)):
                choices_input_ids[choice_id] += self.tokenizer.encode(format_input(context, example.choices), add_special_tokens=False)
                added_length = (len(choices_input_ids[choice_id]) - len(choices_attention_mask[choice_id]))
                context_only_ids = self.tokenizer.encode(format_input(context), add_special_tokens=False)

                if self.dropout_context > 0 and random.random() < self.dropout_context:
                    choices_attention_mask[choice_id] += [0] * (len(context_only_ids) - len(choices_attention_mask[choice_id])) + [1] * (len(choices_input_ids[choice_id]) - len(context_only_ids))
            # elif self.mask_inference:
            #     added_length = len(input_ids) - len(attention_mask)
            #     attention_mask += (~torch.bernoulli(torch.tensor([self.args.mask_prob] * added_length)).bool()).long().tolist()
                else:
                    choices_attention_mask[choice_id] += [1] * (len(choices_input_ids[choice_id]) - len(choices_attention_mask[choice_id]))

            if not self.args.no_explanation:

                explanation = example.explanation
                smoothing_tensor.append(self.args.label_smoothing_no_inference)

                for choice_id in range(len(example.choices)):
                    added_ids = self.tokenizer.encode(format_explanation(explanation[choice_id]), add_special_tokens=False)
                    added_length = len(added_ids) 
                    if self.mask_inference: 
                        if random.random() > self.args.mask_prob:
                            # mask_ratio = random.uniform(self.args.min_mask_ratio, 1.0) 
                            mask_idx = random.sample(range(added_length), int(added_length * self.args.replace_ratio))
                            choices_input_ids[choice_id] += [random.choice(range(len(self.tokenizer))) if _idx in mask_idx else added_ids[_idx] for _idx in range(added_length)]
                            choices_attention_mask[choice_id] += [1] * added_length 
                        else:
                            mask_idx = random.sample(range(added_length), int(added_length * self.args.mask_ratio))
                            choices_input_ids[choice_id] += added_ids
                            choices_attention_mask[choice_id] += [0 if _idx in mask_idx else 1 for _idx in range(added_length)]
                            # choices_attention_mask[choice_id] += (~torch.bernoulli(torch.tensor([self.args.mask_prob] * added_length)).bool()).long().tolist()

                    else:
                        choices_input_ids[choice_id] += added_ids
                        choices_attention_mask[choice_id] += [1] * added_length 
            else:
                smoothing_tensor.append(0)

            choices_input_ids = [ids[:self.args.max_enc_length] for ids in choices_input_ids]
            choices_input_ids = [ids + [self.tokenizer.pad_token_id] * (self.args.max_enc_length - len(ids)) for ids in choices_input_ids]
            choices_attention_mask = [ids[:self.args.max_enc_length] for ids in choices_attention_mask]
            choices_attention_mask = [ids + [0] * (self.args.max_enc_length - len(ids)) for ids in choices_attention_mask]

            choices_tensor = []
            for choice_id, choice in enumerate(example.choices):
                choices_tensor.append(get_label_tensor(choice, self.tokenizer, self.args))
                # if choice == example["answer"]:
                #     task_label = choice_id

            encoder_input_tensor.append(choices_input_ids)
            encoder_attention_mask_tensor.append(choices_attention_mask)
            decoder_label_tensor.append(choices_tensor)
            label_tensor.append(example.answer)

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, encoder_attention_mask_tensor, decoder_label_tensor, label_tensor, smoothing_tensor])


def get_tensor_dataset(split, tokenizer, args):
    data_path = os.path.join('./data', args.dataset, '{}.jsonl'.format(split))

    encoder_input_tensor = []
    encoder_attention_mask_tensor = []
    decoder_label_tensor = []
    task_label_tensor = []

    with open(data_path, 'r') as fr:
        for line_idx, line in tqdm(enumerate(fr), desc='processing {}'.format(data_path)):
            example = json.loads(line)

            # if "choices" in example:
            #     choices = example["choices"]
            # else:
            #     if "context" in example:
            #         choices = ['false', 'true']
            #     else:
            #         choices = ['no', 'yes']
            context=example["context"] if "context" in example else example["question"]

            input_ids = []
            attention_mask = []

            choices_input_ids = [input_ids.copy() for choice in example["choices"]]
            choices_attention_mask = [attention_mask.copy() for ids in choices_input_ids]

            for choice_id in range(len(example["choices"])):
                choices_input_ids[choice_id] += tokenizer.encode(format_input(context, example["choices"]), add_special_tokens=False)
                added_length = (len(choices_input_ids[choice_id]) - len(choices_attention_mask[choice_id]))
                choices_attention_mask[choice_id] += [1] * (len(choices_input_ids[choice_id]) - len(choices_attention_mask[choice_id]))

            if not args.no_explanation:
                if all(isinstance(expl, list) for expl in example["explanation"]):
                    for choice_id in range(len(example["choices"])):
                        expanded_choice_input_ids = [choices_input_ids[choice_id].copy() for expl_id in range(len(example["explanation"][choice_id]))]
                        expanded_attention_mask = [choices_attention_mask[choice_id].copy() for expl_id in range(len(example["explanation"][choice_id]))]
                        for expl_id in range(len(example["explanation"][choice_id])):
                            expanded_choice_input_ids[expl_id] += tokenizer.encode(format_explanation(example["explanation"][choice_id][expl_id]), add_special_tokens=False)
                            added_length = (len(expanded_choice_input_ids[expl_id]) - len(expanded_attention_mask[expl_id]))
                            expanded_attention_mask[expl_id] += [1] * added_length
                        expanded_choice_input_ids = [ids[:args.max_enc_length] for ids in expanded_choice_input_ids]
                        expanded_choice_input_ids = [ids + [tokenizer.pad_token_id] * (args.max_enc_length - len(ids)) for ids in expanded_choice_input_ids]
                        expanded_attention_mask = [ids[:args.max_enc_length] for ids in expanded_attention_mask]
                        expanded_attention_mask = [ids + [0] * (args.max_enc_length - len(ids)) for ids in expanded_attention_mask]
                        choices_input_ids[choice_id] = expanded_choice_input_ids
                        choices_attention_mask[choice_id] = expanded_attention_mask

                else:
                    for choice_id in range(len(example["choices"])):
                        choices_input_ids[choice_id] += tokenizer.encode(format_explanation(example["explanation"][choice_id]), add_special_tokens=False)
                        added_length = (len(choices_input_ids[choice_id]) - len(choices_attention_mask[choice_id]))
                        choices_attention_mask[choice_id] += [1] * added_length 
                    choices_input_ids = [ids[:args.max_enc_length] for ids in choices_input_ids]
                    choices_input_ids = [ids + [tokenizer.pad_token_id] * (args.max_enc_length - len(ids)) for ids in choices_input_ids]
                    choices_attention_mask = [ids[:args.max_enc_length] for ids in choices_attention_mask]
                    choices_attention_mask = [ids + [0] * (args.max_enc_length - len(ids)) for ids in choices_attention_mask]

                choices_input_ids = [ids[:args.max_enc_length] for ids in choices_input_ids]
                choices_input_ids = [ids + [tokenizer.pad_token_id] * (args.max_enc_length - len(ids)) for ids in choices_input_ids]
                choices_attention_mask = [ids[:args.max_enc_length] for ids in choices_attention_mask]
                choices_attention_mask = [ids + [0] * (args.max_enc_length - len(ids)) for ids in choices_attention_mask]

            choices_tensor = []
            for choice_id, choice in enumerate(example["choices"]):
                choices_tensor.append(get_label_tensor(choice, tokenizer, args))

            encoder_input_tensor.append(choices_input_ids)
            encoder_attention_mask_tensor.append(choices_attention_mask)
            decoder_label_tensor.append(choices_tensor)
            task_label_tensor.append(example["answer"])

    encoder_input_tensor = torch.tensor(encoder_input_tensor, dtype=torch.long)
    encoder_attention_mask_tensor= torch.tensor(encoder_attention_mask_tensor, dtype=torch.long)
    decoder_label_tensor = torch.tensor(decoder_label_tensor, dtype=torch.long)
    task_label_tensor = torch.tensor(task_label_tensor, dtype=torch.long)
    for f1, f2, f3, f4 in zip(encoder_input_tensor[:2], encoder_attention_mask_tensor[:2], decoder_label_tensor[:2], task_label_tensor[:2]):
        print("*** Example ***")
        if len(f1.shape) == 3:
            f1 = f1[0]
        for ids in f1:
            print("encoder input: %s" % tokenizer.decode(ids))
        # print("encoder attention mask: %s" % f2)
        for ids in f3:
            print("decoder output: %s" % tokenizer.decode([tid for tid in ids if not tid == -100]))
        print("task label: %s" % f4)

    return TensorDataset(encoder_input_tensor, encoder_attention_mask_tensor, decoder_label_tensor, task_label_tensor)

