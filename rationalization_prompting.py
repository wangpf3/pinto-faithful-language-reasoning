import json
import argparse
import random
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm

torch.set_num_threads(4)

# for REPRODUCIBILITY
set_seed(42)

# ----------------------------------------------------- #
# hyper-parameters
num_return_sequences = 1
generation_length = 128
num_process = 1

def generate(input_seq, model, tokenizer, stop_token_id, args):
    encoder_input = tokenizer(input_seq, truncation=True, return_tensors='pt')
    input_length = len(encoder_input.input_ids[0])

    num_batch = num_return_sequences // args.eval_batch_size
    outputs = []
    for i in range(num_batch):
        with torch.no_grad():
            pred_ids = model.generate(
                  input_ids=encoder_input.input_ids.to(args.device), 
                  attention_mask=encoder_input.attention_mask.to(args.device),
                  max_length=input_length+generation_length,
                  eos_token_id=stop_token_id, # \n\nQ
                  pad_token_id=tokenizer.eos_token_id,
                  early_stopping=True, 
                  num_return_sequences=args.eval_batch_size,
                  num_beams=args.num_beams,
                  do_sample=args.top_k>0 or args.top_p<1.0,
                  top_p=args.top_p,
                  top_k=args.top_k,
                  typical_p=args.typical_p,
                  use_cache=True,
                  temperature=1.0 if args.top_k==0 else 0.7,
                 ) 
        for beam in pred_ids:
            outputs.append(tokenizer.decode(beam[input_length:], skip_special_tokens=True).replace('\n', ''))
    return outputs

def main(args):
    # ----------------------------------------------------- #
    # prepare model
    model_path = 'EleutherAI/gpt-neox-20b'
    model_name = "GPT-neox"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='../../cache') #, use_fast=False)
    n_gpus = 2
    free_in_GB = 32
    max_memory = {i: "{}GB".format(free_in_GB - 16 if i ==args.gpu else free_in_GB + 8) for i in range(args.gpu, args.gpu + n_gpus)}

    from transformers import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(
                model_path, 
                device_map='auto',
                max_memory = max_memory,
                cache_dir='../cache',
                torch_dtype='auto'
            )

    stop_token_id = tokenizer.encode("\n\nQ")[-2]
    model.eval()

    # ----------------------------------------------------- #
    # prepare data
    with open('./prompts/{}.txt'.format(args.dataset), 'r') as fr:
        prompt = json.load(fr)["prompt"].replace("Yes or no: ", "")
    print(prompt)

    for split in args.eval_split.split(','):
        with open('./data/{}/{}.jsonl'.format(args.dataset, split), 'r') as fr:
            all_lines = fr.readlines()

        batch_size = - (len(all_lines) // - num_process)
        testset = all_lines[(args.split*batch_size):(args.split+1)*batch_size]
        if args.debug:
            testset = testset[:2]

        # ----------------------------------------------------- #
        # inference

        output_path = './output/{}/{}.jsonl'.format(
                            args.dataset, 
                            args.split,
                        ) 

        fw = open(output_path, 'w')
        for line in tqdm(testset):
            example = json.loads(line)

            inference = []
            formatted_question = example["question"] if "question" in example else example["context"]
            if "choices" in example:
                formatted_question += "\nAnswer Choices:"
                for choice_id, choice in enumerate(example["choices"]):
                    formatted_question += "\n({}) {}".format(chr(ord('a')+choice_id), choice)
            for choice in example["choices"]:
                input_seq = prompt.format(formatted_question, choice)
                inference.append(generate(input_seq, model, tokenizer, stop_token_id, args)[0])
                if args.debug:
                    print(input_seq)
            output_example = example.copy()
            output_example["explanation"] = inference
            fw.write(json.dumps(output_example) + "\n")
        fw.close()

    # ----------------------------------------------------- #

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--prompt', '-p', type=str)
    parser.add_argument('--eval_split', type=str, default='test,dev,train')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument("--debug", action='store_true')

    # decoding strategy
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--typical_p', type=float, default=1.0)
    parser.add_argument('--eval_batch_size', type=int, default=1)

    # gpu and workers option
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu))

    main(args)

