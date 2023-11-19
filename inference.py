# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
from tqdm import tqdm
import json

from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor

def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


def write_jsonl(path, examples):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--input_path", type=str, required=True, help='input path for .jsonl file')
    parser.add_argument("--output_path", type=str, required=True, help='output path for .jsonl file')
    parser.add_argument("--images_path", type=str, required=True, help='input path for images')

    parser.add_argument("--prompt", type=str, default='Describe this image in great detail.', help='Prompt to use along with image')
    
    parser.add_argument("--english", action='store_true', help='only output English')

    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    parser = CogVLMModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # load model
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device='cuda',
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    print('Welcome to CogVLM-CLI. Enter an image URL or local file path to load an image. Continue inputting text to engage in a conversation. Type "clear" to start over, or "stop" to end the program.')
    with torch.no_grad():
        data = list(read_jsonl(args.input_path))
        with open(args.output_path, 'w') as f:
            for ex in tqdm(data):
                history = None
                cache_image = None
                if world_size > 1:
                    torch.distributed.broadcast_object_list(image_path, 0)
                image_path = ex['images'][0]
                assert image_path is not None
                query = [args.prompt]

                if world_size > 1:
                    torch.distributed.broadcast_object_list(query, 0)
                query = query[0]
                assert query is not None
                try:
                    response, history, cache_image = chat(
                        os.path.join(args.images_path, image_path), 
                        model, 
                        text_processor_infer,
                        image_processor,
                        query, 
                        history=history, 
                        image=cache_image, 
                        max_length=args.max_length, 
                        top_p=args.top_p, 
                        temperature=args.temperature,
                        top_k=args.top_k,
                        invalid_slices=text_processor_infer.invalid_slices,
                        no_prompt=False
                        )
                except Exception as e:
                    print(e)
                    continue
                if rank == 0:
                    print("Model: " + response)
                    f.write(json.dumps(
                      {
                        "id": ex["id"],
                        "response": response
                      }
                      ) + "\n")
                    if tokenizer.signal_type == "grounding":
                        print("Grounding result is saved at ./output.png")
                    


if __name__ == "__main__":
    main()
