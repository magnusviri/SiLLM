import os
import argparse

import mlx.core as mx

import sillm
import sillm.utils as utils

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for generating text with SiLLM.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("-d", "--chdir", default=None, type=str, help="Change working directory")
    parser.add_argument("-c", "--config", default=None, type=str, help="Load YAML configuration file for chat")
    parser.add_argument("-a", "--input_adapters", default=None, type=str, help="Load LoRA adapter weights from .safetensors file")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed for randomization")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("-f", "--flush", type=int, default=5, help="Flush output every n tokens")
    parser.add_argument("-m", "--max_tokens", type=int, default=1024, help="Max. number of tokens to generate")
    parser.add_argument("--template", type=str, default=None, help="Chat template (chatml, llama2, alpaca, etc.)")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for chat template")
    parser.add_argument("-q4", default=False, action="store_true", help="Quantize the model to 4 bits")
    parser.add_argument("-q8", default=False, action="store_true", help="Quantize the model to 8 bits")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Change working directory
    if args.chdir is not None:
        os.chdir(args.chdir)

    # Load YAML configuration file
    if args.config is not None:
        utils.load_yaml(args.config, args)
    
    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Log commandline arguments
    if log_level <= 10:
        utils.log_arguments(args.__dict__)

    # Set random seed
    if args.seed >= 0:
        utils.seed(args.seed)

    # Load model
    model = sillm.load(args.model)

    # Quantize model
    if args.q4 is True:
        model.quantize(bits=4)
    elif args.q8 is True:
        model.quantize(bits=8)

    if args.input_adapters is not None:
        # Convert model to trainable
        model = sillm.TrainableLoRA.from_model(model)

        # Initialize LoRA layers
        model.init_lora()

        # Load and merge adapter file
        model.load_adapters(args.input_adapters)
        model.merge_and_unload_lora()

    generate_args = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "flush": args.flush
    }

    # Set conversation template
    if args.template:
        # Use specified template
        conversation = sillm.Conversation(template=args.template, system_prompt=args.system_prompt)
    else:
        template_name = sillm.Template.guess_template(model.args)
        if template_name:
            # Use template guessed from model type
            conversation = sillm.Conversation(template=template_name, system_prompt=args.system_prompt)
        elif model.tokenizer.has_template:
            # Use template from tokenizer config
            conversation = sillm.AutoConversation(model.tokenizer, system_prompt=args.system_prompt)
        else:
            conversation = None

            logger.warn("No conversation template found")

    # Log memory usage
    utils.log_memory_usage()

    # Input loop
    while True:
        prompt = input("> ")

        if prompt.startswith('.'):
            break
        elif prompt == "":
            if conversation:
                conversation.clear()
            continue

        if conversation:
            prompt = conversation.add_user(prompt)
        
        logger.debug(f"Generating {args.max_tokens} tokens with temperature {args.temperature}")

        response = ""
        for s, metadata in model.generate(prompt, **generate_args):
            print(s, end="", flush=True)
            response += s
        print()

        if conversation:
            conversation.add_assistant(response)

        logger.debug(f"Evaluated {metadata['usage']['prompt_tokens']} prompt tokens in {metadata['timing']['eval_time']:.2f}s ({metadata['usage']['prompt_tokens'] / metadata['timing']['eval_time']:.2f} tok/sec)")
        logger.debug(f"Generated {metadata['usage']['completion_tokens']} tokens in {metadata['timing']['runtime']:.2f}s ({metadata['usage']['completion_tokens'] / metadata['timing']['runtime']:.2f} tok/sec)")