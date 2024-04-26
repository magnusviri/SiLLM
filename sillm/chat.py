import os
import argparse

import sillm
import sillm.utils as utils

import sys
import select
import tty
import termios
from cmd import Cmd

class bcolors:
    USER = '\033[93m'
    AI = '\033[92m'
    DEBUG = '\033[94m'

class NonBlockingConsole(object):

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


    def get_data(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return False


class CommandPrompt(Cmd):
    prompt = f"{bcolors.USER}> "
    intro = f"{bcolors.AI}Type /? or /help for help. Type /quit or /exit to exit."
    inference = None

    def __init__(self):
        super().__init__()
        self.conversation = None
        self.response = None

    def precmd(self, line):
        if line.startswith('/'):
            line = line[1:]
            if line == "?":
                line = "help"
            elif line in ["quit", "exit"]:
                line = "EOF"
        else:
            if line == "EOF":
                line = "EOF"
            else:
                line = f"say {line}"
        return line
    
    def do_EOF(self, line):
        return True

    def do_clear(self, line):
        if self.conversation:
            self.conversation.clear()
            self.response = None

    def do_conversation(self, line):
        if self.conversation:
            print(self.conversation)
            if self.response:
                print("Last assistant response was:")
                print(self.response)

    def do_rewrite(self, line):
        if self.conversation:
            print(f"{bcolors.AI}Changed response to: {line}")
            self.conversation.add_assistant(line)
            self.response = None

    def do_settings(self, line):
        print(self.settings)

    def do_seed(self, line):
        try:
            self.settings["seed"] = int(line)
            if self.settings["seed"] >= 0:
                print(self.settings["seed"])
                utils.seed(self.settings["seed"])
                self.do_clear(None)
    #         else:
    #             utils.seed(None)
        except ValueError:
            print(f"{line} is not an integer")

    def do_temperature(self, line):
        try:
            self.settings["generate_args"]["temperature"] = float(line)
        except ValueError:
            print(f"{line} is not a float")

    def do_max_tokens(self, line):
        try:
            self.settings["generate_args"]["max_tokens"] = int(line)
        except ValueError:
            print(f"{line} is not an integer")

    def do_system_prompt(self, line):
        self.settings["system_prompt"] = line
        self.do_clear(None)
        self.conversation = sillm.Conversation(self.settings["template"], system_prompt=line)


    def do_help(self, line):
        print(f"""{bcolors.AI}
    /? or /help         - this message
    /exit or /quit      - exit/quit
    /clear              - reset conversation
    /rewrite            - change the last reply from the AI
    /seed               - change the seed
    /conversation       - show the conversation
    /settings           - print settings
    /temperature        - change the temperature
    /max_tokens         - change the max tokens
    /system_prompt      - change the system prompt (resets the conversation)

    Press esc to interrupt the AI.
""")

    def do_say(self, line):
        if self.inference is not None:
            if self.conversation and self.response:
                self.conversation.add_assistant(self.response)

            if self.conversation:
                line = self.conversation.add_user(line)

            self.response = self.inference(line, self.settings)

if __name__ == "__main__":
    print(bcolors.DEBUG)

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for generating text with SiLLM.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("-d", "--chdir", default=None, type=str, help="Change working directory")
    parser.add_argument("-c", "--config", default=None, type=str, help="Load YAML configuration file for chat")
    parser.add_argument("-a", "--input_adapters", default=None, type=str, help="Load and merge LoRA adapter weights from .safetensors file")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed for randomization")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("-p", "--repetition_penalty", type=float, default=None, help="Repetition penalty")
    parser.add_argument("-w", "--repetition_window", type=int, default=50, help="Window of generated tokens to consider for repetition penalty")
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

    settings = {
        "seed": args.seed,
        "system_prompt": args.system_prompt,
        "template": args.template,
        "generate_args": {
            "flush": args.flush,
            "max_tokens": args.max_tokens,
            "repetition_penalty": args.repetition_penalty,
            "repetition_window": args.repetition_window,
            "temperature": args.temperature,
        }
    }

    # Load model
    model = sillm.load(args.model)

    if args.input_adapters is not None:
        # Convert model to trainable
        model = sillm.TrainableLoRA.from_model(model)

        lora_config = model.load_lora_config(args.input_adapters)

        # Initialize LoRA layers
        model.init_lora(**lora_config)

        # Load and merge adapter file
        model.load_adapters(args.input_adapters)
        model.merge_and_unload_lora()

    # Quantize model
    if args.q4 is True:
        model.quantize(bits=4)
    elif args.q8 is True:
        model.quantize(bits=8)

    # Init conversation template
    settings["template"] = sillm.init_template(model.tokenizer, model.args, settings["template"])

    # Log memory usage
    utils.log_memory_usage()

    def inference(prompt, settings):
        print(bcolors.DEBUG)
        logger.debug(f"Generating {args.max_tokens} tokens with temperature {args.temperature}")
        print(bcolors.AI)
        response = ""
        with NonBlockingConsole() as nbc:
            for s, metadata in model.generate(prompt, **settings["generate_args"]):
                print(s, end="", flush=True)
                response += s
                if nbc.get_data() == '\x1b':  # x1b is ESC
                    break
        print()
        print(bcolors.DEBUG)
        logger.debug(f"Evaluated {metadata['usage']['prompt_tokens']} prompt tokens in {metadata['timing']['eval_time']:.2f}s ({metadata['usage']['prompt_tokens'] / metadata['timing']['eval_time']:.2f} tok/sec)")
        logger.debug(f"Generated {metadata['usage']['completion_tokens']} tokens in {metadata['timing']['runtime']:.2f}s ({metadata['usage']['completion_tokens'] / metadata['timing']['runtime']:.2f} tok/sec).")
        return response

    # Interactive prompt
    cmdPrompt = CommandPrompt()
    cmdPrompt.settings = settings
    cmdPrompt.inference = inference
    cmdPrompt.do_system_prompt(settings["system_prompt"])
    cmdPrompt.do_seed(settings["seed"])

    try:
        cmdPrompt.cmdloop()
    except (KeyboardInterrupt):
        pass
