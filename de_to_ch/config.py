import argparse


def read_arguments_train():
    parser = argparse.ArgumentParser(description="Run training with following arguments")

    # general configuration
    parser.add_argument('--model_output_dir', default='experiments', type=str)
    parser.add_argument('--exp_name', default='exp', type=str)
    parser.add_argument('--t5', default='t5-small', type=str)
    parser.add_argument('--seed', default=90, type=int)
    parser.add_argument('--toy', default=False, action='store_true')
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--bf16', default=False, action='store_true')

    # training & optimizer configuration
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int)
    parser.add_argument('--eval_acc', default=2000, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--num_epochs', default=5.0, type=float)
    parser.add_argument('--direction', default='de_to_ch', type=str)

    parser.add_argument('--lr_base', default=1e-5, type=float)
    parser.add_argument('--eval_every_n_steps', type=int, default=1000)

    # prediction configuration (run after each epoch)
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')

    args = parser.parse_args()

    print("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        print("argument: {}={}".format(argument, getattr(args, argument)))

    return args
