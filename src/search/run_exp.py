# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse
import numpy as np
import os
import json

import torch

from models import *
from run_manager import RunManager


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--train', action='store_true')

parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--latency', type=str, default=None)

parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--init_lr', type=float, default=0.05)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
# lr_schedule_param

parser.add_argument('--dataset', type=str, default='speech_commands', choices=['imagenet', 'speech_commands'])
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--valid_size', type=int, default=None)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default='bn', choices=['None', 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)

parser.add_argument('--n_worker', type=int, default=32)
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='strong', choices=['normal', 'strong', 'None'])

""" net config """
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument(
    '--net', type=str, default='proxyless_mobile',
    choices=['proxyless_gpu', 'proxyless_cpu', 'proxyless_mobile', 'proxyless_mobile_14']
)
parser.add_argument('--dropout', type=float, default=0)

""" quantization config """
parser.add_argument('--quantize', action='store_true')
parser.add_argument('--n_bits', default='1,2,3,4,5,6,7,8')


# TODO
def fold_batch_norm(state_dict):

    param_names = ["conv.weight", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"]
    layers = {}
    for state_key in state_dict.keys():
        for param_name in param_names:
            if state_key.endswith(param_name) and state_key not in layers.keys():
                layers[state_key.split("." + param_name)[0]] = {}

    for state_key, state_tensor in state_dict.items():
        for param_name in param_names:
            if state_key.endswith(param_name):
                split_state = state_key.split(".")
                layer_name = ".".join(split_state[:-2])
                param_name = ".".join(split_state[-2:])
                break
        layers[layer_name][param_name] = state_tensor

    for layer, params in layers.items():
        if layer == "feature_mix_layer":
            continue
        for l in range(params["conv.weight"].shape[3]):
            for k in range(params["conv.weight"].shape[2]):
                for j in range(params["conv.weight"].shape[1]):
                    for i in range(params["conv.weight"].shape[0]):
                            params["conv.weight"][i][j][k][l] *= params["bn.weight"][i] / np.sqrt(params["bn.running_var"])

        for i in range(params["bn.bias"].shape[0]):
            params["bn.bias"] -= params["bn.weight"][i] * params["bn.running_mean"][i] / np.sqrt(params["bn.running_var"])

        # Reset
        for i in range(params["bn.weight"].shape[0]):
            params["bn.weight"][i] = 1
        params["bn.running_var"] = 1
        for i in range(params["bn.running_mean"].shape[0]):
            params["bn.running_mean"][i] = 0

    return state_dict


# Taken from https://git.spsc.tugraz.at/wroth/nn-discrete-tf/
def quantize_state_dict(state_dict, n_bits):
    conv_weights = {k: v for k, v in state_dict.items() if k.endswith("conv.weight")}
    for name, weight in conv_weights.items():
        min_wt = weight.min()
        max_wt = weight.max()
        num_vals = 2.0 ** n_bits
        step = 2.0 ** -n_bits
        weight = (weight - min_wt) * (num_vals / (max_wt - min_wt))  # transform to [0,num_vals]
        weight = torch.round(weight)
        weight = torch.clamp(weight, 0, num_vals - 1.0)  # clip values
        weight = weight * (step * (max_wt - min_wt)) + min_wt

        state_dict[name] = weight

    return state_dict


# Taken from https://github.com/ARM-software/ML-KWS-for-MCU
def quantize_state_dict_qmn(state_dict, n_bits):
    conv_weights = { k: v for k, v in state_dict.items() if k.endswith("conv.weight")}
    for name, weight in conv_weights.items():
        weight = weight.cpu()
        min_wt = weight.min()
        max_wt = weight.max()
        # find number of integer bits to represent this range
        int_bits = int(np.ceil(np.log2(max(abs(min_wt), abs(max_wt)))))
        frac_bits = n_bits - int_bits - 1  # remaining bits are fractional bits (1-bit for sign)
        assert frac_bits >= 0
        # floating point weights are scaled and rounded, which are used in
        # the fixed-point operations on the actual hardware (i.e., microcontroller)
        quant_weight = np.round(weight * (2 ** frac_bits))
        # To quantify the impact of quantized weights, scale them back to
        # original range to run inference using quantized weights
        quant_weight_float = quant_weight / (2 ** frac_bits)

        state_dict[name] = quant_weight_float

    return state_dict


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.makedirs(args.path, exist_ok=True)

    # prepare run config
    run_config_path = '%s/run.config' % args.path
    if os.path.isfile(run_config_path):
        # load run config from file
        run_config = json.load(open(run_config_path, 'r'))
        if args.dataset == "speech_commands":
            run_config = SpeechCommandsRunConfig(**run_config)
        elif args.dataset == "imagenet":
            run_config = ImagenetRunConfig(**run_config)
        else:
            raise NotImplementedError
        if args.valid_size:
            run_config.valid_size = args.valid_size
    else:
        # build run config from args
        args.lr_schedule_param = None
        args.opt_param = {
            'momentum': args.momentum,
            'nesterov': not args.no_nesterov,
        }
        if args.no_decay_keys == 'None':
            args.no_decay_keys = None
        if args.dataset == "speech_commands":
            run_config = SpeechCommandsRunConfig(
                **args.__dict__
            )
        elif args.dataset == "imagenet":
            run_config = ImagenetRunConfig(
                **args.__dict__
            )
        else:
            raise NotImplementedError
    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))

    # prepare network
    net_config_path = '%s/net.config' % args.path
    if os.path.isfile(net_config_path):
        # load net from file
        from models import get_net_by_name
        net_config = json.load(open(net_config_path, 'r'))
        net = get_net_by_name(net_config['name']).build_from_config(net_config)
    else:
        # build net from args
        if 'proxyless' in args.net:
            from models.normal_nets.proxyless_nets import proxyless_base
            net_config_url = 'https://hanlab.mit.edu/files/proxylessNAS/%s.config' % args.net
            net = proxyless_base(
                net_config=net_config_url, n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout,
            )
        else:
            raise ValueError('do not support: %s' % args.net)

    # build run manager
    run_manager = RunManager(args.path, net, run_config, measure_latency=args.latency)
    run_manager.save_config(print_info=True)

    # load checkpoints
    best_model_path = '%s/checkpoint/model_best.pth.tar' % args.path
    if os.path.isfile(best_model_path):
        init_path = best_model_path
    else:
        init_path = '%s/init' % args.path

    if args.resume:
        run_manager.load_model()
        if args.train and run_manager.best_acc == 0:
            loss, acc1, acc5 = run_manager.validate(is_test=False, return_top5=True)
            run_manager.best_acc = acc1
    elif os.path.isfile(init_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(init_path)
        else:
            checkpoint = torch.load(init_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        run_manager.net.module.load_state_dict(checkpoint)
    elif 'proxyless' in args.net and not args.train:
        from utils.latency_estimator import download_url
        pretrained_weight_url = 'https://hanlab.mit.edu/files/proxylessNAS/%s.pth' % args.net
        print('Load pretrained weights from %s' % pretrained_weight_url)
        init_path = download_url(pretrained_weight_url)
        init = torch.load(init_path, map_location='cpu')
        net.load_state_dict(init['state_dict'])
    else:
        print('Random initialization')

    # train
    if args.train:
        print('Start training')
        run_manager.train(print_top5=True)
        run_manager.save_model()

    output_dict = {}
    # validate
    if run_config.valid_size:
        print('Test model on validation set')
        loss, acc1, acc5 = run_manager.validate(is_test=False, return_top5=True)
        log = 'valid_loss: %f\t valid_acc1: %f\t valid_acc5: %f' % (loss, acc1, acc5)
        run_manager.write_log(log, prefix='valid')
        output_dict = {
            **output_dict,
            'valid_loss': ' % f' % loss, 'valid_acc1': ' % f' % acc1, 'valid_acc5': ' % f' % acc5,
            'valid_size': run_config.valid_size
        }

    # test
    print('Test model on test set')
    loss, acc1, acc5 = run_manager.validate(is_test=True, return_top5=True)
    log = 'test_loss: %f\t test_acc1: %f\t test_acc5: %f' % (loss, acc1, acc5)
    run_manager.write_log(log, prefix='test')
    output_dict = {
        **output_dict,
        'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1, 'test_acc5': '%f' % acc5
    }
    json.dump(output_dict, open('%s/output' % args.path, 'w'), indent=4)

    # load best model, quantize, validate and test
    if args.quantize:
        bits_per_run = [int(i) for i in args.n_bits.split(",")]
        for n_bits in bits_per_run:
            print("Using %s bits for quantization." % n_bits)
            best_model_path = '%s/checkpoint/model_best.pth.tar' % args.path
            if os.path.isfile(best_model_path):
                if torch.cuda.is_available():
                    checkpoint = torch.load(best_model_path)
                else:
                    checkpoint = torch.load(best_model_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
            else:
                raise FileNotFoundError

            # fold batch norm and quantize model
            #checkpoint = fold_batch_norm(checkpoint)
            checkpoint = quantize_state_dict(checkpoint, n_bits)
            run_manager.net.module.load_state_dict(checkpoint)

            output_dict = {}
            # validate
            if run_config.valid_size:
                print('Test {} bit quantized model on validation set'.format(n_bits))
                loss, acc1, acc5 = run_manager.validate(is_test=False, return_top5=True)
                log = '%d bit quant valid_loss: %f\t valid_acc1: %f\t valid_acc5: %f' % (n_bits, loss, acc1, acc5)
                run_manager.write_log(log, prefix='valid')
                output_dict = {
                    **output_dict,
                    'valid_loss': ' % f' % loss, 'valid_acc1': ' % f' % acc1, 'valid_acc5': ' % f' % acc5,
                    'valid_size': run_config.valid_size
                }

            # test
            print('Test {} bit quantized model on test set'.format(n_bits))
            loss, acc1, acc5 = run_manager.validate(is_test=True, return_top5=True)
            log = '%d bit quant test_loss: %f\t test_acc1: %f\t test_acc5: %f' % (n_bits, loss, acc1, acc5)
            run_manager.write_log(log, prefix='test')
            output_dict = {
                **output_dict,
                'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1, 'test_acc5': '%f' % acc5
            }
            json.dump(output_dict, open('%s/output_quantized_%d_bit' % (args.path, n_bits), 'w'), indent=4)