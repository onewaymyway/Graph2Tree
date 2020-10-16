# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
from src.pre_data import *
import time
import torch.optim
from src.expressions_transfer import *
import json
import argparse
import ast
from aiutils.fileutils import readJsonLines
from src.data_utils import create_group, split_by_lens

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--data_set_type", type=str, default="math23k", help="data_set_type")
parser.add_argument("--model_root", type=str, default="model_traintest", help="model_root")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number in batch for training.")
parser.add_argument("--num_epoch", type=int, default=80, help="Number of epoches for fine-tuning.")
parser.add_argument("--if_train", type=ast.literal_eval, default=True, help="if_train")
parser.add_argument("--if_eval", type=ast.literal_eval, default=True, help="if_eval")
parser.add_argument("--do_recover_lang", type=ast.literal_eval, default=False, help="do_recover_lang")
parser.add_argument("--do_save_lang", type=ast.literal_eval, default=True, help="do_save_lang")
args = parser.parse_args()

print("args", args)


def read_json(path):
    with open(path, 'r') as f:
        file = json.load(f)
    return file


batch_size = 64
batch_size = args.batch_size
embedding_size = 128
hidden_size = 512
n_epochs = args.num_epoch
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'

data_set_type = args.data_set_type

model_root = args.model_root

if_train = args.if_train
if_eval = args.if_eval

do_recover_lang = args.do_recover_lang
do_save_lang = args.do_save_lang


def get_train_test_fold(ori_path, prefix, data, pairs, group):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item, pair, g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold


def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a / b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1]) / 100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num


def load_pre_params():
    print("load_pre_params", USE_CUDA)
    if USE_CUDA:
        encoder.load_state_dict(torch.load(model_root + "/encoder"))
        predict.load_state_dict(torch.load(model_root + "/predict"))
        generate.load_state_dict(torch.load(model_root + "/generate"))
        merge.load_state_dict(torch.load(model_root + "/merge"))
    else:
        encoder.load_state_dict(torch.load(model_root + "/encoder", map_location=torch.device('cpu')))
        predict.load_state_dict(torch.load(model_root + "/predict", map_location=torch.device('cpu')))
        generate.load_state_dict(torch.load(model_root + "/generate", map_location=torch.device('cpu')))
        merge.load_state_dict(torch.load(model_root + "/merge", map_location=torch.device('cpu')))


def save_params():
    print("save_params")
    torch.save(encoder.state_dict(), model_root + "/encoder")
    torch.save(predict.state_dict(), model_root + "/predict")
    torch.save(generate.state_dict(), model_root + "/generate")
    torch.save(merge.state_dict(), model_root + "/merge")


def build_math23k_data():
    print("build_math23k_data")
    # data = load_raw_data("data/Math_23K.json")

    data_set = {}
    group_data = read_json("data/Math_23K_processed.json")

    data = load_raw_data("data/Math_23K.json")

    pairs, generate_nums, copy_nums = transfer_num(data)

    temp_pairs = []
    for p in pairs:
        # [句子，表达式，数字列表,数字起始位置]
        temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
    pairs = temp_pairs

    train_fold, test_fold, valid_fold = get_train_test_fold(ori_path, prefix, data, pairs, group_data)

    data_set["generate_nums"] = generate_nums
    data_set["copy_nums"] = copy_nums
    data_set["train_fold"] = train_fold
    data_set["test_fold"] = test_fold
    data_set["valid_fold"] = valid_fold

    return data_set


def build_ape_data():
    print("build_ape_data")
    # data = load_raw_data("data/Math_23K.json")

    data_set = {}

    tests = readJsonLines("data_ape/test.ape.json")
    trains = readJsonLines("data_ape/train.ape.json")
    valids = readJsonLines("data_ape/valid.ape.json")

    data = tests + trains + valids

    pairs, generate_nums, copy_nums = transfer_num(data,1000)
    generate_nums=["1","3.14","2","12","60","10","100","1000","7"]

    temp_pairs = []
    for p in pairs:
        # [句子，表达式，数字列表,数字起始位置]
        temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3], create_group(p[0], p[3])))
    pairs = temp_pairs

    train_fold, test_fold, valid_fold = split_by_lens(temp_pairs, [len(trains), len(tests), len(valids)])

    data_set["generate_nums"] = generate_nums
    data_set["copy_nums"] = copy_nums
    data_set["train_fold"] = train_fold
    data_set["test_fold"] = test_fold
    data_set["valid_fold"] = valid_fold

    return data_set


if data_set_type == "ape":
    data_set = build_ape_data()
else:
    data_set = build_math23k_data()

generate_nums = data_set["generate_nums"]
copy_nums = data_set["copy_nums"]
train_fold = data_set["train_fold"]
test_fold = data_set["test_fold"]
valid_fold = data_set["valid_fold"]

best_acc_fold = []

pairs_tested = test_fold
# pairs_trained = valid_fold
pairs_trained = train_fold

# for fold_t in range(5):
#    if fold_t == fold:
#        pairs_tested += fold_pairs[fold_t]
#    else:
#        pairs_trained += fold_pairs[fold_t]
model_info = ModelInfo()
model_info.set_base_path(model_root)

if do_recover_lang:
    input_lang, output_lang = model_info.recover_lang()
    generate_nums = model_info.generate_nums
    copy_nums = model_info.copy_nums
else:
    input_lang, output_lang = model_info.build_lang(pairs_trained, 5, generate_nums, copy_nums, True)

    if do_save_lang:
        model_info.save_lang()

#print("out_lang",output_lang)

print("build train_pairs")
train_pairs = model_info.build_langed_pairs(pairs_trained, "train")
print("build test_pairs")
test_pairs = model_info.build_langed_pairs(pairs_tested, "test")
print("build pairs complete")
# input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,copy_nums, tree=True)

# print('train_pairs[0]')
# print(train_pairs[0])
# exit()
# Initialize models

op_nums = output_lang.n_words - copy_nums - 1 - len(generate_nums)
input_size = input_lang.n_words

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])


def build_models(model_info):
    model_info.encoder = EncoderSeq(input_size=input_size, embedding_size=embedding_size, hidden_size=hidden_size,
                                    n_layers=n_layers)
    model_info.predict = Prediction(hidden_size=hidden_size, op_nums=op_nums,
                                    input_size=len(generate_nums))
    model_info.generate = GenerateNode(hidden_size=hidden_size, op_nums=op_nums,
                                       embedding_size=embedding_size)
    model_info.merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    model_info.encoder_optimizer = torch.optim.Adam(model_info.encoder.parameters(), lr=learning_rate,
                                                    weight_decay=weight_decay)
    model_info.predict_optimizer = torch.optim.Adam(model_info.predict.parameters(), lr=learning_rate,
                                                    weight_decay=weight_decay)
    model_info.generate_optimizer = torch.optim.Adam(model_info.generate.parameters(), lr=learning_rate,
                                                     weight_decay=weight_decay)
    model_info.merge_optimizer = torch.optim.Adam(model_info.merge.parameters(), lr=learning_rate,
                                                  weight_decay=weight_decay)

    model_info.encoder_scheduler = torch.optim.lr_scheduler.StepLR(model_info.encoder_optimizer, step_size=20,
                                                                   gamma=0.5)
    model_info.predict_scheduler = torch.optim.lr_scheduler.StepLR(model_info.predict_optimizer, step_size=20,
                                                                   gamma=0.5)
    model_info.generate_scheduler = torch.optim.lr_scheduler.StepLR(model_info.generate_optimizer, step_size=20,
                                                                    gamma=0.5)
    model_info.merge_scheduler = torch.optim.lr_scheduler.StepLR(model_info.merge_optimizer, step_size=20, gamma=0.5)

    return model_info


build_models(model_info)
print("build model complete")

encoder = model_info.encoder
predict = model_info.predict
generate = model_info.generate
merge = model_info.merge

encoder_optimizer = model_info.encoder_optimizer
predict_optimizer = model_info.predict_optimizer
generate_optimizer = model_info.generate_optimizer
merge_optimizer = model_info.merge_optimizer

encoder_scheduler = model_info.encoder_scheduler
predict_scheduler = model_info.predict_scheduler
generate_scheduler = model_info.generate_scheduler
merge_scheduler = model_info.merge_scheduler

if not if_train:
    load_pre_params()

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()


def do_eval():
    print("do_eval")
    value_ac = 0
    equation_ac = 0
    eval_total = 0
    start = time.time()
    for test_batch in test_pairs:
        # print(test_batch)
        batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4],
                                               test_batch[5])
        test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                 merge, output_lang, test_batch[5], batch_graph, beam_size=beam_size)
        val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4],
                                                          test_batch[6])
        if val_ac:
            value_ac += 1
        if equ_ac:
            equation_ac += 1
        eval_total += 1
    print(equation_ac, value_ac, eval_total)
    print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
    print("testing time", time_since(time.time() - start))

    return equation_ac, value_ac, eval_total

def batch_creator(datas,batchsize):
    start=0
    total=len(datas)
    while start<total:
        yield datas[start:start+batchsize]
        start+=batchsize

def do_train():
    for epoch in range(n_epochs):
        # encoder_scheduler.step()
        # predict_scheduler.step()
        # generate_scheduler.step()
        # merge_scheduler.step()


        print("epoch:", epoch + 1)


        step_batch_count=50
        step_count=step_batch_count*batch_size

        train_big_batch=batch_creator(train_pairs,step_count)

        train_step=0
        total_step=len(train_pairs)

        for train_small in train_big_batch:

            train_step+=step_count

            start = time.time()
            loss_total = 0

            print("prepare big_batch data")
            input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
                num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches = prepare_train_batch(
                    train_small, batch_size)

            totallen = len(input_lengths)
            print("start big_batch train")
            for idx in range(len(input_lengths)):
                loss = train_tree(
                    input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                    num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
                    encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang,
                    num_pos_batches[idx], graph_batches[idx])
                loss_total += loss
                if idx % 5 == 0:
                    print("progress", idx, totallen, idx / totallen)

            print("loss:", loss_total / len(input_lengths))
            print("training time", time_since(time.time() - start))
            print("progress_big:",train_step,total_step,train_step/total_step)
            print("--------------------------------")
        if epoch % 2 == 0 or epoch > n_epochs - 5:

            equation_ac, value_ac, eval_total = do_eval()
            print("------------------------------------------------------")
            save_params()
            if epoch == n_epochs - 1:
                best_acc_fold.append((equation_ac, value_ac, eval_total))

    a, b, c = 0, 0, 0
    for bl in range(len(best_acc_fold)):
        a += best_acc_fold[bl][0]
        b += best_acc_fold[bl][1]
        c += best_acc_fold[bl][2]
        print(best_acc_fold[bl])
    print(a / float(c), b / float(c))


if if_train:
    do_train()

if if_eval:
    do_eval()
