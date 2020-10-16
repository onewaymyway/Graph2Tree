import re
from aiutils.fileutils import readJsonLines

num_pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")


def text_to_input(seg):
    nums = []
    input_seq = []
    for s in seg:
        pos = re.search(num_pattern, s)
        if pos and pos.start() == 0:
            nums.append(s[pos.start(): pos.end()])
            input_seq.append("NUM")
            if pos.end() < len(s):
                input_seq.append(s[pos.end():])
        else:
            input_seq.append(s)

    nums_fraction = []

    # print("nums",nums)
    # print("input_seq",input_seq)

    for num in nums:
        if re.search("\d*\(\d+/\d+\)\d*", num):
            nums_fraction.append(num)
    nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

    num_pos = []
    for i, j in enumerate(input_seq):
        if j == "NUM":
            num_pos.append(i)
    # print(nums,num_pos)
    assert len(nums) == len(num_pos)

    return input_seq, nums, num_pos, nums_fraction


def equation_to_input(equations, nums, nums_fraction):
    def seg_and_tag(st):  # seg the equation and tag the num
        res = []
        for n in nums_fraction:
            if n in st:
                p_start = st.find(n)
                p_end = p_start + len(n)
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                if nums.count(n) == 1:
                    res.append("N" + str(nums.index(n)))
                else:
                    res.append(n)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
        pos_st = re.search("\d+\.\d+%?|\d+%?", st)
        if pos_st:
            p_start = pos_st.start()
            p_end = pos_st.end()
            if p_start > 0:
                res += seg_and_tag(st[:p_start])
            st_num = st[p_start:p_end]
            if nums.count(st_num) == 1:
                res.append("N" + str(nums.index(st_num)))
            else:
                res.append(st_num)
            if p_end < len(st):
                res += seg_and_tag(st[p_end:])
            return res
        for ss in st:
            res.append(ss)
        return res

    out_seq = seg_and_tag(equations)

    return out_seq


def datas_to_input():
    pass


def get_num_stack(equation, numbers):
    num_stack = []
    for word in equation:
        temp_num = []
        flag_not = True
        if word not in output_lang.index2word:
            flag_not = False
            for i, j in enumerate(numbers):
                if j == word:
                    temp_num.append(i)

        if not flag_not and len(temp_num) != 0:
            num_stack.append(temp_num)
        if not flag_not and len(temp_num) == 0:
            num_stack.append([_ for _ in range(len(pair[2]))])

    num_stack.reverse()
    return num_stack


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res


def create_group(input_seq, num_pos):
    group = []
    d = 3
    for i in num_pos:
        spos = i - d
        for p in range(0, 2 * d + 1):
            p = p + spos
            if p > 0 and p < len(input_seq):
                if p not in group:
                    group.append(p)
    return group


'''
(
['镇海', '雅乐', '学校', '二年级', '的', '小朋友', '到', '一条', '小路', '的', '一边', '植树', '．', '小朋友', '们', '每隔', 'NUM', '米', '种', '一棵树', '（', '马路', '两头', '都', '种', '了', '树', '）', '，', '最后', '发现', '一共', '种', '了', 'NUM', '棵', '，', '这', '条', '小路', '长', '多少', '米', '．'],
['*', '-', 'N1', '1', 'N0'],
['2', '11'],
[16, 34], 
[15, 16, 17, 32, 33, 34, 39, 40, 41]
)

input_seq [2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 11, 12, 13, 7, 14, 15, 1, 16, 17, 18, 19, 20, 21, 22, 17, 23, 24, 25, 26, 27, 28, 29, 17, 23, 1, 30, 26, 31, 32, 10, 33, 34, 16, 13], 
len(input_seq) 44, 
output_seq [0, 1, 8, 5, 7], 
len(output_seq) 5, 
numbers ['2', '11'], 
num_pos [16, 34], 
num_stack [], 
group_num [15, 16, 17, 32, 33, 34, 39, 40, 41]
'''
'''
{
"id": "636507", 
"segmented_text": "有 李 树 5 棵 ， 每 棵 产 李 子 60.8 千 克 ， 桃 树 8 棵 ， 每 棵 产 桃 子 47.5 千 克 ， 收 获 哪 种 水 果 比 较 重 ？ 比 另 一 种 重 多 少 千 克 ？", 
"original_text": "有李树5棵，每棵产李子60.8千克，桃树8棵，每棵产桃子47.5千克，收获哪种水果比较重？比另一种重多少千克？", 
"ans": "76", 
"equation": "x=(47.5*8)-(60.8*5)"}
'''


def ape_file_to_data(file_path):
    lines = readJsonLines(file_path)
    rst = []
    for line in lines:
        seg = line["segmented_text"]
        equation = line["equation"][2:]


def split_by_lens(arr, lens):
    rst = []
    start = 0
    for tlen in lens:
        tarr = arr[start:start + tlen]
        start += tlen
        rst.append(tarr)
    return rst
