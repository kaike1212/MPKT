from ast import Assert
import os, sys
from re import L
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
import numpy as np

# from models.models.CodeBERTClassifier import CodeBERTClassifier
# from models.models.GPT2Classifier import GPT2Classifier

# TODO 修改forget_window 为None 时的问题
class MilestoneDataset(Dataset):
    """Dataset for dkt_forget
        can use to init dataset for: dkt_forget
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """

    def __init__(self, file_path, input_type, folds, forget_window=(1 / 24, 1, 7, 30), qtest=False):
        super(MilestoneDataset, self).__init__()
        self.sequence_path = file_path
        self.input_type = input_type
        self.forget_window = forget_window
        self.qtest=qtest
        folds = list(folds)
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + str(forget_window) + "_milestone_qtest.pkl"
        else:
            processed_data = file_path + folds_str + str(forget_window) + "_milestone.pkl"
        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dgaps, self.max_rgap, self.max_pcount, self.max_acount ,self.dqtest = self.__load_data__(
                    self.sequence_path, folds)
                save_data = [self.dori, self.dgaps, self.max_rgap, self.max_pcount, self.max_acount,self.dqtest ]
            else:
                self.dori, self.dgaps, self.max_rgap, self.max_pcount, self.max_acount = self.__load_data__(
                    self.sequence_path, folds)
                save_data = [self.dori, self.dgaps, self.max_rgap, self.max_pcount, self.max_acount]
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dgaps, self.max_rgap, self.max_pcount, self.max_acount,self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori, self.dgaps, self.max_rgap, self.max_pcount, self.max_acount = pd.read_pickle(processed_data)

        print(
            f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.dori['rseqs'])

    # TODO
    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:

           - ** q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        # q_seqs, qshft_seqs, c_seqs, cshft_seqs = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        mseqs = self.dori["masks"][index]
        dorri = dict()
        for key, tensor in self.dori.items():
            if torch.numel(tensor) == 0:
                dorri[key] = torch.empty_like(tensor)
            else:
                dorri[key] = tensor[index]
        dcur = dict()
        for key, tensor in dorri.items():
            if key in ["masks", "smasks"]:
                continue
            if torch.numel(tensor) == 0:
                dcur[key] = dorri[key]
                dcur["shft_" + key] = dorri[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = dorri["smasks"]

        dcurgaps = dict()
        for key, tensor in self.dgaps.items():
            tensor = tensor[index]
            tensor = torch.LongTensor(tensor)
            seqs = tensor[:-1]
            shft_seqs = tensor[1:]
            dcurgaps[key] = seqs
            dcurgaps["shft_" + key] = shft_seqs

        dcode, describe = torch.empty(()), torch.empty(())
        '''
        dcode = self.dcode[index]
        dcode = torch.cat(dcode, dim=0)
        describe = self.describe[index]
        describe = torch.cat(describe, dim=0)
        '''
        if not self.qtest:
            return dcur, dcurgaps, dcode, describe
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dcurgaps, dcode, describe, dqtest

    def __code_data__(self, sequence_path, folds, code_type="CodeBERT", problem_type="GPT2"):
        sequence_path = sequence_path[:-4] + '.pt'
        tensor_dict = torch.load(sequence_path)
        code, describe = [], []
        code = tensor_dict["code"]
        describe = tensor_dict["describe"]
        now_folds = tensor_dict["folds"]
        target_codes = []
        target_describe = []
        for fold in now_folds:
            if fold.item() in folds:
                index = folds.index(fold.item())
                target_codes.append(code[index])
                target_describe.append(describe[index])

        return target_codes, target_describe

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]):
            pad_val (int, optional): pad value. Defaults to -1.
        Returns:
            (tuple): tuple containing
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}

        dgaps = {"rgaps": [], "pcounts": [], "acounts": []}
        max_rgap, max_pcount, max_acount = 0, 0, 0
        # seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)  # [0:1000]
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests": [], "orirow": []}
        # 找出最大值

        for i, row in df.iterrows():
            # use kc_id or question_id as input
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            # 添加知识时间间隔rgap、序列时间间隔sgap、当前知识掌握状态h以及间隔时间内的成功和失败次数
            rgap, past_counts, apast_counts = self.calC(row)
            dgaps["rgaps"].append(rgap)
            dgaps["pcounts"].append(past_counts)
            dgaps["acounts"].append(apast_counts)
            # TODO max_rgap
            max_rgap = max(rgap) if max(rgap) > max_rgap else max_rgap
            max_pcount = max(max(lst) for lst in past_counts)
            max_acount = max(max(lst) for lst in apast_counts)
            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        for key in dori:
            if key not in ["rseqs"]:  # in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:, :-1] != pad_val) * (dori["rseqs"][:, 1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)

        # dgaps=self.convert_to_longtensor(dgaps)

        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])
        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            return dori, dgaps, max_rgap, max_pcount, max_acount, dqtest
        return dori, dgaps, max_rgap, max_pcount, max_acount

    def log2(self, t):
        import math
        return round(math.log(t + 1, 2))

    # 添加序列时间间隔rgap、知识时间间隔sgap、当前知识掌握状态h以及间隔时间内的成功和失败次数
    def calC(self, row):
        repeated_gap, past_counts, past_acounts = [], [], []
        uid = row["uid"]
        # default: concepts
        skills = row["concepts"].split(",") if "concepts" in self.input_type else row["questions"].split(",")
        timestamps = row["timestamps"].split(",")
        responses = row["responses"].split(",")
        dlastskill, dcount = dict(), dict()


        for s, t, r in zip(skills, timestamps, responses):
            t = int(t)
            r = int(r)
            s = int(s)
            # 上次学习技能的间隔
            if s not in dlastskill or s == -1:
                curRepeatedGap = 0
            else:
                curRepeatedGap = self.log2((t - dlastskill[s]) / 1000 / 60) + 1  # minutes
            if r == 1:
                dlastskill[s] = t
            repeated_gap.append(curRepeatedGap)
            # past_counts.append(pc)

        past_counts, apast_counts = self.Cal_interval(skills, timestamps, responses)
        return repeated_gap, past_counts, apast_counts

    def Cal_interval(self, skills, timestamps, responses):
        # 当前第i个,技能,间隔
        apast_counts, past_counts = [], []
        adskill, dskill = dict(), dict()
        forget_window_milliseconds = [hours * 3600000 * 24 for hours in self.forget_window]
        b, c = len(forget_window_milliseconds) + 1, 2
        for s, t, r in zip(skills, timestamps, responses):
            t, r = int(t), int(r)
            pc = [[0 for _ in range(c)] for _ in range(b)]
            i, j = 0, 0

            s = int(s)
            if s not in dskill or s == -1:
                dskill[s] = []
            if s not in adskill or s == -1:
                adskill[s] = []

            for fw in forget_window_milliseconds:
                count = sum(1 for value in dskill[s] if t - fw <= value)
                acount = sum(1 for value in adskill[s] if t - fw <= value)
                if j == 0:
                    pc[j][0] = self.log2(count)
                    pc[j][1] = self.log2(acount)
                else:
                    pc[j][0] = self.log2(count - pc[j - 1][0])
                    pc[j][1] = self.log2(acount - pc[j - 1][1])
                j = j + 1
            pc[j][0], pc[j][1] = self.log2(len(dskill[s]) - pc[j - 1][0]), self.log2(
                    len(adskill[s]) - pc[j - 1][1])
                # 添加s
            if s != -1:
                dskill[s].append(t)
                if r == 1:
                    adskill[s].append(t)
            past_counts.append([pc[j][0] for j in range(b)])
            apast_counts.append([pc[j][1] for j in range(b)])
        # 之前遇见这个数的次数log2()
        return past_counts, apast_counts
