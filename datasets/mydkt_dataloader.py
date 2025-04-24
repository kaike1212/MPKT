from ast import Assert
import os, sys
from re import L
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
import numpy as np
#from models.models.CodeBERTClassifier import CodeBERTClassifier
#from models.models.GPT2Classifier import GPT2Classifier

delimiter = '#CODE_SNIPPET#'
problem_delimiter = '#PROBLEM_SNIPPET#'

# TODO 修改forget_window 为None 时的问题
class MyDktDataset(Dataset):
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

    def __init__(self, file_path, input_type, folds, concept_num, max_concepts, code_type=None,
                 problem_type=None, forget_window=(1/24,1,7,30), qtest=False):
        super(MyDktDataset, self).__init__()
        self.sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        self.code_type = code_type
        self.problem_type = problem_type
        self.forget_window = forget_window
        folds = list(folds)
        folds_str = "_" + "_".join([str(_) for _ in folds])
        processed_data = file_path + folds_str + "_myforget.pkl"
        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")

            self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount = self.__load_data__(
                self.sequence_path, folds)
            save_data = [self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount]
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount = pd.read_pickle(processed_data)
        # TODO 使dcode和describe分开
        if code_type is not None:
            code_data = file_path + folds_str + "_" + code_type + ".pkl"
            if not os.path.exists(code_data):
                print(f"Start code_preprocessing {file_path} fold: {folds_str}...")
                self.dcode, self.describe = self.__code_data__(self.sequence_path, folds, code_type, problem_type)
                save_data = [self.dcode, self.describe]
                pd.to_pickle(save_data, code_data)
            else:
                print(f"Read code_data from processed file: {code_data}")
                self.dcode, self.describe = pd.read_pickle(code_data)

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
            if key == 'cseqs':
                seqs = dorri[key][:-1, :]
                shft_seqs = dorri[key][1:, :]
            else:
                seqs = dorri[key][:-1] * mseqs
                shft_seqs = dorri[key][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = dorri["smasks"]


        dcurgaps = dict()
        for key, tensor in self.dgaps.items():
            tensor = tensor[index]
            tensor = torch.LongTensor(tensor)
            if key == "sgaps":
                seqs = tensor[:-1] * mseqs
                shft_seqs = tensor[1:] * mseqs
            else:
                seqs = tensor[:-1, :]
                shft_seqs = tensor[1:, :]
            dcurgaps[key] = seqs
            dcurgaps["shft_" + key] = shft_seqs

        dcode, describe = torch.empty(()), torch.empty(())
        '''
        dcode = self.dcode[index]
        dcode = torch.cat(dcode, dim=0)
        describe = self.describe[index]
        describe = torch.cat(describe, dim=0)
        '''

        return dcur, dcurgaps, dcode, describe

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

    def __load_data__(self, sequence_path, folds,  pad_val=-1):
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

        dgaps = {"rgaps": [], "sgaps": [], "pcounts": [], "acounts": []}
        max_rgap, max_sgap, max_pcount = 0, 0, 0
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
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills + [-1] * (self.max_concepts - len(skills))
                    row_skills.append(skills)
                dori["cseqs"].append(row_skills)
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            # 添加序列时间间隔rgap、知识时间间隔sgap、当前知识掌握状态h以及间隔时间内的成功和失败次数
            rgap, sgap, past_counts, apast_counts = self.calC(row)
            dgaps["rgaps"].append(rgap)
            dgaps["sgaps"].append(sgap)
            dgaps["pcounts"].append(past_counts)
            dgaps["acounts"].append(apast_counts)
            max_rgap = max(max(value for sublist in rgap for value in sublist), max_rgap)
            max_sgap = max(sgap) if max(sgap) > max_sgap else max_sgap
            max_pcount = max(max(value for sublist1 in past_counts for sublist2 in sublist1 for value in sublist2),max_pcount)
            interaction_num += dori["smasks"][-1].count(1)

        for key in dori:
            if key not in ["rseqs"]:  # in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:, :-1] != pad_val) * (dori["rseqs"][:, 1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        # TODO
        '''
        torch.cuda.empty_cache()
        batch_size = 200
        for key in dgaps:
            num_samples = len(dgaps[key])
            for i in range(0, num_samples, batch_size):
                batch_data = dgaps[key][i:i + batch_size]
                dgaps[key][i:i + batch_size] = torch.LongTensor(batch_data)
        '''

        # dgaps=self.convert_to_longtensor(dgaps)

        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        return dori, dgaps, max_rgap, max_sgap, max_pcount

    def log2(self, t):
        import math
        return round(math.log(t + 1, 2))

    # 添加序列时间间隔rgap、知识时间间隔sgap、当前知识掌握状态h以及间隔时间内的成功和失败次数
    def calC(self, row):
        repeated_gap, sequence_gap, past_counts = [], [], []
        uid = row["uid"]
        # default: concepts
        # skills = row["concepts"].split(",") if "concepts" in self.input_type else row["questions"].split(",")
        # 算技能
        row_skills = []
        raw_skills = row["concepts"].split(",")
        for concept in raw_skills:
            if concept == "-1":
                skills = [-1] * self.max_concepts
            else:
                skills = [int(_) for _ in concept.split("_")]
                skills = skills + [-1] * (self.max_concepts - len(skills))
            row_skills.append(skills)

        timestamps = row["timestamps"].split(",")
        responses = row["responses"].split(",")
        dlastskill, dcount = dict(), dict()
        pret = None

        for row_skill, t in zip(row_skills, timestamps):
            t = int(t)
            rp, sp, pc = [], [], []
            for s in row_skill:
                s = int(s)
                # 上次学习技能的间隔
                if s not in dlastskill or s == -1:
                    curRepeatedGap = 0
                else:
                    curRepeatedGap = self.log2((t - dlastskill[s]) / 1000 / 60) + 1  # minutes
                dlastskill[s] = t
                rp.append(curRepeatedGap)
                # 上次学习的时间
            if pret == None or t == -1:
                curLastGap = 0
            else:
                curLastGap = self.log2((t - pret) / 1000 / 60) + 1
            pret = t
            # sp.append(curLastGap)

            '''
                dcount.setdefault(s, 0)
                pc.append(self.log2(dcount[s]))
                dcount[s] += 1
            '''

            repeated_gap.append(rp)
            sequence_gap.append(curLastGap)
            # past_counts.append(pc)

        past_counts, apast_counts = self.Cal_interval(row_skills, timestamps, responses)
        return repeated_gap, sequence_gap, past_counts, apast_counts

    def Cal_interval(self, skills, timestamps, responses):
        # 当前第i个,技能,间隔
        apast_counts, past_counts = [], []
        adskill, dskill = dict(), dict()
        forget_window_milliseconds = [hours * 3600000 for hours in self.forget_window]
        a, b, c = self.max_concepts, len(forget_window_milliseconds) + 1, 2
        for row_skills, t, r in zip(skills, timestamps, responses):
            t, r = int(t), int(r)
            pc = [[[0 for _ in range(c)] for _ in range(b)] for _ in range(a)]
            i, j = 0, 0
            for s in row_skills:
                j = 0
                s = int(s)
                if s not in dskill or s == -1:
                    dskill[s] = []
                if s not in adskill or s == -1:
                    adskill[s] = []

                for fw in forget_window_milliseconds:
                    count = sum(1 for value in dskill[s] if t - fw <= value)
                    acount = sum(1 for value in adskill[s] if t - fw <= value)
                    pc[i][j][0] = self.log2(count)
                    pc[i][j][1] = self.log2(acount)
                    j = j + 1
                pc[i][j][0], pc[i][j][1] = self.log2(len(dskill[s])), self.log2(len(adskill[s]))
                # 添加s
                if s != -1:
                    dskill[s].append(t)
                    if r == 1:
                        adskill[s].append(t)
                i = i + 1
            past_counts.append([[pc[i][j][0] for j in range(b)] for i in range(a)])
            apast_counts.append([[pc[i][j][1] for j in range(b)] for i in range(a)])
        #之前遇见这个数的次数log2()
        return past_counts, apast_counts
