import os, sys
import json

from torch.utils.data import DataLoader
import numpy as np

from .forget_dataloader import ForgetDataset
from .forgettotal_dataloader import ForgetTotalDataset
from .gikt_forget_dataloader import GIKTForgetDataset
from .milestone_dataloader import MilestoneDataset
# from .code_dataloader import CodeDataset
# from .problem_dataloader import ProblemDataset
# from .gikt_forget_dataloader import GIKTForgetDataset
from .mydkt_dataloader import MyDktDataset
from .data_loader import KTDataset
from .dkt_forget_dataloader import DktForgetDataset
from .atdkt_dataloader import ATDKTDataset
from .lpkt_dataloader import LPKTDataset
from .lpkt_utils import generate_time2idx
from .mydktdiff_dataloader import MyDktDiffDataset
from .problem_dataloader import ProblemDataset
from .que_data_loader import KTQueDataset
from pykt.config import que_type_models
from .dimkt_dataloader import DIMKTDataset


def init_test_datasets(data_config, model_name, batch_size, diff_level=None, forget_window=None):
    dataset_name = data_config["dataset_name"]
    print(f"model_name is {model_name}, dataset_name is {dataset_name}")
    test_question_loader, test_question_window_loader = None, None
    if model_name in ["dkt_forget", "bakt_time"]:
        test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]),
                                        data_config["input_type"], {-1})
        # test_window_dataset=None
        test_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                               data_config["input_type"], {-1})

        if "test_question_file" in data_config:
            test_question_dataset = DktForgetDataset(
                os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1},
                True)
            test_question_window_dataset = DktForgetDataset(
                os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"],
                {-1}, True)
    elif model_name in ["lpkt"]:
        print(f"model_name in lpkt")
        at2idx, it2idx = generate_time2idx(data_config)
        test_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]), at2idx,
                                   it2idx, data_config["input_type"], {-1})
        test_window_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                                          at2idx, it2idx, data_config["input_type"], {-1})
        test_question_dataset = None
        test_question_window_dataset = None
    elif model_name in ["rkt"] and dataset_name in ["statics2011", "assist2015", "poj"]:
        test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]),
                                 data_config["input_type"], {-1})
        test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                        data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]),
                                              data_config["input_type"], {-1}, True)
            test_question_window_dataset = KTDataset(
                os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"],
                {-1}, True)
    elif model_name in que_type_models:
        test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                    input_type=data_config["input_type"], folds=[-1],
                                    concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
        test_window_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                                           input_type=data_config["input_type"], folds=[-1],
                                           concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
        test_question_dataset = None
        test_question_window_dataset = None
    elif model_name in ["atdkt"]:
        test_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]),
                                    data_config["input_type"], {-1})
        test_window_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                           data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]),
                                                 data_config["input_type"], {-1}, True)
            test_question_window_dataset = ATDKTDataset(
                os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"],
                {-1}, True)
    elif model_name in ["dimkt"]:
        test_dataset = DIMKTDataset(data_config["dpath"], os.path.join(data_config["dpath"], data_config["test_file"]),
                                    data_config["input_type"], {-1}, diff_level=diff_level)
        test_window_dataset = DIMKTDataset(data_config["dpath"],
                                           os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                           data_config["input_type"], {-1}, diff_level=diff_level)
        if "test_question_file" in data_config:
            test_question_dataset = DIMKTDataset(data_config["dpath"],
                                                 os.path.join(data_config["dpath"], data_config["test_question_file"]),
                                                 data_config["input_type"], {-1}, True, diff_level=diff_level)
            test_question_window_dataset = DIMKTDataset(data_config["dpath"], os.path.join(data_config["dpath"],
                                                                                           data_config[
                                                                                               "test_question_window_file"]),
                                                        data_config["input_type"], {-1}, True, diff_level=diff_level)
    elif model_name == "gikt":
        test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                    input_type=data_config["input_type"], folds=[-1],
                                    concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
        test_window_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                                           input_type=data_config["input_type"], folds=[-1],
                                           concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
        test_question_dataset = None
        test_question_window_dataset = None
    elif model_name == "gikt_problem":
        '''
        test_dataset = ProblemDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                      data_config["input_type"], [-1])
        test_window_dataset = ProblemDataset(
            os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
            data_config["input_type"], [-1])
        test_question_dataset = None
        test_question_window_dataset = None
         '''

    elif model_name == "gikt_forget":
        '''
        test_dataset = GIKTForgetDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                         data_config["input_type"], [-1])
        test_window_dataset = GIKTForgetDataset(
            os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
            data_config["input_type"], [-1])
        test_question_dataset = None
        test_question_window_dataset = None
        '''
    elif model_name in ["mydkt", "dkt_forget_window", "dkt_forget_double", "dkt_forget_graph", "dkt_forget_predict",
                        ]:
        test_dataset = MyDktDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                    data_config["input_type"], folds=[-1], concept_num=data_config['num_c'],
                                    max_concepts=data_config['max_concepts'],
                                    forget_window=forget_window)
        test_window_dataset = None
        '''
        test_window_dataset = MyDktDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                                           data_config["input_type"], folds=[-1], concept_num=data_config['num_c'],
                                           max_concepts=data_config['max_concepts'],
                                           forget_window=forget_window)
        '''
        test_question_dataset = None
        test_question_window_dataset = None
    elif model_name in ["dkt_forget_description"]:
        test_dataset = MyDktDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                    data_config["input_type"], folds=[-1], concept_num=data_config['num_c'],
                                    max_concepts=data_config['max_concepts'], code_type="GPT2", problem_type="GPT2",
                                    forget_window=forget_window)
        test_window_dataset = None
        '''
        test_window_dataset = MyDktDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                                           data_config["input_type"], folds=[-1], concept_num=data_config['num_c'],
                                           max_concepts=data_config['max_concepts'],
                                           forget_window=forget_window)
        '''
        test_question_dataset = None
        test_question_window_dataset = None
    elif model_name in ["dkt_forget_diff"]:
        test_dataset = ForgetDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                     data_config["input_type"], folds=[-1], concept_num=data_config['num_c'],
                                     max_concepts=data_config['max_concepts'],
                                     forget_window=forget_window)
        test_window_dataset = None
        '''
        test_window_dataset = MyDktDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                                           data_config["input_type"], folds=[-1], concept_num=data_config['num_c'],
                                           max_concepts=data_config['max_concepts'],
                                           forget_window=forget_window)
        '''
        test_question_dataset = None
        test_question_window_dataset = None
    elif model_name in ["dkt_forget_total", "dkt_forget_ablation","dkt_forget_total_ablation"]:
        test_dataset = ForgetTotalDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                          data_config["input_type"], folds=[-1], concept_num=data_config['num_c'],
                                          max_concepts=data_config['max_concepts'],
                                          forget_window=forget_window)
        test_window_dataset = None
        '''
        test_window_dataset = MyDktDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                                           data_config["input_type"], folds=[-1], concept_num=data_config['num_c'],
                                           max_concepts=data_config['max_concepts'],
                                           forget_window=forget_window)
        '''
        test_question_dataset = None
        test_question_window_dataset = None
    elif model_name in ["dkt_Milestone","gkt_milestone","akt_milestone"]:
        test_dataset = MilestoneDataset(os.path.join(data_config["dpath"], data_config["test_file"]),
                                 data_config["input_type"], {-1},forget_window)
        #test_window_dataset = MilestoneDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
        #                                data_config["input_type"], {-1})
        test_window_dataset=None
        if "test_question_file" in data_config:
            test_question_dataset = MilestoneDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]),
                                              data_config["input_type"], {-1},forget_window, True)
            #test_question_window_dataset = MilestoneDataset(
            #    os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"],
            #    {-1}, forget_window, True)
            test_question_window_dataset=None
    elif model_name == "gkt":
        test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]),
                                 data_config["input_type"], {-1})
        #test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
        #                                data_config["input_type"], {-1})
        test_window_dataset=None
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]),
                                              data_config["input_type"], {-1}, True)
            #test_question_window_dataset = KTDataset(
            #    os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"],
            #    {-1}, True)
            test_question_window_dataset=None
    else:
        test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]),
                                 data_config["input_type"], {-1})
        test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                        data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]),
                                              data_config["input_type"], {-1}, True)
            test_question_window_dataset = KTDataset(
                os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"],
                {-1}, True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if not test_window_dataset is None:
        test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    # test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    if "test_question_file" in data_config:
        print(f"has test_question_file!")
        test_question_loader, test_question_window_loader = None, None
        if not test_question_dataset is None:
            test_question_loader = DataLoader(test_question_dataset, batch_size=batch_size, shuffle=False)
        if not test_question_window_dataset is None:
            test_question_window_loader = DataLoader(test_question_window_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader=None
    test_question_window_loader = None
    return test_loader, test_window_loader, test_question_loader, test_question_window_loader


def update_gap(max_rgap, max_sgap, max_pcount, cur):
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_sgap = cur.max_sgap if cur.max_sgap > max_sgap else max_sgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    return max_rgap, max_sgap, max_pcount


def update_gap_forget(max_rgap, max_sgap, max_pcount, max_acount, cur):
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_sgap = cur.max_sgap if cur.max_sgap > max_sgap else max_sgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    max_acount = cur.max_acount if cur.max_acount > max_acount else max_acount
    return max_rgap, max_sgap, max_pcount, max_acount


def update_total_forget(max_rgap, max_pcount, max_acount, cur):
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    max_acount = cur.max_acount if cur.max_acount > max_acount else max_acount
    return max_rgap, max_pcount, max_acount


def init_dataset4train(dataset_name, model_name, data_config, i, batch_size, code_type=None, problem_type=None,
                       forget_window=None, diff_level=None):
    print(f"dataset_name:{dataset_name}")
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"])
    if model_name in ["dkt_forget", "bakt_time"]:
        max_rgap, max_sgap, max_pcount = 0, 0, 0
        curvalid = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                                    data_config["input_type"], {i})
        curtrain = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                                    data_config["input_type"], all_folds - {i})
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curtrain)
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curvalid)
    elif model_name == "lpkt":
        at2idx, it2idx = generate_time2idx(data_config)
        # json_str = json.dumps(at2idx)
        # with open('at2idx.json', 'w') as json_file:
        #     json_file.write(json_str)
        # json_str_2 = json.dumps(it2idx)
        # with open('it2idx.json', 'w') as json_file2:
        #     json_file2.write(json_str_2)
        curvalid = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]), at2idx,
                               it2idx, data_config["input_type"], {i})
        curtrain = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]), at2idx,
                               it2idx, data_config["input_type"], all_folds - {i})
    elif model_name in ["rkt"] and dataset_name in ["statics2011", "assist2015", "poj"]:
        curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                             data_config["input_type"], {i})
        curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                             data_config["input_type"], all_folds - {i})
    elif model_name in que_type_models:
        curvalid = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds={i},
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
        curtrain = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds=all_folds - {i},
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
    elif model_name in ["atdkt"]:
        curvalid = ATDKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                                data_config["input_type"], {i})
        curtrain = ATDKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                                data_config["input_type"], all_folds - {i})
    elif model_name == "dimkt":
        curvalid = DIMKTDataset(data_config["dpath"],
                                os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                                data_config["input_type"], {i}, diff_level=diff_level)
        curtrain = DIMKTDataset(data_config["dpath"],
                                os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                                data_config["input_type"], all_folds - {i}, diff_level=diff_level)
    elif model_name == "gikt":
        curvalid = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds={i},
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
        curtrain = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds=all_folds - {i},
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
    # TODO 有问题

    elif model_name in ["gikt_code"]:
        '''
        curvalid = CodeDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                               data_config["input_type"], {i}, code_type)
        curtrain = CodeDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                               data_config["input_type"], all_folds - {i}, code_type)
        '''



    elif model_name in ["gikt_problem"]:
        curvalid = ProblemDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                  data_config["input_type"], {i}, code_type=code_type, problem_type=problem_type)
        curtrain = ProblemDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                  data_config["input_type"], all_folds - {i}, code_type=code_type,
                                  problem_type=problem_type)
    elif model_name == "gikt_forget":
        # TODO
        max_rgap, max_sgap, max_pcount = 0, 0, 0
        curvalid = GIKTForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                     data_config["input_type"], {i}, code_type=code_type, problem_type=problem_type,
                                     forget_window=forget_window)
        curtrain = GIKTForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                     data_config["input_type"], all_folds - {i}, code_type=code_type,
                                     problem_type=problem_type, forget_window=forget_window)
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curtrain)
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curvalid)
    elif model_name in ["mydkt", "dkt_forget_window", "dkt_forget_double", "dkt_forget_graph", "dkt_forget_predict",
                        "dkt_forget_description"]:
        # TODO
        max_rgap, max_sgap, max_pcount = 0, 0, 0
        curvalid = MyDktDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                data_config["input_type"], {i}, concept_num=data_config['num_c'],
                                max_concepts=data_config['max_concepts'],
                                code_type=code_type, problem_type=problem_type, forget_window=forget_window)
        curtrain = MyDktDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                data_config["input_type"], all_folds - {i}, concept_num=data_config['num_c'],
                                max_concepts=data_config['max_concepts'],
                                code_type=code_type, problem_type=problem_type, forget_window=forget_window)
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curtrain)
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curvalid)
    elif model_name in ["dkt_forget_diff"]:
        max_rgap, max_sgap, max_pcount, max_acount = 0, 0, 0, 0
        curvalid = ForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                 data_config["input_type"], {i}, concept_num=data_config['num_c'],
                                 max_concepts=data_config['max_concepts'],
                                 forget_window=forget_window)
        curtrain = ForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                 data_config["input_type"], all_folds - {i}, concept_num=data_config['num_c'],
                                 max_concepts=data_config['max_concepts'],
                                 forget_window=forget_window)
        max_rgap, max_sgap, max_pcount, max_acount = update_gap_forget(max_rgap, max_sgap, max_pcount, max_acount,
                                                                       curtrain)
        max_rgap, max_sgap, max_pcount, max_acount = update_gap_forget(max_rgap, max_sgap, max_pcount, max_acount,
                                                                       curvalid)
    elif model_name in ["dkt_forget_total", "dkt_forget_ablation",
                        "dkt_forget_total_ablation"]:
        max_rgap, max_pcount, max_acount = 0, 0, 0
        curvalid = ForgetTotalDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                      data_config["input_type"], {i}, concept_num=data_config['num_c'],
                                      max_concepts=data_config['max_concepts'],
                                      forget_window=forget_window)
        curtrain = ForgetTotalDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                      data_config["input_type"], all_folds - {i}, concept_num=data_config['num_c'],
                                      max_concepts=data_config['max_concepts'],
                                      forget_window=forget_window)
        max_rgap, max_pcount, max_acount = update_total_forget(max_rgap, max_pcount, max_acount,
                                                               curtrain)
        max_rgap, max_pcount, max_acount = update_total_forget(max_rgap, max_pcount, max_acount,
                                                               curvalid)
    elif model_name in ["dkt_Milestone","gkt_milestone","akt_milestone"]:
        max_rgap, max_pcount, max_acount = 0, 0, 0
        curvalid = MilestoneDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                             data_config["input_type"], {i}, forget_window=forget_window)
        curtrain = MilestoneDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                             data_config["input_type"], all_folds - {i},forget_window=forget_window)
        max_rgap, max_pcount, max_acount = update_total_forget(max_rgap, max_pcount, max_acount,
                                                               curtrain)
        max_rgap, max_pcount, max_acount = update_total_forget(max_rgap, max_pcount, max_acount,
                                                               curvalid)
    else:
        curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                             data_config["input_type"], {i})
        curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
                             data_config["input_type"], all_folds - {i})
    train_loader = DataLoader(curtrain, batch_size=batch_size)
    valid_loader = DataLoader(curvalid, batch_size=batch_size)

    try:
        if model_name in ["dkt_forget", "bakt_time"]:
            test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]),
                                            data_config["input_type"], {-1})
            # test_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
            #                                 data_config["input_type"], {-1})
            max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, test_dataset)
        # TODO
        elif model_name in ["gikt_forget"]:
            test_dataset = GIKTForgetDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                             data_config["input_type"], {-1}, code_type, problem_type, forget_window)
            max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, test_dataset)
        elif model_name in ["mydkt", "dkt_forget_window", "dkt_forget_double", "dkt_forget_graph",
                            "dkt_forget_predict", "dkt_forget_description",]:
            test_dataset = MyDktDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                        data_config["input_type"], {-1}, concept_num=data_config['num_c'],
                                        max_concepts=data_config['max_concepts'],
                                        code_type=code_type, problem_type=problem_type, forget_window=forget_window)
            max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, test_dataset)
        elif model_name in ["dkt_forget_diff"]:
            test_dataset = ForgetDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                         data_config["input_type"], {-1}, concept_num=data_config['num_c'],
                                         max_concepts=data_config['max_concepts'],
                                         forget_window=forget_window)
            max_rgap, max_sgap, max_pcount, max_acount = update_gap_forget(max_rgap, max_sgap, max_pcount, max_acount,
                                                                           test_dataset)
        elif model_name in ["dkt_forget_total", "dkt_forget_ablation",
                            "dkt_forget_total_ablation"]:
            test_dataset = ForgetTotalDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                              data_config["input_type"], {-1}, concept_num=data_config['num_c'],
                                              max_concepts=data_config['max_concepts'],
                                              forget_window=forget_window)
            max_rgap, max_pcount, max_acount = update_total_forget(max_rgap, max_pcount, max_acount, test_dataset)
        elif model_name in ["dkt_Milestone","gkt_milestone","akt_milestone"]:
            test_dataset = MilestoneDataset(os.path.join(data_config["dpath"], data_config["test_file"]),
                                        data_config["input_type"], {i}, forget_window=forget_window)
            max_rgap, max_pcount, max_acount = update_total_forget(max_rgap, max_pcount, max_acount,
                                                                   test_dataset)
    #     elif model_name == "lpkt":
    #         test_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), at2idx, it2idx, data_config["input_type"], {-1})
    #         # test_window_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), at2idx, it2idx, data_config["input_type"], {-1})
    #     elif model_name in que_type_models:
    #         test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
    #                         input_type=data_config["input_type"], folds=[-1], 
    #                         concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
    #     else:
    #         test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
    #         # test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
    except:
        pass

    if model_name in ["dkt_forget", "bakt_time", "gikt_forget", "mydkt", "dkt_forget_window", "dkt_forget_double",
                      "dkt_forget_graph", "dkt_forget_predict", "dkt_forget_description", "dkt_forget_diff"]:
        data_config["num_rgap"] = max_rgap + 1
        data_config["num_sgap"] = max_sgap + 1
        data_config["num_pcount"] = max_pcount + 1
    if model_name in ["dkt_forget_diff"]:
        data_config["num_rgap"] = max_rgap + 1
        data_config["num_sgap"] = max_sgap + 1
        data_config["num_pcount"] = max_pcount + 1
        data_config["num_acount"] = max_acount + 1
    if model_name in ["dkt_forget_total", "dkt_forget_ablation",
                      "dkt_Milestone","gkt_milestone","akt_milestone",
                      "dkt_forget_total_ablation"]:
        data_config["num_rgap"] = max_rgap + 1
        data_config["num_pcount"] = max_pcount + 1
        data_config["num_acount"] = max_acount + 1
    if model_name == "lpkt":
        print(f"num_at:{len(at2idx)}")
        print(f"num_it:{len(it2idx)}")
        data_config["num_at"] = len(at2idx) + 1
        data_config["num_it"] = len(it2idx) + 1
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # # test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    # test_window_loader = None
    return train_loader, valid_loader  # , test_loader, test_window_loader
