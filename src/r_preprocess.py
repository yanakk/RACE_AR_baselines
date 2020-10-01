'''
date:   30/09/2020
MODE: 0-Passage     1-Rationale
'''

import os
import json
from nltk import sent_tokenize, word_tokenize, download
import copy
import utils

mode = 1
iteration = 10   # Num of rationales

def tokenize(st):
    #TODO: The tokenizer's performance is suboptimal
    ans = []
    for sent in sent_tokenize(st):
        ans += word_tokenize(sent)
    return " ".join(ans).lower()

def gen_rationale(iteration, dup_ratio, a_words, rationale):
    # TODO: Use rationales to replace the whole passage
    rpt = 0
    for j in range(iteration):
        r_id = dup_ratio.index(max(dup_ratio))
        if rationale.count(a_words[r_id]) == 0:
            rationale.append(a_words[r_id])  # no repeat
        else:
            rpt += 1  # back
        # print(j, r_id, dup_ratio[r_id], a_words[r_id])
        del dup_ratio[r_id]; del a_words[r_id]
    return rationale, rpt

if __name__ == "__main__":

    # difficulty_set = ["middle", "high"]
    # class_set = ["train", "dev", "test"]
    # difficulty_set = ["normal"]
    class_set = ["Fact", "Mainly", "GPurpose", "LPurpose", "Local","Global", "Title"]

    #raw_data = "../data/RACE"
    raw_data = "..\\..\\ye\\LabeledFiles"

    if mode == 0:
        # PASSAGE 
        data = "../data/new_r_rationale"
        cnt = 0
        avg_article_length = 0
        avg_question_length = 0
        avg_option_length = 0
        num_que = 0
        for data_set in class_set:
            p1 = os.path.join(data, data_set)
            if not os.path.exists(p1):
                os.mkdir(p1)
            # for d in difficulty_set:
            new_data_path = os.path.join(data, data_set)
            if not os.path.exists(new_data_path):
                os.mkdir(new_data_path)
            new_raw_data_path = os.path.join(raw_data, data_set)
            for inf in os.listdir(new_raw_data_path):
                print(inf)
                cnt += 1
                obj = json.load(open(os.path.join(new_raw_data_path, inf), "r"))
                obj["article"] = obj["article"].replace("\\newline", "\n")
                obj["article"] = tokenize(obj["article"])
                avg_article_length += obj["article"].count(" ")
                for i in range(len(obj["questions"])):
                    num_que += 1
                    obj["questions"][i] = tokenize(obj["questions"][i])
                    avg_question_length += obj["questions"][i].count(" ")
                    for k in range(4):
                        obj["options"][i][k] = tokenize(obj["options"][i][k])
                        avg_option_length += obj["options"][i][k].count(" ")
                json.dump(obj, open(os.path.join(new_data_path, inf), "w"), indent=4)
        '''print "avg article length", avg_article_length * 1. / cnt
        print "avg question length", avg_question_length * 1. / num_que
        print "avg option length", avg_option_length * 1. / (num_que * 4)'''

    if mode == 1:
        data = "../data/only_r_rationale"
        cnt = 0
        avg_article_length = 0
        avg_question_length = 0
        avg_option_length = 0
        num_que = 0
        for data_set in class_set:
            p1 = os.path.join(data, data_set)
            if not os.path.exists(p1):
                os.mkdir(p1)
            # for d in difficulty_set:
            new_data_path = os.path.join(data, data_set)
            if not os.path.exists(new_data_path):
                os.mkdir(new_data_path)
            new_raw_data_path = os.path.join(raw_data, data_set)
            for inf in os.listdir(new_raw_data_path):
                print(new_raw_data_path, inf)
                cnt += 1
                obj = json.load(open(os.path.join(new_raw_data_path, inf), "r"))
                
                # TODO: Use rationales to replace the whole passage 
                obj["article"] = obj["article"].replace("\\newline", "\n")
                # obj["article"] = tokenize(obj["article"])
                avg_article_length += obj["article"].count(" ")
                a_words = obj["article"].split(' ')
                print('len of mark_ratio',len(obj["mark_ratio"]))
                print('len of article', len(obj["article"]), len(a_words))
                # assert len(a_words) == len(obj["mark_ratio"])
                if len(a_words) != len(obj["mark_ratio"]):
                    print('######## NOT EQUAL #######',len(obj["mark_ratio"]),len(a_words))
                    continue

                dup_ratio = copy.deepcopy(obj["mark_ratio"])
                rationale = []; rpt = iteration

                while(rpt != 0 or len(rationale)==0):
                    iter = rpt
                    rpt = 0
                    for j in range(iter):
                        r_id = dup_ratio.index(max(dup_ratio))
                        if rationale.count(a_words[r_id]) == 0:
                            rationale.append(a_words[r_id])  # no repeat
                        else:
                            rpt += 1  # dup
                        # print(j, r_id, dup_ratio[r_id], a_words[r_id])
                        del dup_ratio[r_id];del a_words[r_id]

                assert len(rationale) == iteration

                # -------------- remove punctuation ----------------
                for i in range(iteration):
                    j = rationale[i]
                    for k in j:
                        if k=='.' or k==',' or k=='"':
                            rationale[i] = j.replace(k,'')

                rationale = ' '.join(rationale)    # convert to string
                print(rationale)

                obj["article"] = rationale; obj["article"] = tokenize(obj["article"])

                for i in range(len(obj["questions"])):
                    num_que += 1
                    obj["questions"][i] = tokenize(obj["questions"][i])
                    avg_question_length += obj["questions"][i].count(" ")
                    for k in range(4):
                        obj["options"][i][k] = tokenize(obj["options"][i][k])
                        avg_option_length += obj["options"][i][k].count(" ")
                json.dump(obj, open(os.path.join(new_data_path, inf), "w"), indent=4)
        '''print "avg article length", avg_article_length * 1. / cnt
        print "avg question length", avg_question_length * 1. / num_que
        print "avg option length", avg_option_length * 1. / (num_que * 4)'''
