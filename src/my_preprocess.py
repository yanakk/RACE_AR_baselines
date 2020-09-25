import os
import json
from nltk import sent_tokenize, word_tokenize, download


def tokenize(st):
    #TODO: The tokenizer's performance is suboptimal
    ans = []
    for sent in sent_tokenize(st):
        ans += word_tokenize(sent)
    return " ".join(ans).lower()

if __name__ == "__main__":

    # difficulty_set = ["middle", "high"]
    # class_set = ["train", "dev", "test"]
    difficulty_set = ["normal"]
    class_set = ["fact", "mainly", "purpose", "title"]

    data = "../data/rationale"
    #raw_data = "../data/RACE"
    raw_data = "..\\..\\ye\\5datasets\\Rationale\\database2.1\\database"

    cnt = 0
    avg_article_length = 0
    avg_question_length = 0
    avg_option_length = 0
    num_que = 0
    for data_set in class_set:
        p1 = os.path.join(data, data_set)
        if not os.path.exists(p1):
            os.mkdir(p1)
        for d in difficulty_set:
            new_data_path = os.path.join(data, data_set, d)
            if not os.path.exists(new_data_path):
                os.mkdir(new_data_path)
            new_raw_data_path = os.path.join(raw_data, data_set, d)
            for inf in os.listdir(new_raw_data_path):
                cnt += 1
                obj = json.load(open(os.path.join(new_raw_data_path, inf), "r"))
                obj["Article"] = obj["Article"].replace("\\newline", "\n")
                obj["Article"] = tokenize(obj["Article"])
                avg_article_length += obj["Article"].count(" ")
                # for i in range(len(obj["Question"])):
                #     # print(obj["Question"])
                #     num_que += 1
                #     obj["Question"][i] = tokenize(obj["Question"][i])
                #     avg_question_length += obj["Question"][i].count(" ")
                #     for k in range(4):
                #         obj["Options"][i][k] = tokenize(obj["Options"][i][k])
                #         avg_option_length += obj["Options"][i][k].count(" ")
                num_que += 1
                temp1 = obj["Question"]
                qs = [['']]
                qs[0] = temp1
                obj["Question"] = qs
                obj["Question"][0] = tokenize(obj["Question"][0])
                avg_question_length += obj["Question"][0].count(" ")

                temp = obj["Options"]
                print(temp,obj["Type"], obj["Id"], data_set)
                op = [['','','','']]
                ia = 0; ij = 0; flag = 0; icnt = 0
                # segment the options
                for i in range(len(temp)):
                    if temp[i] == ' ' and temp[i-1]=='.' and (temp[i-2]=='A' or temp[i-2]=='B' or temp[i-2]=='C' or temp[i-2]=='D'):
                        ia = i
                        flag += 1
                    if temp[i] == '<' and temp[i-1]=='.':
                        ij = i-1
                        flag += 1
                    elif temp[i] == '<':
                        ij = i
                        flag += 1

                    if i == len(temp)-1:
                        flag = 99

                    if flag == 2:
                        op[0][icnt] = temp[ia+1:ij]
                        flag = 0
                        icnt += 1
                    if flag == 99:
                        if temp[-1] == '.':
                            op[0][icnt] = temp[ia + 1: len(temp)-1]
                        else:
                            op[0][icnt] = temp[ia+1:]

                print(op, obj["Type"], obj["Id"], data_set)
                obj["Options"] = op

                for k in range(4):
                    # print(obj["Options"])
                    obj["Options"][0][k] = tokenize(obj["Options"][0][k])
                    avg_option_length += obj["Options"][0][k].count(" ")

                # print(obj["Options"])
                # change the key
                obj.update({'article':obj.pop('Article')})
                obj.update({'options':obj.pop('Options')})
                obj.update({'questions':obj.pop('Question')})
                obj.update({'answers':obj.pop('Answer')})

                json.dump(obj, open(os.path.join(new_data_path, inf), "w"), indent=4)

    '''print "avg article length", avg_article_length * 1. / cnt
    print "avg question length", avg_question_length * 1. / num_que
    print "avg option length", avg_option_length * 1. / (num_que * 4)'''
