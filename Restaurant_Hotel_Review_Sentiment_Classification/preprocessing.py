import pandas as pd
import re
import underthesea
from string import punctuation
import warnings
import numpy as np
punctuation = punctuation + '‘’“”'
vietnamese_words = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ'
warnings.filterwarnings('ignore')


def read_data(path):
    doc = open(path, encoding='utf-8').read().split('\n')
    text = []
    label = []
    i = 1
    while i < len(doc):
        text.append(doc[i])
        label.append(doc[i + 1])
        i += 4

    df = pd.DataFrame({'review': text, 'sentiment': label})
    return df


def adding_label(data):
    label = data["sentiment"].values

    clean_label = []

    for i in range(len(label)):
        sample = label[i].split(",")
        for j in range(len(sample)):
            sample[j] = re.sub("[{}]", "", sample[j].strip())
        clean_label.append(sample)

    y = []
    for i in range(len(clean_label)):
        row = clean_label[i]

        j = 0
        while j < len(row):
            if(row[j+1] == 'positive'):
                y.append([i + 1 , row[j], 1])
            elif(row[j+1] == 'negative'):
                y.append([i + 1 , row[j], -1])
            else:
                y.append([i + 1 , row[j], 0])
            j += 2
    return y


def data_with_sparse_matrix(df):
    all_classes = ["Text"]
    all_classes += list(set([i[1] for i in adding_label(df)]))
    y = adding_label(df)
    new_dat = pd.DataFrame(columns=all_classes)
    all_classes.remove("Text")
    new_dat.Text = df.review
    new_dat[all_classes] = 0
    sparse_matrix = np.zeros((len(df), len(all_classes)))
    dict_class = {}
    index = 0
    for i in all_classes:
        dict_class[i] = index
        index += 1
    sparse_matrix = list(sparse_matrix.astype(int))
    for i in y:
        sparse_matrix[i[0] - 1][dict_class[i[1]]] = i[2]
    for i in y:
        new_dat[i[1]][i[0] - 1] = i[2]
    final_dat = pd.DataFrame({"Text": new_dat.Text, "Label": sparse_matrix})

    data_with_all_classes = new_dat
    data_with_sparse = final_dat
    for i in range(len(final_dat.Label)):
        s = np.sum(final_dat.Label[i])
        if(s>1):
            final_dat.Label[i] = 1
        elif(s<-1):
            final_dat.Label[i] = -1
        else:
            final_dat.Label[i] = 0
    return data_with_sparse, data_with_all_classes


def clean_text(df,option=0):
    #Trong này đã bao gồm normalization từ vựng lẫn 
    stop_words = open('Dataset_For_Work\\vietnamese-stopwords.txt','r',encoding='utf-8').read().split('\n')
    for i in range(len(df.Text)):
        
        df.Text[i] = underthesea.text_normalize(df.Text[i])

        if(option==1):
            df.Text[i] = re.sub(f'[^{vietnamese_words}{punctuation}]',' ',df.Text[i]) #Việc này để loại bỏ các ký tự biểu tượng cảm xúc trong review Restaurant

        df.Text[i] = underthesea.sent_tokenize(df.Text[i])

    for i in range(len(df.Text)):
        for j in range(len(df.Text[i])):

            df.Text[i][j] = re.sub(f'[{punctuation}]',' ',df.Text[i][j])

            for s_w in stop_words:
                df.Text[i][j] = re.sub(f' {s_w} ',' ',df.Text[i][j])

    for i in range(len(df.Text)):
        new_text = ''
        for j in range(len(df.Text[i])):
            if(j==(len(df.Text[i]) - 1)):
                new_text+=df.Text[i][j]
            else:
                new_text+=(df.Text[i][j] + ' ')
        df.Text[i] = re.sub('\s+',' ',new_text)

    # df.to_csv('clean_data.csv')
    return df


def process_data(data):
    sentences = []
    stop_words = open('Dataset_For_Work\\vietnamese-stopwords.txt','r',encoding='utf-8').read().split('\n')
    for item in data:
        processed_item = item.lower()
        for s_t in stop_words:

            processed_item = re.sub(rf" {s_t} ", ' ', processed_item)

        processed_item = re.sub(f'[{punctuation}]',' ',processed_item)
        processed_item = re.sub(f'\s+',' ',processed_item)
        temp = underthesea.sent_tokenize(processed_item)
        sentences += temp

    tokenize_data = [underthesea.word_tokenize(i) for i in sentences]

    return tokenize_data, sentences

def process_data(data):
    data_doc = []
    stop_words = open('Dataset_For_Work\\vietnamese-stopwords.txt','r',encoding='utf-8').read().split('\n')
    for item in data:
        processed_item = item.lower()
        for s_t in stop_words:
            processed_item = re.sub(rf" {s_t} ", ' ', processed_item)
        processed_item = re.sub(f'[{punctuation}]',' ',processed_item)
        processed_item = re.sub(f'\s+',' ',processed_item)
        temp = underthesea.sent_tokenize(processed_item)
        tokenize_data = []
        for i in temp:
          tokenize_data += underthesea.word_tokenize(i)
        data_doc.append(tokenize_data)
    return data_doc