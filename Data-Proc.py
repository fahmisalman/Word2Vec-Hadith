from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import xlrd
import csv
import os


def load_data_excel():
    workbook = xlrd.open_workbook(os.getcwd() + '/Dataset/Data Hadis.xlsx')
    worksheet1 = workbook.sheet_by_index(0)
    x = []
    for i in range(2, worksheet1.nrows):
        x.append(worksheet1.cell(i, 2).value)
    return x


def load_data_csv(loc):
    x = []
    with open(loc) as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(row[2])
    return x


if __name__ == '__main__':

    # x = []
    # x += load_data_csv(os.getcwd() + '/Dataset/had_abudaud.csv')
    # x += load_data_csv(os.getcwd() + '/Dataset/had_bukhari.csv')
    # x += load_data_csv(os.getcwd() + '/Dataset/had_muslim.csv')
    # x += load_data_csv(os.getcwd() + '/Dataset/had_nasai.csv')
    # x += load_data_csv(os.getcwd() + '/Dataset/had_tirmidzi.csv')
    #
    # x_prep = []
    # for i, row in enumerate(x):
    #     x_prep.append(simple_preprocess(row))
    # model = Word2Vec(
    #     x_prep,
    #     size=400,
    #     window=10,
    #     min_count=2,
    #     workers=10,
    #     sg=1,
    #     cbow_mean=0
    # )
    # model.train(x_prep, total_examples=len(x_prep), epochs=10)
    # model.save('sg_hadith_size=1500_window=10.model')
    model = Word2Vec.load('sg_hadith_size=1500_window=10.model')
    w = 'nabi'
    print(model.wv.most_similar(positive=w))
    model = Word2Vec.load('cbow_hadith_size=1500_window=10.model')
    print(model.wv.most_similar(positive=w))

