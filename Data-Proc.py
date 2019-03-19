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
    x = load_data_csv(os.getcwd() + '/Dataset/hadits.csv')
    # x += load_data_csv(os.getcwd() + '/Dataset/had_bukhari.csv')
    # x += load_data_csv(os.getcwd() + '/Dataset/had_muslim.csv')
    # x += load_data_csv(os.getcwd() + '/Dataset/had_nasai.csv')
    # x += load_data_csv(os.getcwd() + '/Dataset/had_tirmidzi.csv')

    sizes = [100]
    windows = [1]

    x_prep = []

    for ii in range(len(sizes)):
        for jj in range(len(windows)):
            x_prep = []
            for i, row in enumerate(x):
                x_prep.append(simple_preprocess(row))

            model = Word2Vec(
                x_prep,
                size=sizes[ii],
                window=windows[jj],
                min_count=2,
                workers=10
            )
            model.train(x_prep, total_examples=len(x_prep), epochs=10)
            model.save('Model/cbow_hadith_size={}_window={}.model'.format(sizes[ii], windows[jj]))
            print('Sizes={}, Window={}'.format(sizes[ii], windows[jj]))
            model = Word2Vec.load('Model/sg_hadith_size={}_window={}.model'.format(sizes[ii], windows[jj]))
            w = 'nabi'
            print(model.wv.most_similar(positive=w))
            model = Word2Vec.load('Model/cbow_hadith_size={}_window={}.model'.format(sizes[ii], windows[jj]))
            print(model.wv.most_similar(positive=w))
            print()
    print(model['nabi'])

