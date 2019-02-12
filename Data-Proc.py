from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import xlrd
import os


def load_data():
    workbook = xlrd.open_workbook(os.getcwd() + '/Dataset/Data Hadis.xlsx')
    worksheet1 = workbook.sheet_by_index(0)
    x = []
    for i in range(2, worksheet1.nrows):
        x.append(worksheet1.cell(i, 2).value)
    return x


if __name__ == '__main__':

    x = load_data()
    x_prep = []
    for i, row in enumerate(x):
        x_prep.append(simple_preprocess(row))
    model = Word2Vec(
        x_prep,
        size=1500,
        window=10,
        min_count=2,
        workers=10)
    model.train(x_prep, total_examples=len(x_prep), epochs=10)
    model.save('hadith_size=1500_window=10.model')
    model = Word2Vec.load('hadith_size=1500_window=10.model')
    w = 'allah'
    print(model.wv.most_similar(positive=w))

