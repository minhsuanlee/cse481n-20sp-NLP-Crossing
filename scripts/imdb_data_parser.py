import os
import csv
import pandas as pd

train_dir = './aclImdb/train/'
test_dir = './aclImdb/test/'

# sent: pos, neg
def parse_txt(data):
    lines = []
    sents = ['neg', 'pos']
    
    for sent in sents:
        temp = []
        d = train_dir if data == 'train' else test_dir
        path = d + sent
        for f in os.listdir(path):
            if f.endswith('.txt'):
                with open(path + '/' + f, 'r') as txt:
                    line = txt.readline()
                    temp.append(line)
        lines.append(temp)
    return lines


def gen_csv(data, lines):
    neg = lines[0]
    pos = lines[1]
    with open(data + '_imdb.csv', 'w') as out:
        writer = csv.writer(out)
        writer.writerow(('review', 'sentiment'))
        for line in neg:
            writer.writerow((line, 0))
        for line in pos:
            writer.writerow((line, 1))


def main():
    train_lines = parse_txt('train')
    gen_csv('train', train_lines)
    train_df = pd.read_csv('train_imdb.csv')
    print('Train DataFrame')
    print(train_df.head())

    test_lines = parse_txt('test')
    gen_csv('test', test_lines)
    test_df = pd.read_csv('test_imdb.csv')
    print('Test DataFrame')
    print(test_df.head())


if __name__ == '__main__':
    main()
