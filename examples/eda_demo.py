# coding=utf-8
# created by msgi on 2020/5/18

from smartnlp.augmentation.eda import EDA

if __name__ == "__main__":
    eda = EDA("./data/stopwords/stopwords.txt")
    aug_sentences = eda.fit_transform("鱼香肉丝好吃的很，你要不要来尝一尝")
    print(aug_sentences)
