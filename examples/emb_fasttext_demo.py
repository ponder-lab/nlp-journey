from smartnlp.embedding.fasttext_model import FastTextModel

if __name__ == '__main__':
    # cbow 模型
    model = FastTextModel('data/tianlong_seg.txt', 'model/fasttext/model.vec', model_type='cbow')
    # skipgram 模型
    # model = FastTextModel('data/tianlong_seg.txt', 'model/fasttext/model')
    print(model.get_nearest_neighbors('段誉', 10))
