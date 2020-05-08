from smartnlp.classfication.deep_classifier import TextHanClassifier

if __name__ == '__main__':
    base_classifier = TextHanClassifier(model_path='./model/han/',
                                        config_path='./model/han/config.pkl',
                                        train=True,
                                        vector_path='/data/GoogleNews-vectors-negative300.bin.gz')
    out = base_classifier.predict(
        ['this is very good movie , i want to watch it again!', 'this is very bad movie , i hate it!'])
    out2 = base_classifier.predict('this is very good movie , i want to watch it again!')
    print(out)
    print(out2)
