import tf_glove

    corpus = f.read()

model = tf_glove.GloVeModel(embedding_size=120, context_size=3)
model.fit_to_corpus(corpus)
print(model.words);
model.train(num_epochs=60)
print(model.words[10]);
print(model.embedding_for(model.words[10]))
model.write_metadata()
