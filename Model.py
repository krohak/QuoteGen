from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Embedding
from keras.optimizers import Adam


class Model:

    def __init__(self,vocab_size,topic):
        self.vocab_size = vocab_size
        self.topic = topic

    def load_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 100))
        model.add(LSTM(100))
        model.add(Dense(self.vocab_size))


        model.add(Activation('softmax'))
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        filename = "trained_weights/QG-%s.hdf5"%self.topic
        try:
            model.load_weights(filename)
            model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

            return model
            
        except Exception as e:
            print(e)
            print("Save model as QG-%s.hdf5"%topic.topic)
