
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.models import Model

# Convolution Parameters
kernel_size = 5
filters = 64
pool_size = 4

def build_model(type, embedded_sequences, labels_index, sequence_input):

    if type == 'cnn':
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        preds = Dense(len(labels_index), activation='softmax')(x)
        _model = Model(sequence_input, preds)
        _model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

    if type == 'lstm':
        x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
        preds = Dense(len(labels_index), activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

    if type == 'cnn_lstm':
        x = Dropout(0.25)(embedded_sequences)
        x = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
        x = MaxPooling1D(pool_size=pool_size)(x)
        x = LSTM(128)(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(len(labels_index), activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

    return _model