from keras.models import Sequential
from keras.optimizers import Adam
from ntm_keras.ntm import NeuralTuringMachine as NTM

model = Sequential()
model.name = "NTM_-_" + controller_model.name

input_dim=10
output_dim = 10


controller_input_dim, controller_output_dim = controller_shape(input_dim, output_dim, 20, 128, 3, read_heads,
write_heads)
controller = Sequential()

controller.name=ntm_controller_architecture
controller.add(LSTM(units=150,
                    stateful=True,
                    implementation=2,   # best for gpu. other ones also might not work.
                    batch_input_shape=(batch_size, None, controller_input_dim)))
controller.add(LSTM(units=controller_output_dim,
                    activation='linear',
                    stateful=True,
                    implementation=2))   # best for gpu. other ones also might not work.

controller.compile(loss='binary_crossentropy', optimizer=sgd,
                 metrics = ['binary_accuracy'], sample_weight_mode="temporal")


ntm = NTM(output_dim, n_slots=50, m_depth=20, shift_range=3,
          controller_model=controller_model,
          return_sequences=True,
          input_shape=(None, input_dim), 
          batch_size = 100)
model.add(ntm)

sgd = Adam(lr=learning_rate, clipnorm=clipnorm)
model.compile(loss='binary_crossentropy', optimizer=sgd,
               metrics = ['binary_accuracy'], sample_weight_mode="temporal")
