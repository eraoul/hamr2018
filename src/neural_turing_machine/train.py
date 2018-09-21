from keras.models import Sequential
from keras.optimizers import Adam
from ntm_keras.ntm import NeuralTuringMachine as NTM
from ntm_keras.ntm import controller_input_output_shape as controller_shape
from keras.layers.recurrent import LSTM

input_dim = 88 #or so, the number of piano keys or a bit less)
output_dim = 86
batch_size = 32

m_depth = 20 #memory depth
n_slots = 128 #number of slots
shift_range = 3
read_heads = 1
write_heads = 1
controller_input_dim, controller_output_dim = controller_shape(input_dim, output_dim, m_depth, n_slots, shift_range, read_heads, write_heads)

controller = Sequential()

controller.name="Two layer LSTM"
controller.add(LSTM(units=1024,
                    stateful=True,
                    implementation=2,   # best for gpu. other ones also might not work.
                    batch_input_shape=(batch_size, None, controller_input_dim)))
controller.add(LSTM(units=controller_output_dim,
                    activation='linear', #this has to be linear if I understand it correctly
                    stateful=True,
                    implementation=2))   # best for gpu. other ones also might not work.



lr = 5e-4
clipnorm = 10
sgd = Adam(lr=lr, clipnorm=clipnorm)
controller.compile(loss='binary_crossentropy', optimizer=sgd,
                 metrics = ['binary_accuracy'], sample_weight_mode="temporal")

model = Sequential()
model.name = "NTM_-_" + controller.name

ntm = NTM(output_dim, n_slots=50, m_depth=20, shift_range=3,
          controller_model=controller,
          return_sequences=True,
          input_shape=(None, input_dim), 
          batch_size = 100)
model.add(ntm)

model.compile(loss='binary_crossentropy', optimizer=sgd,
               metrics = ['binary_accuracy'], sample_weight_mode="temporal")
