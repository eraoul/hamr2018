from keras.models import Sequential
from keras.optimizers import Adam
from ntm_keras.ntm import NeuralTuringMachine as NTM
from ntm_keras.ntm import controller_input_output_shape as controller_shape
from keras.layers.recurrent import LSTM
import numpy as np




def generator():
	input_file = "input.npy"
	target_file = "target.npy"
	inp = np.load(input_file)
	target = np.load(target_file)
	while True:
		example_id = np.random.randint(0, 142)
		yield (np.array([inp[example_id]]), np.array([target[example_id]]))


def model():


	input_dim = 128 #or so, the number of piano keys or a bit less)
	output_dim = 128
	batch_size = 1

	m_depth = 250 #memory depth
	n_slots = 128 #number of slots
	shift_range = 3
	read_heads = 1
	write_heads = 1
	controller_input_dim, controller_output_dim = controller_shape(input_dim, output_dim, m_depth, n_slots, shift_range, read_heads, write_heads)
	print("Creating a following NTM architecture")

	print("Every single one-hot encoding is %d"%input_dim)
	print("Two LSTM layers with input dimenstion %d, output dimenstion %d"%(controller_input_dim,controller_output_dim) )
	print("NTM memory depth is %d, number of slots %d, head shift range %d"%(m_depth,n_slots,shift_range))

	controller = Sequential()

	controller.name="Two layer LSTM"
	controller.add(LSTM(units= 1024,
		            stateful=True,return_sequences=True,
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

	ntm = NTM(output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
		  controller_model=controller,
		  return_sequences=True,
		  input_shape=(250, input_dim), 
		  batch_size = batch_size)
	model.add(ntm)

	model.compile(loss='binary_crossentropy', optimizer=sgd,
		       metrics = ['binary_accuracy'], sample_weight_mode="temporal")
	model.summary()
	return model

num_epochs = 10
sample_generator = generator()
model = model()
model.fit_generator(sample_generator, steps_per_epoch=10, epochs=num_epochs)#, callbacks=callbacks)
