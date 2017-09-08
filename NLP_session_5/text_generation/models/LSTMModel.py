import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np 
from params import *

class LSTMModel:
	"""Build the graph for the model"""

	def __init__(self,vocab_size,look_back=4,hidden_dim=3,batch_size=1024,lr=0.01,nb_layers=1):
		
		self.vocab_size=vocab_size
		self.look_back=look_back
		self.hidden_dim=hidden_dim
		self.batch_size=batch_size
		self.lr=lr
		self.nb_layers=nb_layers
		self.global_step=tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step")


	def _create_placeholders(self):
		"""Creates placeholders for input and output"""

		with tf.name_scope("input_data"):
			self.input_tokens=tf.placeholder(shape=(None,self.look_back), dtype=tf.float32,name='input_tokens')
		with tf.name_scope("output_data"):	
			self.output_tokens=tf.placeholder(shape=(None,self.look_back),dtype=tf.int32,name='output_tokens')


	def _create_recurrent_layers(self):

		with tf.name_scope("recurrent_layers"):

			self.lstms=[tf.contrib.rnn.LSTMCell(
				self.hidden_dim) for i in range(self.nb_layers)]

			self.stacked_lstm=tf.contrib.rnn.MultiRNNCell(
				self.lstms)

			initial_state=state = self.stacked_lstm.zero_state(self.batch_size, dtype=tf.float32)
			
			_output = tf.reshape(self.input_tokens,(-1,self.look_back,1))

			_output,_=tf.nn.dynamic_rnn(self.stacked_lstm,_output,dtype=tf.float32)

			_output=tf.reshape(_output,[-1,self.hidden_dim])			
			
		with tf.name_scope("Softmax_layer"):

			self.W=tf.Variable(tf.random_normal(shape=(self.hidden_dim,self.vocab_size)))
			self.b=tf.Variable(tf.zeros(shape=(self.vocab_size)))

			_output=tf.matmul(_output,self.W)+self.b 

			self.pred_output=_output


	def _create_loss(self):

		#output_tokens_r=tf.reshape(self.output_tokens,(-1,self.look_back))
		one_hot=tf.nn.embedding_lookup(tf.eye(self.vocab_size), self.output_tokens)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred_output, labels=one_hot))


	def _create_optimizer(self):
		self.optimizer=tf.train.AdamOptimizer().minimize(self.loss,global_step=self.global_step)

	def _create_summaries(self):
		with tf.name_scope("summaries"):
			self.train_loss_summary=tf.summary.scalar("train_loss", self.loss)
			self.train_loss_histogram=tf.summary.histogram("histogram_train_loss", self.loss)
			self.summary_train_op = tf.summary.merge((self.train_loss_summary,self.train_loss_histogram))
			self.val_loss_summary=tf.summary.scalar("val_loss",self.loss)
			self.val_loss_histogram=tf.summary.histogram("histogram_val_loss", self.loss)
			self.summary_val_op = tf.summary.merge((self.val_loss_summary,self.val_loss_histogram))


	def build_graph(self):
		"""Build graph for the model"""
		self._create_placeholders()
		self._create_recurrent_layers()
		self._create_loss()
		self._create_optimizer()
		self._create_summaries()


	def train(self,train_data,nb_train_steps=1):
		"""Train the model"""
		print('Training LSTM with cross entropy loss')
		saver = tf.train.Saver()

		with tf.Session() as sess:

			#sess.run(tf.global_variables_initializer())

			ckpt = tf.train.get_checkpoint_state("./save/")


			if ckpt:
				saver.restore(sess,ckpt.model_checkpoint_path)
				print("Model at "+ckpt.model_checkpoint_path+" was restored.")
			else:
				sess.run(tf.global_variables_initializer())

			writer = tf.summary.FileWriter("./save/graph",sess.graph)

			train_data=list(enumerate(train_data))

			initial_step = self.global_step.eval()//len(train_data)

			for i in range(initial_step,initial_step+nb_train_steps):
				for j,data_Xy in train_data:

					global_step=i*len(train_data)+j
					print("Epoch %d out of %d, step %d of %d"%(
						i+1,nb_train_steps+initial_step,global_step, (i+1)*len(train_data)-1),
							end='\r')

					X_batch,y_batch = data_Xy

					feed_dict={self.input_tokens:X_batch,self.output_tokens:y_batch}

					loss_batch,_,summary = sess.run(
						[self.loss,self.optimizer,self.summary_train_op],feed_dict=feed_dict)
					writer.add_summary(summary,global_step=global_step)

				if (i+1)%SKIP_STEP==0:
					saver.save(sess,"./save/step", global_step)

			print("\n")


	def predict(self,data):

		saver = tf.train.Saver()

		with tf.Session() as sess:

			

			ckpt = tf.train.get_checkpoint_state("./save")

			if ckpt:
				saver.restore(sess,ckpt.model_checkpoint_path)
			else:
				sess.run(tf.global_variables_initializer())

			feed_dict={self.input_tokens:data}

			return sess.run([self.pred_output],feed_dict=feed_dict)


	def create_story(self,index_to_char,char_to_index,beginning):

		story=[char_to_index[char] for char in beginning][:self.look_back]

		saver = tf.train.Saver()

		with tf.Session() as sess:


			ckpt = tf.train.get_checkpoint_state("./save/")

			if ckpt:
				saver.restore(sess,ckpt.model_checkpoint_path)
			else: 
				sess.run(tf.global_variables_initializer())

			for i in range(STORY_LENGTH):

				X_batch=[story[-self.look_back:]]
				feed_dict={self.input_tokens:X_batch}

				next_token=sess.run(self.pred_output,feed_dict=feed_dict)
				next_word=np.argmax(next_token[0])

				#print(next_word.shape)

				story+=[next_word]
		
		stories_file=open('./save/story.txt','a')
		story = ''.join([index_to_char[token] for token in story])
		stories_file.write(story+'\n \n')







	def try2(self):

		example=[[3,1,3,4],[1,2,3,4]]

		example2=[[3,1,3,4],[1,2,3,4]]


		sess=tf.Session()

		sess.run(tf.global_variables_initializer())

		print(sess.run([self.loss],feed_dict={self.input_tokens:example,self.output_tokens:example2}))