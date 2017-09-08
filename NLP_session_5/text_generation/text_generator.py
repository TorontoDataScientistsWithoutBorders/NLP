from utils import process_data
from params import *
from models.LSTMModel import LSTMModel

import numpy as np 
import tensorflow as tf 


def main():
	data_train,data_val,data_test,char_to_index,index_to_char=process_data(look_back=30,batch_size=1024,split=[0.7,0.2,0.1],debug=DEBUG)

	vocab_size=len(char_to_index)

	model = LSTMModel(vocab_size,look_back=30,hidden_dim=400,batch_size=1024,lr=1,nb_layers=3)

	model.build_graph()

	model.train(data_train,1)

	model.create_story(index_to_char,char_to_index,"how are you my pretty Baobei, are you having a good day?")

if __name__=='__main__':
	main()