import numpy as np
import os
import cv2
import sys
import tensorflow as tf
import time
import random


class FC(object):
	def __init__(self, sess, is_training=False, checkpoint_dir = None, num_categories = 14):			
		self.sess = sess
		self.num_categories = num_categories
		self.batch_size = 64
		self.num_imgs = 14237*2
		#self.num_imgs = 8400 #300 per category and flipping
		self.img_height = 128
		self.img_width = 128
		self.checkpoint_dir=checkpoint_dir

		if(is_training):
			self.all_imgs = np.zeros((self.num_imgs, self.img_height, self.img_width, 3), dtype=np.uint8)
			self.all_labels = np.zeros((self.num_imgs, self.num_categories), dtype=np.uint8)

		self.build_model()

	def linear(self, input_, output_size, scope):
		shape = input_.get_shape().as_list()
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
			bias = tf.get_variable("bias", [output_size], initializer=tf.zeros_initializer())
			print("linear","in",shape,"out",(shape[0],output_size))
			return tf.matmul(input_, matrix) + bias

	def lrelu(self, x, leak=0.02):
		return tf.maximum(x, leak*x)

	def build_model(self):
		self.input_imgs = tf.placeholder(shape=[self.batch_size, self.img_height, self.img_width, 3], dtype=tf.float32)
		self.input_imgs = tf.contrib.image.rotate(self.input_imgs, tf.random_uniform(shape=[self.batch_size], minval=-0.7, maxval=0.7), interpolation='BILINEAR')
		#self.input_imgs = tf.contrib.image.translate(self.input_imgs, translations=[tf.random_uniform(shape=[self.batch_size], minval=-25, maxval=25), tf.random_uniform(shape=[self.batch_size], minval=-25, maxval=25)])
		self.labels = tf.placeholder(shape=[self.batch_size, self.num_categories], dtype=tf.float32)
		self.keep_prob = tf.placeholder(tf.float32)
		self.logits = self.CNN(self.input_imgs, self.keep_prob, reuse=True)
		self.probs = tf.nn.softmax(self.logits)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels, logits = self.logits))
		self.saver = tf.train.Saver(max_to_keep=4000)
		
		
	def CNN(self, inputs, keep_prob, reuse=False):
		'''conv1 = self.lrelu(tf.nn.conv2d(inputs, tf.Variable(tf.random_normal([7, 7, 3, 32], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv1'))
		conv2 = tf.nn.max_pool(self.lrelu(tf.nn.conv2d(conv1, tf.Variable(tf.random_normal([5, 5, 32, 32], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv2')), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv3 = self.lrelu(tf.nn.conv2d(conv2, tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv3'))
		conv4 = tf.nn.max_pool(self.lrelu(tf.nn.conv2d(conv3, tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv4')), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv5 = self.lrelu(tf.nn.conv2d(conv4, tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv5'))
		conv6 = tf.nn.max_pool(self.lrelu(tf.nn.conv2d(conv5, tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv6')), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv7 = self.lrelu(tf.nn.conv2d(conv6, tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv7'))
		conv8 = tf.nn.max_pool(self.lrelu(tf.nn.conv2d(conv7, tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv8')), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		feature_vector = tf.reshape(conv8, [self.batch_size, 8*8*256])
		h1 = self.lrelu(self.linear(feature_vector, 128, 'layer1'))
		h1_drop = tf.nn.dropout(h1, keep_prob)
		h2 = self.linear(h1, self.num_categories, 'layer2')
		return h2'''
		conv1 = tf.nn.max_pool(self.lrelu(tf.nn.conv2d(inputs, tf.Variable(tf.random_normal([7, 7, 3, 32], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv1')), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv2 = tf.nn.max_pool(self.lrelu(tf.nn.conv2d(conv1, tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv2')), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv3 = tf.nn.max_pool(self.lrelu(tf.nn.conv2d(conv2, tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv3')), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv4 = tf.nn.max_pool(self.lrelu(tf.nn.conv2d(conv3, tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01)), [1, 1, 1, 1], padding='SAME', name='conv4')), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		feature_vector = tf.reshape(conv4, [self.batch_size, 8*8*256])
		h1 = self.lrelu(self.linear(feature_vector, 128, 'layer1'))
		h1_drop = tf.nn.dropout(h1, keep_prob)
		h2 = self.linear(h1, self.num_categories, 'layer2')
		return h2
		
	def train(self, config):
		network_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
		#network_optim = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
		
		self.sess.run(tf.global_variables_initializer())

		counter = 0
		start_time = time.time()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		
		if(could_load):
			counter = checkpoint_counter+1
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed....")

		food_categories = sorted(os.listdir('training_data/'))
		count = 0
		for i in range(len(food_categories)):
			category_samples = sorted(os.listdir('training_data/'+food_categories[i]))
			#category_samples = np.random.choice(category_samples, 300, replace=False)
			for j in range(len(category_samples)):
				self.all_imgs[count] = cv2.resize(cv2.imread('training_data/'+food_categories[i]+'/'+category_samples[j]), (128, 128))
				self.all_labels[count, i] = 1 
				flipHorizontal = cv2.flip(self.all_imgs[count], 1)
				count += 1
				self.all_imgs[count] = flipHorizontal
				self.all_labels[count, i] = 1
				count += 1
		
		permutation = np.arange(self.num_imgs)

		losses = np.zeros((config.epoch))
		
		for i in range(counter, config.epoch):
			np.random.shuffle(permutation)
			self.all_imgs = self.all_imgs[permutation]
			self.all_labels = self.all_labels[permutation]
			batch_num = int(self.num_imgs/self.batch_size)
			total_loss = 0.0
			for j in range(batch_num):
				batch_imgs = self.all_imgs[(j*self.batch_size):((j+1)*self.batch_size)]/255.0
				batch_labels = self.all_labels[(j*self.batch_size):((j+1)*self.batch_size)]

				_, ls = self.sess.run([network_optim, self.loss], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 0.5})
				total_loss += ls

				if(j == (batch_num-1)):
					losses[i] = total_loss
					print("Epoch "+str(i))
					print("Total loss: "+str(total_loss))
					self.save(config.checkpoint_dir, i)
			if(i == (config.epoch-1)):
				np.save("loss_curve.npy", losses)
			

	def test(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		categories = sorted(os.listdir('testing_data/'))
		total_correct = 0	
		total_test_imgs = 0	
		false_alarm_count = np.zeros((len(categories)))
		true_detection_rate = np.zeros((len(categories)))
		true_rejection_rate = np.zeros((len(categories)))
		recognition_rate = np.zeros((len(categories)))
		num_imgs_cat = np.zeros((len(categories)))
		start = time.process_time()
		for i in range(len(categories)):
			category_imgs = sorted(os.listdir('testing_data/'+categories[i]))
			total = len(category_imgs)
			num_imgs_cat[i] = total
			total_test_imgs += total
			correct = 0
			for j in range(len(category_imgs)):
				image = cv2.resize(cv2.imread('testing_data/'+categories[i]+'/'+category_imgs[j]), (128, 128))
				batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
				for k in range(self.batch_size):
					batch_imgs[k] = image/255.0
				batch_labels = np.zeros((self.batch_size, self.num_categories))
				softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
				softmax_dist = softmax_dist[0][0]
				predicted_label = categories[np.argmax(softmax_dist)]
				print(category_imgs[j]+': '+predicted_label)
				if(np.argmax(softmax_dist) == i):
					correct += 1
					total_correct += 1
				else:
					false_alarm_count[np.argmax(softmax_dist)] += 1
			print(categories[i]+': '+str(correct)+'/'+str(total))
			true_detection_rate[i] = correct/total
		total_time = time.process_time()-start
		print("Total Time taken: "+str(total_time))
		print("Time per image: "+str(total_time/total_test_imgs))
		print("Total: "+str(total_correct)+'/'+str(total_test_imgs))
		for i in range(len(categories)):
			correct = total_test_imgs-num_imgs_cat[i]-false_alarm_count[i]
			total = total_test_imgs-num_imgs_cat[i]
			print(categories[i]+': '+str(correct)+'/'+str(total))
			true_rejection_rate[i] = correct/total

		for i in range(len(categories)):
			recognition_rate[i] = 0.5*(true_detection_rate[i]+true_rejection_rate[i])
			print(categories[i]+': '+str(recognition_rate[i]))

		print(np.mean(recognition_rate))

	def test_competition(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
	
		categories = sorted(os.listdir('Test/'))
		print(categories)
		total_correct = 0	
		total_test_imgs = 0	
		false_alarm_count = np.zeros((len(categories)))
		true_detection_rate = np.zeros((len(categories)))
		true_rejection_rate = np.zeros((len(categories)))
		recognition_rate = np.zeros((len(categories)))
		num_imgs_cat = np.zeros((len(categories)))
		multiclass_correct = np.zeros((len(categories)))

		image = cv2.resize(cv2.imread("multi-class/appleStr.jpg"), (128, 128))
		batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
		for k in range(self.batch_size):
			batch_imgs[k] = image/255.0
		batch_labels = np.zeros((self.batch_size, self.num_categories))
		softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
		softmax_dist = softmax_dist[0][0]
		predicted_label = categories[np.argmax(softmax_dist)]
		total_test_imgs += 2
		num_imgs_cat[0] += 1
		num_imgs_cat[12] += 1
		if(predicted_label == 'apples' or predicted_label == 'strawberry'):	
			total_correct += 1
			multiclass_correct[np.argmax(softmax_dist)] += 1
		else:
			false_alarm_count[np.argmax(softmax_dist)] += 1
	
		image = cv2.resize(cv2.imread("multi-class/burger_picture.jpg"), (128, 128))
		batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
		for k in range(self.batch_size):
			batch_imgs[k] = image/255.0
		batch_labels = np.zeros((self.batch_size, self.num_categories))
		softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
		softmax_dist = softmax_dist[0][0]
		predicted_label = categories[np.argmax(softmax_dist)]
		total_test_imgs += 2
		num_imgs_cat[3] += 1
		num_imgs_cat[6] += 1
		if(predicted_label == 'burger' or predicted_label == 'fries'):	
			total_correct += 1
			multiclass_correct[np.argmax(softmax_dist)] += 1
		else:
			false_alarm_count[np.argmax(softmax_dist)] += 1

		image = cv2.resize(cv2.imread("multi-class/Hubcap.jpg"), (128, 128))
		batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
		for k in range(self.batch_size):
			batch_imgs[k] = image/255.0
		batch_labels = np.zeros((self.batch_size, self.num_categories))
		softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
		softmax_dist = softmax_dist[0][0]
		predicted_label = categories[np.argmax(softmax_dist)]
		total_test_imgs += 2
		num_imgs_cat[3] += 1
		num_imgs_cat[6] += 1
		if(predicted_label == 'burger' or predicted_label == 'fries'):	
			total_correct += 1
			multiclass_correct[np.argmax(softmax_dist)] += 1
		else:
			false_alarm_count[np.argmax(softmax_dist)] += 1

		image = cv2.resize(cv2.imread("multi-class/IMG_0094.JPG"), (128, 128))
		batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
		for k in range(self.batch_size):
			batch_imgs[k] = image/255.0
		batch_labels = np.zeros((self.batch_size, self.num_categories))
		softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
		softmax_dist = softmax_dist[0][0]
		predicted_label = categories[np.argmax(softmax_dist)]
		total_test_imgs += 3
		num_imgs_cat[9] += 1
		num_imgs_cat[11] += 1
		num_imgs_cat[12] += 1
		if(predicted_label == 'pizza' or predicted_label == 'salad' or predicted_label == 'strawberry'):	
			total_correct += 1
			multiclass_correct[np.argmax(softmax_dist)] += 1
		else:
			false_alarm_count[np.argmax(softmax_dist)] += 1

		image = cv2.resize(cv2.imread("multi-class/IMG_0406.JPG"), (128, 128))
		batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
		for k in range(self.batch_size):
			batch_imgs[k] = image/255.0
		batch_labels = np.zeros((self.batch_size, self.num_categories))
		softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
		softmax_dist = softmax_dist[0][0]
		predicted_label = categories[np.argmax(softmax_dist)]
		total_test_imgs += 2
		num_imgs_cat[0] += 1
		num_imgs_cat[1] += 1
		if(predicted_label == 'apples' or predicted_label == 'bananas'):	
			total_correct += 1
			multiclass_correct[np.argmax(softmax_dist)] += 1
		else:
			false_alarm_count[np.argmax(softmax_dist)] += 1

		image = cv2.resize(cv2.imread("multi-class/IMG_0453.JPG"), (128, 128))
		batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
		for k in range(self.batch_size):
			batch_imgs[k] = image/255.0
		batch_labels = np.zeros((self.batch_size, self.num_categories))
		softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
		softmax_dist = softmax_dist[0][0]
		predicted_label = categories[np.argmax(softmax_dist)]
		total_test_imgs += 2
		num_imgs_cat[0] += 1
		num_imgs_cat[1] += 1
		if(predicted_label == 'apples' or predicted_label == 'bananas'):	
			total_correct += 1
			multiclass_correct[np.argmax(softmax_dist)] += 1
		else:
			false_alarm_count[np.argmax(softmax_dist)] += 1

		image = cv2.resize(cv2.imread("multi-class/pastasalad.jpg"), (128, 128))
		batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
		for k in range(self.batch_size):
			batch_imgs[k] = image/255.0
		batch_labels = np.zeros((self.batch_size, self.num_categories))
		softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
		softmax_dist = softmax_dist[0][0]
		predicted_label = categories[np.argmax(softmax_dist)]
		total_test_imgs += 2
		num_imgs_cat[8] += 1
		num_imgs_cat[11] += 1
		if(predicted_label == 'pasta' or predicted_label == 'salad'):	
			total_correct += 1
			multiclass_correct[np.argmax(softmax_dist)] += 1
		else:
			false_alarm_count[np.argmax(softmax_dist)] += 1

		image = cv2.resize(cv2.imread("multi-class/pizzasalad.jpg"), (128, 128))
		batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
		for k in range(self.batch_size):
			batch_imgs[k] = image/255.0
		batch_labels = np.zeros((self.batch_size, self.num_categories))
		softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
		softmax_dist = softmax_dist[0][0]
		predicted_label = categories[np.argmax(softmax_dist)]
		total_test_imgs += 2
		num_imgs_cat[9] += 1
		num_imgs_cat[11] += 1
		if(predicted_label == 'pizza' or predicted_label == 'salad'):	
			total_correct += 1
			multiclass_correct[np.argmax(softmax_dist)] += 1
		else:
			false_alarm_count[np.argmax(softmax_dist)] += 1
		
		image = cv2.resize(cv2.imread("multi-class/egg9.jpg"), (128, 128))
		batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
		for k in range(self.batch_size):
			batch_imgs[k] = image/255.0
		batch_labels = np.zeros((self.batch_size, self.num_categories))
		softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
		softmax_dist = softmax_dist[0][0]
		predicted_label = categories[np.argmax(softmax_dist)]
		total_test_imgs += 2
		num_imgs_cat[5] += 1
		num_imgs_cat[13] += 1
		if(predicted_label == 'egg' or predicted_label == 'tomato'):	
			total_correct += 1
			multiclass_correct[np.argmax(softmax_dist)] += 1
		else:
			false_alarm_count[np.argmax(softmax_dist)] += 1

		image = cv2.resize(cv2.imread("multi-class/Egg-Fried-with-Tomatoes.jpg"), (128, 128))
		batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
		for k in range(self.batch_size):
			batch_imgs[k] = image/255.0
		batch_labels = np.zeros((self.batch_size, self.num_categories))
		softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
		softmax_dist = softmax_dist[0][0]
		predicted_label = categories[np.argmax(softmax_dist)]
		total_test_imgs += 2
		num_imgs_cat[5] += 1
		num_imgs_cat[13] += 1
		if(predicted_label == 'egg' or predicted_label == 'tomato'):	
			total_correct += 1
			multiclass_correct[np.argmax(softmax_dist)] += 1
		else:
			false_alarm_count[np.argmax(softmax_dist)] += 1

		image = cv2.resize(cv2.imread("multi-class/strba.jpg"), (128, 128))
		batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
		for k in range(self.batch_size):
			batch_imgs[k] = image/255.0
		batch_labels = np.zeros((self.batch_size, self.num_categories))
		softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
		softmax_dist = softmax_dist[0][0]
		predicted_label = categories[np.argmax(softmax_dist)]
		total_test_imgs += 2
		num_imgs_cat[1] += 1
		num_imgs_cat[12] += 1
		if(predicted_label == 'bananas' or predicted_label == 'strawberry'):	
			total_correct += 1
			multiclass_correct[np.argmax(softmax_dist)] += 1
		else:
			false_alarm_count[np.argmax(softmax_dist)] += 1
		
		for i in range(len(categories)):
			category_imgs = sorted(os.listdir('Test/'+categories[i]))
			total = len(category_imgs)
			num_imgs_cat[i] += total
			total_test_imgs += total
			correct = multiclass_correct[i]
			for j in range(len(category_imgs)):
				image = cv2.resize(cv2.imread('Test/'+categories[i]+'/'+category_imgs[j]), (128, 128))
				batch_imgs = np.zeros((self.batch_size, self.img_height, self.img_width, 3))
				for k in range(self.batch_size):
					batch_imgs[k] = image/255.0
				batch_labels = np.zeros((self.batch_size, self.num_categories))
				softmax_dist = self.sess.run([self.probs], feed_dict = {self.input_imgs: batch_imgs, self.labels: batch_labels, self.keep_prob: 1.0})
				softmax_dist = softmax_dist[0][0]
				predicted_label = categories[np.argmax(softmax_dist)]
				print(category_imgs[j]+': '+predicted_label)
				if(np.argmax(softmax_dist) == i):
					correct += 1
					total_correct += 1
				else:
					false_alarm_count[np.argmax(softmax_dist)] += 1
			print(categories[i]+': '+str(correct)+'/'+str(num_imgs_cat[i]))
			true_detection_rate[i] = correct/num_imgs_cat[i]
		print("Total: "+str(total_correct)+'/'+str(total_test_imgs))
		for i in range(len(categories)):
			correct = total_test_imgs-num_imgs_cat[i]-false_alarm_count[i]
			total = total_test_imgs-num_imgs_cat[i]
			print(categories[i]+': '+str(correct)+'/'+str(total))
			true_rejection_rate[i] = correct/total

		for i in range(len(categories)):
			recognition_rate[i] = 0.5*(true_detection_rate[i]+true_rejection_rate[i])
			print(categories[i]+': '+str(recognition_rate[i]))

		print(np.mean(recognition_rate))

		f = open('competitionResult.txt', 'w')
		for i in range(len(recognition_rate)):
			f.write(str(recognition_rate[i])+' ')
					

	@property
	def model_dir(self):
		return "14_categories"

	def save(self, checkpoint_dir, step):
		model_name = "FC.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
		if(not os.path.exists(checkpoint_dir)):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step = step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if(ckpt and ckpt.model_checkpoint_path):
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0
