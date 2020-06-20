# -*- coding: utf-8 -*-
"""
Code Authors: Dibyanayan Badndypadhyay, Arkadipta De, Baban Gain
MIT LIcensed 2020
SEED = 42
1. Hindi Premise and English Hypothesis (Image Included)
2. Progressive - English Premise and Hindi Hypothesis (Image Included)
"""

# Imports

SEED = 42

from google.colab import drive
from google.colab import files

%tensorflow_version 1.x
import tensorflow as tf

!pip install bert-tensorflow
import bert
from bert import run_classifier

from bert import optimization
from bert import tokenization

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

drive.mount('/content/gdrive')

# BERT Pretrained Model Download

!wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
!unzip multi_cased_L-12_H-768_A-12.zip

# Dataset Loading (Text)

file = '/content/gdrive/My Drive/COLING 2020/dataset (Cleaned).csv'
df = pd.read_csv(file, sep = '\t')

train, test = train_test_split(df, test_size=0.1,random_state = SEED,shuffle = True)

def get_data_eng_hindi(a):
  b_ = list(a['gold_label'])
  lab = []
  for i in b_:
    if i=='contradiction':
        lab.append(0)

    elif i=='neutral':
        lab.append(1)
    elif i== 'entailment':
        lab.append(2)
    else:
        lab.append(3)
  sentence_1 = list(a['english_premise'])
  sentence_2 = list(a['hypo_hindi'])
  raw_data_train = {'sentence1_eng': sentence_1,
              'sentence2_hindi': sentence_2,
          'label': lab}
  df = pd.DataFrame(raw_data_train, columns = ['sentence1_eng','sentence2_hindi','label'])
  return df

def get_data_hindi_eng(a):
  b_ = list(a['gold_label'])
  lab = []
  for i in b_:
    if i=='contradiction':
        lab.append(0)

    elif i=='neutral':
        lab.append(1)
    elif i== 'entailment':
        lab.append(2)
    else:
        lab.append(3)
  sentence_1 = list(a['premise_hindi'])
  sentence_2 = list(a['english_hypo'])
  raw_data_train = {'sentence1_hindi': sentence_1,
              'sentence2_eng': sentence_2,
          'label': lab}
  df = pd.DataFrame(raw_data_train, columns = ['sentence1_hindi','sentence2_eng','label'])
  return df

train_eng_hindi = get_data_eng_hindi(train)
train_hindi_eng = get_data_hindi_eng(train)

test_eng_hindi = get_data_eng_hindi(test)
test_hindi_eng = get_data_hindi_eng(test)

print(train_eng_hindi[0:3])
print(train_hindi_eng[0:3])
print(test_eng_hindi[0:3])
print(test_hindi_eng[0:3])

# Dataset Loading (Image)

'''
#Flickr30K Dataset Attach and Image Preprocess

uploaded = files.upload() #Upload the API Key for Kaggle (Kaggle.json)
!mkdir ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/kaggle.json
!kaggle datasets download -d hsankesara/flickr-image-dataset
!unzip "/content/flickr-image-dataset.zip"

file = '/content/gdrive/My Drive/COLING 2020/dataset.csv'
df = pd.read_csv(file)

# Testing for Proper loading of Image
test_caption = list(df['captionID'])[10][:-2]
image_file = "/content/flickr30k_images/flickr30k_images/"
img = Image.open(image_file + test_caption)
plt.imshow(img)

img_lib = "/content/flickr30k_images/flickr30k_images/"
images = list(df['captionID'])
for i in range(len(images)):
  images[i] = images[i][:-2]   #Last 2 characters contains non relevant hash-values

image_height,image_width = 100,100  #Optimal for RAM Usage

image_array = np.zeros((36072,image_height,image_width,3), dtype = np.float32)  #Because 146 error entries
index = 0
errors = []
for i in images:
  try:
    print("Processing File: "+i)
    img = Image.open(img_lib + i)
    img = img.resize((image_height,image_width))
    img = np.asarray(img, dtype = np.float32)
    image_array[index] = img
    index += 1
  except:
    index += 1
    print("Error at Index: "+ str(index))
    errors.append(index)

np.array(errors).dump(open('Image Error Indices.npy', 'wb'))    #Useful for Sentence Deletion or Manual Image Insertion
images_array = train_images_array/255
train_imgages, test_images = train_test_split(images_array, test_size=0.1,random_state = SEED, shuffle = True)
'''

# Image Numpy File loading Already Resized and Preprocessed
file = '/content/gdrive/My Drive/COLING 2020/image_array_150_150.npy'
images_array = np.load(file)

# Train Test Split for Input in SOTA
train_images, test_images = train_test_split(images_array, test_size=0.1,random_state = SEED, shuffle = True)
print(train_images.shape)
print(test_images.shape)

# Using SOTA Image DNNs for extracting pretrained features InceptionResnetV2 [InceptionResnetV2 Paper](https://arxiv.org/abs/1602.07261)

from tensorflow.keras import applications

Image_Model = applications.InceptionResNetV2(include_top = False, input_shape = (150, 150, 3), weights = "imagenet")
x = Image_Model.output
img_features = tf.keras.layers.Flatten()(x)
Image_model_final = tf.keras.Model(Image_Model.input, img_features)

train_img_features = Image_model_final.predict(train_images)
test_img_features = Image_model_final.predict(test_images)

#Output Shapes
print('Train Image Features Shape = {}'.format(train_img_features.shape))
print('Test Image Features Shape = {}'.format(test_img_features.shape))

#Freeing up Memory by setting reference variable to None after they are used
images_array = None
train_images = None
test_images =  None

# Changing Raw Inpput to Bert Readable Inputs (Train and Test) Function

label_list = [0,1,2,3]

train_InputExamples_eng = train_eng_hindi.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x['sentence1_eng'],
                                                                   text_b = x['sentence2_hindi'],
                                                                   label = x['label']), axis = 1)
train_InputExamples_hindi = train_hindi_eng.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x['sentence1_hindi'],
                                                                   text_b = x['sentence2_eng'],
                                                                   label = x['label']), axis = 1)

test_InputExamples_eng = test_eng_hindi.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x['sentence1_eng'],
                                                                   text_b = x['sentence2_hindi'],
                                                                   label = x['label']), axis = 1)
test_InputExamples_hindi = test_hindi_eng.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x['sentence1_hindi'],
                                                                   text_b = x['sentence2_eng'],
                                                                   label = x['label']), axis = 1)

vocab_file = "multi_cased_L-12_H-768_A-12/vocab.txt"
def create_tokenizer_from_hub_module():

  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=True)

tokenizer = create_tokenizer_from_hub_module()

# Checking BERT Hindi and English Tokenizer

print(tokenizer.tokenize("how are you"))
print(tokenizer.tokenize("एक आदमी गोरा सिर वाली महिला से बात कर रहा है।"))

# Changing Raw Inpput to Bert Readable Inputs (Train and Test) Function

MAX_SEQ_LENGTH = 128
# Convert our train and test features to InputFeatures that BERT understands.
train_features_eng = bert.run_classifier.convert_examples_to_features(train_InputExamples_eng, label_list, MAX_SEQ_LENGTH, tokenizer)
train_features_hindi = bert.run_classifier.convert_examples_to_features(train_InputExamples_hindi, label_list, MAX_SEQ_LENGTH, tokenizer)

MAX_SEQ_LENGTH = 128
# Convert our train and test features to InputFeatures that BERT understands.
test_features_eng = bert.run_classifier.convert_examples_to_features(test_InputExamples_eng, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features_hindi = bert.run_classifier.convert_examples_to_features(test_InputExamples_hindi, label_list, MAX_SEQ_LENGTH, tokenizer)

# CLTE-BERT Custom Model Definition with **Image Input**"""

def create_model_img(img_features,bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings = False):
  """Creates a classification model."""
  model = bert.run_classifier.modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()
  hidden_size = output_layer.shape[-1].value
  old_size = img_features.shape[-1].value
  #output_weights = tf.get_variable("output_weights", [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
  output_weights = tf.get_variable("output_weights", [num_labels, hidden_size*2], initializer=tf.truncated_normal_initializer(stddev=0.02)) #Concatenta
  output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())
  output_weights_img = tf.get_variable("output_weights_img", [hidden_size,old_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
  output_bias_img = tf.get_variable("output_bias_img", [hidden_size],initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    img_features = tf.matmul(img_features, output_weights_img, transpose_b=True)
    img_features = tf.nn.bias_add(img_features, output_bias_img)
    img_features = tf.nn.relu(img_features)

    output_layer = tf.keras.layers.concatenate([output_layer,img_features])  #Text and Image feature fusion

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities,predicted_labels,output_layer)



def model_fn_builder_img(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings = False):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    img_features = features["img_features"]

    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities,predicted_labels,hidden_context) = create_model_img(
        img_features, bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings = False)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = bert.run_classifier.modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    """
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    """
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)

        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss
        }

      eval_metrics = metric_fn(per_example_loss, label_ids, logits, is_real_example)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics)
    else:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities,"labels": predicted_labels, "hidden_context": hidden_context})
    return output_spec

  return model_fn

# CLTE-Progressive-BERT Custom Model Definition with **Image Input**"""

def create_model_progressive_img(img_features,bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings,hidden_context):
  """Creates a classification model."""
  model = bert.run_classifier.modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()
  hidden_size = output_layer.shape[-1].value
  old_size = img_features.shape[-1].value

  #output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
  output_weights = tf.get_variable("output_weights", [num_labels, old_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
  output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())
  output_weights_img = tf.get_variable("output_weights_img", [hidden_size,old_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
  output_bias_img = tf.get_variable("output_bias_img", [hidden_size], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    img_features = tf.matmul(img_features, output_weights_img, transpose_b=True)
    img_features = tf.nn.bias_add(img_features, output_bias_img)
    img_features = tf.nn.relu(img_features)

    output_layer = tf.keras.layers.concatenate([output_layer,img_features])
    output_layer_probs = tf.nn.softmax(output_layer,axis = -1)
    #loss = y_true * log(y_true / y_pred)
    hidden_context = tf.nn.softmax(hidden_context,axis = -1)
    per_example_kd_loss = tf.keras.losses.KLD(hidden_context,output_layer_probs)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    kd_loss_weight = 0.2           #hyperparameter
    per_example_kd_loss = kd_loss_weight*per_example_kd_loss
    per_example_loss += per_example_kd_loss

    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, logits, probabilities,predicted_labels)

def model_fn_builder_img_progressive(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    hidden_context = features["hidden_context"]
    img_features = features["img_features"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities,predicted_labels) = create_model_progressive_img(
        img_features,bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings,hidden_context)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = bert.run_classifier.modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    """
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    """
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = metric_fn(per_example_loss, label_ids, logits, is_real_example)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics)
    else:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities,"labels": predicted_labels})
    return output_spec

  return model_fn

# Input Functions BERT-Progressive with Image

def input_fn_builder_img(img_features,features,seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)
    hidden_shape_img = img_features.shape[-1]

    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),

        "img_features":
            tf.constant(img_features, shape = [num_examples,hidden_shape_img], dtype = tf.float32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def input_fn_builder_pr_img(img_features,features,hidden_context,seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)
    hidden_shape_img = img_features.shape[-1]
    hidden_shape = hidden_context.shape[-1]

    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),

        "img_features":
            tf.constant(img_features, shape = [num_examples,hidden_shape_img], dtype = tf.float32),

        "hidden_context":
            tf.constant(hidden_context, shape = [num_examples,hidden_shape], dtype = tf.float32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn

# Trainer Functions for BERT

Epochs = 10           # Number of Training Epochs

def train_img(img_features,output_dir,input_fn,input_fn_builder_progressive = False,hidden_context = None):
  CONFIG_FILE = "multi_cased_L-12_H-768_A-12/bert_config.json"
  INIT_CHECKPOINT = "multi_cased_L-12_H-768_A-12/bert_model.ckpt"

  BATCH_SIZE = 28
  LEARNING_RATE = 2e-5
  NUM_TRAIN_EPOCHS = Epochs
  # Warmup is a period of time where hte learning rate is small and gradually increases--usually helps training.
  WARMUP_PROPORTION = 0.1
  # Model configs
  SAVE_CHECKPOINTS_STEPS = 6000
  SAVE_SUMMARY_STEPS = 100
  OUTPUT_DIR = output_dir
  # Compute # train and warmup steps from batch size
  num_train_steps = int(len(input_fn) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
  num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
  print(num_train_steps)
  run_config = tf.estimator.RunConfig(
      model_dir=OUTPUT_DIR,
      save_summary_steps=SAVE_SUMMARY_STEPS,
      save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

  # Specify outpit directory and number of checkpoint steps to save
  if input_fn_builder_progressive==False:

    model_fn = model_fn_builder_img(
      bert_config=bert.run_classifier.modeling.BertConfig.from_json_file(CONFIG_FILE),
      num_labels=4, #number of unique labels
      init_checkpoint=INIT_CHECKPOINT,
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=False,
      use_one_hot_embeddings=False
    )

    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={"batch_size": BATCH_SIZE})

    train_input_fn = input_fn_builder_img(
        img_features = img_features,
        features=input_fn,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)


  else:

    model_fn_pr = model_fn_builder_img_progressive(
      bert_config=bert.run_classifier.modeling.BertConfig.from_json_file(CONFIG_FILE),
      num_labels=4, #number of unique labels
      init_checkpoint=INIT_CHECKPOINT,
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=False,
      use_one_hot_embeddings=False
    )



    estimator = tf.estimator.Estimator(
      model_fn=model_fn_pr,
      config=run_config,
      params={"batch_size": BATCH_SIZE})


    train_input_fn = input_fn_builder_pr_img(
        img_features = img_features,
        features=input_fn,
        hidden_context=hidden_context,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

  print(f'Beginning Training!')
#   %timeit

  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  return estimator

# Evaluation Functions for BERT (With and Without Image)
"""
*   CTX = 0 for English Premise and Hindi Hypothesis
*   CTX = 1 for Hindi Premise and English Hypothesis
"""
def evaluate_and_get_hidden_context_img(ctx,img_features_for_test,img_features,estimator,input_fn_for_test,input_fn_for_hidden,is_progressive = False,hidden_context=None):
  MAX_SEQ_LENGTH = 128

  if not is_progressive:
    test_input_fn = input_fn_builder_img(
      features=input_fn_for_test,
      img_features = img_features_for_test,
      seq_length = MAX_SEQ_LENGTH,
      is_training = False,
      drop_remainder = False)
    actual_labels = []
    if ctx ==0:
      for i in test_eng_hindi['label']:
        actual_labels.append(i)
    elif ctx==1:
      for i in test_hindi_eng['label']:
        actual_labels.append(i)
    res = estimator.predict(test_input_fn)
    predicted_labels = []
    for i in res:
      predicted_labels.append(i['labels'])
    estimator.evaluate(input_fn=test_input_fn, steps=None)
    hidden_input_fn = input_fn_builder_img(
        features=input_fn_for_hidden,
        img_features = img_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)
    res = estimator.predict(hidden_input_fn)
    hidden_context = []
    for i in res:
      hidden_context.append(i["hidden_context"])
    hidden_context = np.array(hidden_context)
    return hidden_context, actual_labels,predicted_labels
  else:
    test_input_fn = input_fn_builder_pr_img(
      img_features = img_features_for_test,
      features=input_fn_for_test,
      hidden_context=hidden_context,
      seq_length=MAX_SEQ_LENGTH,
      is_training=False,
      drop_remainder=False)
    estimator.evaluate(input_fn=test_input_fn, steps=None)
    actual_labels = []
    if ctx ==0:
      for i in test_eng_hindi['label']:
        actual_labels.append(i)
    elif ctx==1:
      for i in test_hindi_eng['label']:
        actual_labels.append(i)

    res = estimator.predict(test_input_fn)
    predicted_labels = []

    for i in res:
      predicted_labels.append(i['labels'])
    return actual_labels,predicted_labels


"""
Training and Evaluation 1
"""

# Training for Hindi Premise and English Hypothesis (Image Included)
estimator = train_img(train_img_features,'out_dir_train_hindi_img',train_features_hindi,input_fn_builder_progressive = False,hidden_context = None)

# Evaluation and Hidden Context generation for Hindi Premis and English Hypothesis (Image Included)
"""
*   Hidden Context Obtained
*   Classification Report
"""
hidden_context_hindi_img, act_lab, pred_lab = evaluate_and_get_hidden_context_img(1,test_img_features,train_img_features,estimator,input_fn_for_test = test_features_hindi,input_fn_for_hidden = train_features_hindi,is_progressive = False)

np.array(act_lab).dump(open('HE_Actual_labels_Normal_Image.npy', 'wb'))
np.array(pred_lab).dump(open('HE_Predicted_labels_Normal_Image.npy', 'wb'))

y_true = list(np.load('HE_Actual_labels_Normal_Image.npy', allow_pickle=True))
y_pred = list(np.load('HE_Predicted_labels_Normal_Image.npy', allow_pickle=True))
target_names = ['Contradiction', 'Neutral', 'Entailment','Other']
print(classification_report(y_true, y_pred, target_names=target_names))

np.array(hidden_context_hindi_img).dump(open('Hidden_Context_Hindi_Image.npy', 'wb'))

"""
Training and Evaluation 2
"""

# Progressive Training for English Premise and Hindi Hypothesis (Image Included)
hidden_context_hindi_img = np.load('/content/gdrive/My Drive/COLING 2020/Hidden_Context_Hindi_Image.npy', allow_pickle=True)
estimator = train_img(train_img_features,'out_dir_train_eng_pro_img',train_features_eng,input_fn_builder_progressive = True, hidden_context = hidden_context_hindi_img)

# Evaluation and Hidden Context generation for English Premis and Hindi Hypothesis (Image Included) (Progressive Variant)
"""
*   Hidden Context Obtained
*   Classification Report
"""

Test_batch_size = 3593    # Test Split Size

dummy = np.random.randn(Test_batch_size,768)
act_lab, pred_lab = evaluate_and_get_hidden_context_img(0,test_img_features,train_img_features,estimator,input_fn_for_test = test_features_eng,input_fn_for_hidden = train_features_eng,is_progressive = True,hidden_context=dummy)

np.array(act_lab).dump(open('EH_Actual_labels_Progressive_Image.npy', 'wb'))
np.array(pred_lab).dump(open('EH_Predicted_labels_Progressive_Image.npy', 'wb'))

y_true = list(np.load('EH_Actual_labels_Progressive_Image.npy', allow_pickle=True))
y_pred = list(np.load('EH_Predicted_labels_Progressive_Image.npy', allow_pickle=True))
target_names = ['Contradiction', 'Neutral', 'Entailment','Other']
print(classification_report(y_true, y_pred, target_names=target_names))
