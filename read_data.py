import pandas as pd
import scipy.sparse as sparse
import numpy as np
import re
import gc
from scipy.sparse.linalg import spsolve
import collections
import math

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#%reset -f
cleared = [12442, 12522, 12586, 12602, 12702, 12732, 12770, 12829, 12837, 12879, 13325, 13436, 13508, 13631, 13666, 13697, 13718, 13754, 13809, 13826, 13898, 13960, 14079, 14145, 14218, 14327, 14434, 14501, 14523, 14569, 14798, 14804, 14860, 14957, 15007, 15263, 15277, 15299, 15307, 15385, 15393, 15670, 15678, 15692, 15724, 15836, 15886, 15917, 15922, 15925, 15935, 16096, 16302, 16596, 16641, 16792, 16832, 16847, 16989, 16997, 17010, 17110, 17262, 17309, 17334, 17492, 17572, 17781, 17877, 17906, 17939, 17978]
'''
cleaned_retail_user = pd.read_csv("cleaned_retail.csv", sep='\t') # This may take a couple minutes
invalids =[]
IDs = []
for i in range(len(cleaned_retail_user.CustomerID)):
    #print temp_i
    #print inv
    if( cleaned_retail_user.CustomerID[i] in cleared):
	invalids.append(i)
	IDs.append(cleaned_retail_user.CustomerID[i])

V = pd.DataFrame({'product':valids})
I = pd.DataFrame({'CustomerID':IDs})
merged = pd.concat([I, V], axis=1)
	
cleaned_retail_user.drop(cleaned_retail_user.index[invalids], inplace=True)
cleaned_retail_user.to_csv('cleaned_retail_dropped.csv', sep='\t', encoding='utf-8')
print('dropped')
'''
website_url = 'cleaned_retail_dropped.csv'
cleaned_retail = pd.read_csv(website_url, sep='\t') # This may take a couple minutes
vocabulary = []
for i in range(len(cleaned_retail.Description)):
  vocabulary.append(cleaned_retail.Description[i])

print(len(vocabulary))
vocabulary_size = 3895
'''
total_invoice = int(cleaned_retail.InvoiceNo[405965]) - int(cleaned_retail.InvoiceNo[0])
invoice_array = np.zeros(total_invoice)
invoice_length = np.zeros(1000)
for i in range(total_invoice):
    invoice_array[int(re.sub("[^0-9]","",str(cleaned_retail.InvoiceNo[i])))-536365] +=1

for idx, leng in enumerate(invoice_array):
    invoice_length[int(leng)] +=1 

print invoice_length[:200]
'''
def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  print("building dataset")
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
#print sorted(count)
print(len(dictionary))

#print(invoice_array)
print(cleaned_retail.head())
cleaned_retail.info(memory_usage="deep")

#cleaned_retail.to_csv("cleaned_retail.csv", sep='\t', encoding='utf-8')
temp_i = ""
count = 0
rec_temp = []
rec_vocab = []
combinations = []
for i in range(len(cleaned_retail.InvoiceNo)):
    #print temp_i
    #print inv
    if(temp_i == cleaned_retail.InvoiceNo[i]):
	#print cleaned_retail.Description[i]
	#print i
	if(count == 0):
	    
	    rec_temp.append(cleaned_retail.Description[i-1])
	    rec_temp.append(cleaned_retail.Description[i])
	else:
	    rec_temp.append(cleaned_retail.Description[i])
	count+=1
    else:
	if(count>0):
	    rec_vocab.append(sorted(rec_temp))
	count = 0;
	rec_temp = []
    temp_i = cleaned_retail.InvoiceNo[i]
#print rec_vocab
gc.collect()
del cleaned_retail

labels=[]
inputs=[]
'''
for i in range(len(rec_vocab)):
  #print rec_vocab[i]
  if(len(rec_vocab[i])<5):
    for j, words in enumerate(rec_vocab[i]):
      for k2 in range(len(rec_vocab[i])-1):
        inputs.append(dictionary[words]) 
      for k in range(len(rec_vocab[i])):
        if(j != k):
	  labels.append(dictionary[rec_vocab[i][k]])
	  #combinations.append(float(dictionary[words]) + dictionary[rec_vocab[i][k]]/10000.0)
	  #print rec_vocab[i][k]
  else:
    for j in range(len(rec_vocab[i])-4):
      for k2 in range(4):
	inputs.append(dictionary[rec_vocab[i][j+2]])
      for k in range(5):
	if((j+2) != (j+k)):
	  labels.append(dictionary[rec_vocab[i][j+k]])
'''
for i in range(len(rec_vocab)):
  #print rec_vocab[i]
  for j, words in enumerate(rec_vocab[i]):
    for k2 in range(len(rec_vocab[i])-1):
      inputs.append(dictionary[words]) 
    for k in range(len(rec_vocab[i])):
      if(j != k):
	labels.append(dictionary[rec_vocab[i][k]])
	#combinations.append(float(dictionary[words]) + dictionary[rec_vocab[i][k]]/10000.0)
	#print rec_vocab[i][k]

cleaned_retail_user = pd.read_csv("cleaned_retail_user(train).csv", sep='\t') # This may take a couple minutes
temp_i = ""
count = 0
rec_temp = []
rec_vocab = []
#customer_item_length=[]
#item_length = np.zeros(10000)
for i in range(len(cleaned_retail_user.CustomerID)):
    #print temp_i
    #print inv
    if(temp_i == cleaned_retail_user.CustomerID[i]):
	#print cleaned_retail.Description[i]
	#print i
	if(count == 0):
	    rec_temp.append(cleaned_retail_user.Description[i-1])
	    rec_temp.append(cleaned_retail_user.Description[i])
	else:
	    rec_temp.append(cleaned_retail_user.Description[i])
	count+=1
    else:
	if(count>0):
	    rec_vocab.append(rec_temp)
	#if(count == 11):
	    #customer_item_length.append(cleaned_retail_user.CustomerID[i-1])
	count = 0;
	rec_temp = []
    temp_i = cleaned_retail_user.CustomerID[i]
print len(rec_vocab)
#print customer_item_length[:]
gc.collect()
'''
for i, leng in enumerate(customer_item_length):
    item_length[leng] +=1

print item_length[:1000]
'''
del cleaned_retail_user
'''
for i in range(len(rec_vocab)):
  #print rec_vocab[i]
  if(len(rec_vocab[i])<5):
    for j, words in enumerate(rec_vocab[i]):
      for k2 in range(len(rec_vocab[i])-1):
        inputs.append(dictionary[words]) 
      for k in range(len(rec_vocab[i])):
        if(j != k):
	  labels.append(dictionary[rec_vocab[i][k]])
	  #combinations.append(float(dictionary[words]) + dictionary[rec_vocab[i][k]]/10000.0)
	  #print rec_vocab[i][k]
  else:
    for j in range(len(rec_vocab[i])-4):
      for k2 in range(4):
	inputs.append(dictionary[rec_vocab[i][j+2]])
      for k in range(5):
	if((j+2) != (j+k)):
	  labels.append(dictionary[rec_vocab[i][j+k]])
'''
for i in range(len(rec_vocab)):
  #print rec_vocab[i]
  for j, words in enumerate(rec_vocab[i]):
    for k2 in range(len(rec_vocab[i])-1):
      inputs.append(dictionary[words]) 
    for k in range(len(rec_vocab[i])):
      if(j != k):
	labels.append(dictionary[rec_vocab[i][k]])
	#combinations.append(float(dictionary[words]) + dictionary[rec_vocab[i][k]]/10000.0)
	#print rec_vocab[i][k]


'''

data_comb, count_comb, dictionary_comb, reverse_dictionary_comb = build_dataset(combinations,100000)

count_combinations = list(sorted(count_comb))
count_combinations = count_combinations[1:]
del data_comb, count_comb, dictionary_comb, reverse_dictionary_comb
ref_item = []
target_item = []
ref_item_name = []
target_item_name = []
freq = []
for comb in count_combinations:
    comb = list(comb)
    if(comb[0] != "UNK"):
        ref_item.append(int(comb[0]))
	ref_item_name.append(reverse_dictionary[int(comb[0])])
        target_item.append((comb[0]-int(comb[0]))*10000)
	target_item_name.append(reverse_dictionary[int((comb[0]-int(comb[0]))*10000)])
        freq.append(comb[1])
ref_item = pd.DataFrame({"ref":ref_item})
target_item = pd.DataFrame({"target":target_item})
ref_item_name = pd.DataFrame({"ref_name":ref_item_name})
target_item_name = pd.DataFrame({"target_name":target_item_name})
freq = pd.DataFrame({"freq":freq})
data_merged = pd.concat([ref_item, target_item,ref_item_name,target_item_name,  freq], axis=1)
data_merged.to_csv('sorted_combinations.csv', sep='\t', encoding='utf-8')
#count_combinations.to_csv('sorted_combinations.csv', sep='\t', encoding='utf-8')
del count_combinations
'''
print len(inputs)
print len(labels)
#indices = np.arange(len(labels))
#shuffle the data and labels
#np.random.shuffle(indices)
#inputs = inputs[indices]
#labels = labels[indices]


data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size): #, num_skips, skip_window):
  global data_index
  #assert batch_size % num_skips == 0
  #assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  label = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  #span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  #buffer = collections.deque(maxlen=span)
  #if data_index > len(inputs):
  #  data_index = 0
  #buffer.extend(data[data_index:data_index + span])
  #print(buffer)
  #data_index += span
  for i in range(batch_size):
    if data_index == len(inputs):
      data_index = 0

    batch[i] = inputs[data_index]
    label[i, 0] = labels[data_index]
    data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  #data_index = (data_index + len(data) - span) % len(data)
  return batch, label
'''
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
'''
# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 300  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 3895     # Random set of words to evaluate similarity on.
valid_window = 3895  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  show_emb = tf.Print(valid_embeddings, [valid_embeddings])
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()
num_steps = 115000001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size)
    #print(batch_inputs)
    #print(batch_labels)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    #print loss_val
    average_loss += loss_val

    if step % 2000 == 0:
      #print(feed_dict)
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0
    vocab_words = []
    nearests = []
    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 2300000 == 0:
      sim = similarity.eval()
      show = show_emb.eval()
      p_embeddings = []
      for s in show:
	p_embeddings.append(s)
      #for s in show:
	
      PE = pd.DataFrame({"emb":p_embeddings})
      PE.to_csv('product_emb_300d%d.csv'%(step/2300000), sep='\t', encoding='utf-8')
      for i in xrange(3895):
	#print i
        valid_word = reverse_dictionary[i]
        top_k = 50  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
	#print nearest
        log_str = ''
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
	vocab_words.append(valid_word)
        nearests.append(log_str)
      w = pd.DataFrame({'product':vocab_words})
      n = pd.DataFrame({'nearest':nearests})
      data_merged = pd.concat([w, n], axis=1)
      data_merged.to_csv('product_nearest_300d_%depoch.csv'%(step/2300000), sep='\t', encoding='utf-8')
  final_embeddings = normalized_embeddings.eval()

def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(80, 80))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=100000, n_iter_without_progress=5000,learning_rate=100.0, method='exact',verbose=2)
  plot_only = 1000
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, 'tsne_300d_%depoch.png'%(step/2300000)) #os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
'''
item_lookup = cleaned_retail[['StockCode', 'Description']].drop_duplicates() # Only get unique item/description pairs
item_lookup['StockCode'] = item_lookup.StockCode.astype(str) # Encode as strings for future lookup ease
print(item_lookup.head())

cleaned_retail['CustomerID'] = cleaned_retail.CustomerID.astype(int) # Convert to int for customer ID
cleaned_retail = cleaned_retail[['StockCode', 'Quantity', 'CustomerID']] # Get rid of unnecessary info
grouped_cleaned = cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index() # Group together
grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1 # Replace a sum of zero purchases with a one to
# indicate purchased
grouped_purchased = grouped_cleaned.query('Quantity > 0') # Only get customers where purchase totals were positive
print(grouped_purchased.head())

customers = list(np.sort(grouped_purchased.CustomerID.unique())) # Get our unique customers
products = list(grouped_purchased.StockCode.unique()) # Get our unique products that were purchased
quantity = list(grouped_purchased.Quantity) # All of our purchases
'''
