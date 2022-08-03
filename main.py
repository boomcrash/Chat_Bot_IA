#PARTE 1
#permite el procesamiento de lenguaje natural
import nltk
#permite minimizar las palabras 
from nltk.stem.lancaster import LancasterStemmer
#intanciamos el minimizador
stemmer=LancasterStemmer()
#permite convertir casi lo que sea en arreglos
import numpy
#herramienta de deep learning
import tflearn
import tensorflow
from tensorflow.python.framework import ops
#permite crear bases de datos
import json
#permite crear numeros aleatorios
import random
#permite guardar los modelos de entramientos (mejora la velocidad, ya que no hay que entrenar desde 0)
import pickle
#permite descargar el paquete punkt
nltk.download('punkt')
import requests
import os
import matplotlib as mltp

print("finalizado")

import dload
dload.git_clone("https://github.com/boomcrash/data_bot.git")

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path=dir_path.replace("\\","/")
with open(dir_path+'/data_bot\data_bot-main/data.json','r') as file:
  database=json.load(file)


#PARTE 2  
words=[]
all_words=[]
tags= []
aux= []
auxA= []
auxB= []
training=[]
exit=[]
 
try:
  with open("Entrenamiento/brain.pickle","rb") as pickleBrain:
    all_words,tags,training,exit=pickle.load(pickleBrain)
except:
  for intent in database['intents']:
    #recorremos los tags
    for pattern in intent['patterns']:
      #separamos una frase en palabras
      auxWords=nltk.word_tokenize(pattern)
      #guardamos las palabras
      auxA.append(auxWords)
      auxB.extend(auxWords)
      #guardamos los tags
      aux.append(intent['tag'])
  #simbolos a ignorar
  ignore_words=['?','!','.',',','¿',"'","$","·","-","_","&","%","/","(",")", "=","#"]
  for w in auxB:
    if w not in ignore_words:
      words.append(w)
  #truco para evitar repetidos
  words=sorted(set(words))
  print(words)    
  tags=sorted(set(aux))
  print(tags)
  #convertimos a minuscula
  all_words=[stemmer.stem(w.lower()) for w in words]
  print("SIN ORDENAR")
  print(len(all_words))
  #ordenamos la lista
  ##print("ORDENADO")
  all_words=sorted(list(set(all_words)))
  ##print(all_words)
  #ordenamos los tags
  tags=sorted(tags)
  ##print("TAGS ORDENADO")
  ##print(tags)
  training=[]
  exit=[]
  #creamos una salida falsa
  null_exit=[0 for _ in range(len(tags))]
  #recorremos el auxiliar de palabras
  for i, document in enumerate(auxA):
    bucket=[]
    #hacemos minuscula y quitamos signos
    auxWords=[stemmer.stem(w.lower())for w in document if w!="?"]
    #recorremos las palabras
    for w in all_words:
      if w in auxWords:
        bucket.append(1)
      else:
        bucket.append(0)
    exit_row=null_exit[:]
    exit_row[tags.index(aux[i])]=1
    training.append(bucket)
    exit.append(exit_row)
    ##print(training)
    ##print(exit)
     #pasamos la lista del entrenamiento a un arreglo numpy
  training=numpy.array(training)
  ##print(training)
  exit=numpy.array(exit)
  ##print(exit)
  #crear un archivo pickle para almacenar los datos entrenados
  with open("Entrenamiento/brain.pickle","wb") as pickleBrain:
    pickle.dump((all_words,tags,training,exit),pickleBrain)

#print(training[0])

#print(len(training[0]))
#PARTE 3
#reiniciar red neuronal
tensorflow.compat.v1.reset_default_graph()
tflearn.init_graph(num_cores=8,gpu_memory_fraction=0.5)
#creamos una red neuronal #pasamos longitud de la fila
#creamos las neuronas conectadas
#neuronas de entrada(input layers)
net=tflearn.input_data(shape=[None,len(training[0])])
#neuronas intermedias(hidden layers)
net=tflearn.fully_connected(net,100,activation='Relu')
net=tflearn.fully_connected(net,50)
net=tflearn.dropout(net,0.5)
#neuronas de salida(exit layers)
net=tflearn.fully_connected(net,len(exit[0]), activation='softmax')
#aplicamos regresion a nuestra red
net=tflearn.regression(net, optimizer='adam',learning_rate=0.01, loss='categorical_crossentropy')
model =tflearn.DNN(net)
#print(len(all_words))
#moldeamos el modelo (n_epoch son las veces que se revisa la base de datos para obtener respuestas)
#bath_size son el # de entradas
#show metric muestrav el contenido



if os.path.isfile(dir_path+"/Entrenamiento/model.tflearn.index"):
  model.load(dir_path+"/Entrenamiento/model.tflearn")
else:
  model.fit(training,exit,validation_set=0.1, show_metric=True, batch_size=128,n_epoch=1000)
  model.save("Entrenamiento/model.tflearn")



#PARTE 4
def response(texto):
  if texto=="duerme":
    print("HA SIDO UN GUSTO, VUELVE PRONTO....")
    return False
  else:
    bucket=[0 for _ in range(len(all_words))]
    processed_sentence=nltk.word_tokenize(texto)
    processed_sentence=[stemmer.stem(palabra.lower()) for palabra in processed_sentence]
    for individual_word in processed_sentence:
      for i,palabra in enumerate(all_words):
        if palabra==individual_word:
          bucket[i]=1
    results=model.predict([numpy.array(bucket)])
    index_results=numpy.argmax(results)
    max=results[0][index_results]
    #print(index_results)
    #print(max)
    target=tags[index_results]
    #print(tags)
    #print(target)
    for tagAux in database['intents']:
      if tagAux['tag']==target:
         answer=tagAux['responses']
         answer=random.choice(answer)
    print(answer)
    return True

print("HABLA CONMIGO")
bool=True
while bool==True:
  texto=input()
  bool=response(texto)
