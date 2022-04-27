# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 18:56:16 2022

@author: Juanjo
"""

import matplotlib.pyplot as plt
from functions import import_pdf, preprocessing, generator, noun_sentences, graph_generator, graph_filtration, compresion_hubs
from functions import compresion_sentences, resume_sentences, features_extraction, compose_graph, print_graphs, write_pdf, resumen_final
import spacy
import networkx as nx
import numpy as np
import gensim
from gensim import corpora



#Carga del pdf
path = "./j-r-r-tolkien-lord-of-the-rings-01-the-fellowship-of-the-ring.pdf"
capitulos, titulos = import_pdf(path)

#Preprocesamiento del pdf
titulo_base = "the fellowship of the ring"
capitulos = preprocessing(capitulos, titulo_base, titulos)

#Generación de listas de frases, nombres, grafos, hubs y aristas por grafos de cada capítulo.
nlp = spacy.load('en_core_web_sm')       
sentences_total, nouns_total, G_total, nouns_hubs_total, noun_edges_total = generator(capitulos, nlp)


#Composición de grafo con todos los capítulos
H = compose_graph(G_total)


#Generador de nodos en orden por grado
n_filtro = 0.8 #Eliminición de nodos sin utilidad
grados_hubs = graph_filtration(H, n_filtro)

#Factor de compresión hubs
compresion_factor = 0.2
grados_hubs = compresion_hubs(grados_hubs, compresion_factor)


#Factor compresión frases
compresion_factor = 0.2
lista_index = compresion_sentences(nouns_total, grados_hubs, compresion_factor)


#Unión de frases resumen
resumen = resume_sentences(sentences_total, lista_index)


#Temática por capitulo
chapter_features = features_extraction(sentences_total, titulos)
    

final_resume = resumen_final(resumen, chapter_features, titulos)


#Impresión de grafos
print_graphs(G_total, H) 


#Generación pdf
'''
resumen.encode("utf-8").decode("latin-1")
resume = write_pdf(final_resume, chapter_features, titulos)
'''






