# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:31:45 2022

@author: Juanjo
"""
import os
from PyPDF2 import PdfFileReader, PdfFileWriter
import fitz
import re
import networkx as nx
from itertools import combinations, permutations, product
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer      
from sklearn.decomposition import LatentDirichletAllocation  
from fpdf import FPDF



def import_pdf(path):
    open_pdf = open(path, "rb")
    reader = PdfFileReader(open_pdf)
    
    doc = fitz.open(path)
    toc = doc.get_toc()
    
    capitulo = 1 #Inicializo para nombrar variables dinámicas
    capitulos = [] #Guardo capitulos en una lista de str
    titulos = []
    for i in range(0,len(toc)-1): 
        if toc[i][0] == 2: #SOLO SACO CAPÍTULOS (nivel 2)
            nivel = toc[i][0]
            titulo = toc[i][1]
            titulos.append(titulo)            
            
            pag_inicio = toc[i][2]
            pag_fin = toc[i+1][2]
            #Guardamos cada capítulo en una variable string
            globals()["cap%s" % capitulo] = ""
            for pag in range(pag_inicio-1, pag_fin-1):
                pagina = doc.get_page_text(pag)
                num_pag = str(pag-28)
                pagina = pagina.replace(num_pag, " ")
                globals()["cap%s" % capitulo] += pagina
            
            #Añado cada capítulo a la lista que contendrá todos los capítulos.
            capitulos.append(globals()["cap%s" % capitulo])
            capitulo = capitulo + 1 #Siguiente capítulo
        else: pass

    return capitulos, titulos


def preprocessing(capitulos, titulo_base, titulos):   
    #Eliminación del texto no deseado.
    for i in range(0,len(capitulos)):
        
        #Eliminación de nombres de título y libro en cabeceras.
        titulo_split = titulos[i].split(": ") 
        titulo_chapter = titulo_split[0]
        titulo_nombre = titulo_split[1].upper()
        titulo_nombre = titulo_nombre.replace("ˆ", "")
        titulo_nombre = titulo_nombre.replace("´", "")
      
        #Corrección de codificación
        capitulos[i] = capitulos[i].replace("ˆ ", "")
        capitulos[i] = capitulos[i].replace("´ ", "")
        
        #Eliminación de titulos de capítulos
        capitulos[i] = re.sub(rf"(?i){re.escape(titulo_nombre)}", " ", capitulos[i])
        capitulos[i] = re.sub(rf"(?i){re.escape(titulo_base)}", " ", capitulos[i])
        capitulos[i] = re.sub(r"(?i)(chapter).*", " ", capitulos[i])
        
        #Eliminación de caracteres especiales del texto.
        capitulos[i] = capitulos[i].replace("*", "")
        capitulos[i] = capitulos[i].replace("Mr.", "Mr")
        capitulos[i] = capitulos[i].replace(".’", "’.") #Modifico el orden para que no separe en frases.
        
        #Eliminación de espacios iniciales/finales y saltos de linea intermedios.
        capitulos[i] = capitulos[i].lstrip()
        capitulos[i] = re.sub(r"\n"," ",capitulos[i])
        capitulos[i] = re.sub("\s\s+", " ", capitulos[i])
        
    return capitulos

def generator(capitulos, nlp):
    sentences_total = []
    nouns_total= []
    G_total = []
    nouns_hubs_total = []
    noun_edges_total = []
    
    for i in range(0,len(capitulos)):
        
            
        #División en frases        
        globals()["sentences%s" % i] = capitulos[i].split('.') #División en frases de cada capítulo.
        sentences_total.append(globals()["sentences%s" % i])
        
        #Generador lista de nombres por frase
        globals()["nouns%s" % i] = noun_sentences(globals()["sentences%s" % i], nlp)
        nouns_total.append(globals()["nouns%s" % i])
        
        #Generador del grafo de nombres por capitulos    
        globals()["G%s" % i], globals()["noun_hubs%s" % i], globals()["noun_edges%s" % i] = graph_generator(globals()["nouns%s" % i])
        #G, noun_hubs, noun_edges = graph_generator(nouns)
        G_total.append(globals()["G%s" % i])
        nouns_hubs_total.append(globals()["noun_hubs%s" % i])
        noun_edges_total.append(globals()["noun_edges%s" % i])
    return sentences_total, nouns_total, G_total, nouns_hubs_total, noun_edges_total


def noun_sentences(sentences, nlp):
    nouns = [] #Creación de listas de nombres
    for sentence in sentences:
        token_nouns = []
        doc = nlp(sentence)
        for token in doc:
            if(token.pos_ == 'NOUN'):
                token_nouns.append(token.lemma_.lower())
        
        for ent in doc.ents:
            if ent.label_ == "PERSON" or ent.label_ == "ORG":
                token_nouns.append(ent.text.lower())
        nouns.append(token_nouns) #Las guardo en listas distintas sustantivos y verbos
    return nouns


def graph_generator(nouns):
    #Inicialización del grafo
    G = nx.Graph()   
    #Creación de listas de los hubs
    noun_hubs = [] #Inicializo para no repetir.
    noun_edges = [] #Inicializo lista de aristas bidireccionales sustantivos en misma frase
    for i in range(0,len(nouns)-1):
        for noun in nouns[i]:
            if noun not in noun_hubs: #Sin repetición de hubs
                G.add_node(noun) #Grafico hubs de nombres.
                noun_hubs.append(noun) 
                
            else: pass
    
        noun_combination = combinations(nouns[i],2) #Combinaciones de nombres en frases
    
        for edge in noun_combination:
            noun_edges.append(edge) #Completo lista de aristas sust
    
    
    for combination in noun_edges:
        G.add_edge(combination[0], combination[1]) #Grafico aristas entre nombres
    return G, noun_hubs, noun_edges


def graph_filtration(G, n_filter):
    dic_hubs = sorted(G.degree()) #Dic de grado nodos ordenados.
    grados_hubs = [] #Lo necesito en forma de lista.
    for i in range(0,len(dic_hubs)):
        nodo_grado = []
        nodo_grado.append(dic_hubs[i][1])
        nodo_grado.append(dic_hubs[i][0])
        grados_hubs.append(nodo_grado)
    
    grados_hubs.sort(reverse=True)
     
    filtro = int(len(grados_hubs) * n_filter)
    grados_hubs = grados_hubs[0:filtro]
    return grados_hubs

def compresion_hubs(grados_hubs, compresion_factor):
    compresion = int(len(grados_hubs) * compresion_factor)
    grados_hubs = grados_hubs[0:compresion]
    return grados_hubs


def compresion_sentences(nouns, grados_hubs, compresion_factor):
    lista_index = []
    hubs = []
    for i in range(0,len(grados_hubs)):
        hubs.append(grados_hubs[i][1]) #Lista de hubs
    for i in range(0,len(nouns)):
        for e in range(0,len(nouns[i])):
            ind_pond = [] #Lista para ponderacion, indice y luego ordenar
            ponderacion = 0 #Pondero cada frase por la cantidad de nodos principales
            #for j in range(0,len(nouns[i][e])):
            for noun in nouns[i][e]:
                if noun in hubs:
                    ponderacion = ponderacion + 1 #Sumo uno por cada nodo de la frase que es hub
            ind_pond.append(ponderacion)
            ind_pond.append(i)
            ind_pond.append(e)
            lista_index.append(ind_pond)
    lista_index.sort(reverse=True)
    compresion = int(len(lista_index) * compresion_factor)
    lista_index = lista_index[0:compresion]
    return lista_index
 
       
def resume_sentences(sentences, lista_index):
    resumen = "" #Inicializo resumen
    for i in range(0,len(sentences)):
        for e in range(0,len(sentences[i])):
            for hub in lista_index:
                if hub[1]==i and hub[2]==2: #En orden cronológico
                    resumen = resumen + "." + sentences[i][e] #Añado cada frase que contiene hub
    resumen = resumen[1:-2]
    return resumen


def features_extraction(sentences_total, titulos):
    chapter_features = []
    for i in range(0, len(sentences_total)):        
        count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words="english")        
        doc_term_matrix = count_vect.fit_transform(sentences_total[i])
        LDA = LatentDirichletAllocation(n_components=5, random_state=42) 
        LDA.fit(doc_term_matrix)
        
        first_topic = LDA.components_[0]
        top_topic_words = first_topic.argsort()[-5:]
        print("\n TOP TOPICS {}: ".format(titulos[i]))
        features = []
        for e in top_topic_words:
            print(count_vect.get_feature_names()[e])
            features.append(count_vect.get_feature_names()[e])
        chapter_features.append(features)
    return chapter_features
        
def compose_graph(G_total): #Composición del grafo para el pdf completo  
    H = G_total[0]
    for i in range(1,len(G_total)):
        H = nx.compose(H, G_total[i])
    return H


def print_graphs(G_total, H):
    print("GRAFO GLOBAL: ")
    nx.draw(H, with_labels=True)
    for i in range(0,len(G_total)):
        print("GRAFO {}: ".format(i))
        nx.draw(G_total[i], with_labels=True)
        
def resumen_final(resumen, chapter_features, titulos):
    
    text = resumen + "\n\n\n" + "TOPICS:" + "\n"
    for i in range(0,len(chapter_features)):
        text = text + "\n TOP TOPICS {}: ".format(titulos[i])
        for e in range(0,len(chapter_features[i])):
            text = text + "\n" + chapter_features[i][e]
    return text

def write_pdf(text, chapter_features, titulos):
        
    pdf = FPDF()       
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    pdf.write(5,text)  
    pdf.output("resume.pdf", 'F')

    return text
     
'''
d = dict(G_total[0].degree)
nx.draw(G_total[0], nodelist=d.keys(), node_size=[v * 2 for v in d.values()])
        
'''        

