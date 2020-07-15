# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:36:37 2020

@author: Nagesh K J
"""



#!pip install --upgrade pymupdf
import fitz

import pytesseract
from PIL import Image


import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import spacy



import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
from validator_collection import validators, checkers


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords 
import wikipedia 

from tabula import read_pdf
from tabula import convert_into

# pip install python-docx
from docx import Document

import pandas as pd


import string

import re
import os, docx2txt, io, sys

from datetime import datetime
from os import path
import unidecode




# PATH for storing images found in files

pdf_path = "pdf/"
url_path = "url/"
doc_path =  "doc/"

if not path.exists(pdf_path):
    os.makedirs(pdf_path)
if not path.exists(url_path):
	os.makedirs(url_path)
if not path.exists(doc_path):
	os.makedirs(doc_path)





##################### Text Preprocessing ####################################################################################
# does lemmatization and stop word removal for a string and returns the processed string
def process(l):
  string = ""
  pos=[]
  chunk=[]
  l=re.sub("\n+","\n",l)
  # print(l)
  for i in l.split('.'):
    i=i.lower()#to lowercase
    i=i.strip()#remove whitespaces
    # i=unidecode.unidecode(i)#""remove accented characters from text, e.g. café"""
    w = word_tokenize(i)
    # print('woRD',w)
    ##assigning the parts of speech
    # pos.append(nltk.pos_tag(w))
    # chunk.append(ne_chunk(pos[-1]))



    sw = stopwords.words('english')
    # print("STOPWORDS",sw)
    sw.extend(list(".,/()[]{}|~:?@<=>;#$%&!+-*/'"))
    result = [x for x in w if x not in sw]

    res_lem = []
    for w in result:
      res_lem.append(wordnet_lemmatizer.lemmatize(w))
    text = " ".join(res_lem)
    string += text+". "
  # print(pos,"\n\n",chunk)
  return string

################################Text Preprocessing -- END -- #########################################################################







#################### Image Text Extraction #######################################################

def OCR(path):
    try:
        if(checkers.is_url(path)):
          urllib.request.urlretrieve(path, "tmp."+path.split(".")[-1])
          path="tmp"+path.split(".")[-1]
        text = pytesseract.image_to_string(Image.open(path))
        # print("ocr",text)
        return text
    except Exception as e:
        print(e)
        return ""

#################### Image Text Extraction -- END --  #######################################################








#################################### Web Extraction  #########################################################
#Web Extraction
def getWeb(site):  

	dt="url/"+str(datetime.now())+"/"
	if not os.path.exists(dt):
		os.makedirs(dt)


	response = requests.get(site)
	soup = BeautifulSoup(response.text, 'html.parser')
	img_tags = soup.find_all('img')

	# tables=soup.find_all('table')




	urls = [img['src'] for img in img_tags]
	for url in urls:
	    filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
	    if not filename:
	         print("Regex didn't match with the url: {}".format(url))
	         continue
	    with open(dt + filename.group(1), 'wb') as f:
	        if 'http' not in url:
	            # sometimes an image source can be relative 
	            # if it is provide the base url which also happens 
	            # to be the site variable atm. 
	            url = '{}{}'.format(site, url)
	        response = requests.get(url)
	        f.write(response.content)
	        
	html = urllib.request.urlopen(site).read()

	im=""
	for i in os.listdir(dt):
		if(i.split(".")[-1]!="csv"):
			im+=OCR(dt+i)
	t=text_from_html(html)
	table_extract_url(soup,dt)

	return t+im


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)


#################################### Web Extraction -- END -- ###########################################################






############################## PDF Extraction ###########################################################################

def getPdf(pdf):
    # pdf="op.pdf"

    #http://www.cse.iitm.ac.in/~chester/courses/19e_ns/slides/1_Introduction.pdf -- example
    if(checkers.is_url(pdf)):
      urllib.request.urlretrieve(pdf, "tmp.pdf")
      pdf="tmp.pdf"

    dt="pdf/"+str(datetime.now())+"/"
    if not os.path.exists(dt):
      os.makedirs(dt)
    doc = fitz.open(pdf)

    text=""
    for i in range(len(doc)):
        t=doc.loadPage(i)
        # text+="\n\n\nPagedata##############################\n\n\n"
        text+=t.getText()
        for img in doc.getPageImageList(i):
            xref = img[0]  # check if this xref was handled already?
            pix = fitz.Pixmap(doc, xref)
            image_name=dt+"Image# {}.png".format(xref)
            try:
                if path.exists(image_name):
                    continue
                if pix.n < 5:  # this is GRAY or RGB
                    pix.writePNG(image_name)
                    im_text=OCR(image_name)
                    # text+="\n\nIMAge text###################################\n\n"
                    text+=im_text
                else:  # CMYK needs to be converted to RGB first
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)  # make RGB pixmap copy
                    pix1.writePNG(image_name)
                    im_text=OCR(image_name)
                    # text+="\n\nIMAge text######################################\n\n"
                    text+=im_text
                    pix1 = None  # release storage early (optional)
            except Exception as e:
                print(e)
                sys.exit(0)
            pix = None  # release storage early (optional)
    # print("RESULT>>>>>>>>>\n\n\n")
    table_extract_pdf(pdf,dt)
    return text
    # new()

############################## PDF Extraction -- END --  ###########################################################################






############################# Text file ################################################################

def read_text(f):
  #http://www.google.com/robots.txt 
  #https://raw.githubusercontent.com/geekcomputers/Python/master/Colors/pixel_sort.py
  if(checkers.is_url(f)):
    urllib.request.urlretrieve(f, "tmp.txt")
    f="tmp.txt"
  with open(f,'r') as p:
    return p.read()

############################# Text file -- END -- ################################################################




############################# Doc file ################################################################

def getDoc(filename):

    if(checkers.is_url(filename)):
      urllib.request.urlretrieve(filename, "tmp.docx")
      filename="tmp.docx"

    dt=doc_path+str(datetime.now())+"/"

    if not os.path.exists(dt):
      os.makedirs(dt)
    text = docx2txt.process(filename, dt) 

    for i in os.listdir(dt):
    	if(i.split(".")[-1]!="csv"):
      		text+=OCR(dt+i)

    table_extract_doc(filename,dt)
    return text


############################# Doc file -- END -- ################################################################




############################# Table Extraction ##############################################

def table_extract_pdf(source,path):
	df=read_pdf(source,pages='all')

	# print(df[1],"\n")
	for index,i in enumerate(df):
		k=pd.DataFrame(i)
		k.to_csv(path+"Table# "+str(index)+".csv")

	# convert_into(source,"test.csv",output_format="csv",pages='all')

#-----------------------------------------------------------------------------------------------------------------


def table_extract_doc(source,path):
	document = Document(source)
	tables = []
	for index,table in enumerate(document.tables):
	    df = [['' for i in range(len(table.columns))] for j in range(len(table.rows))]
	    for i, row in enumerate(table.rows):
	        for j, cell in enumerate(row.cells):
	            df[i][j] = cell.text
	        #pd.DataFrame(df).to_csv("Table# "+str(index)+".csv")
	        pd.DataFrame(df).to_csv(path+"Table# "+str(index)+".csv")


#-----------------------------------------------------------------------------------------------------------------


def get_table_headers(table):
    """Given a table soup, returns all the headers"""
    headers = []
    for th in table.find("tr").find_all("th"):
        headers.append(th.text.strip())
    return headers

def get_table_rows(table):
    """Given a table, returns all its rows"""
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = []
        # grab all td tags in this table row
        tds = tr.find_all("td")
        if len(tds) == 0:
            # if no td tags, search for th tags
            # can be found especially in wikipedia tables below the table
            ths = tr.find_all("th")
            for th in ths:
                cells.append(th.text.strip())
        else:
            # use regular td tags
            for td in tds:
                cells.append(td.text.strip())
        rows.append(cells)
    return rows



def table_extract_url(soup,path):
	tables = soup.find_all("table")

	print(f"[+] Found a total of {len(tables)} tables.")	
	# iterate over all tables
	for i, table in enumerate(tables, start=1):
	    # get the table headers
	    headers = get_table_headers(table)
	    # get all the rows of the table
	    rows = get_table_rows(table)
	    # save table as csv file
	    table_name = f"Table# {i}"
	    print(f"[+] Saving {table_name}")
	    pd.DataFrame(rows, columns=headers).to_csv(path+"{}.csv".format(table_name))

#-----------------------------------------------------------------------------------------------------------------


############################# Table Extraction -- END -- ##############################################



############################ Theme Based Extraction ##############################

def theme_based_extraction(text,threshold,theme):
  sent_count=0
  sw = stopwords.words('english')
  theme = theme.lower()
  theme_word_list = word_tokenize(theme) 
  theme_word_list = {w for w in theme_word_list if not w in sw}

  d = dict()
  most_similar_sentences = ""
  for sentence in text.split('.'):
    word_list = { w for w in word_tokenize(sentence) }    
    if(len(word_list) < 7):
      continue
    combined_vector = word_list.union(theme_word_list)  
    l1 = []
    l2 = []
    for w in combined_vector: 
      if w in word_list:
        l1.append(1)  
      else:
        l1.append(0) 
      if w in theme_word_list:
        l2.append(1) 
      else:
        l2.append(0)
    c=0
    for i in range(len(combined_vector)): 
      c+= l1[i]*l2[i]
    l1_up=[i*i for i in l1]
    l2_up=[i*i for i in l2]
    res = float((sum(l1_up)*sum(l2_up))**0.5)
    
    if c == 0:
      c = 0.1

    if res == 0:
      res = 0.1
    cosine = c / res 
    if(cosine not in d.keys()):
      d[cosine]=[]
    if(sentence not in d[cosine]):
      sent_count+=1
      d[cosine].append(sentence)
    # else:
    #   print("\n\n\n\n\n\nUNIQUE\n\n\n\n")

  sorted_cosine = sorted(d.keys(),reverse=True)

  if threshold > sent_count:
    threshold = sent_count


  s=0
  for i in d.values():
    for k in i:
      if(s==threshold):
        break
      most_similar_sentences += k
      most_similar_sentences += ".\n"
      s+=1
    if(s==threshold):
      break
  # print("\n\n\n\n\n\n\n\nSENTENCE COUNT ",sent_count,"threshold ",threshold,"S ",s,"\n\n\n\n\n\n\n\n")
  return most_similar_sentences




'''

############################### New theme based extraction ################################


def proc_sentence(s):
  sw = stopwords.words("english")
  s=s.lower()
  s=s.strip()
  if(len(s)==0):
    return ""


  # print(s)
  w=word_tokenize(s)
  # print(w)

  # sw=stopwords

  # print(sw)
  sw=sw+list(".,/()[]{}|~:?@<=>;#$%&!+-*/'")
  # print(sw)

  result = [wordnet_lemmatizer.lemmatize(x) for x in w if x not in sw]
  result=" ".join(result)

  return result

def process1(l):
  string = ""

  l=re.sub("\n+","\n",l)
  # print(l)

  #if only one sentence
  if("." not in l):
    string=proc_sentence(l)
    return string


  for i in l.split('.'):
    string+=proc_sentence(i)

  return string

def new_theme(text,threshold,theme):
  
  m_theme=process1(theme)
  # m_theme=re.sub("\n","",m_theme)
  # m_theme=re.sub("\."," ",m_theme)
  text=re.sub("\n+"," ",text)
  # print(text,"TEXT")
  cosine={}

  count=0

  sn=1
  #pre processing and finding the cosine-value sentence by sentence instead of whole
  #coine-similarity by using sklearn
  for org_sent in text.split("."):
  

    # print("org_sent",org_sent)
    proc_sent=proc_sentence(org_sent)

    # print("proc_sent",proc_sent)
    if(len(proc_sent)==0):
      continue
    # print(m_theme,i)
    # i=clean_string(i)
    cleaned=[proc_sent,m_theme]

    # print("cleaned",cleaned)
    vectorizer=CountVectorizer().fit_transform(cleaned)

    vectors=vectorizer.toarray()

    # print(vectors)

    vec1=vectors[0].reshape(1,-1)
    vec2=vectors[1].reshape(1,-1)

    # print(vec1,vec2)


    csim=cosine_similarity(vec1,vec2)[0][0]

    # print(csim)

    # if(csim<0.2):
    #   continue

    if(csim not in cosine):
      cosine[csim]=[]


    
    # cosine[csim]=set(cosine[csim])
    if(org_sent in [i[0] for i in cosine[csim]]):
      continue
    cosine[csim].append((org_sent,sn))
    sn+=1
    # cosine[csim]=list(cosine[csim])
    # print(cosine)

  cosine=dict(sorted(cosine.items(),reverse=True))


  most_similar_sentences=""

  sent_list=[sent for ic in cosine.values() for sent in ic]

  # if threshold is ratio
  if(threshold<=1):
  	threshold=int(len(sent_list)*threshold)
  else:
  	if(threshold>len(sent_list)):
  		threshold=len(sent_list)

  sent_list=sent_list[:threshold]

  # print(sent_list)
  a=sorted(sent_list,key=lambda x:x[1])
  a=[i[0] for i in a]
  # print(a)
  most_similar_sentences=".\n".join(a)
  return most_similar_sentences

  # for i in range(threshold):
  #   most_similar_sentences+=sent_list[i]+".\n"
  # return most_similar_sentences

############################ Theme Based Extraction --END-- ##############################
'''





###################### Summary #######################################333333333333#######


# gets the percentage summary and word summary given a string
def get_summary(string,wc):
  nlp = spacy.load('en_core_web_sm')
  doc = nlp(string) 
  #print(wc)
  if(wc<=1):
    summ_words = summarize(string, ratio = wc)
    #print(summ_words)
  # Summary (0.5% of the original content). 
  # summ_words = summarize(string, ratio = 0.5)
  
  # Summary (200 words) 
  else:
    summ_words = summarize(string, word_count = wc)
    #print(summ_words)
  #print(summ_words)
  #f = open("M2_summary.txt", "w")
  #f.write(summ_words)
  #f.close() 
  print(summ_words)
  return summ_words

###################### Summary -- END -- #######################################333333333333#######



# simple theme extraction by finding first and last occurance of theme in sentences.
def simple_theme_extraction(text,theme):
  #print(text,theme)
  l = text.split(".")
  occur_list = []
  #print(l)
  for i in range(len(l)):
    if theme in l[i].lower():
      #print("true")
      occur_list.append(i)
  print(occur_list)
  first_occur = occur_list[0]
  last_occur = occur_list[-1]
  sentences =  l[first_occur : last_occur+1]
  return ''.join(sentences)







