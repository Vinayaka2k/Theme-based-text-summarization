from flask import *  
import requests
app = Flask(__name__)  
 

import mimetypes
import re
import magic

from helper1 import *

def input(ip):
	
	try:
		l=""
		if("." not in list(ip)):
			if(checkers.is_url(ip)):
				print("only web")
				l=getWeb(ip)
			else:
				mt=magic.from_file(ip,mime=True).split('/')
				if(mt[0]=="text"):
					l=read_text(ip)

		else:
			iplist=ip.split(".")
			# print(iplist)
			if(iplist[-1]=='docx' or iplist[-1]=='doc'):
				print(iplist[-1])
				l=getDoc(ip)
			elif(iplist[-1]=='pdf'):
				print(iplist[-1])
				l=getPdf(ip)
			elif(mimetypes.guess_type(ip)[0]!=None and mimetypes.guess_type(ip)[0].split("/")[0]=="text" and iplist[-1]!="html"):
				print("txt")#include ['.txt','.py','.c', no extension...]
				l=read_text(ip)
			elif(checkers.is_url(ip)):
				print("web")
				l=getWeb(ip)
			else:
				return 0
		# print(l)
		return l

	except Exception as e:
		print(e)
		return -1




@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  


 
@app.route('/process', methods = ['POST'])  
def process():  
	if request.method == 'POST': 
		print(request.files['file'].filename)
		theme=request.form.get('theme')

		if(len(str(request.form.get('per')).strip())==0):
			per=50
		else:
			per=float(request.form.get('per'))

		if(len(str(request.form.get('sumi')).strip())==0):
			sumi=50
		else:
			sumi=float(request.form.get('sumi'))

		print(theme)

		if(len(theme.strip())==0 or str(theme).isnumeric()):
			return render_template("file_upload_form.html",data_err="Invalid theme ....")



		if(len(request.files['file'].filename)!=0   or  request.form.get('text') or request.form.get('url')):
			# print("k")
			file=request.files['file'].filename
			if(len(file)!=0):
				f = request.files['file']  
				f.save(file)

				print("File saved",file)
				# k=requests.post(url='http://127.0.0.1:5000/success',json={"file":file}).json()

				text=input(file)
				

				# print(text)
				if(text==-1):
					return render_template("file_upload_form.html",data_err="Processing Error  ....")
				if(text==0):
					return render_template("file_upload_form.html",data_err="Unsupported format ....")


				try:
					ok=text
					extract="extract.txt"
				
					h1="Theme Based extraction and then Text summarization"
					h2="Substring based solution"


					f = open(extract, "w")
					f.write(text)
					f.close()

					#method 1

					#theme
					m11= "M1_theme.txt"
					s = theme_based_extraction(text,per/100,theme)
					f = open(m11, "w")
					f.write(s)
					f.close()

					# summarization
					m12="M1_summary.txt"
					summary = get_summary(s,sumi/100)
					f = open(m12, "w")
					f.write(summary)
					f.close()

					#method2
					'''
					# summarization
					m21="M2_summary.txt"
					summary = get_summary(text,sumi/100)
					f = open(m21, "w")
					f.write(summary)
					f.close()

					# theme
					m22="M2_theme.txt"
					s = new_theme(summary,per/100,theme)
					f = open(m22, "w")
					f.write(s)
					f.close()
					'''

					m21="M2_theme.txt"
					s = simple_theme_extraction(text,theme)
					f = open(m21, "w")
					f.write(s)
					f.close()

					# summarization
					m22="M2_summary.txt"
					summary = get_summary(s,sumi/100)
					f = open(m22, "w")
					f.write(summary)
					f.close()
					
				except Exception as e:
					print(e)
					return jsonify("Server Error.......")


				return render_template("file_upload_form.html",theme=theme,extract=extract,h1=h1,m11=m11,m12=m12,h2=h2,m21=m21,m22=m22)
				# return text
			elif(len(request.form.get('text').strip())!=0):
				# k=requests.post(url='http://127.0.0.1:5000/success',json={"text":request.form.get('text')}).json()
				text=request.form.get('text')

				try:
					ok=text
					extract="extract.txt"
				
					h1="Theme Based extraction and then Text summarization"
					h2="Sub string based solution"


					f = open(extract, "w")
					f.write(text)
					f.close()

					#method 1

					#theme
					m11= "M1_theme.txt"
					s = theme_based_extraction(text,per/100,theme)
					f = open(m11, "w")
					f.write(s)
					f.close()

					# summarization
					m12="M1_summary.txt"
					summary = get_summary(s,sumi/100)
					f = open(m12, "w")
					f.write(summary)
					f.close()

					#method2

					m21="M2_theme.txt"
					s = simple_theme_extraction(text,theme)
					f = open(m21, "w")
					f.write(s)
					f.close()

					# summarization
					m22="M2_summary.txt"
					summary = get_summary(s,sumi/100)
					f = open(m22, "w")
					f.write(summary)
					f.close()
				except Exception as e:
					print(e)
					return jsonify("Server Error.......")
				return render_template("file_upload_form.html",theme=theme,extract=extract,h1=h1,m11=m11,m12=m12,h2=h2,m21=m21,m22=m22)
				# return text
			elif(len(request.form.get('url').strip())!=0):

				ip=request.form.get('url')
				print(ip)

				text=input(ip)
				# text=getWeb(ip)
				# text=input(ip)
				
				# print("HERERERE",text)

				# print(text)
				if(text==-1):
					return render_template("file_upload_form.html",data_err="Processing Error  ....")
				if(text==0):
					return render_template("file_upload_form.html",data_err="Unsupported format ....")



				try:
					ok=text
					extract="extract.txt"
				
					h1="Theme Based extraction and then Text summarization"
					h2="Substring matching solution"


					f = open(extract, "w")
					f.write(text)
					f.close()

					#method 1

					#theme
					m11= "M1_theme.txt"
					s = theme_based_extraction(text,per/100,theme)
					f = open(m11, "w")
					f.write(s)
					f.close()

					# summarization
					m12="M1_summary.txt"
					summary = get_summary(s,sumi/100)
					f = open(m12, "w")
					f.write(summary)
					f.close()

					#method2

					m21="M2_theme.txt"
					s = simple_theme_extraction(text,theme)
					f = open(m21, "w")
					f.write(s)
					f.close()

					#summarization
					m22="M2_summary.txt"
					summary = get_summary(s,sumi/100)
					print(summary)
					f = open(m22, "w",encoding='utf-8')
					f.write(summary)
					f.close()
				except Exception as e:
					print(e)
					return jsonify("Server Error.......")


				return render_template("file_upload_form.html",theme=theme,extract=extract,h1=h1,m11=m11,m12=m12,h2=h2,m21=m21,m22=m22)
			else:
				return render_template("file_upload_form.html",data_err="No input ...")
				# k=requests.post(url='http://127.0.0.1:5000/fail').json()
				# return k
		else:
			return render_template("file_upload_form.html",data_err="No input ...")

			# k=requests.post(url='http://127.0.0.1:5000/fail').json()
			# return k
  


@app.route('/download/<file>')
def download_file(file):
	return send_file(file, as_attachment=True)

@app.route('/view/<file>')
def view_file(file):
	with open(file,"r") as f:
		text=f.read()
		return text

@app.route('/fail', methods = ['POST']) 
def fail():
	return jsonify("Give some input")


if __name__ == '__main__':  
    app.run(host="0.0.0.0",debug = True)  
