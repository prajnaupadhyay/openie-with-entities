import spacy
import networkx as nx
import sys

nlp = spacy.load("en_core_web_sm")

import flair
from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load('ner-fast')


def read_extractions(predictions):
	f1 = open(predictions)
	dict_extract1 = {}
	sentence = ""
	for line in f1:
		line = line[:-1]
		if line.startswith("0") or line.startswith("1"):
			conf = line[:4]
			line = line[7:-1]

			splitline = line.split("; ")
			if sentence in dict_extract1:
				list1 = dict_extract1[sentence]
				list1.append((splitline[0], splitline[1], splitline[2], conf))
				dict_extract1[sentence] = list1
			else:
				list1 = []
				list1.append((splitline[0], splitline[1], splitline[2], conf))
				dict_extract1[sentence] = list1
		else:
			if line == "":
				continue
			else:
				sentence = line.replace("   ", " ").replace("  ", " ").replace('"', "''").replace("’", "'").replace(u'\xa0', u' ').replace('“', "''").replace('”', "''").replace(u'\u202f', u' ').replace("   ", " ")
	return dict_extract1

def read_ent_tagged(dict_extract):
	dict_entities = {}
	for sent in dict_extract:
		dict_entities[sent] = []
		s = Sentence(sent)
		tagger.predict(s)
		entities = s.get_spans('ner')
		for e in entities:
			dict_entities[sent].append(e.text)
	return dict_entities

def filter_openie_original(predictions, output):
	dict_extract1 = read_extractions(predictions)
	dict_entities = read_ent_tagged(dict_extract1)
	o = open(output, "w")
	for s in dict_extract1:
		ent_list = dict_entities[s]
		print(s)
		print(ent_list)
		o.write("\n" + s + "\n")
		for l in dict_extract1[s]:
			ent_exact = False
			ent_ends = False
			obj_ents = {}
			for e in ent_list:
				if e.lower() == l[0].lower():
					ent_exact = True
				if e.lower() in l[2].lower():
					obj_ents[e] = ""
			if len(obj_ents) == 1 and ent_exact:
				ent_obj = list(obj_ents.keys())[0]
				new_obj = ""
				new_rel = ""
				if l[2].lower().endswith(ent_obj.lower()):
					ent_ends = True
					new_obj = ent_obj
					new_rel = l[1] + " " + l[2].replace(new_obj, "")
				elif l[2].lower().endswith(ent_obj.lower() + "."):
					ent_ends = True
					new_obj = ent_obj + "."
					new_rel = l[1] + " " + l[2].replace(new_obj, "")

			if ent_ends and ent_exact:
				o.write(l[3] + ": (" + l[0] + "; " + new_rel.strip() + "; " + new_obj + ")\n")
				print(l[3] + ": (" + l[0] + "; " + new_rel.strip() + "; " + new_obj + ")\n")
			#else:
			#	o.write(l[3] + ": (" + l[0] + "; " + l[1] + "; " + l[2] + ")\n")
			#	print(l[3] + ": (" + l[0] + "; " + l[1] + "; " + l[2] + ")\n")

	o.close()

def gen_dep_paths():
	dict_entities={}
	sentences=[]
	i=0
	o = open("results/sdp_bet_sentences","w")
	o2 = open("results/baseline_sdp.txt","w")
	o1 = open("results/exceptions_segmented.txt","w")
	o3 = open("results/phrase_between.tsv","w")

	#f0 = open("/media/prajna/Files1/inria/relation_extraction/myexp/spacy/pubmed/predictions_mc_comma_replaced.txt.conj")
	f0 = open(sys.argv[1]) # read the conj file

	new_example=False
	count=0
	dict_conj={}
	master_sent=""
	for line in f0:
		line = line[:-1]
		if(new_example):
			if line=="":
				continue
			else:
				master_sent=line
				dict_conj[master_sent]=[]
				#dict_conj[master_sent].append(master_sent)
				new_example=False
		elif count==0:
			master_sent=line
			dict_conj[master_sent]=[]
			count=count+1
		else:
			if(line==""):
				new_example=True
			else:
				dict_conj[master_sent].append(line)
	'''		
	for m in dict_conj:
		print("master: "+m)
		for mm in dict_conj[m]:
			print("child: "+str(mm))
	#print(dict_conj)
	'''

	# tag entities in the conj file
	for sent in dict_conj:
		dict_entities[sent] = []
		s = Sentence(sent)
		tagger.predict(s)
		entities = s.get_spans('ner')
		for e in entities:
			#print(e.text)
			dict_entities[sent].append((e.text,e.start_position))
		for conjs in dict_conj[sent]:
			if conjs not in dict_entities:
				dict_entities[conjs]=[]
			s = Sentence(conjs)
			tagger.predict(s)
			entities = s.get_spans('ner')
			for e in entities:
				dict_entities[conjs].append((e.text,e.start_position))
				#print(e.text)
	'''
		
	f = open("/media/prajna/Files1/inria/relation_extraction/myexp/spacy/pubmed/predictions_mc_comma_replaced.txt.conj_ent_tagged.txt")
	for line in f:
		line = line[:-1]
		if(line.startswith("Sentence is: ")):
			line = line.replace("Sentence is: ","")
			dict_entities[line]=[]
			sentences.append(line)
			i=i+1
		elif(line.startswith("Entity")):
			splitline = line.split("', ")
			ent = splitline[0].split(": '")[1]
			ent_list = dict_entities[sentences[i-1]]
			ent_list.append(ent.lower())
			dict_entities[sentences[i-1]]=ent_list
	
	'''

	print("read tagged entities")
	print("size of dictionary: "+str(len(dict_entities)))

	n=0
	for s1 in dict_conj:
		o.write("\n"+s1+"\n")
		unique_extractions={}
		phrase_between = {}
		if(len(dict_conj[s1])==0):
			dict_conj[s1].append(s1)
		for s in dict_conj[s1]:
			n=n+1
			if(n % 1000==0):
				print(n)
			doc = nlp(s)
			print(doc)
			edges = []
			dict_pos_tags={}
			for token in doc:
				dict_pos_tags[token.lower_]=token.pos_
				for child in token.children:
					edges.append(('{0}'.format(token.lower_),'{0}'.format(child.lower_)))
			graph = nx.Graph(edges)
			if(len(dict_entities[s])==0):
				continue

			for i in range(len(dict_entities[s])-1):
				entity1 = dict_entities[s][i][0]
				split_entity = entity1.split(" ")
				ent_word1=split_entity[len(split_entity)-1]
				split_entity = ent_word1.split("-")
				ent_word1=split_entity[0].lower()
				#print("ent word 1: "+ent_word1)
				for j in range(i+1,len(dict_entities[s])):
					entity2 = dict_entities[s][j][0]
					split_entity1 = entity2.split(" ")
					ent_word2=split_entity1[len(split_entity1)-1]
					split_entity = ent_word2.split("-")
					ent_word2=split_entity[0].lower()
					#print("ent word 2: "+ent_word2)
					try:
						sp = nx.shortest_path(graph, source=ent_word1, target=ent_word2)
						print("we are here: entity1: "+entity1+", entity2: "+entity2+", shortest path: "+str(sp)+", length: "+str(len(sp)-1)+"\n")
						if(ent_word1 in sp):
							sp.remove(ent_word1)
						if(ent_word2 in sp):
							sp.remove(ent_word2)
						start_string=entity1+" ; "
						phrase_bet = entity1+" ; "

						contains_verb=False
						for t in sp:
							if(dict_pos_tags[t]=="VERB" or dict_pos_tags[t]=="AUX"):
								contains_verb=True
						listToStr = ' '.join([str(elem) for elem in sp])
						start_string = start_string + str(listToStr)+" ; "+entity2
						phrase_bet = phrase_bet + s[dict_entities[s][i][1]+len(entity1)+1:dict_entities[s][j][1]]+" ; "+entity2
						if(contains_verb):
							if start_string in unique_extractions:
								continue
							else:
								unique_extractions[start_string]=""
							if phrase_bet not in phrase_between:
								phrase_between[phrase_bet] = ""
						else:
							print(s+"\t"+start_string)
					except nx.NetworkXNoPath:
						o1.write("no path, sentence is: "+s+", entity1: "+entity1+", entity2: "+entity2+"\n")
					except nx.NodeNotFound:
						o1.write("node not found, sentence is: "+s+", entity1: "+entity1+", entity2: "+entity2+"\n")
					#print(nx.shortest_path_length(graph, source=ent_word1, target=ent_word2))
					#print(nx.shortest_path(graph, source=ent_word1, target=ent_word2))
		print("length of unique_extractions: "+str(len(unique_extractions)))
		for t in unique_extractions:
			o.write(t+"\n")
			t1 = t.split(" ; ")
			#print(len(t1))
			o2.write(s1+"\t1.00\t")
			o2.write(t1[1]+"\t"+t1[0]+"\t"+t1[2])
			#for i in range(0,len(t1)):
			#	o2.write("\t"+t1[i])
			o2.write("\n")
		for t in phrase_between:
			t1 = t.split(" ; ")
			o3.write(s1+"\t1.00\t"+t1[1]+"\t"+t1[0]+"\t"+t1[2]+"\n")


	o1.close()
	o.close()
	o2.close()
	o3.close()

#filter_openie_original(sys.argv[1], sys.argv[2])
gen_dep_paths()
