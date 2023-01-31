import flair
from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load('ner-fast')
import operator

import sys
import re


def proccess_input(file):
   rb = open(file, "r", encoding="utf8")
   predicates = {}
   subjects = []
   objects = []
   extractions = []
   line = rb.readline()
   while line:
      sentence = line  # each sentence contains several extracted triples
      extraction = rb.readline()
      while len(extraction) > 1:
         predicate = extraction.split(";")[1].strip()
         predicate = re.sub(r'[^\w\s]', '', predicate)
         predicate = predicate.lower()
         if len(predicate) == 0:
            extraction = rb.readline()
            continue
         if predicate not in predicates:
            predicates[predicate] = 1
         else:
             predicates[predicate]+=1
         sub = extraction.split(";")[0].strip()
         sub = sub[sub.find("(") + 1:]
         obj = extraction.split(";")[2].strip()
         obj = obj[0:len(obj) - 1]

         subjects.append(Sentence(sub))
         objects.append(Sentence(obj))
         extractions.append(extraction)
         extraction = rb.readline()

      line = rb.readline()

   return predicates, subjects, objects, extractions

def generateExtractions():
    ent_sub_obj = 0
    per_org_pairs = 0
    predicates, subjects, objects, extractions = proccess_input(sys.argv[1])
    total_extractions = len(extractions)
    sorted_ex = {k: v for k, v in sorted(predicates.items(), key=lambda item: item[1], reverse=True)}
    o = open("relation_sorted_openie6_1.tsv", "w")
    for r in sorted_ex:
        o.write(r + ", " + str(predicates[r]) + "\n")
    o.close()

    tagger.predict(subjects, mini_batch_size=256)
    tagger.predict(objects, mini_batch_size=256)

    o = open("extractions_per_org_openie6_1", "w")

    for i in range(len(subjects)):
        ent1 = None
        ent2 = None
        entities_sub = subjects[i].get_spans('ner')
        if len(entities_sub) == 1:
            ent1 = entities_sub[0].text
        entities_obj = objects[i].get_spans('ner')
        if len(entities_obj) == 1:
            ent2 = entities_obj[0].text
        if ent1 and ent2:
            ent_sub_obj += 1
            ent1_type = entities_sub[0].get_label("ner").value
            ent2_type = entities_obj[0].get_label("ner").value
            if str(ent1_type) == "PER" and str(ent2_type) == "ORG":
                per_org_pairs += 1
                # print("ent1 type: " +ent1_type)
                # print("ent2 type: " +ent2_type)
                o.write(extractions[i][7:-2])
    o.close()
    print(total_extractions)
    print(ent_sub_obj)
    print(per_org_pairs)
    print(str(total_extractions)+"\t"+str(ent_sub_obj)+"\t"+str(per_org_pairs)+"\n")

generateExtractions()
