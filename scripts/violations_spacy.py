import sys
import json
import spacy

spacy.require_gpu()
nlp = spacy.load("en_core_web_sm")


def proccess_input(file):
    rb = open(file, "r", encoding="utf8")
    predicates = []
    subjects = []
    objects = []
    sentences = {}
    line = rb.readline()
    num_extractions = 0
    sentences_list = []
    count = 0
    while line:
        sentence = line.strip()  # each sentence contains several extracted triples
        extraction = rb.readline()
        sentences_list.append(sentence)
        sentences[sentence] = []
        count = count + 1
        #if count % 100 == 0:
        #    print("processed 100 lines")
        while len(extraction) > 1:
            num_extractions += 1
            predicate = extraction.split(";")[1].strip()
            sub = extraction.split(";")[0].strip()
            sub = sub[sub.find("(")+1:]
            obj = extraction.split(";")[2].strip()
            obj = obj[0:len(obj)-1]
            predicates.append(predicate)
            subjects.append(sub)
            objects.append(obj)
            extraction = rb.readline()
            sentences[sentence].append((sub, obj))

        line = rb.readline()
    sentences1 = {}
    predicates_ent = []
    sub_ents = []
    obj_ents = []
    for sentence_index, spacy_sentence in enumerate(nlp.pipe(sentences_list, batch_size=10000)):
        sentences1[spacy_sentence] = sentences[sentences_list[sentence_index]]
    for sentence_index, spacy_sentence in enumerate(nlp.pipe(predicates, batch_size=10000)):
        predicates_ent.append(spacy_sentence.ents)
    for sentence_index, spacy_sentence in enumerate(nlp.pipe(subjects, batch_size=10000)):
        sub_ents.append(spacy_sentence.ents)
    for sentence_index, spacy_sentence in enumerate(nlp.pipe(objects, batch_size=10000)):
        obj_ents.append(spacy_sentence.ents)

    print("finished tagging")
    print(len(predicates))

    violations_c1 = 0  # Entities as subject or object
    violations_c2 = 0  # Entity exclusivity
    violations_c3 = 0  # Entity in relation penalty.
    violations_c4 = 0  # Entity segmentation penalty.
    for i in range(len(predicates)):
        predicate = predicates[i]
        sub = subjects[i]
        obj = objects[i]
        # the predicate should not contain an entity
        entities_pred = predicates_ent[i]
        if len(entities_pred):
            violations_c3 += 1
            #print("violation c3 "+ str(entities_pred))
        # Entity exclusivity: the object/subject contains just one entity
        entities_obj = obj_ents[i]
        if len(entities_obj) > 1:
            violations_c2 += 1
            #print("violation c2 " + str(entities_obj))
        entities_sub = sub_ents[i]
        if len(entities_sub) > 1:
            violations_c2 += 1
            #print("violation c2 " + str(entities_sub))


    num_entities = 0
    for s in sentences1.keys():
        num_entities += len(s)
    count_c4 = 0
    for s in sentences1.keys():
        for entity in s.ents:
            e = entity.text
            e_set = set(e.split(" "))
            found = 0
            found_str = e
            for (sub, obj) in sentences1[s]:
                count_c4 += 1
                if e in sub and len(e) and len(sub):
                    found = 1
                    found_str = e + " ||| " + sub
                if e in obj and len(e) and len(obj):
                    found = 1
                    found_str = e + " ||| " +obj
                o_set = set(obj.split(" "))
                s_set = set(sub.split(" "))
                if e in obj or e in sub:
                    continue
                if len(e_set.intersection(o_set))==min(len(e_set),len(o_set)) and e_set!=o_set or len(e_set.intersection(s_set))==min(len(e_set),len(s_set)) and e_set!= s_set:
                    violations_c4 += 1
                    #print("violation_c4 "+ str(e_set) + str(o_set)+str(s_set))

            if found == 0:
                violations_c1 += 1
            #else:
            #    print("not violations_c1 "+found_str)



    return {"number_sentences": len(sentences), "number_extractions": num_extractions, "number_entities": num_entities,
            "violations_c1":violations_c1, "percentage_violations_c1": violations_c1*1.0/num_entities,
            "violations_c2":violations_c2, "percentage_violations_c2": violations_c2*1.0/(2*num_extractions),
            "violations_c3":violations_c3, "percentage_violations_c3": violations_c3*1.0/num_extractions,
            "violations_c4":violations_c4, "percentage_violations_c4": violations_c4*1.0/count_c4}


if __name__ == '__main__':
    results = proccess_input(sys.argv[1])
    results['input'] = sys.argv[1]
    print(results)
    # Step 2
    with open(sys.argv[2], 'w', encoding="utf-8") as output_file:
        # Step 3
        json.dump(results, output_file)

