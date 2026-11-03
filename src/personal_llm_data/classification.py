"""Strict existing-label JSON classification metrics; no GPU dependencies."""
from collections import Counter
import json


def parse_label(text, labels):
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('Duplicate JSON key')
            value[key]=item
        return value
    try:
        value=json.loads(text,object_pairs_hook=unique)
    except (ValueError,TypeError):return None
    if not isinstance(value,dict) or set(value)!={'label'} or not isinstance(value['label'],str):return None
    return value['label'] if value['label'] in labels else None


def score_labels(records,predictions,labels):
    if not labels or len(set(labels))!=len(labels) or any(not isinstance(x,str) or not x for x in labels):
        raise ValueError('Explicit unique nonempty string vocabulary required')
    if not records or len({r['id'] for r in records})!=len(records):raise ValueError('Nonempty unique test IDs required')
    if [r['id'] for r in records]!=[p.get('id') for p in predictions]:raise ValueError('Full ordered test IDs must match')
    matrix=Counter()
    for row,prediction in zip(records,predictions):
        reference=row['messages'][-1]['content']
        truth=parse_label(reference,labels)
        if truth is None or prediction.get('reference')!=reference:raise ValueError('Invalid/mismatched reference label')
        result=parse_label(prediction.get('prediction'),labels)
        matrix[(truth,result)]+=1
    classes={}
    for label in labels:
        tp=matrix[label,label]
        support=sum(count for (gold,_),count in matrix.items() if gold==label)
        guessed=sum(count for (_,guess),count in matrix.items() if guess==label)
        classes[label]=dict(support=support,precision=tp/guessed if guessed else 0,recall=tp/support if support else 0,
                            f1=2*tp/(support+guessed) if support+guessed else 0)
    return dict(samples=len(records),labels=list(labels),accuracy=sum(matrix[x,x] for x in labels)/len(records),
                macro_f1=sum(c['f1'] for c in classes.values())/len(labels),per_class=classes,
                invalid_outputs=sum(count for (_,guess),count in matrix.items() if guess is None),
                invalid_rate=sum(count for (_,guess),count in matrix.items() if guess is None)/len(records),
                confusion={gold:{guess:matrix[gold,guess] for guess in labels} | {'invalid':matrix[gold,None]} for gold in labels})
