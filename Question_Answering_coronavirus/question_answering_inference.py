# -*- coding: utf-8 -*-
"""
"""
import os
import csv
import json
import requests
import argparse
import pandas as pd
from question_answering_main import answer_prediction



def process_text(q):
    '''
    Function to clean up the text
    '''
    stop_words = ['advertisement']
    query_tokens = [x for x in q.split() if x.lower() not in stop_words]
    processed_sentence = ' '.join(query_tokens)
    return processed_sentence

    
def search_and_format_text(processed_query,  source):
    '''
    Function to format the paragraphs
    '''
    with open(source,'r') as f:
        text = f.read()

    final_paragraphs = text.split('\n\n')

    cleaned_paras = []
    for para in final_paragraphs:
        sentences = para.split('.')
        cleaned_para = '.'.join([process_text(x) for x in sentences])
        cleaned_paras.append(cleaned_para)

    return cleaned_paras


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ques', '--question', type=str, default='abc')
    parser.add_argument('-src','--source',type=str)
    
    args = vars(parser.parse_args())
    query = args['question']
    source = args['source']
    
    paras_to_search = search_and_format_text(query, source)

    questions,answers,para,probs = answer_prediction(paras_to_search, query,'bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin','bert_config.json')
    questions_final,answers_final,paras_final,probs_final = zip(*sorted(zip(questions,answers,para,probs),reverse=True))
   
    result_str = []
    count = 0

    for i in range(int(len(answers_final))):
        if count < 3:
            if (answers_final[i] == 'empty'):
                pass
            if answers_final[i] != 'empty':
                count += 1
                result_str.append(answers_final[i])

            
    print(list(set(result_str)))

    for i,a in enumerate(answers_final):
        if a == 'empty':
            pass
        else:
            if probs_final[i] < 0.5:
                print('No answer')
            else:
                print(paras_final[i] + '\n')
                print(questions_final[i]+'\n')
                print(a+'\n')
                print(str(probs_final[i])+'\n')
                print('------------------------------------\n')

