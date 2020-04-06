import math
import torch
import argparse
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from termcolor import colored, cprint
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,whitespace_tokenize)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 example_id,
                 para_text,
                 qas_id,
                 question_text,
                 doc_tokens,
                unique_id):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.example_id = example_id
        self.para_text = para_text
        self.unique_id = unique_id
        

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        
        return s



def read_paragraphs(paragraphs,ques_text):
    '''
    Convert paragraph to tokens and returns question_text
    '''    
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    i = 0
    examples = []
    for entry in paragraphs:
        example_id = entry['id']
        paragraph_text = entry['text']
        doc_tokens = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
        
        qas_id = i
        question_text = entry['ques']
        
        
        example = SquadExample(example_id = example_id,
                qas_id=qas_id,
                para_text = paragraph_text,               
                question_text=question_text,
                doc_tokens=doc_tokens,
                unique_id = i)
        i+=1
        examples.append(example)

    return examples



def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_is_max_context,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids):
 
        self.doc_span_index = doc_span_index
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.token_is_max_context = token_is_max_context
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        




def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""


    features = []
    unique_id = 1
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        ### Truncate the query if query length > max_query_length..
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)


            input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            
            features.append(InputFeatures(unique_id = unique_id,
                            example_index = example_index,
                            doc_span_index=doc_span_index,
                            tokens=tokens,   
                            token_is_max_context=token_is_max_context,
                            token_to_orig_map=token_to_orig_map,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids))
            unique_id += 1

            
    
    return features


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
   
    return best_indexes



def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

    
    
_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
["feature_index", "start_index", "end_index", "start_logit", "end_logit"])


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_logit", "end_logit"])

def predict(examples, all_features, all_results, max_answer_length):

    n_best_size = 10
    
    ### Adding index to feature ###
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
     
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
        
        
    all_predictions = collections.OrderedDict()
   
    
    
    for example in examples:
        index = 0
        features = example_index_to_features[example.unique_id]
        prelim_predictions = []
       
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                    for end_index in end_indexes:
                     #### we remove the indexes which are invalid @
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):

                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue

                        prelim_predictions.append(
                                        _PrelimPrediction(
                                            feature_index=feature_index,
                                            start_index=start_index,
                                            end_index=end_index,
                                            start_logit=result.start_logits[start_index],
                                            end_logit=result.end_logits[end_index]))


        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True) 
            
    
         
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
                
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, True)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        
        
    
        if not nbest:
                nbest.append(
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1
        
        
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

    
        probs = _compute_softmax(total_scores)
        nbest_json = []
        prob_json = []
        for (i, entry) in enumerate(nbest):

            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)
            prob_json.append(probs[i])

        assert len(nbest_json) >= 1
        all_predictions[example] = nbest_json[0]["text"]
        all_predictions['prob'+str(example)] = nbest_json[0]["probability"]
        index=+1
    return all_predictions
        

                

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])




def answer_prediction(paras,question,model,config_file,max_seq_length=384,doc_stride=128,max_query_length=64,max_answer_length=60):
    
    
    #para_file = 'Input_file.txt'
    model_path = model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    ### Raeding paragraph
    ## Reading question
#     f = open(ques_file, 'r')
#     ques = f.read()
#     f.close()
     
    ## input_data is a list of dictionary which has a paragraph and questions
    #para_list = para.split('\n\n')
    #print(paras)
    input_data = []
    i = 1
    for i,para in enumerate(paras):
       # print(para)
        paragraphs = {}
        #splits = para.split('\nQuestions:')
        paragraphs['id'] = i
        paragraphs['text'] = para
        paragraphs['ques']= question
        input_data.append(paragraphs)
           
    
    examples = read_paragraphs(input_data,question)
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', do_lower_case=True)
    
    
    eval_features = convert_examples_to_features(
            examples = examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length)
    
    
    
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    
    ### Loading Pretrained model for QnA 
    config = BertConfig(config_file)
    model = BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(model_path,map_location='cpu'))
    model.to(device)
   

    pred_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    pred_sampler = SequentialSampler(pred_data)
    pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=10)
    
    predictions = []

    for input_ids, input_mask, segment_ids, example_indices in tqdm(pred_dataloader):
        model.eval()
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
            
                    
        features=[]
        example = []
        all_results = []
       
        for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits =   batch_end_logits[i].detach().cpu().tolist()
                feature = eval_features[example_index.item()]
                unique_id = int(feature.unique_id)
                features.append(feature)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))
                
       
        output = predict(examples, features, all_results,max_answer_length)
        predictions.append(output)
 
   
    ### For printing the results ####
    final_preds = []
    final_paras = []
    final_probs = []
    final_scores = []
    final_ques = []
    index = None
    for i,example in enumerate(examples):
        if index!= example.example_id:
            index = example.example_id
#          
        ques_text = colored(example.question_text, 'blue')

        prediction = predictions[math.floor(example.unique_id/12)][example]

        prob = predictions[math.floor(example.unique_id/12)]['prob'+str(example)]

        final_ques.append(ques_text)
        final_preds.append(prediction)
        final_paras.append(example.para_text)
        final_probs.append(prob)
        
    return final_ques,final_preds,final_paras,final_probs

#if __name__ == "__main__":
#    answer_prediction()
