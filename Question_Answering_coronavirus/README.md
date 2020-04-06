### Question Answering using CNN articles on Coronavirus

### Installation Instructions
Please install pytorch and transformers using the guide on huggingface repo
https://github.com/huggingface/transformers#installation


### Contents

1. question_answering_inference.py - This is a wrapper file which has supporting functions to process the input and outputs. In this file, we load the text from the file path, clean the text (remove the stop words after converting all words to lowercase), convert the text into paragraphs and then pass it on to the answer_prediction function in the question_answering_main.py. The answer_prediction function returns the answers and their probabilities. Then, we filter the answers based on a probability threshold and then display the answers with their probability.

2. question_answering_main.py - This file is the main file which has all the functions required to use the pre-trained model and tokenizer to predict the answer given the paragraphs and questions.

### Running the code

```
python question_answering_inference.py - ques 'How many confirmed cases are in Mexico?' - source 'sample2.txt'
```


