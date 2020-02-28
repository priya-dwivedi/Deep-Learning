# Fine tuning a GPT-2 model on your own input text

## First preprocess the data
Use the notebook data_cleanup/clean_harry_book.ipynb
The clean up steps can be improved by more manual inspection of the quality of cleaned text. This is sufficient to get started with the training.

Copy cleaned train and val text to examples/input_data

## Start Training the GPT-2 model

``` 
python run_lm_finetuning.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --do_train \
    --train_data_file='input_data/train_harry.txt' \
    --do_eval \
    --eval_data_file='input_data/val_harry.txt'\
    --overwrite_output_dir\
    --block_size=200\
    --per_gpu_train_batch_size=1\
    --save_steps 5000\
    --num_train_epochs=2
  ```


## Inference
```
cd examples
```

```
python run_generation.py --model_type gpt2 --model_name_or_path output --length 300 --prompt "Standing in the doorway, illuminated by the shivering flames in Lupin’s hand, was a cloaked figure that towered to the ceiling."
```
