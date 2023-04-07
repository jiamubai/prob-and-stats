#download data
import gdown
eval_url = 'https://drive.google.com/file/d/1-yahnUKwjDuJIsPuVupJLBhAYtL4JxQx'
eval_output = 'eval.txt'
gdown.download(eval_url, output, quiet=False)

train_url = 'https://drive.google.com/file/d/1oP9nb8gqfx2030DF5SGRsvyCNbsT6PyM'
train_output = 'train.txt'
gdown.download(train_url, train_output, quiet=False)