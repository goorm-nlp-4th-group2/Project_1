import re
import pandas as pd
import datasets
import torch

from glob import glob

def cleaning(inputs : str) -> str :
    """
    전처리 함수입니다.
    입력으로 문자열을 받고 전처리해 반환합니다.
    """
    inputs = inputs.lower()

    # 아래는 약어를 원문으로 복원하는 작업들입니다.

    inputs = inputs.replace('+', 'and') 
    inputs = inputs.replace('&', 'and')
    inputs = inputs.replace("=", 'is')
    inputs = inputs.replace("w/", 'with')
    inputs = inputs.replace("@", "at")
    inputs = inputs.replace("a\+", "best")
    inputs = inputs.replace(" a \+", "best")

    if "a \+" in ' ' + inputs :
        inputs = inputs.replace("a \+", "best")

    inputs = inputs.replace("b/c", 'because')
    inputs = inputs.replace("a/c", 'air conditioner')
    inputs = inputs.replace("24/7", 'always')

    # 여기서부턴 서로 다른 작업들입니다.

    inputs = inputs.replace("1/2", 'half')              # 영문으로 표현하는게 나을 듯 해 대체했습니다.
    inputs = inputs.replace("1/3", 'half')              
    inputs = inputs.replace("1/4", 'quarter')           # 영문으로 표현하는게 나을 듯 해 대체했습니다.
    inputs = inputs.replace("5/5", 'perfect')           # 5/5가 들어간건 평점을 이야기하더라구요. 변경했습니다.
    inputs = inputs.replace("10/10", 'perfect')         # 평점
    inputs = inputs.replace("4/5", 'nearly perfect')    # 평점
    inputs = inputs.replace("3/4", 'almost')            # 영어로 표현하고자 했습니다.

    # 여기서부턴 필요없는 문자들을 제거합니다.

    inputs = re.sub("`|''", '', inputs)             # 특수문자 `와 ''를 없앱니다. (따옴표 하나는 문장부호 정도로 남겨두는게 어떨까 싶어서 우선은 남겨뒀습니다.)
    inputs = re.sub('""', '', inputs)               # 특수문자 ""를 없앱니다.
    inputs = re.sub("\.{2,}", '.', inputs)          # 마침표가 2번 이상 반복될 경우 하나로 바꿉니다. (ex 점심 별로야... -> 점심 별로야.)
    inputs = re.sub("[ .]{2,}", ' .', inputs)       # 몇몇 경우 공백과 마침표가 함께 반복되는 경우가 있어 이것도 두 번 이상 반복될 경우 하나로 바꿉니다.
    inputs = re.sub(":|\)|\(|!", '', inputs)       # 특수문자 :와 )와 (와 !를 없앱니다.

    inputs = re.sub("\.", ' ', inputs)
    inputs = re.sub("\$|\#|\*|\%|~", ' ', inputs)
    inputs = re.sub("\[|\]", ' ', inputs)
    inputs = re.sub("/", ' ', inputs)
    inputs = re.sub(";", '', inputs)
    inputs = re.sub("-", ' ', inputs)

    inputs = re.sub("\s{2,}", ' ', inputs)          # 공백이 2개 이상일 경우 공백을 하나로 바꿉니다.
    inputs = inputs.strip()                         # 좌우 공백 제거

    return inputs

def load_data(path, min_num_of_tokens, max_num_of_tokens, tokenizer) -> dict :
    """
    ./RawData 디렉토리에 저장되어있는 데이터 파일을 불러옵니다.

    arguments
        path : 불러올 디렉토리의 경로를 문자열로 받습니다.
        min_num_of_tokens : 문장 내 토큰 최소 개수가 이보다 적으면 제외합니다.
        max_num_of_tokens : 문장 내 토큰 최대 개수가 이보다 많으면 제외합니다.
        tokenizer : 사용할 사전학습 모델의 tokenizer        
        ex) load_data("./Model", 8, 22, your_tokenizer)

    return : 위에서 불러온 데이터를 데이터 유형과 pandas DataFrame을 각각 key, value로 갖는 dictionary로 반환합니다.
    """
    path_list = glob(path + "/*")
    train = []
    valid = []
    for p in path_list :
        if "test" in p :
            continue
        temp = pd.read_table(p, header = None).rename({0 : "sentence"}, axis = "columns")
        temp.loc[:, "labels"] = 1 if '1' in p else 0
        check_encoded = temp.sentence.apply(lambda x : tokenizer.encode(x))
        idx = (min_num_of_tokens <= check_encoded.apply(len)) & (check_encoded.apply(len) <= max_num_of_tokens)
        temp = temp.loc[idx, :]
        if 'train' in p :
            train.append(temp)
        else :
            valid.append(temp)

    final_train = pd.concat(train)
    final_train = final_train.drop_duplicates().reset_index(drop = True)

    final_valid = pd.concat(valid)
    final_valid = final_valid.drop_duplicates().reset_index(drop = True)
    return {"train" : final_train, "valid" : final_valid}

def __tokenizing(inputs, tokenizer, training):
    """
    데이터를 입력받아 문장을 tokenizing합니다.
    해당 함수는 아래 get_dataset의 입력으로 사용됩니다.

    arguments :
        inputs : pandas DataFrame을 입력받습니다.
                 해당 DataFrame은 sentence라는 이름의 column을 가지고 있어야 합니다.
                 학습 단계일 경우 labels라는 이름의 column도 가지고 있어야 합니다.
        tokenizer : 사용할 사전학습 모델의 tokenizer를 입력으로 받습니다.
        training : 현재 데이터가 훈련용인지 추론용인지 입력받습니다.
                   훈련용일 경우 True, 추론용일 경우 False를 인자로 받습니다.
    
    return : 해당 사전학습 모델에서 사용해야 할 입력과 필요한 경우 labels를 키로 가진 dictionary를 반환합니다.
    """

    model_inputs = tokenizer(inputs["sentence"])
    if training :
        model_inputs["labels"] = inputs["labels"]
    return model_inputs

def get_dataset(inputs, tokenizer, collator, batch_size, training) :
    """
    pandas Dataframe을 입력받아 torch dataset으로 변환합니다.
    중간에 parallel processing을 위해 huggingface dataset으로 변환 후 작업합니다.

    arguments :
        inputs : pandas DataFrame을 입력받습니다.
                 해당 DataFrame은 sentence라는 이름의 column을 가지고 있어야 합니다.
                 학습 단계일 경우 labels라는 이름의 column도 가지고 있어야 합니다.
        tokenizer : 사용할 사전학습 모델의 tokenizer를 입력으로 받습니다.
        collator : 입력 데이터의 길이를 task에 따라 맞춰주는 collator function을 인자로 받습니다.
        batch_size : 학습 또는 추론에 사용할 batch size를 인자로 받습니다.
        training : 현재 데이터가 훈련용인지 추론용인지 입력받습니다.
                   훈련용일 경우 True, 추론용일 경우 False를 인자로 받습니다.
    
    return : torch dataset
    """
    inputs = datasets.Dataset.from_pandas(inputs)
    tokenized_inputs = inputs.map(__tokenizing,
                                  batched = True,
                                  fn_kwargs = {"training" : training,
                                               "tokenizer" : tokenizer})
    
    if training :
        columns = tokenizer.model_input_names + ["labels"]
    else :
        columns = tokenizer.model_input_names
    
    tokenized_inputs.set_format("torch", columns = columns)
    torch_dataset = torch.utils.data.DataLoader(tokenized_inputs,
                                                batch_size = batch_size,
                                                shuffle = training,
                                                collate_fn = collator)
    return torch_dataset