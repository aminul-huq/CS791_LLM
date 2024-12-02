import torch
from PIL import Image
import json
import re
from typing import Optional
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from PIL import Image
from tqdm.notebook import tqdm
from transformers import ViltConfig
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering


def id_from_filename(filename: str) -> Optional[int]:
    filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")
    match = filename_re.fullmatch(filename)
    if match is None:
        return None
    return int(match.group(1))

def get_score(count: int) -> float:
    return min(1.0, count / 3)

def collate_fn(batch):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # create padded pixel values and corresponding pixel mask
    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels)

    return batch

class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, questions, annotations, id_to_filename, processor, config):
        self.questions = questions
        self.annotations = annotations
        self.id_to_filename = id_to_filename
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        questions = self.questions[idx]
        image = Image.open(self.id_to_filename[annotation['image_id']])
        text = questions['question']
        if image.mode != "RGB":  # Convert grayscale or other modes to RGB
            image = image.convert("RGB")

        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k,v in encoding.items():
          encoding[k] = v.squeeze()
        # add labels
        labels = annotation['labels']
        scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(self.config.id2label))
        for label, score in zip(labels, scores):
            targets[label] = score
            #print(targets[label])
        encoding["labels"] = targets

        return encoding

# Opening questions JSON file
def load_data(image_root,q_root,a_root, config):
    #fq = open('/home/mpervin/LLM/Datasets/v2_OpenEnded_mscoco_val2014_questions.json')
    #root = '/home/mpervin/LLM/Datasets/val2014'
    #a_root = open('/home/mpervin/LLM/Datasets/v2_mscoco_val2014_annotations.json')
    
    #load images
    file_names = [f for f in tqdm(listdir(image_root)) if isfile(join(image_root, f))]
    filename_to_id = {image_root + "/" + file: id_from_filename(file) for file in file_names}
    id_to_filename = {v:k for k,v in filename_to_id.items()}
    
    #load questions
    data_questions = json.load(q_root)
    questions = data_questions['questions']
    print('data questions keys:',data_questions.keys())
    print("Number of questions:", len(questions))
    
    #load annotations
    data_annotations = json.load(a_root)
    annotations = data_annotations['annotations']
    print('data annotations keys:',data_annotations.keys())
    print("Number of annotations:", len(annotations))
    
   
    for annotation in tqdm(annotations):
        answers = annotation['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer["answer"]
            answer_count[answer_] = answer_count.get(answer_, 0) + 1
        labels = []
        scores = []
        for answer in answer_count:
            if answer not in list(config.label2id.keys()):
                continue
            labels.append(config.label2id[answer])
            score = get_score(answer_count[answer])
            scores.append(score)
        annotation['labels'] = labels
        annotation['scores'] = scores
        
    return questions, annotations, id_to_filename


