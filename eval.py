import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import warnings
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
import utils

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

class TestDataset(Dataset):
    # Flickr8k dataset.
    def __init__(self, img_dir, split_dir, ann_dir, vocab_file, transform=None):
        # Args:
        #     img_dir (string): Directory with all the images.
        #     ann_dir (string): Directory with all the tokens
        #     split_dir (string): Directory with all the file names which belong to a certain split(train/dev/test)
        #     vocab_file (string): File which has the entire vocabulary of the dataset.
        #     transform (callable, optional): Optional transform to be applied
        #         on a sample.
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.split_dir = split_dir
        self.SOS = self.EOS = None
        self.vocab = None
        self.vocab_size = None
        self.images = self.captions = []
        self.all_captions = {}
        self.preprocess_files(self.split_dir, self.ann_dir, vocab_file)
        
        if(transform == None):
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    
    def preprocess_files(self, split_dir, ann_dir, vocab_file):
        # all_captions = {}
        with open(split_dir, "r") as split_f:
            sub_lines = split_f.readlines()
        
        with open(ann_dir, "r") as ann_f:
            for line in ann_f:
                if line.split("#")[0] + "\n" in sub_lines:
                    image_file = line.split('#')[0]
                    caption = utils.clean_description(line.replace("-", " ").split()[1:])
                    if image_file in self.all_captions:
                        self.all_captions[image_file].append(caption)
                    else:
                        self.all_captions[image_file] = [caption]

        self.images = list(self.all_captions.keys())
        self.captions = list(self.all_captions.values())
        assert(len(self.images) == len(self.captions))
        assert(len(self.captions[-1]) == 5)
        vocab = []
        with open(vocab_file, "r") as vocab_f:
            for line in vocab_f:
                vocab.append(line.strip())
        
        self.vocab_size = len(vocab) + 2 # The +2 is to accomodate for the SOS and EOS
        self.SOS = 0
        self.EOS = self.vocab_size - 1
        self.vocab = vocab        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name, caps = self.images[idx], self.captions[idx]
        img_name = os.path.join(self.img_dir, img_name)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'captions': caps}

def collater(batch):
    images = torch.stack([item['image'] for item in batch])
    all_caps = [item['captions'] for item in batch]
    return images, all_caps

def get_output_sentence(model, device, images, vocab):
    # hypotheses = []
    with torch.no_grad():
        torch.cuda.empty_cache()

        images = images.to(device)
        target_eos = len(vocab) + 1
        target_sos = 0

        b_1 = model(images, on_max='halt')
        captions_cand = b_1[..., 0]

        cands = captions_cand.T
        cands_list = cands.tolist()
        for i in range(len(cands_list)): #Removes sos tags
            cands_list[i] = list(filter((target_sos).__ne__, cands_list[i]))
            cands_list[i] = list(filter((target_eos).__ne__, cands_list[i]))

        # hypotheses += cands_list
    return cands_list

def score(ref, hypo):
    # ref, dictionary of reference sentences (id, sentence)
    # hypo, dictionary of hypothesis sentences (id, sentence)
    # score, dictionary of scores
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def get_references_and_hypotheses(model, device, dataset, dataloader):
    references = []
    hypotheses = []
    assert(len(dataset.captions) == len(dataset.images))
    with torch.no_grad():
        for data in tqdm(dataloader):
            torch.cuda.empty_cache()
            images, captions = data
            references += captions
            hypotheses += get_output_sentence(model, device, images, dataset.vocab)

        for i in range(len(references)):
            hypotheses[i] = " ".join([dataset.vocab[j - 1] for j in hypotheses[i]])
        assert(len(references) == len(hypotheses))
        return references, hypotheses

def get_pycoco_metrics(model, device, dataset, dataloader):
    references, hypotheses = get_references_and_hypotheses(model, device, dataset, dataloader)
    hypo = {idx: [h] for idx, h in enumerate(hypotheses)}
    ref = {idx: [" ".join(l) for l in r] for idx, r in enumerate(references)}
    metrics = score(ref, hypo)
    return metrics

def print_metrics(model, device, dataset, dataloader):
    references, hypotheses = get_references_and_hypotheses(model, device, dataset, dataloader)
    # BLEU scores
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, hypotheses)

    print('BLEU-1 ({})\t'
          'BLEU-2 ({})\t'
          'BLEU-3 ({})\t'
          'BLEU-4 ({})\t'.format(bleu_1, bleu_2, bleu_3, bleu_4))
    
    # Meteor score
    total_m_score = 0.0
    for i in range(len(references)):
        actual = [" ".join(ref) for ref in references[i]]
        total_m_score += meteor_score(actual, " ".join(hypotheses[i]))
    m_score = total_m_score/len(references)
    print('Meteor Score: {}'.format(m_score))
    metrics = {
        'bleu_1': bleu_1,
        'bleu_2': bleu_2,
        'bleu_3': bleu_3,
        'bleu_4': bleu_4,
        'meteor': m_score
    }
    return metrics


"""
BLEU (Bilingual Evaluation Understudy) is a widely used evaluation metric for machine translation. It compares the generated translation to a reference translation and assigns a score based on how similar they are. The score ranges from 0 to 1, where 1 is a perfect match and 0 is a completely dissimilar translation.

The basic idea behind BLEU is to calculate the n-gram (i.e unigram, bigram, trigram, etc) overlaps between the generated translation and reference translation. The more overlaps there are, the higher the BLEU score will be. The score is then modified by a brevity penalty term to account for the length of the generated translation.

In more detail, for a given generated sentence and a reference sentence,

The first step is to calculate the number of matching n-grams for each n from 1 to N (usually N is 4), where N-gram is a contiguous sequence of n items from a given sample of text or speech, where n is the order of the model.
Then it calculates the precision score for each n-gram by dividing the number of matching n-grams by the total number of n-grams in the generated sentence.
After that, it calculate the brevity penalty
Finally, it calculates the BLEU score using the geometric mean of the precision scores for all n-grams, modified by the brevity penalty.
It is worth noting that BLEU is not without its criticisms, it doesn't consider the meaning of the sentences and it is considered a weak metric for evaluating the quality of text generation.


METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a widely used evaluation metric for machine translation, similar to BLEU. It is an automatic evaluation metric that is designed to overcome some of the limitations of BLEU by considering not just the n-gram overlap between the generated and reference translations, but also the alignment between words in the two translations.

The basic idea behind METEOR is to calculate the harmonic mean of unigram precision and recall, where precision is the proportion of matched words in the generated translation that are also present in the reference translation, and recall is the proportion of words in the reference translation that are also present in the generated translation.

In more detail, for a given generated sentence and a reference sentence,

The first step is to align the words in the generated and reference sentences using a word alignment algorithm.
Then, it computes the unigram precision and recall scores by comparing the aligned words in the generated and reference sentences.
The unigram F-measure is calculated as the harmonic mean of precision and recall.
METEOR also calculates a penalty for fragmentation, which is a penalty term for the number of chunks in the alignment between the generated and reference sentences.
Finally, it calculates the METEOR score by combining the unigram F-measure and the fragmentation penalty.
METEOR is considered to be a more robust metric than BLEU as it takes into account the alignment between words in the generated and reference translations, which can help to better evaluate the fluency and coherence of the generated translations.



ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation) is a commonly used metric for evaluating the quality of text summarization systems. It calculates the similarity between a generated summary and a reference summary by computing the longest common subsequence (LCS) between the two. The LCS is a measure of how much of the generated summary is shared with the reference summary.

The ROUGE-L score is calculated as follows:

The LCS between the generated summary and the reference summary is computed
The LCS is divided by the length of the reference summary
The result is multiplied by 100 to get the ROUGE-L score, which is expressed as a percentage
For example, if the LCS between the generated summary and the reference summary is 10 words, and the reference summary has 20 words, the ROUGE-L score is 50%.

It's important to note that ROUGE-L only considers the longest common subsequence of words and does not take into account their order. There are other version of ROUGE such as ROUGE-N, that considers n-grams and ROUGE-W that consider the longest common contiguous subsequence (LCSS) of words.



CIDEr (Consensus-based Image Description Evaluation) is a metric for evaluating the quality of image captioning systems. It's similar to ROUGE for text summarization, but it's designed specifically for image captioning.

The CIDEr score is calculated as follows:

The captions generated by the image captioning model are compared to the reference captions.
Each word in the generated caption is matched to the closest word in the reference caption using cosine similarity.
The matched words are assigned a score based on the cosine similarity and the IDF (inverse document frequency) of the word. The score is highest for words that are both similar and rare.
The scores for all matched words are summed to give the CIDEr score.
CIDEr score consider the similarity between generated captions and the reference captions, it also takes into account the rarity of the words used. This allows it to give more weight to words that are both similar and rare, which are thought to be more important for describing images.

It's important to note that CIDEr can be more challenging to optimize for as it's more sensitive to small changes in the generated captions.



In transformer models, the word embeddings are typically learned as part of the model training process, rather than being pre-trained and then used as input. This is done using an embedding layer that is part of the transformer model architecture.

The embedding layer is typically the first layer in the transformer model, and it maps the input words to a high-dimensional vector space. The vectors are learned during the training process and are optimized to represent the meaning of the words in a way that is useful for the specific task the transformer model is being used for.

Here is an example of how the word embeddings are learned in a transformer model:

The input to the transformer model is a sequence of words. Each word is first passed through the embedding layer, which maps the word to a high-dimensional vector.
These embeddings are then passed through the multi-head attention layer, where the attention mechanism is applied to calculate the representation of the input.
The representations are then passed through the feed-forward layers, which are used to learn the final representation of the input.
The final representation is then used for the specific task such as language modeling or text classification.
The embeddings are learned during the training process by optimizing the parameters of the transformer model to minimize a loss function that measures the difference between the model's predictions and the true labels. By the end of training, the embeddings will have learned to represent the meaning of
"""