import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import warnings
import eval
import bleu
import utils
import string
import copy
import argparse
import os, random
from matplotlib import pyplot as plt
from models import *
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

img_dir = '../input/flickr8k/Images/'
ann_dir = '../input/flickr8k-text/Flickr8k.token.indonesia.txt'
train_dir = '../input/flickr8k-text/Flickr_8k.trainImages.txt'
val_dir = '../input/flickr8k-text/Flickr_8k.devImages.txt'
test_dir = '../input/flickr8k-text/Flickr_8k.testImages.txt'
vocab_file = './vocab.txt'

SEED = 123
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use training's class and function
class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, split_dir, ann_dir, vocab_file, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.split_dir = split_dir
        self.SOS = self.EOS = None
        self.word_2_token = None
        self.vocab_size = None
        self.image_file_names, self.captions, self.tokenized_captions = self.tokenizer(self.split_dir, self.ann_dir)
        if (transform == None):
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def tokenizer(self, split_dir, ann_dir):
        image_file_names = []
        captions = []
        tokenized_captions = []
        with open(split_dir, "r") as split_f:
            sub_lines = split_f.readlines()
        with open(ann_dir, "r") as ann_f:
            for line in ann_f:
                if line.split("#")[0] + "\n" in sub_lines:
                    caption = utils.clean_description(line.replace("-", " ").split()[1:])
                    image_file_names.append(line.split()[0])
                    captions.append(caption)
        vocab = []
        # for caption in captions:
        #     for word in caption:
        #         if word not in vocab:
        #             vocab.append(word)
        with open(vocab_file, "r") as vocab_f:
            for line in vocab_f:
                vocab.append(line.strip())
        self.vocab_size = len(vocab) + 2
        self.SOS = 0
        self.EOS = self.vocab_size - 1
        self.word_2_token = dict(zip(vocab, list(range(1, self.vocab_size - 1))))

        for caption in captions:
            temp = []
            for word in caption:
                temp.append(self.word_2_token[word])
            temp.insert(0, self.SOS)
            temp.append(self.EOS)
            tokenized_captions.append(temp)

        assert (len(image_file_names) == len(captions))
        return image_file_names, captions, tokenized_captions

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name, cap_tok, caption = self.image_file_names[idx], self.tokenized_captions[idx], self.captions[idx]
        img_name, instance = img_name.split('#')
        img_name = os.path.join(self.img_dir, img_name)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        cap_tok = torch.tensor(cap_tok)
        sample = {'image': image, 'cap_tok': cap_tok, 'caption': caption}
        return sample

def collater(batch):
    cap_lens = torch.tensor([len(item['cap_tok']) for item in batch])  # Includes SOS and EOS as part of the length
    caption_list = [item['cap_tok'] for item in batch]
    # padded_captions = pad_sequence(caption_list, padding_value=9631)
    images = torch.stack([item['image'] for item in batch])
    return images, caption_list, cap_lens

def display_sample(sample):
    image = sample['image']
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    image = inv_normalize(image)
    caption = ' '.join(sample['caption'])
    cap_tok = sample['cap_tok']
    plt.figure()
    plt.imshow(image.permute(1, 2, 0))
    print("Caption: ", caption)
    print("Tokenized Caption: ", cap_tok)
    plt.show()

def predict(model, device, image_path, vocab_file='vocab.txt'):
    vocab = []
    with open(vocab_file, "r") as vocab_f:
        for line in vocab_f:
            vocab.append(line.strip())
    # image_path = os.path.join(img_dir, image_name)
    image = Image.open(image_path)
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    # model.to(device)
    hypotheses = eval.get_output_sentence(model, device, image, vocab)

    for i in range(len(hypotheses)):
        hypotheses[i] = [vocab[token - 1] for token in hypotheses[i]]
        hypotheses[i] = " ".join(hypotheses[i])

    return hypotheses

parser = argparse.ArgumentParser(description='Training Script for Encoder + Transformer Decoder')
parser.add_argument('--vocab-file', type=str, help='text file of all words', default='vocab.txt')
parser.add_argument('--img-path', type=str, help='complete path to image file', default='')
parser.add_argument('--model-path', type=str, help='complete path to model file (.pt)', default='')
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--batch-size', type=int, help='batch size', default=64)
parser.add_argument('--batch-size-val', type=int, help='batch size validation', default=64)
parser.add_argument('--encoder-type', choices=['resnet18', 'resnet50', 'resnet101', 'vgg16', 'inception_v3',
                                               'efficientnet_b0', 'efficientnet_b1', 'googlenet'],
                    default='resnet18',
                    help='Network to use in the encoder (default: resnet18)')
parser.add_argument('--fine-tune', type=int, choices=[0, 1], default=0)
parser.add_argument('--beam-width', type=int, default=4)
parser.add_argument('--num-epochs', type=int, default=35)
parser.add_argument('--experiment-name', type=str, default="autobestmodel")
parser.add_argument('--num-tf-layers', help="Number of transformer layers", type=int, default=3)
parser.add_argument('--num-heads', help="Number of heads", type=int, default=2)
parser.add_argument('--beta1', help="Beta1 for Adam", type=float, default=0.9)
parser.add_argument('--beta2', help="Beta2 for Adam", type=float, default=0.999)
parser.add_argument('--dropout-trans', help="Dropout_Trans", type=float, default=0.1)
parser.add_argument('--smoothing', help="Label smoothing", type=int, default=1)
parser.add_argument('--Lepsilon', help="Label smoothing epsilon", type=float, default=0.1)
parser.add_argument('--use-checkpoint', help="Use checkpoint or start from beginning", type=int, default=0)
parser.add_argument('--checkpoint-name', help="Checkpoint model file name", type=str, default=None)

args = parser.parse_args()

encoder_type = args.encoder_type
decoder_type = transformer
warmup_steps = 4000
n_iter = 1

if encoder_type == 'resnet18' or encoder_type == 'vgg16':
    CNN_channels = 512
elif 'efficientnet' in encoder_type:
    CNN_channels = 1280
elif encoder_type == 'googlenet':
    CNN_channels = 1024
else:
    CNN_channels = 2048

max_epochs = args.num_epochs
beam_width = args.beam_width

word_embedding_size = 1024
attention_dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lamda = 1.
batch_size = args.batch_size
batch_size_val = args.batch_size_val
grad_clip = 5.
transformer_layers = args.num_tf_layers
heads = args.num_heads
beta1 = args.beta1
beta2 = args.beta2

# print("Label smoothing set to: ", bool(args.smoothing))
learning_rate = 0.00001
# learning_rate = (CNN_channels*(-0.5)) * min(n_iter(-0.5), n_iter(warmup_steps**(-1.5)))
decoder_hidden_size = CNN_channels
dropout = args.dropout_trans

batch_size_val = args.batch_size_val
grad_clip = 5.
transformer_layers = args.num_tf_layers
heads = args.num_heads
beta1 = args.beta1
beta2 = args.beta2

vocab = []
with open(args.vocab_file, "r") as vocab_f:
    for line in vocab_f:
        vocab.append(line.strip())

train_data = Flickr8kDataset(img_dir, train_dir, ann_dir, vocab_file)
val_data = eval.TestDataset(img_dir, val_dir, ann_dir, vocab_file)
test_data = eval.TestDataset(img_dir, test_dir, ann_dir, vocab_file)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collater)
val_dataloader = DataLoader(val_data, batch_size=batch_size_val, shuffle=False, collate_fn=eval.collater)
test_dataloader = DataLoader(test_data, batch_size=batch_size_val, shuffle=False, collate_fn=eval.collater)

encoder_class = Encoder
decoder_class = TransformerDecoder

model = EncoderDecoder(encoder_class, decoder_class, train_data.vocab_size, target_sos=train_data.SOS,
                       target_eos=train_data.EOS, fine_tune=bool(args.fine_tune), encoder_type=args.encoder_type,
                       encoder_hidden_size=CNN_channels,
                       decoder_hidden_size=decoder_hidden_size,
                       word_embedding_size=word_embedding_size, attention_dim=attention_dim, decoder_type=decoder_type,
                       cell_type='lstm', beam_width=beam_width, dropout=dropout,
                       transformer_layers=transformer_layers, num_heads=heads)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             betas=(beta1, beta2))  # used to experiment with (0.9, 0.98) for transformer
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
model.to(device)
model.eval()

pred = predict(model, device, args.img_path)
print("predicted caption: \n",pred[0], "\n")
img = Image.open(args.img_path)
width, height = img.size[:2]

if height > width:
    baseheight = 320
    hpercent = (baseheight/float(img.size[1]))
    wsize = int((float(img.size[0])*float(hpercent)))
    img = img.resize((wsize, baseheight), Image.ANTIALIAS)
else:
    basewidth = 320
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)

display(img)