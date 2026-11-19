import torch
import random
from data_loading import tensorFromSentence

SOS_token = 0
EOS_token = 1

def evaluate(encoder, decoder, sentence, input_lang, output_lang, device):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, device, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('Source  >', pair[0])
        print('Target  =', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang, device)
        output_sentence = ' '.join(output_words)
        print('Predict <', output_sentence)
        print('')

def evaluateUserInput(encoder, decoder, input_lang, output_lang, device, input_text):
    #pair = random.choice(pairs)
    print('Input   >', input_text)
    #print('=', pair[1])
    output_words, _ = evaluate(encoder, decoder, input_text, input_lang, output_lang, device)
    output_sentence = ' '.join(output_words)
    print('Predict <', output_sentence)
    print('')
