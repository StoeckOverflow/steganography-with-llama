from .codec import Codec
from llama_cpp import Llama
from typing import List, Union
from io import StringIO
from nltk.corpus import stopwords
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizer
from ..utils import decode_secret

class SynonymHashCodec(Codec):
    '''
    ToDo: Start and End Token integration
    Source: https://github.com/ku-nlp
    '''

    def __init__(self, llm: Llama, disable_tqdm):
        super().__init__(llm)
        #self._tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained('resources/bert-base-cased', local_files_only=True)
        #self._model = BertForMaskedLM.from_pretrained('resources/bert-base-cased', local_files_only=True)
        self._tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self._model = BertForMaskedLM.from_pretrained('bert-base-cased')
        self._STOPWORDS: List[str] = stopwords.words('english')
        self.disable_tqdm = disable_tqdm
    
    def encode_single_string(self, single_feed: str,  binary_secret: str, mask_interval: int = 3, score_threshold: float = 0.01) -> tuple[str, str]:
        assert set(binary_secret) <= set('01')
        message_io = StringIO(binary_secret)
        processed = self._preprocess_text(single_feed, mask_interval)
        input_ids = processed['input_ids']
        masked_ids = processed['masked_ids']
        sorted_score, indices = processed['sorted_output']
        for i_token, token in enumerate(masked_ids):
            if token != self._tokenizer.mask_token_id:
                continue
            ids = indices[i_token]
            scores = sorted_score[i_token]
            candidates = self._pick_candidates_threshold(ids, scores, score_threshold)
            #print(self._tokenizer.convert_ids_to_tokens(candidates))
            if len(candidates) < 2:
                continue
            replace_token_id = self._block_encode_single(candidates, message_io).item()
            #print('replace', replace_token_id, self._tokenizer.convert_ids_to_tokens([replace_token_id]))
            input_ids[i_token] = replace_token_id
        encoded_message: str = message_io.getvalue()[:message_io.tell()]
        message_io.close()
        stego_text = self._tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return stego_text, encoded_message
    
    def encode_newsfeed(self, news_feed: list[str], binary_secret: str, labeled_for_training_flag=False, **kwargs) -> str:
        doctored_newsfeed = []
        for feed in news_feed:
            stego_text, binary_secret_string = self.encode_single_string(feed, binary_secret)
            doctored_newsfeed.append(stego_text)
            binary_secret = binary_secret[len(binary_secret_string):]
            if len(binary_secret) == 0:
                break
        if labeled_for_training_flag:
            return doctored_newsfeed + news_feed[len(doctored_newsfeed):], len(news_feed[len(doctored_newsfeed):])
        else:
            return doctored_newsfeed + news_feed[len(doctored_newsfeed):]
    
    def decode_single_string(self, newsfeed:str, mask_interval: int = 3, score_threshold: float = 0.01) -> str:
        decoded_message: List[str] = []
        processed = self._preprocess_text(newsfeed, mask_interval)
        input_ids = processed['input_ids']
        masked_ids = processed['masked_ids']
        sorted_score, indices = processed['sorted_output']
        for i_token, token in enumerate(masked_ids):
            if token != self._tokenizer.mask_token_id:
                continue
            ids = indices[i_token]
            scores = sorted_score[i_token]
            candidates = self._pick_candidates_threshold(ids, scores, score_threshold)
            if len(candidates) < 2:
                continue
            chosen_id: int = input_ids[i_token].item()
            print(self._block_decode_single(candidates, chosen_id))
            decoded_message.append(self._block_decode_single(candidates, chosen_id))
        return ''.join(decoded_message)
    
    def decode_newsfeed(self, newsfeed: list[str], **kwargs):
        remaining_decoded_secret = []
        for feed in newsfeed:
            decoded_message = self.decode_single_string(feed)
            remaining_decoded_secret.append(decoded_message)
        return decode_secret(''.join(remaining_decoded_secret))    

    def _predict(self, input_ids: Union[Tensor, List[List[int]]]):
        self._model.eval()
        with torch.no_grad():
            output = self._model(input_ids.unsqueeze(0))['logits'][0]
            softmaxed_score = F.softmax(output, dim=1)  # [word_len, vocab_len]
            return softmaxed_score.sort(dim=1, descending=True)
    
    def _preprocess_text(self, sentence: str, mask_interval: int) -> dict:
        encoded_ids = self._tokenizer([sentence], return_tensors='pt').input_ids[0]
        masked_ids = self._mask(encoded_ids.clone(), mask_interval)
        sorted_score, indices = self._predict(masked_ids)        
        return { 'input_ids': encoded_ids, 'masked_ids': masked_ids, 'sorted_output': (sorted_score, indices) }

    def _mask(self, input_ids: Union[Tensor, List[List[int]]], mask_interval: int) -> Tensor:
        length = len(input_ids)
        tokens: List[str] = self._tokenizer.convert_ids_to_tokens(input_ids)
        offset = mask_interval // 2 + 1
        mask_count = offset
        for i, token in enumerate(tokens):
            # Skip initial subword
            if i + 1 < length and self._is_subword(tokens[i + 1]): continue
            if not self._substitutable_single(token): continue
            if mask_count % mask_interval == 0:
                input_ids[i] = self._tokenizer.mask_token_id
            mask_count += 1
        return input_ids

    def _encode_topk(self, ids: List[int], message: StringIO, bits_per_token: int) -> int:
        k = 2**bits_per_token
        candidates: List[int] = []
        for id in ids:
            token = self._tokenizer.convert_ids_to_tokens(id)
            if not self._substitutable_single(token):
                continue
            candidates.append(id)
            if len(candidates) >= k:
                break
        return self._block_encode_single(candidates, message)

    def _pick_candidates_threshold(self, ids: Tensor, scores: Tensor, threshold: float) -> List[int]:
        filtered_ids: List[int] = ids[scores >= threshold]
        def filter_fun(idx: Tensor) -> bool:
            return self._substitutable_single(self._tokenizer.convert_ids_to_tokens(idx.item()))
        return list(filter(filter_fun, filtered_ids))

    def _substitutable_single(self, token: str) -> bool:
        if self._is_subword(token): return False
        if token.lower() in self._STOPWORDS: return False
        if not token.isalpha(): return False
        return True

    @staticmethod
    def _block_encode_single(ids: List[int], message: StringIO) -> int:
        assert len(ids) > 0
        if len(ids) == 1:
            return ids[0]
        capacity = len(ids).bit_length() - 1
        bits_str = message.read(capacity)
        if len(bits_str) < capacity:
            padding: str = '0' * (capacity - len(bits_str))
            bits_str = bits_str + padding
            message.write(padding)
        index = int(bits_str, 2)
        return ids[index]

    @staticmethod
    def _block_decode_single(ids: List[int], chosen_id: int) -> str:
        if len(ids) < 2:
            return ''
        capacity = len(ids).bit_length() - 1
        index = ids.index(chosen_id)
        return format(index, '0' + str(capacity) +'b')

    @staticmethod
    def _is_subword(token: str) -> bool:
        return token.startswith('##')