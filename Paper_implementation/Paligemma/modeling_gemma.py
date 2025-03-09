import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip_vision import SiglipVisionConfig, SiglipVisionModel

class KVCache():

  def __init__(self) -> None:
    self.key_cache: List[torch.Tensor] = []
    self.value_cache: List[torch.Tensor] = []

  def num_items(self) -> int:
    if len(self.key_cache) == 0:
      return 0
    else:
      #the shape of the key_cache is [batch_size, num_heads_kv, seq_len, head_dim]
      return self.key_cache[0].shape[-2]

  def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      layer_idx: int,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(self.key_cache) <= layer_idx:
      #if we never added anything to the kv_cache of this layer lets create it.
      self.key_cache.append(key_states)
      self.value_cache.append(value_states)

    else:
      #... otherwise we concatenate the new keys with the existing ones
      #each tensor has shape [batch_size, num_heads_kv, seq_len, head_dim]
      self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim =-2)
      self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim = -2)
    #....and then we return all the existing keys+ new ones
    return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaConfig():

  def __init__(
      self,
      vocab_size,
      hidden_size,
      intermediate_size,
      num_hidden_layers,
      num_attention_heads,
      num_key_value_heads,
      head_dim = 256,
      max_position_embedding = 8192,
      rms_norm_eps = 1e-6,
      rope_theta = 10000.0,
      attention_bias = False,
      attention_dropout = 0.0,
      pad_token_id = None,
      **kwargs
  ):

    super().__init__()
    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embedding
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.head_dim = head_dim
    self.num_key_value_heads = num_key_value_heads
    self.rms_norm_eps = rms_norm_eps
    self.rope_theta = rope_theta
    self.attention_bias = attention_bias
    self.attention_dropout = attention_dropout
    self.pad_token_id = pad_token_id

#implementing PaligemmaConfig
class PaliGemmaConfig():

  def __init__(self,
               vision_config = None,
               text_config= None,
               ignore_index = -100,
               image_token_index = 256000,
               vocab_size = 257152,
               projection_dim = 2048,
               hidden_size = 2048,
               pad_token_id = None,
               **kwargs,
               ):

    super().__init__()
    self.ignore_index = ignore_index
    self.image_token_index = image_token_index
    self.vocab_size = vocab_size
    self.projection_dim = projection_dim
    self.hidden_size = hidden_size
    self.vision_config = vision_config
    self.is_encoder_decoder = False
    self.pad_token_id = pad_token_id

    self.vision_config = SiglipVisionConfig(**vision_config)
    self.text_config = text_config

    self.text_config = GemmaConfig(**text_config, pad_token_id = pad_token_id)
    self.vocab_size = vocab_size

    self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size)**2

    self.vision_config.projection_dim = projection_dim

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  batch, num_key_value_heads, seq_len , head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:,:,None, :,:].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class GemmaRotatoryEmbedding(nn.Module):

  def __init__(self, dim , max_position_embeddings = 2048, base = 10000, device = None):
    super().__init__()

    self.dim = dim #it is set as head_dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base

    #calculate the theta according to the formulat theta_i = base^(2i/dim) where i = 0,2,4,6,.... dim //2
    inv_freq = 1.0 / self.base ** (torch.arange(0,self.dim,2,dtype = torch.int64).float()/ self.dim)
    self.register_buffer("inv_freq", tensor = inv_freq, persistent = False)

  @torch.no_grad()
  def forward(self, x, position_ids, seq_len = None):
    #x = [batch_size, num_attention_heads,seq_len, head_size]
    self.inv_freq.to(x.device)
    #copy the inv_freq tensor for batch in the sequence
    #inv_freq_expanded = [batch_size, head_dim //2,1]
    inv_freq_expanded = self.inv_freq[None,:,None].float().expand(position_ids.shape[0],-1,1)
    #postion_ids_expanded : [batch_size, 1,seq_len]
    position_ids_expanded = position_ids[:,None,:].float()
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

    with torch.autocast(device_type= device_type, enabled = False):
      #Multiply each theta by the position(which is the argument of the sin and cos functions)
      #freqs: [batch_size, head_dim//2,1] @[batch_size, 1, seq_len] -> [batch_size, seq_len, head_dim //2]
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float().transpose(1,2))
      #emb: [batch_size, seq_len, head_dim]
      emb = torch.cat((freqs, freqs),dim =-1)
      #cos,sin = [batch_size,seq_len, head_dim]
      cos = emb.cos()
      sin = emb.sin()

    return cos.to(dtype = x.dtype), sin.to(dtype = x.dtype)

def rotate_half(x):
  #build the [-x2,x1,-x4,x3....] tensor for the sin part of the positional encoding
  x1 = x[...,:x.shape[-1]//2] #Take the first half of the last dimensions
  x2 = x[..., x.shape[-1]//2:] #takes the second half of the last dimension

  return torch.cat((-x2,x1), dim = -1)

def apply_rotatory_pos_emb(q,k,cos,sin, unsqueeze_dim = 1):
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)

  #apply the formula (34) to the rotatory positional emcoding paper
  q_embed = (q* cos) + (rotate_half(q)*sin)
  k_embed = (k* cos) + (rotate_half(k) * sin)

  return q_embed, k_embed


class GemmaAttention(nn.Module):

  def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
    super().__init__()
    
    self.config = config
    self.layer_idx = layer_idx

    self.attention_dropout = config.attention_dropout
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = config.head_dim
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True

    assert self.hidden_size % self.num_heads == 0, "hidden size is not divisible by num heads"

    #Number of heads = 8
    #Hidden_Size = 1024
    # Head_dim = 1024/8 = 128
    #Wq = [1024, 8*128] = [1024,1024]
    #Wk = [1024, 4 *128] = [1024,512]
    #Wv = [1024, 4*128] = [1024,512]
    #this technique is known as grouped query attention
    #where suppose for two query heads there will be one key head
    #In multi-query you have only one head in key and one in value

    self.q_proj = nn.Linear(self.hidden_size , self.num_heads * self.head_dim, bias = config.attention_bias)
    self.k_proj = nn.Linear(self.hidden_size , self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
    self.v_proj = nn.Linear(self.hidden_size , self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
    self.o_proj = nn.Linear(self.num_heads* self.head_dim , self.hidden_size, bias = config.attention_bias)

    self.rotatory_emb = GemmaRotatoryEmbedding(
        self.head_dim,
        max_position_embeddings = self.max_position_embeddings,
        base = self.rope_theta,
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
      kv_cache: Optional[KVCache] = None,
      **kwargs,
  )-> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    batch_size , q_len,_ = hidden_states.size() #[batch_size, seq_len, hidden_size]
    #[batch_size, seq_len, num_head_q*head_dim]
    query_states = self.q_proj(hidden_states)
    #[batch_size, seq_len, num_heads_kv*head_dim]
    key_states  = self.k_proj(hidden_states)
    #[batch_size, seq_len, num_heads_kv*head_dim]
    value_states  = self.v_proj(hidden_states)
    #[batch_size, num_heads_q, seq_len,head_dim]
    query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1,2)
    #[batch_size, num_heads_q, seq_len,head_dim]
    key_states = key_states.view(batch_size,q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
    #[batch_size, num_heads_q, seq_len,head_dim]
    value_states = value_states.view(batch_size,q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

    #[batch_size, seq_len, head_dim], [batch_size, seq_len, head_dim]
    cos,sin = self.rotatory_emb(value_states, position_ids, seq_len = None)
    #[batch_size,num_heads_q, seq_len, head_dim], [batch_size, num_heads_kv, seq_len, head_dim]
    query_states, key_states = apply_rotatory_pos_emb(query_states,key_states, cos,sin)

    if kv_cache is not None:
      key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

    #Repeat the key and values to match the number of heads to the query
    #its kind of reversing the grouped query attention cuz we dont have seperate cuda kernels
    #but this could be leveraged in flash attention
    #this output will have same number of heads as of the query_states
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    #Perform teh calculation as usual , Q* k^T / sqrt(head_dim) , shape: [batch_size, num_heads_q, se_len_q, seq_len_kv]
    attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)

    assert attention_mask is not None
    attn_weights = attn_weights + attention_mask


    #apply the softmax
    #[batch_size, num_heads_q, seq_len_q, seq_len_kv]
    attn_weights = nn.Softmax(attn_weights, dim = -1, dtype = torch.float32).to(query_states.dtype)
    #apply the dropout
    attn_weights = nn.dropout(attn_weights, p = self.attention_dropout, training = self.training)
    #multiply by the value. [batch_size, num_heads_q, seq_len_q, seq_len_kv] * [batch_size, num_heads_kv, seq_len_kv, head_dim]->
    # [batch_size,num_heads_q, seq_len_q, head_dim ]
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
      raise ValueError(
          f"attn_output should be of size {(batch_size, self.num_heads, q_len, self.head_dim)} but is"
          f"{attn_output.size()}"

      )

    #make sure the sequence length is the second dimension .
    #[batch_size, num_heads_q, seq_len, head_dim] -> [batch_size, seq_len, num_heads_q, head_dim]
    attn_output = attn_output.transpose(1,2).contigous()
    #concatenate all the heads together. [batch_Size, seq_len_q, num_heads_q, head_dim]
    #[batch_size, seq_len_q, num_heads_q * head_dim]
    attn_output = attn_output.view(batch_size, q_len,-1)

    #multiply by W_o [batch_size, seq_len_q, hidden_size]
    #just to mix the results so the value should not be of just the concatenated heads
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


class GemmaMLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
    self.up_proj = nn.Linear(self.hidden_size , self.intermediate_size, bias = False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias = False)

  def forward(self, x):
    # Equivalent to:
    # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
    # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
    # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
    # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
    # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
    return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

#coding Gemma Decoder layer
class GemmaDecoderLayer(nn.Module):

  def __init__(self, config:GemmaConfig, layer_idx: int):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = GemmaAttention(config = config, layer_idx = layer_idx)

    self.mlp = GemmaMLP(config)
    self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
    self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)

  def forward(self,
              hidden_states : torch.Tensor,
              attention_mask: Optional[torch.Tensor] = None,
              position_ids : Optional[torch.LongTensor] = None,
              kv_cache :Optional[KVCache] = None,
              ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    residual = hidden_states
    #[batch_size, seq_len, hidden_size]
    hidden_states = self.input_layernorm(hidden_states)

    #[batch_size, seq_len,hidden_size]
    hidden_states, _ = self.self_attn(
        hidden_states = hidden_states,
        attention_mask = attention_mask,
        position_ids = position_ids,
        kv_cache = kv_cache
    )

    #[batch_size, seq_len,hidden_size]
    hidden_states = hidden_states + residual

    #[batch_size, seq_len,hidden_size]
    residual = hidden_states

    #[batch_size, seq_len,hidden_size]
    hidden_states = self.post_attention_layernorm(hidden_states)
    #[batch_size, seq_len,hidden_size]
    hidden_states = self.mlp(hidden_states)

    #[batch_size, seq_len,hidden_size]
    hidden_states = residual + hidden_states

    return hidden_states

class GemmaRMSNorm(nn.Module):
  def __init__(self, dim : int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.zeros(dim)) #gamma parameter

  def _norm(self,x):
    return x*torch.rsqrt(x.pow(2).mean(-1, keepdim = True)+ self.eps)

  def forward(self,x):
    output = self._norm(x.float())
    #LLama does x.to(float16) * w whilst Gemma is (x*w).to(float16)

    output = output* (1.0 + self.weight.float())
    return output.type_as(x)

#coding gemma
class GemmaModel(nn.Module):

  def __init__(self, config: GemmaConfig):
    super().__init__()
    self.config = config
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    #padding_idx is used to make sure the length of the sentence is equal to all other sentence
    #to fit it with same length
    #This padding_idx is not considered by the model thats why we put it into the embedding layer 
    self.embed_tokens= nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList(
        [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
    )

    
    self.norm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)

  def get_input_embeddings(self):
    return self.embed_tokens

  def forward(self,
              attention_mask: Optional[torch.Tensor] = None,
              position_ids : Optional[torch.LongTensor] = None,
              inputs_embeds: Optional[torch.FloatTensor] = None,
              kv_cache :Optional[KVCache] = None,
              ) -> torch.FloatTensor:

    #[batch_size,seq_len_hidden_size]
    hidden_states = inputs_embeds
    #[batch_size, seq_len, hidden_size]
    normalizer = torch.tensor(self.config.hidden_size**0.5 , dtype = hidden_states.dtype)
    hidden_states = hidden_states * normalizer


    for decoder_layer in self.layers:

      #[batch_size, seq_len, hidden_size]
      hidden_states = decoder_layer(
          hidden_states,
          attention_mask = attention_mask,
          position_ids = position_ids,
          kv_cache = kv_cache,
      )


    #[batch_size, seq_len, hidden_size]
    hidden_states = self.norm(hidden_states)

    #[batch_size, seq_len, hidden_size]
    return hidden_states




#coding gemma model
#causalLM -> language model + Linear layer
class GemmaForCausalLM(nn.Module):
    """
    GemmaForCausalLM is a class for causal language modeling using the Gemma architecture.
    Args:
        config (Config): Configuration object containing model parameters.
    Attributes:
        config (Config): Configuration object containing model parameters.
        model (GemmaModel): The underlying Gemma model.
        vocab_size (int): Size of the vocabulary.
        lm_head (nn.Linear): Linear layer for language modeling head.
    Methods:
        get_input_embeddings():
            Returns the input embeddings from the underlying Gemma model.
        tie_embedding():
            Ties the weights of the language modeling head to the input embeddings.
        forward(attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None) -> Tuple:
            Performs a forward pass through the model.
            Args:
                attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on padding token indices.
                position_ids (Optional[torch.LongTensor]): Indices of positions of each input sequence tokens in the batch.
                input_embeds (Optional[torch.FloatTensor]): Precomputed input embeddings.
                kv_cache (Optional[KVCache]): Key-value cache for faster decoding.
            Returns:
                return_data (dict): Dictionary containing:
                    - "logits" (torch.Tensor): Logits for the next token prediction.
                    - "kv_cache" (Optional[KVCache]): Updated key-value cache if provided.
    """

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_embedding(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                input_embeds: Optional[torch.FloatTensor] = None,
                kv_cache :Optional[KVCache] = None,
                ) -> Tuple:

        #input_embeds : [batch_size, seq_len, hidden_size]
        #outputs: [batch_size,seq_len, hidden_size]

        outputs = self.model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            input_embeds = input_embeds,
            kv_cache = kv_cache,
        )

        #pass the embeddings to the logit layer to predict next token
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            #return the updated cache
            return_data['kv_cache'] = kv_cache

        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
  def __init__(self, config: PaliGemmaConfig):
    super().__init__()
    self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias = True)

  def forward(self,image_features):

    #[batch_size, num_patches, embed_dim] -> [batch_size, num_patches, projection_dim]
    hidden_states = self.linear(image_features)
    return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):

  def __init__(self, config: PaliGemmaConfig):
    super().__init__()
    self.config = config
    self.vision_tower = SiglipVisionModel(config.vision_config)
    self.multi_modal_projection = PaliGemmaMultiModalProjector(config)
    self.vocab_size = config.vocab_size

    language_model = GemmaForCausalLM(config.text_config)
    self.language_model = language_model

    self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

  def tie_weights(self):
    return self.language_model.tie_weights()

  def _merge_input_ids_with_image_features(
      self, image_features:torch.Tensor,
      input_embeds: torch.Tensor, input_ids: torch.Tensor,
      attention_mask : torch.Tensor, kv_cache: Optional[KVCache] = None
  ):
    _, _, embed_dim = image_features.shape
    batch_size,sequence_length = input_ids.shape
    dtype, device = input_embeds.dtype, input_embeds.device
    #shape: [batch_size, seq_len,hidden_size]
    scaled_image_features = image_features / (self.config.hidden_size**0.5)

    #combine the embeddings of the image tokens , the text tokens and mask out all the padding tokens
    final_embedding = torch.zeros(batch_size,sequence_length, embed_dim, dtype = input_embeds.dtype, device = input_embeds.device)
    #shape: [batch_size, seq_len] . True for text tokens
    text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
    # [567,567,567,567,567,1,65,65,65,65,65,2]
    # [0,0,0,0,0,1,1,1,1,1,1,1]
    #shape: [batch_size, seq_len]. True for image tokens
    image_mask = input_ids == self.config.image_token_index
    #[1,1,1,1,1,1,0,0,0,0,0,0,0]
    #shape: [batch_size, seq_len]. True for padding tokens
    pad_mask = input_ids == self.pad_token_id
    #[0,0,0,0,0,0,0,0]

    #we need to expand the masks to the embedding dimension otherwise we cant use them in torch.where
    text_mask_expanded = text_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
    image_mask_expanded = image_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
    pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1,-1,embed_dim)

    #add the text embedding
    final_embedding = torch.where(text_mask_expanded,input_embeds, final_embedding)
    #Insert image embeddings , we can't use torch.where because the sequence length of scaled_image_features is not equal to sequence length of final embeddings
    final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
    #zero out padding tokens
    final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

    #### CREATE THE ATTENTION MASK #####

    dtype , device = input_embeds.dtype, input_embeds.device
    min_dtype = torch.finfo(dtype).min
    q_len = input_embeds.shape(1)

    #so this part is when we are passing the text and image embedding first time to the model 
    #then we dont need to mask out anything because we are just passing the embeddings and we wont have any cache
    if kv_cache is None or kv_cache.num_items() == 0:
      #Do not mask any token because we're in the prefill phase
      #This only works when we have no padding
      causal_mask = torch.full(
          (batch_size, q_len, q_len), fill_value = 0, dtype = dtype, device = device
      )
    else:
      #since we are generating tokens, the query must be one single token
      #you would have the other tokens in the kv_cache
      assert q_len == 1
      kv_len = kv_cache.num_items()+ q_len
      #Also in this case we don't need to mask anything , since each query should be able to attend all previous
      #This only works when we have no padding
      causal_mask = torch.full(
          (batch_size, q_len, kv_len), fill_value = 0, dtype = dtype, device =device
      )

    #we are not adding any mask in the paligemma as the image tokens and the text tokens needs to access the future tokens too
    # and currently we are doing the inferencing so we dont need to mask out cuz with the kv cache we will using the previous value
    # and it will only generate one single contexualized embedding which will have the knowledge of previous values
    # only when you train the model you need to mask out to compare the answer and everything


    #Add the head dimension
    #[Batch_size, Q_len, kv_len] -> [batch_size, num_heads_Q, Q_len, KV_len]
    causal_mask = causal_mask.unsqueeze(1)

    if kv_cache is not None and kv_cache.num_items() > 0:
      #the position of the query is just the last position
      position_ids = attention_mask.cumsum(-1)[:,-1]
      if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    else:
      #create a position_ids based on the size of the attention_mask
      #for masked tokens use the number 1 as the position
      position_ids = (attention_mask.cumsum(-1)).masked_fill((attention_mask ==0),1).to(device)

    return final_embedding, causal_mask, position_ids

  def forward(
      self,
      input_ids: torch.LongTensor = None,
      pixel_values: torch.FloatTensor = None,
      attention_mask: Optional[torch.Tensor] = None,
      kv_cache: Optional[KVCache] = None
  )-> Tuple:

    assert torch.all(attention_mask ==1), "The input cannot be padded."

    #1. Extract the input embeddings
    #shape [batch_size, seq_len, hidden_size]
    #here we pass the input id to the model '<image_token>...<image_token><bos><sentence_token>....\n' to convert
    #it into embeddings based on the seq_len and the dimension
    input_embeds = self.language_model.get_input_embeddings()(input_ids)


    #2. Merge text and images
    #[Batch_size, channels , height, width] -> [batch_size, NUM_PATCHES, EMBED_DIM]
    #contexualized image embeddings is generated after we pass the image through the vision tower
    selected_image_features = self.vision_tower(pixel_values.to(input_embeds.dtype)) 
    #we make the image embeddings to the same size of the language model
    #Linear Projector(Linear layer)
    #Here the contexualized embedding of the image is made equal to the size thats needed by the language model
    image_features = self.multi_modal_projection(selected_image_features)


    #merge the embeddings of the text tokens with the image tokens
    input_embeds, attention_mask , position_ids = self._merge_input_ids_with_image_features(
                                                    image_features,
                                                    input_embeds,
                                                    input_ids,
                                                    attention_mask ,
                                                    kv_cache
                                                     )

    outputs = self.language_model(
        attention_mask = attention_mask,
        position_ids = position_ids,
        input_embeds = input_embeds,
        kv_cache = kv_cache
    )

    return outputs

