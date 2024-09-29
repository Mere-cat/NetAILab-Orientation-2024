# Topic3 Assignment
Topic3: Sequence Model

## Assignment List:
* Assignment3.ipynb

## Report

作業希望完成模型挖空的 TO-DOs，以建立一 seq2seq 的機器翻譯模型（且加入了 attention 機制）。

在作業中，首先要完成

### TO-DO 1: Implement Padding
```python
def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (4-6 Lines)
    list_len = [len(i) for i in sents]
    max_len = max(list_len)

    for sub_list in sents:
        sub_list += [pad_token] * (max_len - len(sub_list))
        sents_padded.append(sub_list)

    ### END YOUR CODE
    return sents_padded
```
在迴圈中迭代所有的句子，並對每個句子補上 padding token 到最長句子的長度

### TO-DO 3: Implement NMT Model - 定義各神經層
```python
### YOUR CODE HERE (~8 Lines)
### TODO - Initialize the following variables:
###     self.encoder = nn.LSTM() (Bidirectional LSTM with bias)
###     self.decoder = nn.LSTMCell() (LSTM Cell with bias)
###     self.h_projection = nn.Linear() (Linear Layer with no bias), called W_{h}.
###     self.c_projection = nn.Linear() (Linear Layer with no bias), called W_{c}.
###     self.att_projection = nn.Linear() (Linear Layer with no bias), called W_{attProj}.
###     self.combined_output_projection = nn.Linear() (Linear Layer with no bias，注意 false), called W_{u}.
###     self.target_vocab_projection = nn.Linear() (Linear Layer with no bias), called W_{vocab}.
###     self.dropout = nn.Dropout() (Dropout Layer)
###
### Use the following docs to properly initialize these variables:
###     LSTM:
###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
###     LSTM Cell:
###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
###     Linear Layer:
###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
###     Dropout Layer:
###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout

self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True)     # (Bidirectional LSTM with bias)
self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size)       # (LSTM Cell with bias)
self.h_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)   # (Linear Layer with no bias), called W_{h}.
self.c_projection = nn.Linear(2*hidden_size, hidden_size, bias=False, device = self.device)   # (Linear Layer with no bias), called W_{c}.
self.att_projection = nn.Linear(2*hidden_size, hidden_size, bias=False, device = self.device) # (Linear Layer with no bias，注意 false), called W_{attProj}.
self.combined_output_projection = nn.Linear(3*hidden_size, hidden_size, bias=False) # (Linear Layer with no bias，注意 false), called W_{u}.
self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)   # (Linear Layer with no bias), called W_{vocab}.
self.dropout = nn.Dropout(dropout_rate)  # (Dropout Layer)
### END YOUR CODE
```
依照註解的指示，填入參數完成此 NMT 模型的架構：
* encoder: 一整組雙向 [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)。需填入 input_size 與 hidden_size，同時將雙向設置為 True。
    * `input_size=embed_size`: 輸入的大小，即原始語句 embedding 後的大小（直接使用傳入__init__的參數）
    * `hidden_size=hidden_size`: 輸出 hidden state 的大小（直接使用傳入__init__的參數）
* decoder: 單一 [LSTMCell](https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html)。亦需填入 input_size 與 hidden_size，由於 input 會讀入 encoder
    * `input_size=embed_size + hidden_size`: seq2seq 中，原是傳入 decoder 產生的嵌入向量，但由於引入了 attention 機制，我們還會再傳入一 hidden state，因此 input size 會是兩者加總
    * `hidden_size=hidden_size`: 輸出 hidden state 的大小（直接使用傳入__init__的參數）
* h_projection: 參數 W_{h} 的線性層。需填入 in_features 與 out_features
    * `in_feature=2*hidden_size`: 要乘上往前與往後的 hidden state
    * `out_feature=hidden_size`: 最終輸出成一 1*h 維度的 hidden layer
* c_projection: 同上
* att_projection: 同上
* combined_output_projection: 這裡要 concatenate attention output 和 hidden state。
    * `in_feature=3*hidden_size`: attention output 出來的大小也是 hidden_size，加上雙向的 hidden state，這裡的 input 大小會是三個 hidden_size
    * `out_feature=hidden_size`: 最終輸出成一 1*h 維度的向量 v_{t}
* target_vocab_projection: 向量 o_{t}（v_{t} 經過 tanh 與 dropout 後的結果） 經 softmax 處理前，乘上的向量
    * `in_feature=hidden_size`: 這時傳入的 o_{t} 已在上一步降為成 hidden_size
    * `out_feature=len(vocab.tgt)`: 輸出大小即目標單字的長度
* dropout: 傳入 dropout_rate



### TO-DO 4: Implement NMT Model - Encode & Decode
#### Encode
```python
### YOUR CODE HERE (8~10 Lines)
### TODO:
###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
###         that there is no initial hidden state or cell for the decoder.
###     2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
###         - Note that the shape of the tensor returned by the encoder is (src_len, b, h*2) and we want to
###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
###     3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
###         - `init_decoder_hidden`:
###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
###             This is h_0^{dec}. Here b = batch size, h = hidden size
###         - `init_decoder_cell`:
###             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
###             Apply the c_projection layer to this in order to compute init_decoder_cell.
###             This is c_0^{dec}. Here b = batch size, h = hidden size
###
### See the following docs, as you may need to use some of the following functions in your implementation:
###     Pack the padded sequence X before passing to the encoder:
###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
###     Pad the packed sequence, enc_hiddens, returned by the encoder:
###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
###     Tensor Concatenation:
###         https://pytorch.org/docs/stable/torch.html#torch.cat
###     Tensor Permute:
###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute

# 1. Construct Tensor `X` of source sentences with shape (src_len, b, e)
X = self.model_embeddings.source(source_padded.T).permute(1, 0, 2)
X_packed = pack_padded_sequence(X, source_lengths)

# 2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applyin
enc_hiddens, (last_hidden, last_cell) = self.encoder(X_packed)
enc_hiddens = pad_packed_sequence(enc_hiddens)[0].permute(1, 0, 2)

# 3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
state = torch.cat([*last_hidden], dim=1), torch.cat([*last_cell], dim=1)
dec_init_state = self.h_projection(state[0]), self.c_projection(state[1])

### END YOUR CODE
```
#### Decode
```python
### YOUR CODE HERE (~9 Lines)
### TODO:
###     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
###         which should be shape (b, src_len, h),
###         where b = batch size, src_len = maximum source length, h = hidden size.
###         This is applying W_{attProj} to h^enc.
###     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
###     3. Use the torch.split function to iterate over the time dimension of Y.
###         Within the for loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
###             - Squeeze Y_t into a tensor of dimension (b, e).
###             - Construct Ybar_t by concatenating Y_t with o_prev on their last dimension
###             - Use the step function to compute the the Decoder's next (cell, state) values
###               as well as the new combined output o_t.
###             - Append o_t to combined_outputs
###             - Update o_prev to the new o_t.
###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
###
### Note:
###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
###      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
###
### You may find some of these functions useful:
###     Zeros Tensor:
###         https://pytorch.org/docs/stable/torch.html#torch.zeros
###     Tensor Splitting (iteration):
###         https://pytorch.org/docs/stable/torch.html#torch.split
###     Tensor Dimension Squeezing:
###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
###     Tensor Concatenation:
###         https://pytorch.org/docs/stable/torch.html#torch.cat
###     Tensor Stacking:
###         https://pytorch.org/docs/stable/torch.html#torch.stack


# 1. Apply the attention projection layer to `enc_hiddens`        
enc_hiddens_proj = self.att_projection(enc_hiddens)

# 2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e)
Y = self.model_embeddings.target(target_padded.T).permute(1, 0, 2)

# 3. Use the torch.split function to iterate over the time dimension of Y
for Y_t in torch.split(Y, 1):
    # Perform time-steps
    Y_t = Y_t.squeeze()
    Ybar_t = torch.cat([Y_t, o_prev], dim=1)
    dec_state, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
    combined_outputs.append(o_t)
    o_prev = o_t

# 4. Use torch.stack to convert combined_outputs
combined_outputs = torch.stack(combined_outputs)
### END YOUR CODE
```

### TO-DO 5: Implement NMT Model - Step
```python
### YOUR CODE HERE (~3 Lines)
### TODO:
###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
###     3. Compute the attention scores e_t, a Tensor shape (b, src_len).
###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
###
###       Hints:
###         - dec_hidden is shape (b, h) and corresponds to h^dec_t
###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
###         - Use batched matrix multiplication (torch.bmm) to compute e_t (be careful about the input/ output shapes!)
###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
###
### Use the following docs to implement this functionality:
###     Batch Multiplication:
###        https://pytorch.org/docs/stable/torch.html#torch.bmm
###     Tensor Unsqueeze:
###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
###     Tensor Squeeze:
###         https://pytorch.org/docs/stable/torch.html#torch.squeeze

# 1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
dec_state = self.decoder(Ybar_t, dec_state)

# 2. Split dec_state into its two parts (dec_hidden, dec_cell)
dec_hidden, dec_cell = dec_state

# 3. Compute the attention scores e_t, a Tensor shape (b, src_len)
e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(-1)).squeeze(-1)
### END YOUR CODE

# Set e_t to -inf where enc_masks has 1
if enc_masks is not None:
    e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))




### YOUR CODE HERE (~6 Lines)
### TODO:
###     1. Apply softmax to e_t to yield alpha_t
###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
###         attention output vector, a_t.
#$$     Hints:
###           - alpha_t is shape (b, src_len)
###           - enc_hiddens is shape (b, src_len, 2h)
###           - a_t should be shape (b, 2h)
###           - You will need to do some squeezing and unsqueezing.
###     Note: b = batch size, src_len = maximum source length, h = hidden size.
###
###     3. Concatenate dec_hidden with a_t to compute tensor U_t
###     4. Apply the combined output projection layer to U_t to compute tensor V_t
###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
###
### Use the following docs to implement this functionality:
###     Softmax:
###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
###     Batch Multiplication:
###        https://pytorch.org/docs/stable/torch.html#torch.bmm
###     Tensor View:
###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
###     Tensor Concatenation:
###         https://pytorch.org/docs/stable/torch.html#torch.cat
###     Tanh:
###         https://pytorch.org/docs/stable/torch.html#torch.tanh

# 1. Apply softmax to e_t to yield alpha_t
alpha_t = F.softmax(e_t, dim=1)

# 2. Use batched matrix multiplication between alpha_t and enc_hiddens
a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(1)

# 3. Concatenate dec_hidden with a_t to compute tensor U_t
U_t = torch.cat([dec_hidden, a_t], dim=1)

# Apply the combined output projection layer to U_t to compute tensor V_t
V_t = self.combined_output_projection(U_t)

# Compute tensor O_t by first applying the Tanh function and then the dropout layer
O_t = self.dropout(torch.tanh(V_t))
### END YOUR CODE
```

### Reference
[Code and written solutions of the assignments of the Stanford CS224N](https://github.com/floriankark/cs224n-win2223)