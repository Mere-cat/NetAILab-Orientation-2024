# Topic5: GNN

## Environment Setting
若是在 server 上跑（conda 環境）而非在 colab，有幾點要注意:
1. 建立新 conda 環境（或使用現存的）
2. 下載 [PyTorch](https://pytorch.org/): 
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
3. 下載原本筆記本中用 pip 下載的 [PyG 套件](https://anaconda.org/pyg/repo)：
    * [torch_scatter](https://anaconda.org/pyg/pytorch-scatter)
        ```bash
        conda install pyg::pytorch-scatter
        ```
    * [torch-sparse](https://anaconda.org/pyg/pytorch-sparse)
        ```bash
        conda install pyg::pytorch-sparse
        ```
    * [torch-cluster](https://anaconda.org/pyg/pytorch-cluster)
        ```bash
        conda install pyg::pytorch-cluster
        ```
    * **[失敗]** [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
        ```bash
        conda install pyg -c pyg
        ```
        使用 conda 安裝會[報錯](https://github.com/pyg-team/pytorch_geometric/discussions/7866)，但即使最後再安裝 pyg 還是不行
        只能暫時用 pip
        ```bash
        pip install torch_geometric
        ```

## 2-2-ApplyGNN
嘗試修改模型結構 (EX: 更換模型, 調整 Link pred 預測邊的方式, 調整鄰居層數 等等)，觀察其對模型表現的影響

我進行以下更動：
### 更動模型：
```python
def __init__(self, in_channels, hidden_channels, out_channels):
    super().__init__()
    ############################################################################
    # TODO: Your code here! 
    # create you GNN layer here. 
    # try to use different GNN backbone layer or stacking multiple layer to boost performance
    self.conv1 = SAGEConv(in_channels, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, out_channels)
    
    self.dropout = torch.nn.Dropout(p=0.5)
    ############################################################################

def forward(self, x, edge_index):
    ############################################################################
    # TODO: Your code here! 
    # Apply the forward pass according to your GNN layers
    # you shoud return the embedding of each node (x has shape [num_nodes, dim])    
    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = self.dropout(x)  # Dropout to regularize

    x = self.conv2(x, edge_index)
    ############################################################################
    return x
```
* 增加 dropout 以及將第一層 Conv 改為 SAGEConv
* 嘗試增加鄰居層數，但效果反而不好，故最後仍使用兩層

### 調整 Link pred criterion
```python

############################################################################
# TODO: Your code here! 
# initiate your GNN model and select the criterion for link prediction

model = MyGNN(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
# criterion = torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.MSELoss()
############################################################################
```
* 將 criterion 改成 MSELoss()

### Result
Final Test: 0.9184

## 3-GNN_Aggregation_Function
完成 GraphSAGE 的 Message Passing