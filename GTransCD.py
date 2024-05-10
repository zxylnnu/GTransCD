import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class embedding(nn.Module):

    def __init__(self, node_input_dim, edge_input_dim, out_dim):
        super(embedding, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim

        self.node_embedding = nn.Linear(node_input_dim, out_dim)
        self.bn_node = nn.BatchNorm1d(num_features=out_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, edge_input_dim)
        self.bn_edge = nn.BatchNorm1d(num_features=edge_input_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, node_feature, edge_feature):
        node_feature_embedded = self.node_embedding(node_feature)
        node_feature_embedded = self.relu(self.bn_node(node_feature_embedded))
        edge_feature_embedded = self.edge_embedding(edge_feature)
        edge_feature_embedded = self.relu(self.bn_edge(edge_feature_embedded))

        return node_feature_embedded, edge_feature_embedded

def attention(q, k, v, gate, e, dropout):

    dim = k.shape[1]
    d_k = dim ** 0.5
    scores = torch.matmul(q, k.permute([1, 0])) / d_k
    ue = scores + e
    attm = F.softmax(ue)
    attm = F.dropout(attm, dropout)
    out = gate * attm
    out = torch.matmul(out, v)

    return out

class Att(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, e_dim: int):
        super().__init__()

        self.Q = nn.Linear(in_dim, out_dim, bias=True)
        self.K = nn.Linear(in_dim, out_dim, bias=True)
        self.V = nn.Linear(in_dim, out_dim, bias=True)
        self.proj_e = nn.Linear(e_dim, e_dim, bias=True)
        self.e_gate = nn.Linear(e_dim, e_dim, bias=True)

    def forward(self, h, e):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        e_gate = self.e_gate(e)
        e_gate = F.sigmoid(e_gate)
        proj_e = self.proj_e(e)

        outputs_n = attention(Q_h, K_h, V_h, e_gate, proj_e, dropout=0.2)

        return outputs_n

class Transformer(nn.Module):

    def __init__(self, in_dim, out_dim, e_dim, dropout):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.e_channels = e_dim
        self.dropout = dropout

        self.attention = Att(in_dim, out_dim, e_dim)
        self.lh0 = nn.Linear(out_dim, out_dim)
        self.lh1 = nn.Linear(out_dim, out_dim * 2)
        self.lh2 = nn.Linear(out_dim * 2, out_dim)
        self.bn_h = nn.BatchNorm1d(out_dim)

    def forward(self, h, e):

        h_attn_out = self.attention(h, e)
        h = h_attn_out.view(-1, self.out_channels)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.lh0(h)
        h = h_attn_out + h
        h = self.bn_h(h)
        h_res = h
        h = self.lh1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.lh2(h)
        h = h_res + h
        h = self.bn_h(h)

        return h

class myGCN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, adjacency_matrix: torch.Tensor):
        super(myGCN, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition1 = nn.LeakyReLU(inplace=True)
        self.A_in = adjacency_matrix
        self.lg0 = nn.Sequential(nn.Linear(input_dim, 128))
        self.lg1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.lambda_ = nn.Parameter(torch.zeros(1))
        self.I = torch.eye(adjacency_matrix.shape[0], adjacency_matrix.shape[1], requires_grad=False, device=device,
                               dtype=torch.float32)


    def forward(self, H):
        H = self.BN(H)
        H_xx1 = self.lg0(H)
        e = torch.matmul(H_xx1, H_xx1.t())
        e_softmax = F.softmax(e, dim=1)
        A = e_softmax * self.A_in + self.lambda_ * self.I

        out = self.lg1(H)  # refer to: https://ieeexplore.ieee.org/document/9547387
        out = self.Activition1(torch.mm(e_softmax, out))
        return out, A

class gcf(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim):

        super(gcf, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv_gates = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                    out_channels=2 * self.hidden_dim,
                                    kernel_size=1)

        self.conv_can = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=self.hidden_dim,
                                  kernel_size=1)

    def forward(self, input_tensor, h_cur):

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next

class gtrans(nn.Module):

    def __init__(self, height: int, width: int, changel: int, class_count: int,
                 hierarchy_matrices, adjacency_matrices):
        super(gtrans, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.S_mat = hierarchy_matrices
        self.A_mat = adjacency_matrices
        self.S_mat_Hat_T = []

        layer_channels = 128
        temp = hierarchy_matrices
        A_m, _ = self.A_mat.shape

        self.CNN_head_layer = nn.Conv2d(in_channels=self.channel, out_channels=layer_channels, kernel_size=1)
        self.S_mat_Hat_T=(temp / (torch.sum(temp, 0, keepdim=True, dtype=torch.float32))).t()
        self.GraphConv = myGCN(layer_channels, layer_channels, self.A_mat)
        self.CNN_tail_layer = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.ch_ln = nn.Linear(64, 16)
        self.Softmax = nn.Sequential(nn.Linear(16, self.class_count), nn.Softmax(-1))
        self.GCF = gcf(input_size=(self.height, self.width), input_dim=256, hidden_dim=256)
        self.Embedding = embedding(layer_channels, A_m, layer_channels)
        self.trans = Transformer(in_dim=layer_channels, out_dim=layer_channels, e_dim=A_m, dropout=0.2)

    def forward(self, x: torch.Tensor, x_be: torch.Tensor, x_af: torch.Tensor):

        (h, w, c) = x.shape
        x    = torch.unsqueeze(x.permute([2, 0, 1]), 0)
        x_be = torch.unsqueeze(x_be.permute([2, 0, 1]), 0)
        x_af = torch.unsqueeze(x_af.permute([2, 0, 1]), 0)

        H_0    = self.CNN_head_layer(x)
        H_0_be = self.CNN_head_layer(x_be)
        H_0_af = self.CNN_head_layer(x_af)

        H_0 = torch.squeeze(H_0, 0).permute([1, 2, 0])
        F3 = H_0.reshape([h * w, -1])

        H_i = torch.mm(self.S_mat_Hat_T, F3)
        H_i, A_i = self.GraphConv(H_i)

        H_i, A_i = self.Embedding(H_i, A_i)

        H_i = self.trans(H_i, A_i)

        H_i = torch.mm(self.S_mat, H_i)
        H_i = torch.cat([H_i, F3], dim=-1)

        cnn_tail_input = H_i.reshape([1, h, w, -1]).permute([0, 3, 1, 2])

        cnn_tail_biin = torch.cat([H_0_af, H_0_be], dim=1)

        H_tail = self.GCF(cnn_tail_input, cnn_tail_biin)

        H_tail = self.CNN_tail_layer(H_tail)
        H_tail = F.relu(H_tail)
        final_features = torch.squeeze(H_tail, 0).permute([1, 2, 0]).reshape([h * w, -1])
        final_features = self.ch_ln(final_features)

        Y = self.Softmax(final_features)
        return Y

